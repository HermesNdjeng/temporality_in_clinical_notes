# -*- coding: utf-8 -*-
"""
Demo: Cross-sentence TLINKs from verb tense (French) — FIXED
- Accepts non-finite heads (e.g., participles) and climbs the AUX chain.
- Adds TLINK_BEFORE/AFTER/SIMULTANEOUS based on tense.
- Verbose prints to inspect decisions.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Literal

import spacy

def load_fr_model():
    for name in ["fr_dep_news_trf", "fr_core_news_md", "fr_core_news_sm"]:
        try:
            print(f"[INFO] Trying spaCy model: {name}")
            return spacy.load(name), name
        except Exception:
            continue
    print("[ERROR] No French spaCy model is installed.")
    print("        Install one, e.g.:  python -m spacy download fr_core_news_md")
    raise SystemExit(1)

NLP, NLP_NAME = load_fr_model()
print(f"[OK] Using spaCy model: {NLP_NAME}")

Relation = Literal["BEFORE", "AFTER", "SIMULTANEOUS"]
TenseClass = Literal["PAST", "PRESENT", "FUTURE", "UNKNOWN"]

@dataclass
class SimpleEvent:
    id: str
    text: str
    start: int
    end: int
    source_timex: Optional[str] = None
    negation: bool = False
    hypothesis: bool = False
    family: bool = False

@dataclass
class SimpleTimex:
    id: str
    text: str
    start: int
    end: int

@dataclass
class RelationEdge:
    source: str
    target: str
    type: str
    evidence: str

PAST_TENSES    = {"Imp", "Past", "Pqp"}   # Imparfait, Passé(simple/part), Plus-que-parfait (rare tag)
FUTURE_TENSES  = {"Fut"}
PRESENT_TENSES = {"Pres"}

def is_finite(tok) -> bool:
    return tok.pos_ in {"VERB", "AUX"} and "VerbForm=Fin" in tok.morph

def aux_chain_recursive(tok) -> List["spacy.tokens.Token"]:
    """
    Collect AUX tokens connected to tok (children and AUX head), recursively,
    so we see chains like: introduit(Part) <- été(Part) <- avait(Fin,Imp)
    """
    chain = []
    seen = {tok}
    stack = [tok]
    while stack:
        t = stack.pop()
        # children that are AUX/aux:pass
        for c in t.children:
            if (c.pos_ == "AUX" or c.dep_ in {"aux", "aux:pass"}) and c not in seen:
                chain.append(c)
                seen.add(c)
                stack.append(c)
        # sometimes the AUX can be the HEAD of a participle:
        h = t.head
        if h is not None and (h.pos_ == "AUX" or h.dep_ in {"aux", "aux:pass"}) and h not in seen:
            chain.append(h)
            seen.add(h)
            stack.append(h)
    return chain

def get_governing_token(event_head) -> Optional["spacy.tokens.Token"]:
    """
    Prefer a governing finite VERB/AUX by climbing heads; if none, return the event head itself
    (even if non-finite Part/Inf). This lets us infer tense from auxiliaries.
    """
    if event_head is None:
        return None
    cur = event_head
    visited = set()
    while cur is not None and cur not in visited:
        visited.add(cur)
        if is_finite(cur):
            return cur
        nxt = cur.head if cur.head is not cur else None
        cur = nxt
    # fallback: use the event head itself (e.g., 'introduit' Part)
    return event_head

def classify_tense(governor) -> TenseClass:
    """
    Tense logic:
    - If finite AUX with participle child → Past (unless AUX Fut → Future)
    - If finite AUX with infinitive child → Future (futur proche)
    - If non-finite VERB (Part/Inf), read AUX chain (parents/children) to infer:
        * AUX Fut → Future
        * AUX Pres/Imp/Past + Part → Past
    - Else, use governor's own Tense (Pres/Fut/Imp/Past/Pqp).
    """
    if governor is None:
        return "UNKNOWN"

    gv_tense = set(governor.morph.get("Tense"))
    gv_form  = set(governor.morph.get("VerbForm"))
    gv_mood  = set(governor.morph.get("Mood"))

    # Conditional: let hypothesis module resolve — skip
    if "Cond" in gv_mood:
        return "UNKNOWN"

    # Utility
    def has_child_with_form(form: str) -> bool:
        return any(form in set(c.morph.get("VerbForm")) for c in governor.children if c.pos_ in {"VERB","AUX"})

    chain = aux_chain_recursive(governor)

    # Case A: finite AUX as governor
    if is_finite(governor) and governor.pos_ == "AUX":
        if has_child_with_form("Part"):
            # Futur antérieur if AUX Fut; else passé composé/plus-que-parfait
            if "Fut" in gv_tense:
                return "FUTURE"
            return "PAST"
        if has_child_with_form("Inf"):
            return "FUTURE"

    # Case B: non-finite governor (e.g., participle 'introduit')
    if {"Part","Inf","Ger"} & gv_form:
        # Look across the AUX chain for a finite AUX
        chain_tenses = [set(a.morph.get("Tense")) for a in chain]
        # If any AUX is Fut → Future (futur antérieur/proche)
        if any("Fut" in t for t in chain_tenses):
            return "FUTURE"
        # If we see Présent/Imparfait/Past on an AUX with a participle governor → Past
        if any( (t & (PRESENT_TENSES | PAST_TENSES)) for t in chain_tenses ):
            return "PAST"
        return "UNKNOWN"

    # Case C: simple finite verb (non-AUX) — use its own tense
    if "Fut" in gv_tense:
        return "FUTURE"
    if "Pres" in gv_tense:
        return "PRESENT"
    if gv_tense & PAST_TENSES:
        return "PAST"
    return "UNKNOWN"

def tense_to_relation(t: TenseClass) -> Optional[Relation]:
    return {
        "PRESENT": "SIMULTANEOUS",
        "PAST": "BEFORE",
        "FUTURE": "AFTER",
        "UNKNOWN": None,
    }[t]

def add_cross_sentence_tense_tlinks(raw_text: str,
                                    events: List[SimpleEvent],
                                    timexes: List[SimpleTimex]) -> List[RelationEdge]:
    edges: List[RelationEdge] = []
    doc = NLP(raw_text)

    def same_sentence(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        a = doc.char_span(a_start, a_end, alignment_mode="expand")
        b = doc.char_span(b_start, b_end, alignment_mode="expand")
        if a is None or b is None:
            return False
        return a.sent == b.sent

    for ev in events:
        if not ev.source_timex:
            print(f"\n[SKIP] {ev.id}: no source_timex → precondition not met.")
            continue
        tx = next((t for t in timexes if t.id == ev.source_timex), None)
        if tx is None:
            print(f"\n[SKIP] {ev.id}: source_timex {ev.source_timex} not found.")
            continue

        print("\n" + "="*80)
        print(f"[EVENT] {ev.id}: '{ev.text}' [{ev.start}:{ev.end}]")
        print(f"[TIMEX] {tx.id}: '{tx.text}' [{tx.start}:{tx.end}]")

        if same_sentence(ev.start, ev.end, tx.start, tx.end):
            print("[INFO] Event and Timex are in the SAME sentence → rule does not fire.")
            continue

        if ev.negation or ev.hypothesis or ev.family:
            print("[INFO] Event flagged (negation/hypothesis/family) → skipping.")
            continue

        ev_span = doc.char_span(ev.start, ev.end, alignment_mode="expand")
        if ev_span is None:
            print("[WARN] Could not create char_span for event → skipping.")
            continue

        head = ev_span.root
        gov = get_governing_token(head)
        print(f"[HEAD] event root: '{head.text}'  (pos={head.pos_}, morph={head.morph})")
        if gov is not None:
            print(f"[GOV ] chosen governor: '{gov.text}' (pos={gov.pos_}, morph={gov.morph})")
        else:
            print(f"[GOV ] chosen governor: <none>")

        chain = aux_chain_recursive(gov) if gov is not None else []
        if chain:
            print("[AUX ] chain (parents/children):")
            for a in chain:
                print(f"       - '{a.text}' (pos={a.pos_}, morph={a.morph})")
        else:
            print("[AUX ] chain: <empty>")

        tclass = classify_tense(gov)
        rel = tense_to_relation(tclass)
        print(f"[CLASSIFY] tense={tclass} → relation={rel}")

        if rel is None:
            print("[INFO] relation is None (UNKNOWN/COND) → no TLINK added.")
            continue

        edge = RelationEdge(
            source=ev.id,
            target=tx.id,
            type=f"TLINK_{rel}",
            evidence="governing_verb_tense"
        )
        edges.append(edge)
        print(f"[EDGE] + {edge}")

    print("\n" + "-"*80)
    print("[RESULT] Edges added:")
    for e in edges:
        print(f"  {e.source} --{e.type}--> {e.target}  ({e.evidence})")
    print("-"*80 + "\n")
    return edges

# ----------------- Test on your example -----------------
text1 = (
"""Dernière hospitalisation en juin 2024 pour une exacerbation de BPCO non hypercapnique et sans critère de gravité sur probable trigger infectieux d'évolution favorable sous AUGMENTIN et corticothérapie. En raison de la fréquence des exacerbations de l'AZITHROMYCINE avait été introduit.

Majoration de la dyspnée depuis 8 jours accompagnée de toux grasses sans expectoration possible et sans fièvre.

Elle consulte aux urgences le 01/09 devant l'impossibilité d'évacuer ses sécrétions malgré son ALPHA et les séances de kiné respiratoire. L'examen clinique est sans particularité. Les débits d'oxygène sont inchangés. La biologie retrouve une CRP à 5.9 mg/L. Dans ce contexte, il n'est pas introduit d'antibiothérapie ni de corticothérapie.

Observation de fievre pendant 05 semaines succesives.

Transfert dans le service de pneumologie pour la suite de la prise en charge.
"""
)

# TIMEX: "juin 2024"
m_timex = re.search(r"\bjuin\s+2024\b", text1, flags=re.IGNORECASE)
if not m_timex:
    raise RuntimeError("Could not find 'juin 2024' in text1.")
T1 = SimpleTimex(
    id="T1",
    text=text1[m_timex.start():m_timex.end()],
    start=m_timex.start(),
    end=m_timex.end()
)

# EVENT: "AZITHROMYCINE ... avait été introduit"
m_event = re.search(r"azithromycine[^.]*avait été introduit[e]?", text1, flags=re.IGNORECASE)
if not m_event:
    raise RuntimeError("Could not find the 'AZITHROMYCINE ... avait été introduit' event in text1.")
E1 = SimpleEvent(
    id="E1",
    text=text1[m_event.start():m_event.end()],
    start=m_event.start(),
    end=m_event.end(),
    source_timex="T1"  # pre-linked to the timex
)

print("\n" + "#"*80)
print("# TEST: text1 — expect plus-que-parfait → PAST → TLINK_BEFORE")
print("#"*80)

_ = add_cross_sentence_tense_tlinks(text1, [E1], [T1])
