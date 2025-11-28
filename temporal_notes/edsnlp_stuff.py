import edsnlp, edsnlp.pipes as eds
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline # type: ignore
from spacy.tokens import Token
import datetime
import pytz
import re
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Any, Set
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Literal  # add Literal
import spacy  # NEW: only for POS/morph if available

# Try to load a spaCy French model for POS/morph features.
# If not available, the tense rule silently skips.
try:
    morph_nlp = spacy.load("fr_dep_news_trf")
except Exception:
    morph_nlp = None
    print("[INFO] No spaCy 'fr_core_news_md' model found. Tense-based TLINKs will be skipped.")

# -------- 1. Texte d'entrée --------
text1 = (
"""Dernière hospitalisation en juin 2024 pour une exacerbation de BPCO non hypercapnique et sans critère de gravité sur probable trigger infectieux d'évolution favorable sous AUGMENTIN et corticothérapie. En raison de la fréquence des exacerbations de l'AZITHROMYCINE avait été introduit.


Majoration de la dyspnée depuis 8 jours accompagnée de toux grasses sans expectoration possible et sans fièvre.

Elle consulte aux urgences le 01/09 devant l'impossibilité d'évacuer ses sécrétions malgré son ALPHA et les séances de kiné respiratoire. L'examen clinique est sans particularité. Les débits d'oxygène sont inchangés. La biologie retrouve une CRP à 5.9 mg/L. Dans ce contexte, il n'est pas introduit d'antibiothérapie ni de corticothérapie.

Observation de fievre pendant 05 semaines succesives.

Transfert dans le service de pneumologie pour la suite de la prise en charge.
"""
)

text2 = (
"""Dernière hospitalisation en janvier 2024 pour une exacerbation cardio-respiratoire avec facteur déclenchant infectieux. Elle a été vu en consultation pneumologique en mars 2024 par Dr BRAVOS, qui avait programmée une hospitalisation pour bilan de VNI en juin.

Majoration de la dyspnée depuis quelques jours avec fièvre et toux grasse au domicile.

Elle consulte aux urgences le 2/06. Pas de majoration des débits de base en O2, pas de signe de détresse respiratoire.

La biologie retrouve une CRP à 5.6 mg/l.

Il est instauré un traitement par AUGMENTIN et solupred.

La patiente ne présentait pas de signe de gravité clinique ou biologique mais devant l'inquiétude de sa fille, elle est hospitalisée en pneumologie.
Le 02/06 la patiente est eupneique sous 1L d'O2, saturation à 97%. Elle est apyrétique.

Murmure vésiculaire atténué de façon diffuse, sibilant expiratoire diffus. Pas de ronchis ni crépitant.

Bdc atténué, pas de souffle perçus. Pas de signe de décompensation cardiaque. Pas d'oedème des membres inférieurs.

Abdomen souple dépressible et indolore. Trouble du transit à type de constipation, connu.

Pas de symptôme fonctionnel urinaire.


**Evolution dans le service:**

Poursuite de l'antibiothérapie par AUGMENTIN probabiliste pour un total de 5 jours. L'examen cyto-biologique des crachats n'a pas mis en évidence de pathogène. Poursuite également de la corticothérapie orale pour un total de trois jours.

Evolution clinique favorable avec franche diminution de la purulence et de la quantité des crachats, diminution de la spasticité et amélioration de la dyspnée. La patiente réalise bien son drainage bronchique par ALPHA 300, elle a également bénéficié d'une prise en charge par le kinésithérapeute du service.

Par ailleurs, la patiente rapporte faire des exacerbations à répétition avec au domicile une prise d'antibiotique environ tous les mois depuis plusieurs mois, raison pour laquelle nous introduisons un traitement par AZITHROMYCINE à raison de trois fois par semaine pour tenter de diminuer la fréquence des exacerbations. L'ECG réalisé à cette occasion ne retrouve pas de QT allongé.

Nous profitons également de cette hospitalisation pour réaliser un bilan respiratoire avec une oxymétrie sous VNI et une gazométrie de réveil.

**L'oxymétrie sous VNI** sous 1 l d'O² est très satisfaisante avec une saturation moyenne à 93 %, une saturation minimum à 91 %.

**La gazométrie de réveil** retrouve un normopH à 7.42, une pCO² légèrement haute à 52, une pO² à 85 sous 1 l.

La lecture de carte de la VNI retrouve :

- Des paramètres suivant : PI 19, PE 6, FR de sécurité à 12, sous 2 L d'oxygène

- Une excellente observance à 8.37H/jour avec une absence de fuite.

- Un volume courant médian satisfaisant à 620 ml

- Un IAH machine à 0.1.

- Au niveau des paramètres fins on retrouve un TI entre 0.6 et 1.4 sec, un taux de déclenchement inspiratoire spontané à 35 %, un taux de déclenchement expiratoire spontané à 33 %. Une fréquence respiratoire médiane à 12.

L'ensemble de ces données sont satisfaisantes, nous ne modifions pas les paramètres de la VNI.

Enfin la patiente a également rencontré l'assistante sociale du service pour faire un point sur les aides à la maison.

**Au total**, exacerbation de BPCO non hypercapnique et sans critère de gravité sur probable trigger infectieux, d'évolution favorable sous antibiothérapie probabiliste et corticothérapie.

Finalement la patiente rejoint son domicile le 10/06. Nous la remettons sous la surveillance de sa pneumologue référente le DR BRAVOS avec qui la prochaine consultation est prévue le 05/12.

Bien confraternellement.
"""
)

text3 = (
"""Motif de prescription : KR:Suivie par insuffisance respiratoire mixte sur asthme et BPCO sous VNI nocturne et 1L/min au repos, 2L/min à l'effort, ALPHA 300. Suivi depuis 2015.
Exacerbation sans trigger identifié avec impossibilité d'évacuation des sécrétions.
10/09/2024 L Czerewyk
1lO2 SpO2 95% 83bpm
KR+Alpha 300
sécrétion bien mobilisé, pas d'expectoration

09/09/2024 L Czerewyk
sous 1l O2 SpO2 97% 78bpm
KR+Apha 300
tousse bcp avec expectoration epaisse

06/09/2024 L Czerewyk
O2 1l, SpO2 94% 87bpm
KR+ alpha 300 avec sérum physiologique
sécretion bien mobilise, pas d'expectoration
crachée le matin selon patient

05/09/2024 vbosle
KR sous 1 l 02 sp02=95% + alfa 300
vient de faire son aérosol:2 sécrétions remontées

04/09/2024 V BOSLE
KR sous 1 L 02 sp02=93%
AFE+Elpr pas de sécrétion remontée
A crachée hier AM pour ECBC me dit elle

03/09/2024 vpetit

KR+ alpha 300, sous 1l/min, pas d'expectoration.

Validé par INTERIM CZEREWYK LILIA
"""
)


text4 = (
""" 
Dernière hospitalisation en juin 2024 pour une exacerbation de BPCO non hypercapnique et sans critère de gravité sur probable trigger infectieux d'évolution favorable sous AUGMENTIN et corticothérapie. En raison de la fréquence des exacerbations, l'AZITHROMYCINE avait été introduit.

Le 01/09/2024, la patiente consulte aux urgences devant l'impossibilité d'évacuer ses sécrétions malgré son ALPHA et les séances de kiné respiratoire. Au total, exacerbation de BPCO/asthme sans surinfection chez une patiente de 90 ans. 

Le 02/06/2024, la patiente est hospitalisée pour une exacerbation de BPCO. L'évolution clinique est favorable avec une franche diminution de la purulence et de la quantité des crachats, diminution de la spasticité et amélioration de la dyspnée. La patiente rapporte faire des exacerbations à répétition avec au domicile une prise d'antibiotique environ tous les mois depuis plusieurs mois, raison pour laquelle un traitement par AZITHROMYCINE a été introduit pour tenter de diminuer la fréquence des exacerbations.

Le 09/09/2024, nous avons tenté un arrêt de la corticothérapie avec récidive immédiate du tableau spastique avec impossibilité d'expectoration. La reprise de la prednisolone a été effectuée le même jour.

Le 31/08/2024, la patiente présente une dyspnée s'intensifiant ces dernières 24 heures, avec une impossibilité d'évacuer ses sécrétions. Elle est hospitalisée en pneumologie pour kiné respiratoire et caugh assist. 

Au total, l'historique des exacerbations de la patiente montre une nécessité d'adaptation du traitement pour mieux gérer ces événements, notamment par l'introduction de l'AZITHROMYCINE et la surveillance de la corticothérapie.
"""
)
# -------- 2. NER HuggingFace (DrBERT-CASM2) pour récupérer les spans --------
MODEL_NAME = "medkit/DrBERT-CASM2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True) # type: ignore 
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME) # type: ignore
ner_pipe = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)

# ======= SUPPRIME l'ancien bloc unique raw_ents/terms/nlp =======
# (Remplacé par make_nlp_for_text pour créer un pipeline indépendant par document)

def make_nlp_for_text(raw_text: str, min_score: float = 0.1) -> Tuple[Any, Dict[str, List[str]]]:
    """
    Construit un pipeline edsnlp local pour un texte donné :
    - NER (DrBERT)
    - Fusion des entités contiguës/splittées (incluant tokens '##')
    - Construction matcher
    """
    raw_ents_local = ner_pipe(raw_text)

    # -------- Fusion entités (gère overlaps + morceaux '##') --------
    def merge_entities(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ents_sorted = sorted(ents, key=lambda e: e["start"])
        merged: List[Dict[str, Any]] = []
        for ent in ents_sorted:
            if "start" not in ent or "end" not in ent:
                continue
            if not merged:
                merged.append(dict(ent))
                continue
            prev = merged[-1]
            same_label = ent.get("entity_group") == prev.get("entity_group")
            contiguous = ent["start"] <= prev["end"] + 1
            subword = ent.get("word","").startswith("##")
            overlap = ent["start"] <= prev["end"]
            if (same_label and (contiguous or overlap)) or (subword and contiguous):
                # Étend la fenêtre
                prev["end"] = max(prev["end"], ent["end"])
                prev["score"] = max(prev.get("score", 0), ent.get("score", 0))
            else:
                merged.append(dict(ent))
        return merged

    raw_ents_local = merge_entities(raw_ents_local)

    spans_by_label: Dict[str, Set[str]] = {}
    for ent in raw_ents_local:
        if ent.get("score", 0) < min_score:
            continue
        start, end = ent["start"], ent["end"]
        span_text = raw_text[start:end]
        # Nettoyage sous-mots '##'
        norm_text = re.sub(r'\s*##', '', span_text)
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        if len(norm_text) < 2:
            continue
        label = ent.get("entity_group", "MISC")
        spans_by_label.setdefault(label, set()).add(norm_text.lower())

    if not spans_by_label:
        spans_by_label = {"dummy": {"placeholder"}}

    terms_local = {label: list(vals) for label, vals in spans_by_label.items()}

    nlp_local = edsnlp.blank("fr") # type: ignore
    nlp_local.add_pipe("eds.normalizer")
    nlp_local.add_pipe("eds.sentences")
    nlp_local.add_pipe(eds.matcher(terms=terms_local, attr="LOWER"))
    nlp_local.add_pipe(eds.sections())
    nlp_local.add_pipe(eds.dates())
    nlp_local.add_pipe("eds.negation")
    nlp_local.add_pipe("eds.hypothesis")
    nlp_local.add_pipe("eds.family")
    nlp_local.add_pipe(eds.history(use_sections=True, use_dates=True))
    return nlp_local, terms_local # type: ignore

# (Optionnel) Fonction de debug pour voir les terms par doc
def debug_terms(doc_id: str, terms: Dict[str, List[str]]):
    print(f"[TERMS] {doc_id}")
    for k, v in terms.items():
        print(f"  {k}: {v}")

# ================== Timeline utilities (deduplicated) ==================
# Data classes & patterns defined once (were duplicated before)

@dataclass
class SimpleEvent:
    id: str
    text: str
    start: int
    end: int
    label: str
    negation: Optional[bool]
    hypothesis: Optional[bool]
    family: Optional[bool]
    history: Optional[bool]
    history_str: Optional[str]
    anchor_date: Optional[str] = None
    source_timex: Optional[str] = None

@dataclass
class SimpleTimex:
    id: str
    text: str
    start: int
    end: int
    kind: str            # DATE | RELATIVE | DURATION
    value: Optional[str] # YYYY-MM-DD si absolu
    interval: Optional[Dict[str, str]] = None
    role: str = "NORMAL" # NORMAL | IGNORE | SKIP
    resolved: bool = False

@dataclass
class ImpactZone:
    date: str
    start: int
    end: int

@dataclass
class RelationEdge:
    source: str
    target: str
    type: str
    evidence: str

# === Règles relatives spécifiques ===
# Reconnaît l'expression "depuis quelques jour(s)" (approximatif)
PAT_DEP_FEW_DAYS = re.compile(r'\bdepuis\s+quelques\s+jour(s)?\b', re.I)
FEW_DAYS_APPROX = 3  # Nombre de jours approximatif pour "quelques jours"

# Patterns
REL_QUANT = re.compile(r'\b(\d+|un|une)\s+(jour|jours|semaine|semaines|mois|an|ans)\b', re.I)
SIGNAL_AFTER = re.compile(r'\b(apres|après|plus tard|suivant(e)?|le lendemain)\b', re.I)
SIGNAL_BEFORE = re.compile(r'\b(avant|plus tot|plus tôt|précédent(e)?|precedent(e)?|la veille)\b', re.I)
J_REL = re.compile(r'\bJ\s*([+-])\s*(\d+)\b', re.I)

UNIT_DAYS = {"jour":1,"jours":1,"semaine":7,"semaines":7,"mois":30,"an":365,"ans":365}

ANCHOR_EVENT_TERMS = {
    "hospitalisation","hospitalisé","hospitalisee","consultation","consultée","consulté",
    "admis","admission","admise","sortie","traitement","perfusion"
}
# ===== Verb & tense helpers (NEW) =====

Relation = Literal["BEFORE", "AFTER", "SIMULTANEOUS"]
TenseClass = Literal["PAST", "PRESENT", "FUTURE", "UNKNOWN"]

PAST_TENSES = {"Imp", "Past", "Pqp"}   # Imparfait, Passé simple/participle tag (Past), Plus-que-parfait
FUTURE_TENSES = {"Fut"}                # Futur
PRESENT_TENSES = {"Pres"}              # Présent

# ===== Improved verb helpers (REPLACE your _get_finite_verb / _classify_tense) =====
Relation = Literal["BEFORE", "AFTER", "SIMULTANEOUS"]
TenseClass = Literal["PAST", "PRESENT", "FUTURE", "UNKNOWN"]


def _is_finite(tok: Token) -> bool:
    return True and tok.pos_ in {"VERB", "AUX"} and "VerbForm=Fin" in tok.morph

def _aux_chain_recursive(tok: Token) -> List[Token]:
    """
    Collect AUX tokens connected to tok both via children and via HEAD recursively.
    Handles chains like: introduit(Part) <- été(Part) <- avait(Fin,Imp)
    """
    chain = []
    seen = {tok}
    stack = [tok]
    while stack:
        t = stack.pop()
        # children that are AUX or aux deps
        for c in t.children:
            if (c.pos_ == "AUX" or c.dep_ in {"aux", "aux:pass"}) and c not in seen:
                chain.append(c); seen.add(c); stack.append(c)
        # head that is AUX or aux dep
        h = t.head
        if  (h.pos_ == "AUX" or h.dep_ in {"aux", "aux:pass"}) and h not in seen:
            chain.append(h); seen.add(h); stack.append(h)
    return chain

def _get_governing_token(event_head: Optional[Token]) -> Optional[Token]:
    """
    Prefer a governing VERB/AUX by climbing heads; return the first finite one if any,
    otherwise the nearest VERB/AUX (even if non-finite). Only fall back to the original
    token if no verb/aux exists.
    """
    if event_head is None:
        return None
    cur = event_head
    visited: Set[Token] = set()
    first_verb_or_aux = None
    while cur is not None and cur not in visited:
        visited.add(cur)
        if cur.pos_ in {"VERB", "AUX"}:
            if first_verb_or_aux is None:
                first_verb_or_aux = cur
            if _is_finite(cur):
                return cur  # best case
        nxt = cur.head if cur.head is not cur else None
        cur = nxt
    # fallback: non-finite verb/aux if we saw one, else the event head itself
    return first_verb_or_aux if first_verb_or_aux is not None else event_head

def _classify_tense(governor: Optional[Token]) -> TenseClass: # type: ignore
    """
    Tense logic:
    - Finite AUX + participle child → PAST (unless AUX Fut → FUTURE)
    - Finite AUX + infinitive child → FUTURE (futur proche)
    - Non-finite (Part/Inf): inspect AUX chain (parents/children):
        * any AUX Fut → FUTURE
        * any AUX Pres/Imp/Past (+ Part governor) → PAST
    - Else: use governor's own Tense (Pres/Fut/Imp/Past).
    Also: if governor itself is Part with Tense=Past → PAST (spaCy sometimes sets this).
    """
    if governor is None:
        return "UNKNOWN"

    gv_tense = set(governor.morph.get("Tense"))
    gv_form  = set(governor.morph.get("VerbForm"))
    gv_mood  = set(governor.morph.get("Mood"))

    if "Cond" in gv_mood:
        return "UNKNOWN"

    def has_child_form(form: str) -> bool:
        return any(form in set(c.morph.get("VerbForm")) for c in governor.children if c.pos_ in {"VERB","AUX"})

    chain = _aux_chain_recursive(governor)

    # Finite AUX as governor
    if _is_finite(governor) and governor.pos_ == "AUX":
        if has_child_form("Part"):
            return "FUTURE" if "Fut" in gv_tense else "PAST"
        if has_child_form("Inf"):
            return "FUTURE"

    # Non-finite governor (e.g., participle 'introduit')
    if {"Part","Inf","Ger"} & gv_form:
        # direct hint: some models put Tense=Past on participles
        if "Part" in gv_form and "Past" in gv_tense:
            return "PAST"
        chain_tenses = [set(a.morph.get("Tense")) for a in chain]
        if any("Fut" in t for t in chain_tenses):
            return "FUTURE"
        if any(t & (PRESENT_TENSES | PAST_TENSES) for t in chain_tenses):
            return "PAST"
        return "UNKNOWN"

    # Simple finite non-AUX
    if "Fut" in gv_tense:
        return "FUTURE"
    if "Pres" in gv_tense:
        return "PRESENT"
    if gv_tense & PAST_TENSES:
        return "PAST"
    return "UNKNOWN"

def _tense_to_relation(t: TenseClass) -> Optional[Relation]:
    return {"PRESENT":"SIMULTANEOUS","PAST":"BEFORE","FUTURE":"AFTER","UNKNOWN":None}[t]


def _add_cross_sentence_tense_tlinks(raw_text: str,
                                     events: List[SimpleEvent],
                                     timexes: List[SimpleTimex],
                                     edges: List[RelationEdge]) -> None:
    """
    For events already linked to a timex (via source_timex) but NOT in the same sentence,
    infer BEFORE/AFTER/SIMULTANEOUS from the governing verb tense and add a TLINK.
    Skips if spaCy morph model is unavailable.
    """
    if morph_nlp is None:
        return

    morph_doc = morph_nlp(raw_text)

    # quick helper for sentence equality on morph_doc
    def same_sentence(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        a = morph_doc.char_span(a_start, a_end, alignment_mode="expand")
        b = morph_doc.char_span(b_start, b_end, alignment_mode="expand")
        if a is None or b is None:
            return False
        return a.sent == b.sent

    for ev in events:
        if not ev.source_timex:
            continue
        # find that timex
        tx = next((t for t in timexes if t.id == ev.source_timex), None)
        if tx is None:
            continue

        # Only when not in the same sentence
        if same_sentence(ev.start, ev.end, tx.start, tx.end):
            continue

        # skip negation/hypothesis/family (as per your policy)
        if ev.negation or ev.hypothesis or ev.family:
            continue

        ev_span = morph_doc.char_span(ev.start, ev.end, alignment_mode="expand")
        if ev_span is None:
            continue

        head = ev_span.root
        gov = _get_governing_token(head)            # NEW: accept participle/infinitive
        tclass = _classify_tense(gov)               # NEW: uses recursive AUX chain
        rel = _tense_to_relation(tclass)
        if rel is None:
            continue
        
        # Remove any EVENT_TIME* edges between this event and this timex
# (covers EVENT_TIME, EVENT_TIME_CAND, EVENT_TIME_PREV, EVENT_TIME_NEXT)
        edges[:] = [
            e for e in edges
            if not (e.source == ev.id and e.target == tx.id and e.type.startswith("EVENT_TIME"))
        ]
        edges.append(RelationEdge(
            source=ev.id,
            target=tx.id,
            type=f"TLINK_{rel}",  # e.g., TLINK_BEFORE / TLINK_AFTER / TLINK_SIMULTANEOUS
            evidence="governing_verb_tense"
        ))

def build_timeline_from_text(
    nlp,
    raw_text: str,
    doc_id: str,
    note_datetime: Optional[datetime.datetime],
    out_dir: Path = Path("data/graph_per_documents"),
    save: bool = True,
):
    """
    Construit la timeline pour un texte.
    out_dir : dossier UNIQUE où sauvegarder les JSON (par défaut data/graph_per_documents).
    Aucun JSON n'est écrit dans graphs/ (réservé aux HTML de visualisation).
    """
    out_dir = Path(out_dir)
    doc = nlp(raw_text)
    if note_datetime:
        doc._.note_datetime = note_datetime

    # Timex extraction
    timexes: List[SimpleTimex] = []
    tid = 0
    dates_sp = doc.spans.get("dates", [])
    dur_sp = doc.spans.get("durations", [])

    for s in dates_sp:
        tid += 1
        dt = s._.date.to_datetime(
            note_datetime=note_datetime,
            infer_from_context=True,
            tz="Europe/Paris",
            default_day=15,
        )
        value = dt.date().isoformat() if dt else None
        timexes.append(SimpleTimex(
            id=f"T{tid}", text=s.text, start=s.start_char, end=s.end_char,
            kind="DATE" if value else "RELATIVE", value=value, interval=None, resolved=bool(value)
        ))

    for s in dur_sp:
        tid += 1
        timexes.append(SimpleTimex(
            id=f"T{tid}", text=s.text, start=s.start_char, end=s.end_char,
            kind="DURATION", value=None, interval=None, resolved=False
        ))

    # Sentence helper
    def sentence_of(offset: int):
        for sent in doc.sents:
            if sent.start_char <= offset < sent.end_char:
                return sent
        return None

    # Roles
    for t in timexes:
        sent = sentence_of(t.start)
        seg = (sent.text if sent else "").lower()
        if any(k in seg for k in ["reçu","réception","reception","obtenu"]) and any(k in seg for k in ["rapport","information","cas","résumé","resume","summary"]):
            t.role = "IGNORE"
        if "antécédent" in seg or "antécédents" in seg:
            t.role = "SKIP"

    # Events
    events: List[SimpleEvent] = []
    eid = 0
    for ent in doc.ents:
        if ent.label_.lower() not in ("problem","treatment","test"):
            continue
        eid += 1
        events.append(SimpleEvent(
            id=f"E{eid}", text=ent.text, start=ent.start_char, end=ent.end_char,
            label=ent.label_,
            negation=getattr(ent._,"negation",None),
            hypothesis=getattr(ent._,"hypothesis",None),
            family=getattr(ent._,"family",None),
            history=getattr(ent._,"history",None),
            history_str=getattr(ent._,"history_",None),
        ))

    # Distance heuristic
    def distance_score(ev: SimpleEvent, tx: SimpleTimex) -> Tuple[int,int,int,int]:
        a, b = sorted([(ev.start, ev.end), (tx.start, tx.end)], key=lambda x: x[0])
        segment = doc.text[a[1]:b[0]]
        return (segment.count(";"),
                len(re.findall(r'\b(et|and)\b', segment, re.I)),
                segment.count(","),
                len(segment.split()))

    edges: List[RelationEdge] = []

    # Direct associations (SUPPR suppression du filtre role)
    for ev in events:
        sent = sentence_of(ev.start)
        if not sent:
            continue
        cands = [t for t in timexes if t.start >= sent.start_char and t.end <= sent.end_char]
        if not cands:
            continue
        cands.sort(key=lambda t: distance_score(ev, t))
        best = cands[0]
        edges.append(RelationEdge(ev.id, best.id, "EVENT_TIME_CAND", "same sentence"))
        if best.kind == "DATE" and best.resolved and best.value:
            ev.anchor_date = best.value
            ev.source_timex = best.id
            edges[-1].type = "EVENT_TIME"

    # Helper last absolute (retire filtre role == IGNORE)
    def last_abs_before(pos: int) -> Optional[str]:
        abs_dates = [t for t in timexes if t.resolved and t.value and t.kind == "DATE"]
        if abs_dates:
            abs_dates.sort(key=lambda x: pos - x.start)
            return abs_dates[0].value
        return note_datetime.date().isoformat() if note_datetime else None

    # Relative resolution (retire test role == IGNORE)
    for t in timexes:
        if t.resolved:
            continue
        low = t.text.lower()

        # --- NEW: "depuis quelques jours" -> DURATION approximative se terminant à note_datetime ---
        if PAT_DEP_FEW_DAYS.search(low):
            ref = note_datetime.date().isoformat() if note_datetime else last_abs_before(t.start)
            if ref:
                end_dt = datetime.datetime.fromisoformat(ref).date()
                start_dt = (end_dt - datetime.timedelta(days=FEW_DAYS_APPROX)).isoformat()
                t.kind = "DURATION"
                t.interval = {"start": start_dt, "end": end_dt.isoformat() if hasattr(end_dt, 'isoformat') else ref}
                t.value = end_dt.isoformat() if hasattr(end_dt, 'isoformat') else ref
                t.resolved = True
                continue

        # J+/-N
        m = J_REL.search(low)
        if m:
            sign, num = m.group(1), int(m.group(2))
            ref = last_abs_before(t.start)
            if ref:
                base = datetime.datetime.fromisoformat(ref)
                delta = num if sign == "+" else -num
                t.value = (base + datetime.timedelta(days=delta)).date().isoformat()
                t.kind = "DATE"; t.resolved = True
                continue
        # Quantified relative
        q = REL_QUANT.search(low)
        if q:
            qty_raw, unit = q.group(1).lower(), q.group(2).lower()
            qty = 1 if qty_raw in ("un","une") else int(qty_raw)
            days = qty * UNIT_DAYS.get(unit, 0)
            if days:
                ref = last_abs_before(t.start)
                if ref:
                    base = datetime.datetime.fromisoformat(ref)
                    if "depuis" in low:
                        start_dt = (base - datetime.timedelta(days=days)).date().isoformat()
                        end_dt = base.date().isoformat()
                        t.kind = "DURATION"
                        t.interval = {"start": start_dt, "end": end_dt}
                        t.value = end_dt
                        t.resolved = True
                    else:
                        direction = +1 if SIGNAL_AFTER.search(low) else (-1 if SIGNAL_BEFORE.search(low) else 0)
                        if direction:
                            t.value = (base + datetime.timedelta(days=direction * days)).date().isoformat()
                            t.kind = "DATE"; t.resolved = True

    # -------- NEW RULE: backfill events with the closest previous timex in the paragraph --------
    for ev in events:
        if ev.anchor_date:
            continue
        already = any(e.source == ev.id and e.type.startswith("EVENT_TIME") for e in edges)
        if already:
            continue
        prev_candidates = [t for t in timexes if t.start < ev.start]
        if not prev_candidates:
            continue
        prev_candidates.sort(
            key=lambda tt: (
                0 if (tt.resolved and tt.value) else 1,
                ev.start - tt.start
            )
        )
        chosen = prev_candidates[0]
        edges.append(RelationEdge(ev.id, chosen.id, "EVENT_TIME_PREV", "previous timex in paragraph"))
        if chosen.resolved and chosen.value:
            ev.anchor_date = chosen.value
            ev.source_timex = chosen.id

    # -------- NEW RULE: single-timex paragraph association --------
    # Si le paragraphe ne contient qu'un seul timex (role != IGNORE), l'associer à tous les events.
    # - Events avant le timex reçoivent EVENT_TIME_NEXT (analogue à EVENT_TIME_PREV mais vers le futur)
    # - Events après (ou chevauchant) : EVENT_TIME (si résolu) sinon EVENT_TIME_CAND
    valid_timexes = list(timexes)
    if len(valid_timexes) == 1:
        single_tx = valid_timexes[0]
        for ev in events:
            # déjà lié ?
            if any(r.source == ev.id and r.target == single_tx.id and r.type.startswith("EVENT_TIME")
                   for r in edges):
                continue
            if ev.start < single_tx.start:
                rel_type = "EVENT_TIME_NEXT"
            else:
                rel_type = "EVENT_TIME" if (single_tx.resolved and single_tx.value) else "EVENT_TIME_CAND"
            edges.append(RelationEdge(ev.id, single_tx.id, rel_type, "single timex paragraph"))
            # ancrage si possible
            if rel_type in ("EVENT_TIME", "EVENT_TIME_NEXT") and single_tx.resolved and single_tx.value and not ev.anchor_date:
                ev.anchor_date = single_tx.value
                ev.source_timex = single_tx.id

    # Impact zones (retire filtre role)
    abs_seq = [t for t in timexes if t.resolved and t.value and t.kind == "DATE"]
    abs_seq.sort(key=lambda x: x.start)
    zones: List[ImpactZone] = []
    for i, t in enumerate(abs_seq):
        z_start = t.start
        z_end = abs_seq[i + 1].start if i + 1 < len(abs_seq) else len(doc.text)
        zones.append(ImpactZone(t.value, z_start, z_end))

    # Final anchor assignment (inchangé, fonctionne avec tous les timex)
    for ev in events:
        if ev.anchor_date:
            continue
        linked_ids = [e.target for e in edges if e.source == ev.id and e.type.startswith("EVENT_TIME")]
        for tx_id in linked_ids:
            tt = next((tt for tt in timexes if tt.id == tx_id), None)
            if tt and tt.resolved and tt.value:
                ev.anchor_date = tt.value
                ev.source_timex = tt.id
                break
        if ev.anchor_date:
            continue
        for z in zones:
            if z.start <= ev.start < z.end:
                ev.anchor_date = z.date
                break

    # --- NEW RULE: liens PROBLEM–TREATMENT dans la même phrase ---
    # Pour chaque phrase : relier chaque couple (problem, treatment).
    # Type différent selon cohérence de la négation.
    sent_spans = list(doc.sents)
    existing_pt_pairs = set()
    for sent in sent_spans:
        sent_events = [e for e in events if sent.start_char <= e.start < sent.end_char]
        problems = [e for e in sent_events if e.label.lower() == "problem"]
        treatments = [e for e in sent_events if e.label.lower() == "treatment"]
        if not problems or not treatments:
            continue
        for pe in problems:
            for te in treatments:
                pair_key = tuple(sorted([pe.id, te.id]))
                if pair_key in existing_pt_pairs:
                    continue
                existing_pt_pairs.add(pair_key)
                same_neg = (pe.negation == te.negation)
                r_type = "PROBLEM_TREATMENT_SAME_NEG" if same_neg else "PROBLEM_TREATMENT_DIFF_NEG"
                edges.append(RelationEdge(
                    pe.id,
                    te.id,
                    r_type,
                    "same sentence"
                ))

    # BEFORE edges
    chron = [e for e in events if e.anchor_date]
    chron.sort(key=lambda e: (e.anchor_date, e.start))
    for a, b in zip(chron, chron[1:]):
        if a.anchor_date < b.anchor_date:
            edges.append(RelationEdge(a.id, b.id, "BEFORE", "chronological order"))
    
        # --- NEW: cross-sentence TLINKs driven by verb tense ---
    try:
        _add_cross_sentence_tense_tlinks(raw_text, events, timexes, edges)
    except Exception as ex:
        # guardrail: never break the pipeline due to this optional step
        print(f"[WARN] tense TLINKs skipped due to error: {ex}")


    timeline = {
        "doc_id": doc_id,
        "note_datetime": note_datetime.date().isoformat() if note_datetime else None,
        "events": [asdict(e) for e in events],
        "timex": [asdict(t) for t in timexes],
        "relations": [asdict(r) for r in edges],
    }

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"timeline_rules_simple_{doc_id}.json"
        out_path.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {doc_id} -> {out_path}")
    return timeline

def split_paragraphs(raw_text: str) -> List[Tuple[str, int, int]]:
    """
    Retourne une liste de tuples (paragraph_text, start_offset, idx)
    Un paragraphe = blocs séparés par >=1 ligne vide.
    """
    parts = []
    cursor = 0
    idx = 0
    # Normalisation fin de lignes
    blocks = re.split(r'\n\s*\n+', raw_text)
    pos = 0
    for block in blocks:
        block_stripped = block.strip()
        if not block_stripped:
            # avancer le curseur d'autant (inclure séparateurs)
            pos += len(block) + 2
            continue
        # Trouver l'offset réel de ce bloc dans le texte original (cherche depuis pos)
        start_in_doc = raw_text.find(block, pos)
        if start_in_doc == -1:
            start_in_doc = pos
        parts.append((block_stripped, start_in_doc, idx))
        idx += 1
        pos = start_in_doc + len(block)
    return parts

def build_timeline_paragraphwise(
    raw_text: str,
    doc_id: str,
    note_datetime: Optional[datetime.datetime],
    paragraph_max: Optional[int] = None,
    out_dir: Path = Path("data/graph_per_documents"),
    save: bool = True,
):
    """
    Traite le document paragraphe par paragraphe.
    Chaque paragraphe est NER + timeline isolée (dates relatives résolues UNIQUEMENT dans le paragraphe).
    Les timelines de paragraphes sont fusionnées ensuite (offsets réalignés).
    """
    paragraphs = split_paragraphs(raw_text)
    if paragraph_max:
        paragraphs = paragraphs[:paragraph_max]

    all_events: List[Dict] = []
    all_timex: List[Dict] = []
    all_edges: List[Dict] = []

    global_eid = 0
    global_tid = 0

    for p_text, base_offset, p_idx in paragraphs:
        # Pipeline spécifique sur le paragraphe
        local_nlp, _ = make_nlp_for_text(p_text)
        tl = build_timeline_from_text(
            local_nlp,
            p_text,
            f"{doc_id}_p{p_idx}",
            note_datetime,
            out_dir=Path("_ignore_tmp"),  # évite création
            save=False,
        )

        # Re-numérotation timex
        timex_id_map = {}
        for t in tl["timex"]:
            global_tid += 1
            new_id = f"T{global_tid}"
            timex_id_map[t["id"]] = new_id
            t["id"] = new_id
            t["start"] += base_offset
            t["end"] += base_offset
            all_timex.append(t)

        # Re-numérotation events
        event_id_map = {}
        for e in tl["events"]:
            global_eid += 1
            new_id = f"E{global_eid}"
            event_id_map[e["id"]] = new_id
            e["id"] = new_id
            e["start"] += base_offset
            e["end"] += base_offset
            # Remap source_timex si présent
            if e.get("source_timex") and e["source_timex"] in timex_id_map:
                e["source_timex"] = timex_id_map[e["source_timex"]]
            all_events.append(e)

        # Zones (dates d'impact)
        for z in tl.get("zones", []):
            z["start"] += base_offset
            z["end"] += base_offset
            all_zones.append(z)

        # Relations (conserver uniquement intra-paragraphe; pas de BEFORE cross-paragraph)
        for r in tl["relations"]:
            s = r["source"]; t = r["target"]
            if s.startswith("E") and s in event_id_map:
                r["source"] = event_id_map[s]
            if t.startswith("E") and t in event_id_map:
                r["target"] = event_id_map[t]
            if s.startswith("T") and s in timex_id_map:
                r["source"] = timex_id_map[s]
            if t.startswith("T") and t in timex_id_map:
                r["target"] = timex_id_map[t]
            all_edges.append(r)

    # --- NEW RULE: Anchor orphan events to closest previous absolute timex ---
    # Build a list of (paragraph_idx, [timex_ids], [timex_values]) for absolute timex
    para_abs_timex = []
    para_events = []
    for p_idx, (p_text, base_offset, _) in enumerate(paragraphs):
        # Find events and timex in this paragraph
        para_ev = [e for e in all_events if base_offset <= e["start"] < base_offset + len(p_text)]
        para_tx = [t for t in all_timex if base_offset <= t["start"] < base_offset + len(p_text)]
        abs_tx = [t for t in para_tx if t["kind"] == "DATE" and t.get("resolved") and t.get("value")]
        para_abs_timex.append(abs_tx)
        para_events.append(para_ev)

    for idx, (evs, txs) in enumerate(zip(para_events, para_abs_timex)):
        if txs:
            continue  # Paragraph has at least one absolute timex, skip
        # Find previous paragraph with absolute timex
        prev_idx = idx - 1
        while prev_idx >= 0 and not para_abs_timex[prev_idx]:
            prev_idx -= 1
        if prev_idx < 0:
            continue  # No previous absolute timex found
        # Use the last absolute timex in the previous paragraph
        prev_abs = para_abs_timex[prev_idx][-1]
        for e in evs:
            if not e.get("anchor_date"):
                e["anchor_date"] = prev_abs["value"]
                e["source_timex"] = prev_abs["id"]
                # Add a relation for traceability
                all_edges.append({
                    "source": e["id"],
                    "target": prev_abs["id"],
                    "type": "EVENT_TIME_PREV",
                    "evidence": "closest previous absolute timex (cross-paragraph)"
                })

    merged_timeline = {
        "doc_id": doc_id,
        "note_datetime": note_datetime.date().isoformat() if note_datetime else None,
        "paragraph_mode": True,
        "events": all_events,
        "timex": all_timex,
        "relations": all_edges,
    }

    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"timeline_rules_simple_{doc_id}.json"
        out_path.write_text(json.dumps(merged_timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK][PARA] {doc_id} -> {out_path}")

    return merged_timeline

# -------- Multi-text processing (indépendant) --------
multi_texts = [
    ("doc1", text1, datetime.datetime(2024, 9, 2, tzinfo=pytz.timezone("Europe/Paris"))),
    ("doc2", text2, datetime.datetime(2024, 6, 10, tzinfo=pytz.timezone("Europe/Paris"))),
    ("doc3", text3, datetime.datetime(2024, 6, 10, tzinfo=pytz.timezone("Europe/Paris"))),
    ("doc4", text4, datetime.datetime(2024, 9, 2, tzinfo=pytz.timezone("Europe/Paris"))),
]

# Remplace l’appel précédent par le mode paragraphe
for doc_id, raw_txt, ndt in multi_texts:
    build_timeline_paragraphwise(
        raw_txt,
        doc_id,
        ndt,
        paragraph_max=None,              # limiter p.ex. à 5 si besoin
        out_dir=Path("data/graph_per_documents"),
        save=True,
    )
