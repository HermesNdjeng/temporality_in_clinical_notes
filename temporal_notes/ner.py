# TODO: REFACTOR THIS CODE; too much to type ignore
from pathlib import Path
import re, json, datetime
from typing import List, Tuple, Dict, Any, Optional

import edsnlp   # pyright: ignore[reportMissingTypeStubs]
from spacy.language import Language 
from spacy.tokens import Doc
from transformers import pipeline, AutoTokenizer  # type: ignore[reportMissingTypeStubs]

PAGE_BREAK = "\f"  # keep page boundaries internally

# ---------------------------
# Config
# ---------------------------
MAX_TOKENS = 128      # model context (per model card)
STRIDE_TOKENS = 32    # token overlap
EVENT_TIMEX_MAX_DIST = 200  # max char distance to link event -> timex (fallback if not same sentence)

# ---------------------------
# 1) Markdown cleaning
# ---------------------------
def read_md(md_path: Path) -> str:
    t = md_path.read_text(encoding="utf-8")
    t = re.sub(r"```.*?```", "", t, flags=re.S)      # code fences
    t = re.sub(r"`[^`]+`", "", t)                    # inline code
    t = t.replace("**", "").replace("* ", "- ")      # formatting
    lines: List[str] = []
    for raw in t.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", raw.strip())
        if m:
            title = m.group(2).strip()
            if re.match(r"(?i)^page\s+\d+$", title):
                lines.append(PAGE_BREAK)
            else:
                lines.append(title)
        else:
            lines.append(raw)
    t = "\n".join(lines).replace("☐", "").replace("○", "-")
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ---------------------------
# 2) NLP components
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained("medkit/DrBERT-CASM2", use_fast=True) #type:ignore
hf_ner = pipeline(
    "token-classification",
    model="medkit/DrBERT-CASM2",
    tokenizer=tokenizer, # type:ignore
    aggregation_strategy="simple",
)

def build_eds_dates() -> Language:
    nlp = edsnlp.blank("fr") # type: ignore
    nlp.add_pipe("eds.normalizer") # type: ignore
    nlp.add_pipe("eds.sentences") # type: ignore
    nlp.add_pipe("eds.dates") # type: ignore
    return nlp # type: ignore

eds_dates_nlp: Language = build_eds_dates()

# ---------------------------
# 3) Token-based windowing for NER
# ---------------------------
def token_windows(text: str, max_tokens: int = MAX_TOKENS, stride_tokens: int = STRIDE_TOKENS) -> List[Tuple[int,int]]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False) # type: ignore
    offsets = enc["offset_mapping"] # type: ignore
    n = len(offsets) # type: ignore
    if n == 0:
        return []
    windows: List[Tuple[int,int]] = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        start_char = offsets[i][0] # type: ignore
        end_char = offsets[j - 1][1] # type: ignore
        windows.append((start_char, end_char)) # type: ignore
        if j == n:
            break
        i = max(j - stride_tokens, i + 1)
    return windows

# ---------------------------
# 4) Post-processing helpers (merge subwords, dedupe)
# ---------------------------
def _strip_hashes(w: str) -> str:
    return w.lstrip("#")

def merge_subwords(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ents:
        return ents
    ents = sorted(ents, key=lambda e: (e["start"], e["end"]))
    merged: List[Dict[str, Any]] = []
    cur = ents[0].copy()

    def is_alnum(c: str) -> bool:
        return c.isalnum() or c in "-_/."

    for nxt in ents[1:]:
        contiguous = nxt["start"] == cur["end"]
        same_label = nxt["label"] == cur["label"]
        looks_subword = nxt["text"].startswith("##") or (
            cur["text"] and is_alnum(cur["text"][-1]) and is_alnum(nxt["text"][0])
        )
        if contiguous and same_label and looks_subword:
            cur["text"] = cur["text"] + _strip_hashes(nxt["text"])
            cur["end"] = nxt["end"]
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged

def dedupe_and_merge_overlaps(ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not ents:
        return ents
    ents = sorted(ents, key=lambda e: (e["start"], e["end"], e["label"]))
    out: List[Dict[str, Any]] = []
    for e in ents:
        if out and e["label"] == out[-1]["label"]:
            if e["start"] == out[-1]["start"] and e["end"] == out[-1]["end"] and e["text"] == out[-1]["text"]:
                continue
            inter = min(e["end"], out[-1]["end"]) - max(e["start"], out[-1]["start"])
            if inter > 0 and inter >= 0.5 * min(e["end"] - e["start"], out[-1]["end"] - out[-1]["start"]):
                out[-1]["start"] = min(out[-1]["start"], e["start"])
                out[-1]["end"] = max(out[-1]["end"], e["end"])
                if len(e["text"]) > len(out[-1]["text"]):
                    out[-1]["text"] = e["text"]
                continue
        out.append(e)
    return out

# ---------------------------
# 5) Temporal utils (strict parse-based comparison)
# ---------------------------
DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}")
DATE_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+\d{2}h\d{2}m")

def parse_norm_to_date(s: Optional[str]) -> Optional[datetime.date]:
    """
    Return datetime.date if norm is a full date (YYYY-MM-DD),
    optionally followed by a time like ' 14h44m'. Otherwise None.
    """
    if not s:
        return None
    m = DATE_TIME.match(s)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(1))
        except Exception:
            return None
    m = DATE_ONLY.match(s)
    if m:
        try:
            return datetime.date.fromisoformat(m.group(0))
        except Exception:
            return None
    return None

def compare_dates(d1: datetime.date, d2: datetime.date) -> str:
    if d1 < d2:
        return "BEFORE"
    if d1 > d2:
        return "AFTER"
    return "OVERLAP"

# ---------------------------
# 6) Rule-based temporal linking
# ---------------------------

def sentences_spans(doc: Doc) -> List[Tuple[int, int]]:
    return [(s.start_char, s.end_char) for s in doc.sents]

def in_same_sentence(a_start: int, a_end: int, b_start: int, b_end: int, sent_spans: List[Tuple[int,int]]) -> bool:
    for s_start, s_end in sent_spans:
        if a_start >= s_start and a_end <= s_end and b_start >= s_start and b_end <= s_end:
            return True
    return False

def link_event_to_timex(events: List[Dict[str, Any]], timex: List[Dict[str, Any]],
                        sent_spans: List[Tuple[int,int]], max_dist: int = EVENT_TIMEX_MAX_DIST) -> Dict[int, int]:
    """
    Returns mapping: event_index -> timex_index (best candidate) if any.
    Prefers same-sentence; otherwise nearest within max_dist chars.
    """
    mapping: Dict[int, int] = {}
    for i, ev in enumerate(events):
        best_j = None
        best_score = 10**9
        for j, tx in enumerate(timex):
            if in_same_sentence(ev["start"], ev["end"], tx["start"], tx["end"], sent_spans):
                dist = abs((ev["start"] + ev["end"])//2 - (tx["start"] + tx["end"])//2)
                if dist < best_score:
                    best_score = dist
                    best_j = j
        if best_j is None:
            # fallback: nearest by char distance within max_dist
            for j, tx in enumerate(timex):
                dist = abs((ev["start"] + ev["end"])//2 - (tx["start"] + tx["end"])//2)
                if dist < best_score and dist <= max_dist:
                    best_score = dist
                    best_j = j
        if best_j is not None:
            mapping[i] = best_j
    return mapping

def tlinks_timex_timex(timex: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build TLINKs only between timexes that resolve to a full calendar date.
    Partials (2015-??-??), month-only, relative (e.g., '-730 days'), or durations are ignored.
    """
    edges: List[Dict[str, Any]] = []

    # Pre-compute resolvable (full) dates for each timex:
    parsed: List[Optional[datetime.date]] = []
    for t in timex:
        d = None
        val = t.get("norm")
        if isinstance(val, str):
            d = parse_norm_to_date(val)
        parsed.append(d)

    for i in range(len(timex)):
        if parsed[i] is None:
            continue
        for j in range(i + 1, len(timex)):
            if parsed[j] is None:
                continue
            rel = compare_dates(parsed[i], parsed[j]) # type: ignore
            edges.append({"u": f"t{i}", "v": f"t{j}", "rel": rel})
    return edges

def tlinks_event_timex(events: List[Dict[str, Any]], timex: List[Dict[str, Any]],
                       ev2tx: Dict[int, int]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for ei, tj in ev2tx.items():
        edges.append({"u": f"e{ei}", "v": f"t{tj}", "rel": "OCCURS_ON"})
    return edges

def tlinks_event_event_from_anchors(ev2tx: Dict[int, int],
                                    timex: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Infer event↔event TLINKs only when BOTH events are anchored to timexes
    that resolve to full calendar dates.
    """
    edges: List[Dict[str, Any]] = []

    # Precompute parsed dates for each timex
    parsed: List[Optional[datetime.date]] = []
    for t in timex:
        d = None
        val = t.get("norm")
        if isinstance(val, str):
            d = parse_norm_to_date(val)
        parsed.append(d)

    # Pairwise compare events through their anchors when both anchors are full dates
    items = list(ev2tx.items())  # (event_idx, timex_idx)
    for a in range(len(items)):
        ei, ti = items[a]
        di = parsed[ti]
        if di is None:
            continue
        for b in range(a + 1, len(items)):
            ej, tj = items[b]
            dj = parsed[tj]
            if dj is None:
                continue
            rel = compare_dates(di, dj)
            edges.append({"u": f"e{ei}", "v": f"e{ej}", "rel": rel})
    return edges

# ---------------------------
# 7) Processing
# ---------------------------
def process_markdown_file(md_path: Path, out_dir: Path, preview_chars: int = 10000) -> None:
    # Read + clean
    cleaned = read_md(md_path)

    # Preview
    print(f"\n--- CLEANED TEXT PREVIEW: {md_path.name} ---")
    preview = cleaned[:preview_chars].replace(PAGE_BREAK, "<PAGE_BREAK>")
    print(preview + ("..." if len(cleaned) > preview_chars else ""))

    # Dates (and sentences)
    doc = eds_dates_nlp(cleaned)
    timex: List[Dict[str, Any]] = []
    for s in doc.spans.get("dates", []): # type: ignore
        norm_obj = s._.get("date")  # AbsoluteDate / RelativeDate / Duration / str / None # type: ignore
        kind = "unknown"
        parts = {"y": None, "m": None, "d": None}
        rel_seconds = None

        if hasattr(norm_obj, "mode"):  # eds objects # type: ignore
            mode = getattr(norm_obj, "mode", None) # type: ignore
            if str(mode) == "absolute":
                kind = "absolute"
                parts["y"] = getattr(norm_obj, "year", None) # type: ignore
                parts["m"] = getattr(norm_obj, "month", None) # type: ignore
                parts["d"] = getattr(norm_obj, "day", None) # type: ignore
            elif str(mode) == "relative":
                kind = "relative"
                try:
                    td = norm_obj.to_duration()  # timedelta # type: ignore
                    rel_seconds = td.total_seconds()  # type: ignore
                except Exception:
                    rel_seconds = None
            elif str(mode) == "duration":
                kind = "duration"
        else:
            # keep "unknown"; we'll still store the string in "norm" for parsing attempts

            pass

        timex.append({
            "text": s.text,  # type: ignore
            "norm": str(norm_obj) if norm_obj is not None else None,  # readable, JSON-safe # type: ignore
            "kind": kind,                     # absolute | relative | duration | unknown
            "parts": parts,                   # for absolute/partial (informative only here)
            "rel_seconds": rel_seconds,       # for relative (informative only here)
            "start": s.start_char, # type: ignore
            "end": s.end_char, # type: ignore
        })

    sent_spans = sentences_spans(doc)

    # Events (token-windowed NER)
    events: List[Dict[str, Any]] = []
    for c_start, c_end in token_windows(cleaned):
        piece = cleaned[c_start:c_end]
        preds = hf_ner(piece)  # each piece already <= 128 tokens
        for ent in preds:
            events.append({
                "text": ent["word"],
                "label": ent["entity_group"].lower(),  # problem | test | treatment
                "start": int(ent["start"]) + c_start,
                "end": int(ent["end"]) + c_start,
            })

    # Post-process: merge subwords, then dedupe
    events = merge_subwords(events)
    events = dedupe_and_merge_overlaps(events)

    # ---- PRELIMINARY TEMPORAL EDGES ----
    # Event -> Timex (nearest; prefer same sentence)
    ev2tx = link_event_to_timex(events, timex, sent_spans, max_dist=EVENT_TIMEX_MAX_DIST)
    edges_et = tlinks_event_timex(events, timex, ev2tx)

    # Timex <-> Timex (compare full dates only)
    edges_tt = tlinks_timex_timex(timex)

    # Event <-> Event inferred from anchors (full dates only)
    edges_ee = tlinks_event_event_from_anchors(ev2tx, timex)

    # Package nodes + edges
    nodes_events = [
        {"id": f"e{i}", "type": "EVENT", "label": ev["label"], "text": ev["text"],
         "start": ev["start"], "end": ev["end"]}
        for i, ev in enumerate(events)
    ]
    nodes_timex = [
        {"id": f"t{i}", "type": "TIMEX", "text": tx["text"], "norm": tx["norm"],
         "kind": tx["kind"], "parts": tx["parts"], "rel_seconds": tx["rel_seconds"],
         "start": tx["start"], "end": tx["end"]}
        for i, tx in enumerate(timex)
    ]
    edges = edges_tt + edges_et + edges_ee

    out = {
        "id": md_path.stem,
        "events": events,
        "timex": timex,
        "graph": {
            "nodes": nodes_events + nodes_timex,
            "edges": edges
        }
    }
    out_path = out_dir / f"{md_path.stem}.hf_eds.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_path}")

# ---------------------------
# 8) Driver
# ---------------------------
def run(md_dir: str, out_dir: str):
    in_p, out_p = Path(md_dir), Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    files = sorted(in_p.glob("*.md"))
    if not files:
        print(f"No .md files in {md_dir}")
        return
    for md in files:
        process_markdown_file(md, out_p)

if __name__ == "__main__":
    run(md_dir="data/concatenated_docs", out_dir="data/annotated_docs")
