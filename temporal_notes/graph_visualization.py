import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, cast

from pyvis.network import Network

EventDict = Dict[str, Any]
TimexDict = Dict[str, Any]
RelationDict = Dict[str, Any]
TimelineData = Dict[str, Any]

"""
Force-directed graph visualization (Solution B).

Input: JSON produced by timeline_rules_simple.json (your timeline).
Usage:
    python temporal_notes/visualize_timeline_force.py \
        data/graph_per_documents/timeline_rules_simple.json \
        data/graph_per_documents/timeline_force.html

Rules implemented:
- Exclude events where family == True.
- Nodes: Events + Timex (DATE + resolved DURATION only).
- Edges: EVENT_TIME + BEFORE.
- Suppress BEFORE edges between events sharing the same anchor_date (reduces clutter).
- Styling cues:
  * Event color by label: problem (tomato), treatment (steelblue), test (goldenrod).
  * Negation => red border.
  * Hypothesis => dashed border.
  * History (history_str == ATCD) => muted fill.
  * Timex DATE => diamond (color gray), DURATION => box (purple if resolved interval).
"""

COLOR_EVENT = {
    "problem": "#e74c3c",
    "treatment": "#1f77b4",
    "test": "#d4aa00",
}
COLOR_MUTED = "#cccccc"
COLOR_NEG_BORDER = "#ff0033"
COLOR_TIMEX_DATE = "#555555"
COLOR_TIMEX_DURATION = "#8e44ad"
COLOR_PT_SAME_NEG = "#2ecc71"      # vert
COLOR_PT_DIFF_NEG = "#ff9800"      # orange
COLOR_EVENT_TIME_PREV = "#6a9ff5"  # bleu clair pour distinguer EVENT_TIME_PREV
COLOR_EVENT_TIME_NEXT = "#6a9ff5"   # vert/teal pour NEXT (à adapter si besoin)
COLOR_TLINK_SIMULTANEOUS = "#ff6b6b"     # Rouge corail pour simultanéité
COLOR_EVENT_TIME_SIMULTANEOUS = "#4ecdc4" # Turquoise pour simultanéité temporelle
COLOR_EVENT_BEFORE_TIMEX = "#45b7d1"     # Bleu clair pour événement avant timex
COLOR_TIMEX_BEFORE_EVENT = "#96ceb4"     # Vert clair pour timex avant événement
COLOR_TENSE_UNKNOWN = "#95a5a6"          # Gris pour tense inconnue
COLOR_TLINK_BEFORE = "#1221f3"  # Orange for TLINK_BEFORE


def load_json(path: Path) -> TimelineData:
    return json.loads(path.read_text(encoding="utf-8"))


def build_graph(data: TimelineData) -> Network:
    net = Network(
        height="800px",
        width="100%",
        notebook=False,
        directed=True,
        bgcolor="#ffffff",
        font_color="#222", #type: ignore
    )
    net.force_atlas_2based(gravity=-50, central_gravity=0.01,
                           spring_length=100, spring_strength=0.05)

    events: List[EventDict] = [
        e for e in cast(Iterable[EventDict], data.get("events", [])) if not e.get("family")
    ]
    timex_list: List[TimexDict] = list(cast(Iterable[TimexDict], data.get("timex", [])))
    timex_map: Dict[str, TimexDict] = {t["id"]: t for t in timex_list}
    relations: List[RelationDict] = list(cast(Iterable[RelationDict], data.get("relations", [])))

    # Timex référencés
    referenced_timex_ids: Set[str] = set()
    for r in relations:
        if r.get("type", "").startswith("EVENT_TIME"):
            if r["source"] in timex_map:
                referenced_timex_ids.add(r["source"])
            if r["target"] in timex_map:
                referenced_timex_ids.add(r["target"])

    timex_filtered: List[TimexDict] = []
    for t in timex_list:
        if (
            t["id"] in referenced_timex_ids
            or (t["kind"] == "DATE" and t.get("value"))
            or (t["kind"] == "DURATION" and t.get("interval"))
        ):
            timex_filtered.append(t)

    # Timex nodes
    for t in timex_filtered:
        shape = "diamond" if t["kind"] == "DATE" else "box"
        color = COLOR_TIMEX_DATE if t["kind"] == "DATE" else COLOR_TIMEX_DURATION
        label = t.get("value") or t.get("text") or t["id"]
        net.add_node(
            t["id"],
            label=label,
            title="<br/>".join([
                f"<b>{t['id']}</b>",
                f"Texte: {t.get('text','')}",
                f"Kind: {t.get('kind')}",
                f"Value: {t.get('value','')}",
                f"Interval: {t.get('interval','')}",
                f"Role: {t.get('role','NORMAL')}",
            ]),
            shape=shape,
            color=color,
            font={"color": "#ffffff" if t["kind"] == "DATE" else "#f5f5f5"},
        )

    # Event nodes
    existing_nodes: Set[str] = set()
    for e in events:
        base_color = COLOR_EVENT.get(e["label"].lower(), "#7f8c8d")
        if e.get("history_str") == "ATCD":
            base_color = COLOR_MUTED
        border = COLOR_NEG_BORDER if e.get("negation") else "#333333"
        dashes = True if e.get("hypothesis") else False
        short = e["text"] if len(e["text"]) <= 30 else e["text"][:27] + "..."
        net.add_node(
            e["id"],
            label=short,
            title="<br/>".join([
                f"<b>{e['id']}</b> ({e.get('label')})",
                f"Texte: {e['text']}",
                f"Negation: {e.get('negation')}",
                f"Hypothesis: {e.get('hypothesis')}",
                f"History: {e.get('history_str')}",
                f"Anchor: {e.get('anchor_date')}",
                f"Source timex: {e.get('source_timex')}",
            ]),
            shape="ellipse",
            color=base_color,
            borderWidth=2,
            font={"color": "#ffffff"},
            **({"dashes": True} if dashes else {}),
        )
        node = net.get_node(e["id"]) #type: ignore
        if node is None:
            continue
        node_dict: Dict[str, Any] = cast(Dict[str, Any], node)
        node_dict["color"] = {
            "border": border,
            "background": base_color,
            "highlight": {"border": border, "background": base_color},
            "hover": {"border": border, "background": base_color},
        }
        existing_nodes.add(e["id"])

    # BEFORE filter (different dates only)
    anchor_map: Dict[str, Optional[str]] = {e["id"]: e.get("anchor_date") for e in events}
    keep_before: Set[Tuple[str, str]] = {
        (r["source"], r["target"])
        for r in relations
        if r.get("type") == "BEFORE"
        and r["source"] in anchor_map
        and r["target"] in anchor_map
        and anchor_map[r["source"]] != anchor_map[r["target"]]
    }

    # Enhanced Edges section
    for r in relations:
        st = r.get("type", "")
        s_raw = r.get("source")
        t_raw = r.get("target")
        if not isinstance(s_raw, str) or not isinstance(t_raw, str):
            continue
        s, t = s_raw, t_raw

        # Sécurise la condition (parenthèses)
        if (s not in existing_nodes or t not in existing_nodes) and not st.startswith("EVENT_TIME") and not st.startswith("TLINK"):
            continue

        # Existing relations
        if st == "EVENT_TIME":
            net.add_edge(s, t, color="#4c6ef5", title="EVENT_TIME", width=2, arrows="to")
        elif st == "EVENT_TIME_PREV":
            net.add_edge(s, t, color=COLOR_EVENT_TIME_PREV, title="EVENT_TIME_PREV", width=2, arrows="to", dashes=True)
        elif st == "EVENT_TIME_NEXT":
            net.add_edge(s, t, color=COLOR_EVENT_TIME_NEXT, title="EVENT_TIME_NEXT", width=2, arrows="to", dashes=True)
        elif st == "EVENT_TIME_CAND":
            net.add_edge(s, t, color="#90b4f9", title="EVENT_TIME_CAND", width=1, arrows="to", dashes=True)
        elif st == "BEFORE" and (s, t) in keep_before:
            net.add_edge(s, t, color="#999999", title="BEFORE", width=1, dashes=True, arrows="to")
        elif st == "PROBLEM_TREATMENT_SAME_NEG":
            net.add_edge(s, t, color=COLOR_PT_SAME_NEG, title="PROBLEM_TREATMENT (same neg)", width=3, smooth=True)
        elif st == "PROBLEM_TREATMENT_DIFF_NEG":
            net.add_edge(s, t, color=COLOR_PT_DIFF_NEG, title="PROBLEM_TREATMENT (diff neg)", width=3, dashes=True, smooth=True)
        
        # NEW: Tense-based relations
        elif st == "TLINK_SIMULTANEOUS":
            net.add_edge(s, t, color=COLOR_TLINK_SIMULTANEOUS, title="TLINK_SIMULTANEOUS (verb tense)", 
                        width=2, arrows="to", smooth={"type": "curvedCW", "roundness": 0.2})
        elif st == "TLINK_BEFORE":
            net.add_edge(s, t, color=COLOR_TLINK_BEFORE, title="TLINK_BEFORE (verb tense)", 
                        width=2, arrows="to", dashes=[8,3])
        elif st == "EVENT_TIME_SIMULTANEOUS":
            net.add_edge(s, t, color=COLOR_EVENT_TIME_SIMULTANEOUS, title="EVENT_TIME_SIMULTANEOUS (present tense)", 
                        width=2, arrows="to", smooth={"type": "curvedCW", "roundness": 0.1})
        elif st == "EVENT_BEFORE_TIMEX":
            net.add_edge(s, t, color=COLOR_EVENT_BEFORE_TIMEX, title="EVENT_BEFORE_TIMEX (past tense)", 
                        width=2, arrows="to", dashes=[5,5])
        elif st == "TIMEX_BEFORE_EVENT":
            net.add_edge(s, t, color=COLOR_TIMEX_BEFORE_EVENT, title="TIMEX_BEFORE_EVENT (future tense)", 
                        width=2, arrows="to", dashes=[10,5])
        elif st.endswith("_TENSE_UNKNOWN"):
            # Handle modified relations with unknown tense
            base_type = st.replace("_TENSE_UNKNOWN", "")
            net.add_edge(s, t, color=COLOR_TENSE_UNKNOWN, title=f"{base_type} (tense unknown)", 
                        width=1, arrows="to", dashes=[2,2])

    return net

def main() -> None:
    """
    Nouvel usage:
      1) Avec répertoire de sortie + plusieurs JSON:
         python temporal_notes/graph_visualization.py out_dir file1.json file2.json file3.json
      2) Sans arguments: cherche automatiquement data/graph_per_documents/timeline_rules_simple_*.json
         et écrit les HTML dans data/graph_per_documents/graphs/
    """
    out_dir: Path
    json_files: List[Path]
    if len(sys.argv) >= 3:
        out_dir = Path(sys.argv[1]).resolve()
        json_files = [Path(p).resolve() for p in sys.argv[2:]]
    else:
        print("[INFO] Aucun argument fourni, recherche auto des timelines.")
        out_dir = Path("data/graph_per_documents/graphs").resolve()
        json_files = sorted(Path("data/graph_per_documents").glob("timeline_rules_simple_*.json"))
        if not json_files:
            print("[ERREUR] Aucun fichier timeline_rules_simple_*.json trouvé.")
            return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Répertoire sortie: {out_dir}")
    print(f"[INFO] Fichiers détectés: {len(json_files)}")

    for jf in json_files:
        if not jf.exists():
            print(f"[WARN] Introuvable: {jf}")
            continue
        try:
            data = load_json(jf)
        except Exception as e:
            print(f"[ERREUR] Lecture JSON {jf}: {e}")
            continue

        net = build_graph(data)
        stem = jf.stem  # ex: timeline_rules_simple_doc1
        html_path = out_dir / f"{stem}.html"
        net.write_html(str(html_path), open_browser=False)
        print(f"[OK] {jf.name} -> {html_path.name} (Nodes={len(net.nodes)} Edges={len(net.edges)})")

    print("[FIN] Génération terminée.")

if __name__ == "__main__":
    main()
