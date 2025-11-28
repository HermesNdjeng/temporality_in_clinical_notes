import os
import json
from typing import Dict, Any

ONTOLOGY_SPEC = """
Ontology (only elements present in the JSON)

Top-level keys
- note_datetime (string, YYYY-MM-DD): Reference document date used for resolving relative expressions.
- events (array<Event>): Clinical mentions.
- timex (array<Timex>): Temporal expressions.
- zones (array<Zone>): Sequential impact intervals derived from absolute timex.
- relations (array<Relation>): Typed links between events and/or timex.

Event object (entries in events)
- id (E#): Unique event identifier.
- text: Surface string as extracted.
- start (int): 0-based character offset (inclusive).
- end (int): 0-based character offset (exclusive).
- label (problem | treatment | test): Event category.
- negation (bool): True = explicitly negated; False = affirmed.
- hypothesis (bool): True = speculative / conditional; False = asserted.
- family (bool): True = refers to a family member (kept for trace, often excluded downstream).
- history (bool): Raw detector flag for historical context.
- history_str (CURRENT | ATCD | UNKNOWN): Normalized temporal status (ATCD = past history).
- anchor_date (string | null, YYYY-MM-DD): Final resolved date assigned; null if not anchored.
- source_timex (T# | null): Timex id that directly anchored the event (via EVENT_TIME); null if anchor came from zone logic or absent.

Semantics:
- If anchor_date present and source_timex null -> inferred from zone (not a direct timex link).
- Multiple events can share same anchor_date if linked to same timex or same zone.

Timex object (entries in timex)
- id (T#): Unique temporal expression identifier.
- text: Original surface form.
- start / end (int): Character span.
- kind (DATE | DURATION): Type after normalization (RELATIVE may already have been converted; here only DATE/DURATION appear).
- value (string | null, YYYY-MM-DD): Resolved absolute date (for DATE) or representative endpoint (often interval end) for some durations; null if unresolved.
- interval (object | null): For durations; { "start": YYYY-MM-DD, "end": YYYY-MM-DD }. Null if not a resolved duration.
- role (NORMAL): Processing role (in this JSON all are NORMAL).
- resolved (bool): True if value (DATE) or interval (DURATION) successfully determined.

Semantics:
- DATE with resolved=true + value usable for anchoring.
- DURATION unresolved (value=null, interval=null) does not anchor events directly.

Zone object (entries in zones)
- date (YYYY-MM-DD): Anchor date governing this zone.
- start (int): Character offset where influence begins (at timex).
- end (int): Character offset where influence ends (next absolute timex start or document end).

Semantics:
- Provides fallback anchor_date for events whose start lies in [start, end) and lack direct EVENT_TIME-based anchoring.

Relation object (entries in relations)
- source (E#): Origin node id.
- target (T# or E#): Destination node id.
- type (EVENT_TIME | BEFORE): Relation category present.
- evidence (string): Short provenance tag (e.g., "same sentence", "chronological order").

Relation types present
- EVENT_TIME (Event -> Timex):
  Meaning: Event directly associated with that temporal expression (typically same sentence). If timex resolved (value present), event’s anchor_date = timex.value and source_timex = timex.id.
- BEFORE (Event -> Event):
  Meaning: Temporal precedence: source event occurs earlier than target event (constructed by chronological ordering of anchored events; same-date links usually suppressed except when ordering chain emitted).

Derived logic implicit in a typical JSON
- Events sharing the same timex have identical anchor_date.
- Events without direct EVENT_TIME may receive anchor_date from a zone (source_timex null).
- Unresolved durations do not anchor events unless further rules resolve them.

Constraints
- ids in relations must exist in events/timex.
- anchor_date should match either a timex value or a zone date.
- BEFORE edges should not introduce cycles (expected DAG over anchored events).
"""

def get_ontology_spec() -> str:
    return ONTOLOGY_SPEC

LLM_TEMPORAL_AUDIT_PROMPT = """
RECOMMENDATIONS FOR LLM (to be followed strictly):
- When comparing events, always analyze the full event text and its clinical meaning, not just token or substring overlap.
- If two events are related (e.g., "BPCO" and "exacerbation de BPCO"), explain their clinical relationship in the details. For example, "exacerbation de BPCO" is an acute episode of the chronic disease "BPCO".
- If an anchor_date mismatch could be explained by disease progression (e.g., "BPCO" at an earlier date, "exacerbation de BPCO" at a later date), downgrade the severity to WARNING and explain that this may reflect normal clinical evolution.
- In all explanations, use the event/timex texts and their context to justify the issue or lack thereof.
- Prefer to explain why a difference is or is not a true error, using clinical logic (e.g., chronic vs. acute, progression, recurrence).
- For each issue, always provide a short, explicit explanation of the semantic/clinical relationship between the events/timexes involved.
- If you are unsure due to lack of context, use severity=INFO and code=INSUFFICIENT_DATA.
- Whenever you mention an event or timex (in details, example_ids, explanations, etc.), always use both its ID and its text (for events) or text/value (for timex), e.g. "E6 (dyspnée aggravée)", "T2 (01/09)". Never use generic terms like "the equivalent event"—always be explicit with the event/timex names and texts.

You are a clinical timeline consistency auditor.

Ontology:
{ontology}

Input:
JSON_A (GROUND TRUTH):
{json_a}

JSON_B (TO EVALUATE):
{json_b}

Illness of interest: {illness}

Goal:
Compare JSON_B (the system output) against JSON_A (the ground truth) and list TEMPORAL INCOHERENCES and STRUCTURAL ISSUES in JSON_B, using JSON_A as reference related to the illness {illness}.
Work strictly from provided data. Do not invent clinical facts.

Instructions:
- Whenever you mention an event or timex in any output field (including example_ids, details, etc.), always provide both its ID and its text (for events) or text/value (for timex), e.g. "E6 (dyspnée aggravée)", "T2 (01/09)".
- Do NOT include a "recommendations" section in the output.
- Focus your analysis on JSON_B, using JSON_A as the reference for expected values, order, and structure.
- Do NOT report "MISSING_NODE" errors: if an event or timex is present in B but not in A, or vice versa, this is not an error and should not be included in the issues list.
- When comparing events, do not rely only on token or substring matching. Consider the full meaning and context of the event text. For example, "BPCO" and "exacerbation de BPCO" are not the same event: "BPCO" refers to the underlying disease, while "exacerbation de BPCO" refers to an acute episode. Only report an anchor_date mismatch if the events are truly semantically equivalent (same clinical episode or mention).
- Use your best judgment to avoid false positives due to partial or generic text overlap.
- When describing or explaining any issue, always leverage the semantic meaning of the event and timex texts. If two events are related (e.g., "BPCO" and "exacerbation de BPCO"), explain the clinical or temporal relationship in your details field, and downgrade the severity to WARN if the difference could be explained by disease progression or clinical context.
- In the "details" field for each issue, provide a short explanation using the event/timex texts, their semantic relationship, and the possible clinical interpretation (e.g., "In JSON_B, 'exacerbation de BPCO' may represent a progression of 'BPCO' from JSON_A, so the anchor_date mismatch may reflect disease evolution rather than a strict error.").
- If a difference is likely due to semantic overlap or clinical evolution (e.g., a chronic disease and its acute exacerbation), prefer severity=WARNING and explain why.

Tasks:

1. Cross-JSON comparison (A = ground truth, B = system):
   a. Events (by (label, lowercased text and meaning)) that appear in both JSONs and refer to the same clinical episode, but have different anchor_date (report as error in B, or as warning if semantic/clinical evolution is plausible).
   b. BEFORE chains that differ: for matching event pairs (by (label, text, and meaning)), if order is reversed or missing in B compared to A, report as error in B.
   c. Presence of EVENT_TIME in A but missing or downgraded to null anchor in B (report as error in B).

2. Temporal logic checks (on B):
   a. An event in B anchored earlier than a timex it references (start/end offsets sanity if available).
   b. BEFORE edges in B inconsistent with dates (source.anchor_date >= target.anchor_date).
   c. DURATION interval inconsistencies in B: interval.start > interval.end; anchor_date outside interval.
   d. Non-monotonic chains in B: A BEFORE B, B BEFORE C, but A anchor_date > C anchor_date.

3. Classification:
   For each issue produce:
   - code: SHORT_UPPER_SNAKE (e.g., ANCHOR_MISMATCH, BEFORE_CYCLE)
   - severity: one of INFO|WARN|ERROR (ERROR if data integrity or chronology violation; WARN if probable semantic/clinical evolution or heuristic issue; INFO if optimization)
   - json: B (unless BOTH is strictly necessary)
   - details: concise explanation (always include event/timex text with IDs, and explain the semantic/clinical relationship if relevant)
   - example_ids: list of involved IDs, always formatted as "E6 (dyspnée aggravée)", "T2 (01/09)", etc.

4. Output JSON schema:
{{
  "summary": {{
    "totals": {{ "ERROR": n, "WARN": n, "INFO": n }}
  }},
  "issues": [
    {{
      "code": "...",
      "severity": "...",
      "json": "B",
      "details": "...",
      "example_ids": ["E1 (texte de l'événement)", "T3 (texte ou value du timex)", "E7->E9 (texte1->texte2)"]
    }}
  ]
}}

Constraints:
- Do not hallucinate; if data insufficient, mark severity=INFO with code=INSUFFICIENT_DATA.
- Always include the event/timex text with its ID everywhere an ID appears.
- Do NOT output a recommendations section.
- Do NOT output any "MISSING_NODE" errors.
- Use semantic and contextual understanding of event meaning, not just string overlap.
- Return ONLY the JSON described (no prose, no markdown).
Begin now.
"""

def build_llm_temporal_audit_prompt(json_a: str, json_b: str, illness: str = "") -> str:
    """
    Injects JSON strings, ontology, and illness into the prompt template.
    """
    return LLM_TEMPORAL_AUDIT_PROMPT.format(
        ontology=ONTOLOGY_SPEC.strip(),
        json_a=json_a.strip(),
        json_b=json_b.strip(),
        illness=illness
    )

def run_temporal_audit(json_a_path: str, json_b_path: str, illness: str = "", model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Run the temporal audit prompt against two timeline JSON files using OpenAI API.
    Requires environment variable OPENAI_API_KEY (do NOT hardcode the key).
    Returns the parsed JSON response from the LLM (or raises ValueError if parsing fails).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install --upgrade openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY environment variable with your OpenAI key.")

    client = OpenAI(api_key=api_key)

    with open(json_a_path, "r", encoding="utf-8") as fa:
        json_a_raw = fa.read()
    with open(json_b_path, "r", encoding="utf-8") as fb:
        json_b_raw = fb.read()

    prompt = build_llm_temporal_audit_prompt(json_a_raw, json_b_raw, illness=illness)

    # Call the chat completion endpoint
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a precise JSON-only temporal validation engine."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = completion.choices[0].message.content or ""
    content = content.strip()
    # --- Correction : retire les balises markdown éventuelles ---
    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    if content.startswith("```"):
        content = content.removeprefix("```").strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    # --- Fin correction ---
    try:
        result = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{content}") from e
    return result

if __name__ == "__main__":
    # Example usage:
    # export OPENAI_API_KEY=sk-...
    # python temporal_notes/graph_comparaison.py doc1.json doc2.json [illness] [output.json]
    import sys
    if len(sys.argv) == 5:
        out = run_temporal_audit(sys.argv[1], sys.argv[2], illness=sys.argv[3], model="gpt-4o")
        print(json.dumps(out, ensure_ascii=False, indent=2))
        with open(sys.argv[4], "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    elif len(sys.argv) == 4:
        out = run_temporal_audit(sys.argv[1], sys.argv[2], illness=sys.argv[3], model="gpt-4o")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif len(sys.argv) == 3:
        out = run_temporal_audit(sys.argv[1], sys.argv[2], illness="", model="gpt-4o")
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print("Usage: python graph_comparaison.py timeline_A.json timeline_B.json [illness] [output.json]")

