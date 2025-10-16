# Temporal Narratives in Clinical Notes
### A Framework for Evaluating Temporal Coherence in AI-Generated Medical Reports

**Author:** Hermes Ndjeng
**Supervision:** Dr. Anuradha Kar
**Affiliations:** Arkhn · Aivancity Paris-Cachan
**Date:** 2025

---

## Abstract
This repository implements the methodology described in
*"Temporal Graphs for Ensuring Clinical Consistency in AI-Generated Medical Reports."*
The project proposes a **temporal-graph-based framework** for assessing the chronological reliability of AI-generated clinical summaries.

Unlike classical language evaluation metrics, this approach focuses on **temporal reasoning**—how accurately models capture the sequence of problems, treatments, and tests within patient histories.
By representing both ground-truth and generated narratives as **temporal graphs**, the framework allows systematic detection of temporal errors such as anchor mismatches, misordered chains, or missing time references.

---

## Research Objectives

1. **Quantify temporal coherence** in AI-generated medical summaries.
2. **Construct interpretable temporal graphs** that encode clinical events and their relations.
3. **Develop a hybrid extraction pipeline** combining transformer-based NER and rule-based temporal linking.
4. **Compare AI-generated vs. ground-truth timelines** to identify inconsistencies.
5. **Support visual and statistical analysis** of temporal reliability across models.

---

## Methodological Overview

### 1. Data Sources and Ethics
- **MIMIC-III** dataset (public, de-identified critical-care records).
- **Real-world French hospital records** (pseudonymized with [EDS-Pseudo](https://github.com/aphp/eds-pseudo)).
- All patient data used here are **synthetic or anonymized**.
- OCR performed with **dots.ocr** to preserve document layout and context markers.

### 2. Temporal Graph Construction (`edsnlp_stuff.py`, `ner.py`)
The pipeline converts unstructured French medical notes into structured temporal graphs:

| Stage | Method | Description |
|--------|---------|-------------|
| **Clinical Entity Extraction** | `DrBERT-CASM2` via `ner.py` | Identifies *Problems*, *Tests*, and *Treatments* in free text. |
| **Temporal Expression Detection** | `edsnlp.dates` | Recognizes absolute and relative dates, normalizing them to ISO-8601. |
| **Temporal Linking** | `edsnlp_stuff.py` | Associates events with their nearest time anchors and propagates through “temporal impact zones.” |
| **Graph Assembly** | internal JSON schema | Builds a directed graph of events, timexes, and relations (e.g., `EVENT_TIME`, `BEFORE`). |

The resulting JSONs (`timeline_rules_simple_*.json`) serve as the canonical representation of a patient’s trajectory.

### 3. Temporal Graph Visualization (`graph_visualization.py`)
Graphs are rendered as interactive PyVis networks where:
- Nodes = events (problems/tests/treatments) or timexes.
- Colors denote semantic categories; borders indicate negations or hypotheses.
- Edges encode temporal relations and clinical links (problem–treatment pairs).

This visualization supports **qualitative analysis** of temporal reasoning performance.

### 4. AI-Generated Summaries (`ai_generation.py`)
To test temporal accuracy, summaries are generated using a constrained LLM prompt:
- The model is instructed to act as a **French-speaking hospital physician**.
- Each clinical event must include an **explicit temporal marker**.
- Prompts focus on a given illness (e.g., BPCO exacerbations).
- Outputs are short (< 350 words), temporally explicit summaries.

Generated texts are then re-converted into temporal graphs for comparison.

### 5. Temporal Graph Comparison (`graph_comparaison.py`)
Pairs of graphs—reference vs. AI output—are automatically compared to detect:
- **Anchor mismatches:** events linked to inconsistent dates.
- **Broken chains:** missing or reversed BEFORE relations.
- **Redundant or missing nodes:** omissions or hallucinations.

The system produces structured audit reports highlighting mismatched relations, useful for LLM evaluation and fine-tuning.

### 6. Error Distribution Analysis (`plot_error_code_hist.py`)
Error logs from graph comparison are aggregated and visualized to quantify:
- Frequency of each error type (`ANCHOR_MISMATCH`, `BEFORE_CHAIN_MISMATCH`, etc.).
- Distribution of errors per document or model.
- Overall temporal reliability score.

This statistical layer supports **quantitative benchmarking** of temporal reasoning.

---

## 🧠 Conceptual Architecture

┌──────────────────────────┐
│  Clinical Documents      │
│ (MIMIC-III / Real Data)  │
└────────────┬─────────────┘
▼
OCR & Pseudonymization  →  Narrative Section Selection
▼
Named Entity Recognition (DrBERT-CASM2)
▼
Temporal Expression Detection (edsnlp.dates)
▼
Event–Timex Linking & Temporal Zones
▼
Temporal Graph JSON
▼
AI-Generated Summary (LLM)
▼
Temporal Graph Comparison
▼
Error Metrics & Visualization

---

## 🧮 Experimental Outputs

| Output Type | Description |
|--------------|-------------|
| `timeline_rules_simple_*.json` | Temporal graphs representing patient event sequences. |
| `.html` PyVis networks | Interactive visualization of event–time relations. |
| `.json` audit files | Comparison results between AI and ground truth. |
| `.png` / `.pdf` | Statistical plots of error distributions. |
| Generated summaries | LLM-produced medical reports with explicit time anchors. |

---

## 📊 Evaluation Metrics

| Code | Interpretation | Severity |
|------|----------------|-----------|
| `ANCHOR_MISMATCH` | Incorrect event–date link | ⚠️ Moderate |
| `BEFORE_CHAIN_MISMATCH` | Broken chronological sequence | ❌ High |
| `MISSING_EVENT` | Missing clinical event | ❌ High |
| `ZONE_ANCHOR_ERROR` | Wrong inherited temporal zone | ⚠️ Moderate |
| `TEMPORAL_SIGNAL_ERROR` | Ambiguous temporal cue | ⚠️ Low |

Each error is mapped to a severity score, enabling temporal reliability scoring for different model generations.

---

## 🧰 Technology Stack

| Layer | Tools |
|-------|-------|
| **Language Models** | DrBERT-CASM2 · GPT-4o / Llama-3-70B |
| **NLP Toolkit** | spaCy · EDS-NLP |
| **Data Handling** | Pandas · JSON · Markdown |
| **Visualization** | PyVis · Matplotlib · Graphviz |
| **Evaluation** | Python 3.11 · Poetry environment |
| **Pseudonymization** | EDS-Pseudo |
| **OCR** | dots.ocr |

---

## 🔬 Scientific Relevance

This framework contributes to the growing field of **temporal reasoning in clinical NLP** by:
- Bridging symbolic and neural approaches (rule-based + transformer).
- Providing interpretable graph-based representations of temporal knowledge.
- Offering an **auditable evaluation layer** for AI-generated clinical text.

It complements existing benchmarks like *TIMER* and *Test of Time* by targeting real French hospital documentation and leveraging hybrid temporal extraction pipelines.

---

## 📎 Citation

If you use this framework or methodology, please cite:

> Ndjeng, H., & Kar, A. (2025). *Temporal Graphs for Ensuring Clinical Consistency in AI-Generated Medical Reports.*

---

## 📜 License & Data Policy

All code is released under the **MIT License**.
The included data are **synthetic or anonymized** and intended for research demonstration only.
When adapting the pipeline to real patient records, ensure full compliance with GDPR and institutional data governance policies.

---

## 📬 Contact
Hermes Ndjeng
📧 hermes.ndjeng@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/hermes-yan-ntjam-ndjeng-99a241217/) · [GitHub](https://github.com/HermesNdjeng)

---


⸻

Would you like me to append a “Reproducibility & Folder Structure” section (detailing how to re-run each experiment and what outputs to expect) for inclusion in a thesis appendix or repository archive?
