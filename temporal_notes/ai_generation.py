import os
import re
import argparse
from pathlib import Path
from typing import Dict, List
from openai import OpenAI

PAGE_BREAK = "\f"
def read_markdown(md_path: Path) -> str:
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

def load_markdown_texts(md_dir: Path, doc_ids: List[str]) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for did in doc_ids:
        p = md_dir / f"{did}.md"
        if p.exists():
            try:
                texts[did] = read_markdown(p)
            except Exception as e:
                print(f"[WARN] Lecture/clean MD échouée {p}: {e}")
        else:
            print(f"[WARN] Fichier MD manquant: {p}")
    return texts

def load_all_markdown(md_dir: Path) -> Dict[str, str]:
    texts: Dict[str, str] = {}
    for p in sorted(md_dir.glob("*.md")):
        did = p.stem
        try:
            texts[did] = read_markdown(p)
        except Exception as e:
            print(f"[WARN] Échec lecture {p}: {e}")
    return texts

def build_context_only_text(raw_texts: Dict[str, str]) -> str:
    parts: List[str] = []
    for did, txt in raw_texts.items():
        parts.append(f"[DOC {did}]\n{txt.strip()}")
    return "\n\n".join(parts)

def build_prompt(illness: str, patient_context: str) -> Dict[str, str]:
    system_msg = f"""
Tu es un médecin hospitalier francophone expérimenté.
Objectif : produire une synthèse centrée sur un terme cible (maladie ou traitement particulier) à partir de notes cliniques historiques.
Contraintes :
- Inclure UNIQUEMENT les phrases ou informations qui mentionnent explicitement ou décrivent un lien direct (prescription, effet, indication, résultat, complication, arrêt, échec, succès, surveillance, adaptation, etc.) avec le terme cible : "{illness}".
- Exclure toute phrase ou information qui ne mentionne pas "{illness}" ou qui n'a pas de lien causal, thérapeutique, diagnostique ou contextuel direct avec "{illness}".
- Ne pas inclure de contexte général, d'autres hospitalisations, de symptômes, de suivi ou d'évolution qui ne sont pas explicitement reliés à "{illness}".
- Si une phrase mentionne plusieurs traitements, ne conserver que la partie pertinente pour "{illness}".
- Ne pas reformuler ni résumer le contenu non lié : simplement l'omettre.
- Pour CHAQUE événement ou action mentionné(e) en lien avec "{illness}", il est OBLIGATOIRE de préciser explicitement la date ou la période à laquelle il/elle s'est produit(e) (ex : "En juin 2024, ...", "Le 02/06/2024, ...", "Pendant 5 jours, ..."). Aucune phrase ne doit être produite sans référence temporelle explicite.
- La synthèse doit être concise, cohérente et strictement centrée sur le rôle clinique, les effets et le contexte de "{illness}" dans la prise en charge du patient.
- Présenter la synthèse sous forme de paragraphes (blocs de texte séparés par une ligne vide), et surtout PAS sous forme de liste numérotée ou à puces.
- Ne rien ajouter avant ni après les paragraphes.
Format de sortie :
- Paragraphes séparés par UNE ligne vide maximum.
""".strip()

    user_msg = f"""
Terme cible : {illness}

Documents cliniques (historique) :
----------------------------------
{patient_context}
----------------------------------

Tâche : Génère la synthèse demandée (≤ 350 mots). Commencer directement par la section 1.
""".strip()
    return {"system": system_msg, "user": user_msg}

def generate_section(client: OpenAI, model: str, system_msg: str, user_msg: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    content = resp.choices[0].message.content
    return content.strip() if content is not None else ""

def slugify_illness(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "maladie"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md-dir", type=Path, default=Path("data/concatenated_docs"),
                        help="Répertoire contenant les fichiers Markdown *.md")
    parser.add_argument("--illness", required=True, help="Maladie cible (ex: BPCO)") 
    parser.add_argument("--out-dir", type=Path, default=Path("data/discharge_summaries"),
                        help="Répertoire de sortie (défaut: data/discharge_summaries)")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY manquant.")

    raw_texts = load_all_markdown(args.md_dir)
    if not raw_texts:
        raise SystemExit("Aucun fichier .md trouvé dans le répertoire.")

    context = build_context_only_text(raw_texts)
    prompt_parts = build_prompt(args.illness, context)

    client = OpenAI(api_key=api_key)
    section = generate_section(client, args.model, prompt_parts["system"], prompt_parts["user"])

    slug = slugify_illness(args.illness)
    out_path = args.out_dir / f"synthese_{slug}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(section, encoding="utf-8")
    print(f"[INFO] Synthèse enregistrée dans : {out_path}")

if __name__ == "__main__":
    main()