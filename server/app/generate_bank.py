# app/generate_bank.py
"""Offline: generate CANDIDATE quiz questions for every notion and write them
to a spreadsheet the trainers can review. Nothing here reaches the learners
until import_bank.py ingests the rows marked « validé ».

  MISTRAL_API_KEY=... python generate_bank.py --per-kc 6 --out bank_candidates.xlsx
  (add --provider openai to use OPENAI_API_KEY)

Columns: id, kc_id, notion, chapitre, page, question, A, B, C, D, reponse,
justification, difficulte, statut (à remplir : validé / rejeté), commentaire."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("STATE_LOCAL_DIR", "/tmp/its-bank-state")

from app.orchestrator import INSTR_QCM, KcGraph, KC_GRAPH_PATH  # noqa: E402
from app.content import doctrine  # noqa: E402
from app.providers import ProviderChoice, current_provider, run_structured  # noqa: E402
from app.schemas import QcmList, normalize_questions  # noqa: E402

HEADERS = ["id", "kc_id", "notion", "chapitre", "page", "question", "A", "B", "C", "D", "reponse", "justification", "difficulte", "statut", "commentaire"]


async def generate_for_kc(graph: KcGraph, kc_id: str, per_kc: int) -> list[dict]:
    kc = graph.nodes[kc_id]
    pages = graph.kc_pages(kc)
    doc = doctrine()
    rules = doc.rules_for_pages(pages)
    prompt = (
        f"Rédige {per_kc} questions à choix multiples variées sur la notion « {kc.title} » ({kc.id}), pages {pages} du mémento : "
        f"{max(1, per_kc // 2)} faciles (reconnaissance, définitions), le reste moyennes ou difficiles (application, comparaison, distracteurs plausibles). "
        f"Points de doctrine à couvrir si possible : {rules[:10]}\n\n{doc.pages_text(pages, max_chars=7000)}\n\n"
        "Contraintes : question de moins de 240 caractères en français ; 4 propositions courtes et distinctes ; une seule juste ; answer = lettre ; "
        "explanation = une phrase avec la page ; page = page source ; kc_id = identifiant fourni ; pas de doublon ; pas de réponse visible dans l'énoncé."
    )
    out = await run_structured("Bank-QCM", INSTR_QCM, prompt, QcmList, None)
    questions = normalize_questions(out.questions, kc.id)
    for i, q in enumerate(questions):
        q["difficulty"] = "easy" if i < max(1, per_kc // 2) else "medium"
        q["module"] = graph.module_title(kc.id)
        q["kc_title"] = kc.title
    return questions


def write_xlsx(rows: list[dict], path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "candidats"
    ws.append(HEADERS)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for q in rows:
        c = q["choices"]
        ws.append([q["id"], q["kc_id"], q.get("kc_title", ""), q.get("module", ""), q.get("page") or "", q["text"], c[0], c[1], c[2], c[3],
                   q["answer"], q.get("explanation", ""), q.get("difficulty", ""), "", ""])
    dv = DataValidation(type="list", formula1='"validé,rejeté,à corriger"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"N2:N{max(2, len(rows) + 1)}")
    widths = {"F": 60, "G": 28, "H": 28, "I": 28, "J": 28, "L": 50, "C": 28, "D": 28}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    wb.save(path)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kc", type=int, default=6)
    ap.add_argument("--provider", default="mistral", choices=["mistral", "openai", "openrouter"])
    ap.add_argument("--out", default="bank_candidates.xlsx")
    ap.add_argument("--kc", default="", help="comma-separated kc ids (default: all)")
    args = ap.parse_args()
    current_provider.set(ProviderChoice(provider=args.provider))
    graph = KcGraph.load(KC_GRAPH_PATH)
    kc_ids = [k for k in args.kc.split(",") if k] or graph.kc_ids()
    rows: list[dict] = []
    for kc_id in kc_ids:
        try:
            qs = await generate_for_kc(graph, kc_id, args.per_kc)
            print(f"{kc_id} {graph.nodes[kc_id].title}: {len(qs)} questions")
            rows.extend(qs)
        except Exception as exc:  # noqa: BLE001
            print(f"{kc_id}: FAILED {exc}", file=sys.stderr)
    out = Path(args.out)
    if out.suffix.lower() == ".xlsx":
        write_xlsx(rows, out)
    else:
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(rows)} candidate questions to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
