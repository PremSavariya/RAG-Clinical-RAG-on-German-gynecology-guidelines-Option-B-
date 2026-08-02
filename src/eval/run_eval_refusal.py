"""Refusal eval for trap questions only (out-of-scope medical topics).

Success = the model returns the fixed refusal phrase, not a guessed answer.
Unlike Hit/Recall or answer quality, this measures "know when not to answer."
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.questions import QUESTIONS
from src.rag.generate import answer_question


def run_eval_refusal(*, out_path: Path | None = None) -> dict:
    """Run full RAG on traps only; score fraction that correctly refuse."""
    traps = [q for q in QUESTIONS if q.get("trap")]
    rows = []
    for q in traps:
        result = answer_question(q["question"])
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "refused": result.refused,
                "answer": result.answer,
            }
        )

    n = len(rows)
    report = {
        "n": n,
        "metric": "refusal",
        "definition": "Fraction of trap questions that return the refusal phrase.",
        "refusal_correctness": sum(1 for r in rows if r["refused"]) / max(n, 1),
        "rows": rows,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
