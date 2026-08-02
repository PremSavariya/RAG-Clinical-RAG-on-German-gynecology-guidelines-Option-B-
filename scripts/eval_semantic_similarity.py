"""Semantic similarity: embed expected_evidence vs answer, then cosine score.

Unlike LLM-as-judge (subjective 1–10), this is a local embedding distance —
no second LLM call. Reads a prior answers JSONL; does not re-run RAG.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EMBEDDINGS_MODEL_NAME
from src.eval.load_answer_rows import load_answer_rows
from src.ollama_client import embed_texts

DEFAULT_REPORT = ROOT / "reports" / "answers_finals.jsonl"


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [−1, 1]; NaN if vectors are empty or mismatched."""
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cosine semantic similarity: expected_evidence vs answer"
    )
    parser.add_argument(
        "--from-report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"answers.jsonl or eval JSON with rows (default: {DEFAULT_REPORT})",
    )
    args = parser.parse_args()
    report_path = args.from_report
    if not report_path.is_file():
        raise SystemExit(f"Report not found: {report_path}")

    rows = load_answer_rows(report_path)
    print(f"Source report: {report_path}")
    print(f"Embedding model: {EMBEDDINGS_MODEL_NAME}")
    print(f"Scoring semantic similarity from {len(rows)} row(s)...")

    per_question: list[dict] = []
    skipped: list[dict] = []

    for row in rows:
        qid = row.get("id")
        question = row.get("question") or ""
        # Traps belong in refusal eval, not answer-quality similarity.
        if row.get("trap") or not row.get("answerable", True):
            skipped.append({"id": qid, "reason": "trap_or_unanswerable", "question": question})
            continue

        expected = (row.get("expected_evidence") or "").strip()
        generated = (row.get("answer") or "").strip()
        if not expected:
            skipped.append({"id": qid, "reason": "no_expected_evidence", "question": question})
            continue
        if not generated:
            skipped.append({"id": qid, "reason": "no_answer", "question": question})
            continue

        # Same embedding model as indexing; for_query=False = passage style.
        vecs = embed_texts([expected, generated], for_query=False)
        score = cosine(vecs[0], vecs[1])

        per_question.append(
            {
                "id": qid,
                "question": question,
                "semantic_similarity": score if score == score else None,
                "refused": row.get("refused"),
                "expected_evidence": expected,
                "generated_answer": generated,
            }
        )

    scores = [
        r["semantic_similarity"]
        for r in per_question
        if r["semantic_similarity"] is not None
    ]
    mean_s = sum(scores) / len(scores) if scores else None

    out_report = {
        "metric": "semantic_similarity",
        "definition": (
            "Cosine similarity between embeddings of expected_evidence and answer "
            "from answers.jsonl or an eval JSON (local EMBEDDINGS_MODEL_NAME). "
            "No re-generation."
        ),
        "source_report": str(report_path.resolve()),
        "embedding_model": EMBEDDINGS_MODEL_NAME,
        "n_scored": len(scores),
        "n_skipped": len(skipped),
        "mean_semantic_similarity": mean_s,
        "per_question": per_question,
        "skipped": skipped,
    }

    out = ROOT / "reports" / "eval_semantic_similarity_finals.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "mean_semantic_similarity": mean_s,
        "n_scored": len(scores),
        "n_skipped": len(skipped),
        "source_report": str(report_path),
        "embedding_model": EMBEDDINGS_MODEL_NAME,
        "per_question": [
            {"id": r["id"], "semantic_similarity": r["semantic_similarity"]}
            for r in per_question
        ],
        "out": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
