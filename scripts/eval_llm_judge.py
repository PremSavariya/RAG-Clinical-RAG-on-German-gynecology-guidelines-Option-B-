"""LLM-as-judge: score how well each answer matches expected_evidence (1–10).

Reads a prior answers JSONL (or eval JSON with rows). Does not re-run retrieval.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LLM_MODEL_NAME
from src.ollama_client import chat

DEFAULT_REPORT = ROOT / "reports" / "answers_finals.jsonl"

JUDGE_SYSTEM = """Du bewertest RAG-Antworten zu deutschen Leitlinien.
Vergleiche die Antwort NUR mit der erwarteten Evidenz (kein externes Wissen).
Antworte NUR als JSON: {"correctness": 1-10}
Skala: 10=vollständig korrekt, 5=teilweise, 1=falsch/fehlt."""


def _load_rows(path: Path) -> list[dict]:
    """Load answers.jsonl or an eval JSON that has a rows list."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return list(json.loads(text).get("rows") or [])


def _parse_json(text: str) -> dict:
    """Parse judge JSON; tolerate extra text around the object."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def judge(question: str, expected: str, answer: str, *, model: str) -> dict:
    """Ask the judge LLM for a single correctness score in 1..10."""
    user = f"Frage:\n{question}\n\nErwartete Evidenz:\n{expected}\n\nAntwort:\n{answer}"
    raw = chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=50,
        model=model,
    )
    data = _parse_json(raw)
    score = int(data.get("correctness", 1))
    return {"correctness": max(1, min(10, score))}


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge correctness 1-10 vs expected_evidence")
    parser.add_argument("--from-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--judge-model", default=LLM_MODEL_NAME)
    args = parser.parse_args()
    if not args.from_report.is_file():
        raise SystemExit(f"Report not found: {args.from_report}")

    rows = _load_rows(args.from_report)
    print(f"Source: {args.from_report} | judge={args.judge_model} | n={len(rows)}")

    scored: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        qid, question = row.get("id"), row.get("question") or ""
        if row.get("trap") or not row.get("answerable", True):
            skipped.append({"id": qid, "reason": "trap_or_unanswerable"})
            continue
        expected = (row.get("expected_evidence") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not expected or not answer:
            skipped.append({"id": qid, "reason": "missing_fields"})
            continue
        result = judge(question, expected, answer, model=args.judge_model)
        scored.append(
            {
                "id": qid,
                "question": question,
                **result,
                "answer": answer,
                "expected_evidence": expected,
            }
        )
        print(f"Q{qid}: {result['correctness']}/10")

    scores = [r["correctness"] for r in scored]
    out_report = {
        "metric": "llm_judge_correctness",
        "definition": "LLM scores answer vs expected_evidence only (1-10). No labels/rationale.",
        "source_report": str(args.from_report.resolve()),
        "judge_model": args.judge_model,
        "n_scored": len(scores),
        "n_skipped": len(skipped),
        "mean_correctness": (sum(scores) / len(scores)) if scores else None,
        "per_question": scored,
        "skipped": skipped,
    }
    out = ROOT / "reports" / "eval_llm_judge.json"
    out.write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "mean_correctness": out_report["mean_correctness"],
                "n_scored": len(scores),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
