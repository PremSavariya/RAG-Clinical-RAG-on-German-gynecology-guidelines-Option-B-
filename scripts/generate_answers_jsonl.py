"""Generate reports/answers.jsonl for LLM-judge and semantic similarity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.questions import QUESTIONS
from src.rag.generate import answer_question


def main() -> None:
    out = ROOT / "reports" / "answers.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for q in QUESTIONS:
            if q.get("trap") or not q.get("answerable", True):
                continue
            result = answer_question(q["question"])
            row = {
                "id": q["id"],
                "question": q["question"],
                "answer": result.answer,
                "expected_evidence": q.get("expected_evidence") or "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
            print(f"Q{q['id']} done")
    print(json.dumps({"n": n, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
