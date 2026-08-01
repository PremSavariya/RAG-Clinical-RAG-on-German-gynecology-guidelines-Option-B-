"""Run refusal eval on trap questions (Q10–12)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval_refusal import run_eval_refusal


def main() -> None:
    out = ROOT / "reports" / "eval_refusal.json"
    report = run_eval_refusal(out_path=out)
    print(json.dumps(
        {
            "n": report["n"],
            "refusal_correctness": report["refusal_correctness"],
            "per_question": [
                {"id": r["id"], "refused": r["refused"]} for r in report["rows"]
            ],
            "out": str(out),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
