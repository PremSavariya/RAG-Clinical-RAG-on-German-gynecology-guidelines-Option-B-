"""Chunk-id Hit@k / Recall@k + keyword Precision@k (like evaluate.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval_chunk_ids import run_eval_chunk_ids


def main() -> None:
    out = ROOT / "reports" / "eval_recall_chunk_ids_4.json"
    report = run_eval_chunk_ids(out_path=out)

    print("=== Chunk-id retrieval summary ===")
    print(json.dumps(
        {
            "retrieval_gold": report["retrieval_gold"],
            "retrieval_labeled": report["retrieval_labeled"],
            "definition": report["definition"],
            "out": str(out),
        },
        ensure_ascii=False,
        indent=2,
    ))

    print("\n=== Per-question hit_ids & rates ===")
    for row in report["rows"]:
        print(f"\nQ{row['id']} gold={row.get('gold')} labeled={bool(row.get('relevant_chunk_ids'))}")
        print(f"  question: {row['question'][:80]}…")
        print(f"  relevant_chunk_ids: {row['relevant_chunk_ids']}")
        print(f"  hit_ids: {row['hit_ids']}")
        print(f"  found_ids: {row['found_ids']}")
        if row.get("relevant_chunk_ids") or row.get("evidence_keywords"):
            print(
                f"  hit@3={row.get('hit@3')} recall@3={row.get('recall@3')} "
                f"precision@3={row.get('precision@3')}"
            )
            print(
                f"  hit@5={row.get('hit@5')} recall@5={row.get('recall@5')} "
                f"precision@5={row.get('precision@5')}"
            )


if __name__ == "__main__":
    main()
