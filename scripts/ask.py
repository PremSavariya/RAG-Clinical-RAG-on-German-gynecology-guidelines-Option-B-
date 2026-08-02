"""CLI: ask one German question against the indexed guidelines.

Uses the same answer_question() path as the web UI (config-driven retrieve + generate).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.generate import answer_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="German question")
    args = parser.parse_args()
    q = args.question or input("Frage: ").strip()
    result = answer_question(q)
    print(result.answer)
    # Show which chunks the answer was grounded on (useful for debugging retrieval).
    print("\n--- retrieved ---")
    for h in result.hits:
        page = h.metadata.get("page")
        print(f"[{h.id}] score={h.score:.3f} page={page} type={h.metadata.get('type')}")


if __name__ == "__main__":
    main()
