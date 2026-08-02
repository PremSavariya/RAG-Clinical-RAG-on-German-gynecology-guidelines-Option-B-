"""Load prior answers for offline evals (judge / semantic similarity).

Accepts either:
  - answers.jsonl: one object per line (from generate_answers_jsonl.py)
  - eval JSON: a report with a rows[] list

Does not re-run retrieval or generation — only reads saved answers.
"""
from __future__ import annotations

import json
from pathlib import Path


def load_answer_rows(path: Path) -> list[dict]:
    """Return a list of answer rows from .jsonl or .json."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return list(data.get("rows") or [])
