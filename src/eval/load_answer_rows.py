"""Load answer rows from answers.jsonl or an eval JSON with rows[]."""
from __future__ import annotations

import json
from pathlib import Path


def load_answer_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return list(data.get("rows") or [])
