from __future__ import annotations

from src.config import CFG


def chunk_elements(records: list[dict]) -> list[dict]:
    """Element-aware chunking: keep tables whole when small; split long text."""
    max_chars = CFG["chunking"]["max_chars"]
    overlap = CFG["chunking"]["overlap"]
    chunks: list[dict] = []

    for rec in records:
        text = rec["text"]
        if rec["type"] == "Table" or len(text) <= max_chars:
            chunks.append(_as_chunk(rec, text, 0))
            continue
        for i, piece in enumerate(_split(text, max_chars, overlap)):
            chunks.append(_as_chunk(rec, piece, i))
    return chunks


def _as_chunk(rec: dict, text: str, part: int) -> dict:
    return {
        "id": f"{rec['id']}_c{part}",
        "doc_id": rec["doc_id"],
        "type": rec["type"],
        "page": rec["page"],
        "text": text,
        "path": rec.get("path"),
        "source": rec["source"],
    }


def _split(text: str, max_chars: int, overlap: int) -> list[str]:
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]
