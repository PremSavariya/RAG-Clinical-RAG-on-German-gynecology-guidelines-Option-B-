"""Step 3 of ingest: chunks → Ollama embeddings → Chroma collection.

This is what retrieve.py queries later. Cosine space matches how we convert
Chroma distances to similarity scores (1 - distance).
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb

from src.config import CHROMA_DIR
from src.ingest.chunk import chunk_elements
from src.ollama_client import embed_texts

COLLECTION = "guidelines"


def load_elements(jsonl: Path) -> list[dict]:
    rows = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def get_collection(reset: bool = False):
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def index_elements(jsonl: Path, *, reset: bool = False, batch_size: int = 32) -> int:
    """Chunk elements.jsonl, embed as passages, upsert into Chroma."""
    records = load_elements(jsonl)
    chunks = chunk_elements(records)
    col = get_collection(reset=reset)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed_texts([c["text"] for c in batch], for_query=False)
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings,
            metadatas=[
                {
                    "doc_id": c["doc_id"],
                    "type": c["type"],
                    "page": c["page"] if c["page"] is not None else -1,
                    "source": c["source"],
                    "path": c["path"] or "",
                }
                for c in batch
            ],
        )
    return len(chunks)
