"""Thin OpenAI-compatible client pointed at local Ollama.

embed_texts → dense retrieval / indexing / semantic similarity
chat        → answer generation and LLM-as-judge
"""
from __future__ import annotations

from openai import OpenAI

from src.config import (
    EMBEDDINGS_MODEL_NAME,
    LLM_MODEL_NAME,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    is_e5_model,
)


def get_client() -> OpenAI:
    """Ollama speaks the OpenAI HTTP API at OLLAMA_BASE_URL."""
    return OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)


def embed_texts(texts: list[str], *, for_query: bool = False) -> list[list[float]]:
    """Embed strings with EMBEDDINGS_MODEL_NAME (one call per text for reliability)."""
    if not texts:
        return []
    prepared = [_prefix(t, for_query=for_query) for t in texts]
    client = get_client()
    # Ollama embedding endpoint accepts one input at a time more reliably
    vectors: list[list[float]] = []
    for text in prepared:
        resp = client.embeddings.create(model=EMBEDDINGS_MODEL_NAME, input=text)
        vectors.append(resp.data[0].embedding)
    return vectors


def chat(
    messages: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 512,
    model: str | None = None,
) -> str:
    """Chat completion; model defaults to LLM_MODEL_NAME from .env."""
    client = get_client()
    resp = client.chat.completions.create(
        model=model or LLM_MODEL_NAME,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _prefix(text: str, *, for_query: bool) -> str:
    # Only E5 family needs asymmetric prefixes; bge-m3 does not.
    if not is_e5_model():
        return text
    return f"query: {text}" if for_query else f"passage: {text}"
