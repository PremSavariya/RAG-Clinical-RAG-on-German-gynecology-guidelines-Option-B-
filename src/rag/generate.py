"""RAG answer step: retrieve chunks → prompt the LLM → optional citations."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import CFG
from src.ollama_client import chat
from src.rag.prompts import SYSTEM_PROMPT, build_user_prompt, is_refusal
from src.rag.retrieve import Hit, retrieve


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    hits: list[Hit]


def _format_citations(hits: list[Hit]) -> str:
    """Build a Quellen block from the top retrieved chunks (not from the LLM)."""
    lines = []
    for h in hits[:3]:
        src = h.metadata.get("source", "?")
        page = h.metadata.get("page", "?")
        lines.append(f"[{h.id}] (source={src}, S. {page})")
    return "Quellen:\n" + "\n".join(lines)


def answer_question(
    question: str,
    *,
    top_k: int | None = None,
    temperature: float | None = None,
    model: str | None = None,
) -> Answer:
    """Retrieve context, generate a German answer, append citations unless refused."""
    hits = retrieve(question, top_k=top_k)
    contexts = [
        {
            "id": h.id,
            "text": h.text,
            "source": h.metadata.get("source"),
            "page": h.metadata.get("page"),
            "type": h.metadata.get("type"),
        }
        for h in hits
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, contexts)},
    ]
    temp = CFG["generation"]["temperature"] if temperature is None else temperature
    text = chat(
        messages,
        temperature=temp,
        max_tokens=CFG["generation"]["max_tokens"],
        model=model,
    )
    # Refusal = out-of-scope / not in context; do not attach Quellen then.
    refused = is_refusal(text)
    if text and not refused and hits:
        text = text.rstrip() + "\n\n" + _format_citations(hits)
    return Answer(question=question, answer=text, refused=refused, hits=hits)
