"""Small FastAPI UI for asking the local RAG system.

Thin wrapper around src.rag.generate.answer_question.
Optional form overrides: LLM, top_k, temperature (eval/CLI unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config import CFG, EMBEDDINGS_MODEL_NAME, LLM_MODEL_NAME
from src.rag.generate import answer_question

TEMPLATES_DIR = ROOT / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app = FastAPI(title="Nixi RAG", docs_url=None, redoc_url=None)

_SNIPPET = 280
LLM_CHOICES = ("qwen2.5:1.5b", "qwen2.5:3b")
DEFAULT_TOP_K = int(CFG["retrieval"]["top_k"])
DEFAULT_TEMPERATURE = float(CFG["generation"]["temperature"])


def _hit_view(h) -> dict:
    text = h.text or ""
    return {
        "id": h.id,
        "source": h.metadata.get("source") or "",
        "page": h.metadata.get("page"),
        "score": round(float(h.score), 4),
        "type": h.metadata.get("type") or "",
        "snippet": text if len(text) <= _SNIPPET else text[:_SNIPPET].rstrip() + "…",
    }


def _base_ctx(
    *,
    question: str = "",
    answer=None,
    refused=None,
    hits=None,
    error=None,
    llm_model: str | None = None,
    top_k: int | None = None,
    temperature: float | None = None,
) -> dict:
    chosen = llm_model or LLM_MODEL_NAME
    if chosen not in LLM_CHOICES:
        chosen = LLM_CHOICES[0] if LLM_MODEL_NAME not in LLM_CHOICES else LLM_MODEL_NAME
    return {
        "question": question,
        "answer": answer,
        "refused": refused,
        "hits": hits or [],
        "llm_model": chosen,
        "llm_choices": LLM_CHOICES,
        "embed_model": EMBEDDINGS_MODEL_NAME,
        "top_k": DEFAULT_TOP_K if top_k is None else top_k,
        "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
        "error": error,
    }


def _parse_top_k(raw: str | None) -> int:
    try:
        k = int(raw) if raw is not None and str(raw).strip() != "" else DEFAULT_TOP_K
    except ValueError:
        k = DEFAULT_TOP_K
    return max(1, min(20, k))


def _parse_temperature(raw: str | None) -> float:
    try:
        t = float(raw) if raw is not None and str(raw).strip() != "" else DEFAULT_TEMPERATURE
    except ValueError:
        t = DEFAULT_TEMPERATURE
    return max(0.0, min(1.0, t))


def _parse_model(raw: str | None) -> str:
    m = (raw or "").strip() or LLM_MODEL_NAME
    return m if m in LLM_CHOICES else (
        LLM_MODEL_NAME if LLM_MODEL_NAME in LLM_CHOICES else LLM_CHOICES[0]
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", _base_ctx())


@app.post("/ask", response_class=HTMLResponse)
async def ask(
    request: Request,
    question: str = Form(...),
    llm_model: str = Form(default=""),
    top_k: str = Form(default=""),
    temperature: str = Form(default=""),
) -> HTMLResponse:
    q = (question or "").strip()
    model = _parse_model(llm_model)
    k = _parse_top_k(top_k)
    temp = _parse_temperature(temperature)

    if not q:
        return templates.TemplateResponse(
            request,
            "index.html",
            _base_ctx(
                question=q,
                error="Bitte eine Frage eingeben.",
                llm_model=model,
                top_k=k,
                temperature=temp,
            ),
        )

    try:
        # Same RAG path as scripts/ask.py; form can override model / top_k / temp.
        result = answer_question(q, top_k=k, temperature=temp, model=model)
        return templates.TemplateResponse(
            request,
            "index.html",
            _base_ctx(
                question=q,
                answer=result.answer,
                refused=result.refused,
                # Hide chunk list on refusal — nothing useful to cite.
                hits=[] if result.refused else [_hit_view(h) for h in result.hits],
                llm_model=model,
                top_k=k,
                temperature=temp,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — show in UI for local demo
        return templates.TemplateResponse(
            request,
            "index.html",
            _base_ctx(
                question=q,
                error=f"{type(exc).__name__}: {exc}",
                llm_model=model,
                top_k=k,
                temperature=temp,
            ),
        )
