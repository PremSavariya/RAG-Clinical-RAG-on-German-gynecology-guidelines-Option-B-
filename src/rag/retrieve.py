from __future__ import annotations

from dataclasses import dataclass

import chromadb
from rank_bm25 import BM25Okapi

from src.config import CFG, CHROMA_DIR
from src.ollama_client import embed_texts

COLLECTION = "guidelines"

# Lazy-loaded cross-encoder (optional; see retrieval.rerank in config).
_cross_encoder = None

# Out-of-scope traps — do not expand (keeps refusal path clean).
_TRAP_MARKERS = (
    "schlaganfall",
    "thrombolyse",
    "pneumonie",
    "antibiotikum",
    "hypertonie",
)

# Triage / follow-up intents — more specific than plain age-band screening.
_TRIAGE_MARKERS = (
    "hpv-positiv",
    "hpv positiv",
    "zytologisch unauff",
    "zytologisch negativ",
    "pap-negativ",
    "pap negativ",
)

# Age-band expansions skipped when a triage marker already matched (avoids
# pulling primary screening interval chunks ahead of 12-month follow-up).
_AGE_BAND_MARKERS = (
    "20 bis 34",
    "20-34",
    "20–34",
    "ab 35",
    "ab35",
    "35 jahren",
    "35 jahre",
)

# Simple German guideline expansions: (substring markers, phrase to append).
_EXPANSIONS: list[tuple[tuple[str, ...], str]] = [
    (
        _TRIAGE_MARKERS,
        "Ko-Testung Wiederholung nach 12 Monaten Kontrolle",
    ),
    (
        ("selbstabnahme", "selbstentnahme", "self-sampling", "eigenentnahme"),
        "Non-Responder nicht an der regulären Früherkennung",
    ),
    (
        ("impfung", "impfstoff", "vakzin"),
        "Primärprävention ersetzt Screening nicht",
    ),
    (
        ("20 bis 34", "20-34", "20–34"),
        "zytologische Untersuchung jährlich Screening",
    ),
    (
        ("ab 35", "ab35", "35 jahren", "35 jahre"),
        "Ko-Testung HPV Zytologie 3-Jahres-Intervall",
    ),
]

# Prefer the Prävention guideline over Diagnostik/Therapie for screening Qs.
_SCREENING_MARKERS = (
    "screening",
    "früherkenn",
    "frueherkenn",
    "zytolog",
    "hpv",
    "ko-test",
    "kotest",
    "zervix",
    "20 bis 34",
    "20-34",
    "20–34",
    "ab 35",
    "selbstabnahme",
    "selbstentnahme",
    "organisiert",
)

_PRAEVENTION_HINTS = ("015-027", "praevention", "prävention")


def expand_query(query: str) -> str:
    """Append light clarifying phrases for known guideline intents. No extra deps."""
    q = (query or "").strip()
    if not q:
        return q
    lower = q.lower()
    if any(m in lower for m in _TRAP_MARKERS):
        return q
    triage = any(m in lower for m in _TRIAGE_MARKERS)
    extras: list[str] = []
    for markers, phrase in _EXPANSIONS:
        if triage and any(m in _AGE_BAND_MARKERS for m in markers):
            continue
        if any(m in lower for m in markers):
            extras.append(phrase)
    if not extras:
        return q
    seen: set[str] = set()
    uniq = []
    for p in extras:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return f"{q} {' '.join(uniq)}"


@dataclass
class Hit:
    id: str
    text: str
    score: float
    metadata: dict


def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


def retrieve(query: str, *, top_k: int | None = None) -> list[Hit]:
    top_k = top_k or CFG["retrieval"]["top_k"]
    candidate_k = CFG["retrieval"]["candidate_k"]
    col = _collection()
    original = (query or "").strip()
    expanded = expand_query(original)

    q_emb = embed_texts([expanded], for_query=True)[0]
    dense = col.query(
        query_embeddings=[q_emb],
        n_results=min(candidate_k, col.count() or 1),
        include=["documents", "metadatas", "distances"],
    )
    dense_hits = _from_chroma(dense)

    if not CFG["retrieval"].get("hybrid", True):
        candidates = _lexical_boost(original, expanded, dense_hits)
    else:
        # BM25 over all stored docs (guideline PDFs are small).
        all_docs = col.get(include=["documents", "metadatas"])
        bm25_hits = _bm25(expanded, all_docs, k=candidate_k)
        fused = _rrf(dense_hits, bm25_hits, top_k=candidate_k)
        candidates = _lexical_boost(original, expanded, fused)

    if CFG["retrieval"].get("rerank", False):
        return _cross_encoder_rerank(original, candidates, top_k=top_k)
    return candidates[:top_k]


def _from_chroma(result: dict) -> list[Hit]:
    hits: list[Hit] = []
    ids = result["ids"][0]
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    for i, doc, meta, dist in zip(ids, docs, metas, dists):
        # chroma cosine distance: similarity = 1 - distance
        hits.append(Hit(id=i, text=doc, score=1.0 - float(dist), metadata=meta or {}))
    return hits


def _bm25(query: str, store: dict, *, k: int) -> list[Hit]:
    docs = store.get("documents") or []
    ids = store.get("ids") or []
    metas = store.get("metadatas") or []
    if not docs:
        return []
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_s = max(scores) or 1.0
    return [
        Hit(id=ids[i], text=docs[i], score=float(scores[i]) / max_s, metadata=metas[i] or {})
        for i in ranked
        if scores[i] > 0
    ]


def _rrf(a: list[Hit], b: list[Hit], *, top_k: int, k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    payload: dict[str, Hit] = {}
    for rank, hit in enumerate(a):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        payload[hit.id] = hit
    for rank, hit in enumerate(b):
        scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank + 1)
        payload.setdefault(hit.id, hit)
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    out: list[Hit] = []
    for i in ordered:
        h = payload[i]
        out.append(Hit(id=h.id, text=h.text, score=scores[i], metadata=h.metadata))
    return out


def _is_praevention(hit: Hit) -> bool:
    meta = hit.metadata or {}
    blob = f"{meta.get('doc_id', '')} {meta.get('source', '')} {hit.id}".lower()
    return any(h in blob for h in _PRAEVENTION_HINTS)


def _prefer_praevention(query: str) -> bool:
    lower = (query or "").lower()
    if any(m in lower for m in _TRAP_MARKERS):
        return False
    return any(m in lower for m in _SCREENING_MARKERS)


def _phrase_bonus(expanded_query: str, text: str) -> float:
    """Small lexical bump when chunk text matches intent phrases already in the query."""
    q = expanded_query.lower()
    t = (text or "").lower()
    bonus = 0.0
    if any(p in q for p in ("12 monat", "wiederholung", "kontrolle")):
        if any(p in t for p in ("12 monat", "12-monat", "nach 12", "einem jahr", "wiederhol")):
            bonus += 0.025
    if any(p in q for p in ("3-jahres", "3 jahres", "jährlich", "jaehrlich")):
        if any(p in t for p in ("3 jahr", "drei jahr", "alle 3", "jährlich", "jaehrlich")):
            bonus += 0.015
    if "20" in q and "34" in q:
        if "20" in t and "34" in t:
            bonus += 0.015
    return bonus


def _lexical_boost(original: str, expanded: str, hits: list[Hit]) -> list[Hit]:
    """Light post-RRF boost: Prävention preference + intent phrase overlap (keeps all hits)."""
    prefer = _prefer_praevention(original) or _prefer_praevention(expanded)
    scored: list[tuple[float, Hit]] = []
    for h in hits:
        score = float(h.score)
        if prefer and _is_praevention(h):
            score += 0.02
        score += _phrase_bonus(expanded, h.text or "")
        scored.append((score, h))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        Hit(id=h.id, text=h.text, score=score, metadata=h.metadata) for score, h in scored
    ]


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        model_name = CFG["retrieval"].get("rerank_model") or "BAAI/bge-reranker-v2-m3"
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


def _cross_encoder_rerank(query: str, hits: list[Hit], *, top_k: int) -> list[Hit]:
    """Score (query, passage) pairs with a multilingual cross-encoder; return top_k."""
    if not hits:
        return []
    model = _get_cross_encoder()
    pairs = [(query, h.text or "") for h in hits]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(scores, hits), key=lambda x: float(x[0]), reverse=True)
    return [
        Hit(id=h.id, text=h.text, score=float(s), metadata=h.metadata)
        for s, h in ranked[:top_k]
    ]
