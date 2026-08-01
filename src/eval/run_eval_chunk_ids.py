"""Chunk-id retrieval eval: Hit@k / Recall@k vs gold ids; keyword Precision@k."""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.questions_chunk_ids import QUESTIONS_CHUNK_IDS
from src.rag.retrieve import Hit, retrieve


def _normalize(text: str) -> str:
    t = (text or "").lower()
    return (
        t.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
        .replace("–", "-")
        .replace("—", "-")
    )


def _passage_matches(text: str, keyword_groups: list[list[str]]) -> bool:
    """True if every group has at least one term appearing in the chunk text."""
    if not keyword_groups:
        return False
    norm = _normalize(text)
    for group in keyword_groups:
        if not any(_normalize(term) in norm for term in group):
            return False
    return True


def _is_relevant(hit: Hit, keyword_groups: list[list[str]]) -> bool:
    return _passage_matches(hit.text or "", keyword_groups)


def _hit_at_k(hit_ids: list[str], gold: list[str], k: int) -> float:
    """1 if any gold id appears in top-k hits, else 0."""
    if not gold:
        return float("nan")
    top = set(hit_ids[:k])
    return 1.0 if top.intersection(gold) else 0.0


def _recall_at_k(hit_ids: list[str], gold: list[str], k: int) -> float:
    """|gold ∩ top-k| / |gold|."""
    if not gold:
        return float("nan")
    top = set(hit_ids[:k])
    return len(top.intersection(gold)) / len(gold)


def _precision_at_k(hits: list[Hit], keyword_groups: list[list[str]], k: int) -> float:
    """Fraction of top-k chunks whose text matches evidence_keywords (like run_eval)."""
    if k <= 0 or not keyword_groups:
        return float("nan")
    top = hits[:k]
    return sum(1 for h in top if _is_relevant(h, keyword_groups)) / k


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) == r.get(key)]  # drop NaN
    return sum(vals) / max(len(vals), 1)


def _block(rows: list[dict], k_values: tuple[int, ...]) -> dict:
    block: dict = {"n": len(rows)}
    for k in k_values:
        block[f"hit@{k}"] = _avg(rows, f"hit@{k}")
        block[f"recall@{k}"] = _avg(rows, f"recall@{k}")
        block[f"precision@{k}"] = _avg(rows, f"precision@{k}")
    return block


def run_eval_chunk_ids(
    *,
    k_values: tuple[int, ...] = (3, 5),
    out_path: Path | None = None,
) -> dict:
    rows: list[dict] = []
    for q in QUESTIONS_CHUNK_IDS:
        hits: list[Hit] = retrieve(q["question"], top_k=max(k_values))
        hit_ids = [h.id for h in hits]
        gold = list(q.get("relevant_chunk_ids") or [])
        keyword_groups = [list(g) for g in (q.get("evidence_keywords") or [])]
        found = [i for i in hit_ids if i in gold]

        metrics: dict = {}
        for k in k_values:
            top_found = [i for i in hit_ids[:k] if i in gold]
            metrics[f"hit@{k}"] = _hit_at_k(hit_ids, gold, k)
            metrics[f"recall@{k}"] = _recall_at_k(hit_ids, gold, k)
            metrics[f"precision@{k}"] = _precision_at_k(hits, keyword_groups, k)
            metrics[f"found_ids@{k}"] = top_found
            metrics[f"relevant_hit_ids@{k}"] = [
                h.id for h in hits[:k] if _is_relevant(h, keyword_groups)
            ]

        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "gold": q.get("gold", False),
                "trap": q.get("trap", False),
                "question": q["question"],
                "hit_ids": hit_ids,
                "hit_pages": [h.metadata.get("page") for h in hits],
                "hit_scores": [round(h.score, 4) for h in hits],
                "relevant_chunk_ids": gold,
                "evidence_keywords": keyword_groups,
                "found_ids": found,
                "expected_evidence": q.get("expected_evidence"),
                "gold_pages": list(q.get("gold_pages") or []),
                **metrics,
            }
        )

    labeled = [r for r in rows if r.get("relevant_chunk_ids")]
    gold_rows = [r for r in labeled if r.get("gold")]
    answerable_labeled = [r for r in labeled if r["answerable"]]

    report = {
        "n": len(rows),
        "metric": "chunk_id_retrieval",
        "definition": (
            "Hit@k / Recall@k use relevant_chunk_ids (exact Chroma ids). "
            "Hit@k = 1 if any gold id in top-k else 0. "
            "Recall@k = |gold ∩ top-k| / |gold|. "
            "Precision@k matches evaluate.py: (# top-k hits whose text matches all "
            "evidence_keywords groups) / k. "
            "Hit/Recall averages exclude questions without relevant_chunk_ids; "
            "Precision@k averages drop NaN (no keywords)."
        ),
        "retrieval_gold": {
            **_block(gold_rows, k_values),
            "note": "Averaged over gold questions with non-empty relevant_chunk_ids (Q1–3).",
        },
        "retrieval_labeled": {
            **_block(answerable_labeled, k_values),
            "note": "Averaged over answerable questions with non-empty relevant_chunk_ids.",
        },
        "rows": rows,
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
