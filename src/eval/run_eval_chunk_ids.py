"""Chunk-id retrieval eval: Hit@k / Recall@k vs gold Chroma ids (no Precision)."""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.questions_chunk_ids import QUESTIONS_CHUNK_IDS
from src.rag.retrieve import Hit, retrieve


def _hit_at_k(hit_ids: list[str], gold: list[str], k: int) -> float:
    """Hit@k = 1 if any gold chunk id appears in the top-k results, else 0."""
    if not gold:
        return float("nan")
    top = set(hit_ids[:k])
    return 1.0 if top.intersection(gold) else 0.0


def _recall_at_k(hit_ids: list[str], gold: list[str], k: int) -> float:
    """Recall@k = |gold ∩ top-k| / |gold| (how many gold ids we found)."""
    if not gold:
        return float("nan")
    top = set(hit_ids[:k])
    return len(top.intersection(gold)) / len(gold)


def _avg(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if r.get(key) == r.get(key)]  # drop NaN
    return sum(vals) / max(len(vals), 1)


def _block(rows: list[dict], k_values: tuple[int, ...]) -> dict:
    """Average Hit@k and Recall@k over a subset of questions."""
    block: dict = {"n": len(rows)}
    for k in k_values:
        block[f"hit@{k}"] = _avg(rows, f"hit@{k}")
        block[f"recall@{k}"] = _avg(rows, f"recall@{k}")
    return block


def run_eval_chunk_ids(
    *,
    k_values: tuple[int, ...] = (3, 5),
    out_path: Path | None = None,
) -> dict:
    """Run retrieve() for each labeled question and score against relevant_chunk_ids."""
    rows: list[dict] = []
    for q in QUESTIONS_CHUNK_IDS:
        # Same retrieve path as the app (hybrid / rerank from config).
        hits: list[Hit] = retrieve(q["question"], top_k=max(k_values))
        hit_ids = [h.id for h in hits]
        gold = list(q.get("relevant_chunk_ids") or [])
        found = [i for i in hit_ids if i in gold]

        metrics: dict = {}
        for k in k_values:
            top_found = [i for i in hit_ids[:k] if i in gold]
            metrics[f"hit@{k}"] = _hit_at_k(hit_ids, gold, k)
            metrics[f"recall@{k}"] = _recall_at_k(hit_ids, gold, k)
            metrics[f"found_ids@{k}"] = top_found

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
            "Averages exclude questions without relevant_chunk_ids."
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
