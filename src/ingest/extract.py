from __future__ import annotations

import json
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from src.config import ARTIFACTS_DIR, CFG


def extract_pdf(pdf_path: Path, *, strategy: str | None = None) -> Path:
    """Partition PDF with Unstructured; write tables + elements.jsonl (text/tables only)."""
    strategy = strategy or CFG["extract"]["strategy"]
    languages = CFG["extract"]["languages"]
    doc_id = pdf_path.stem
    out_dir = ARTIFACTS_DIR / doc_id
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        languages=languages,
        infer_table_structure=True,
        extract_images_in_pdf=False,
    )

    records: list[dict] = []
    tbl_i = 0
    for idx, el in enumerate(elements):
        etype = type(el).__name__
        if etype == "Image":
            continue

        meta = el.metadata.to_dict() if el.metadata else {}
        page = meta.get("page_number")
        text = (el.text or "").strip()
        path = None

        if etype == "Table":
            html = meta.get("text_as_html") or text
            path = str(tbl_dir / f"p{page or 0}_{tbl_i}.html")
            Path(path).write_text(html, encoding="utf-8")
            text = _html_to_plain(html) or text
            tbl_i += 1

        if not text:
            continue

        records.append(
            {
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "type": etype,
                "page": page,
                "text": text,
                "path": path,
                "source": pdf_path.name,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "elements.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return jsonl


def _html_to_plain(html: str) -> str:
    # light strip; enough for embedding/search
    import re

    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p>", "\n", text, flags=re.I)
    text = re.sub(r"</tr>", "\n", text, flags=re.I)
    text = re.sub(r"</t[dh]>", " | ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()
