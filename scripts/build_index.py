"""Build local vector index from PDFs in data/pdfs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PDF_DIR
from src.ingest.extract import extract_pdf
from src.ingest.index import index_elements


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract + index guideline PDFs")
    parser.add_argument("--pdf", type=Path, help="Single PDF path (default: all in data/pdfs)")
    parser.add_argument("--reset", action="store_true", help="Reset Chroma collection")
    parser.add_argument("--strategy", default=None, help="Unstructured strategy override")
    args = parser.parse_args()

    pdfs = [args.pdf] if args.pdf else sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {PDF_DIR}. Place guideline PDFs there first.")

    reset = args.reset
    total = 0
    for pdf in pdfs:
        print(f"Extracting {pdf.name} ...")
        jsonl = extract_pdf(pdf, strategy=args.strategy)
        print(f"Indexing {jsonl} ...")
        n = index_elements(jsonl, reset=reset)
        print(f"  -> {n} chunks")
        total += n
        reset = False  # only reset once
    print(f"Done. Indexed {total} chunks from {len(pdfs)} PDF(s).")


if __name__ == "__main__":
    main()
