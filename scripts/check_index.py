"""Load the built index and run retrieval-only smoke queries.

Usage:
    python scripts/check_index.py
    python scripts/check_index.py "how to apply for income certificate"

Reports OCR coverage of the indexed screenshot chunks and, for each query, the
top dense hits. Needs no Gemini key.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from index import SearchIndex

DEFAULT_QUERIES = (
    "how to apply for income certificate",
    "what documents are needed for nativity certificate",
)

PREVIEW_CHARS = 110


def summarise(index: SearchIndex) -> None:
    image_chunks = [chunk for chunk in index.chunks if chunk["chunk_type"] == "image"]
    with_ocr = [chunk for chunk in image_chunks if chunk.get("ocr_text", "").strip()]
    confidences = [chunk["ocr_confidence"] for chunk in with_ocr if chunk["ocr_confidence"]]

    print(f"chunks          : {len(index.chunks)}")
    print(f"vectors         : {index.faiss.ntotal} (dim {index.faiss.d})")
    print(f"image chunks    : {len(image_chunks)}")
    print(f"  with OCR text : {len(with_ocr)}")
    if confidences:
        print(f"  mean OCR conf : {sum(confidences) / len(confidences):.1f}")
    services = sorted({chunk["service_code"] for chunk in index.chunks})
    print(f"services        : {', '.join(services)}")


def run_query(index: SearchIndex, query: str, top_k: int = 3) -> None:
    print(f"\nquery: {query}")
    hits = index.dense_search(query, k=top_k)
    if not hits:
        print("  no dense hits")
        return
    for chunk_id, score in hits:
        chunk = index.chunks[chunk_id]
        preview = " ".join(chunk["text"].split())[:PREVIEW_CHARS]
        print(f"  {score:6.3f}  {chunk['chunk_id']:<22} {chunk['service_name']} p{chunk['page']}")
        print(f"          {preview}")
        if chunk.get("image_path"):
            print(f"          crop: {chunk['image_path']}")


def main() -> int:
    config = load_config(ROOT / "config.yaml")
    index = SearchIndex(config)
    index.load()

    summarise(index)
    for query in sys.argv[1:] or DEFAULT_QUERIES:
        run_query(index, query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
