"""Build the offline e-Sevai Saathi index."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from ingest import ingest_all
from index import SearchIndex

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    config = load_config(ROOT / "config.yaml")
    chunks, discovered = ingest_all(config)
    if not chunks:
        raise SystemExit("No chunks created. Put PDFs in data/manuals first.")
    services_path = config.services_file
    existing = json.loads(services_path.read_text(encoding="utf-8")) if services_path.exists() else {}
    for code, meta in discovered.items():
        existing[code] = {"name": meta["name"], "manual_file": meta["manual_file"], "manual_year": meta["manual_year"]}
    services_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    SearchIndex(config).build(chunks)
    print(f"Built index with {len(chunks)} chunks for {len(discovered)} manuals.")


if __name__ == "__main__":
    main()
