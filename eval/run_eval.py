"""Run the starter evaluation set."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline import Pipeline


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def fact_match(answer: str, alternatives: str) -> bool:
    parts = [p.strip() for p in alternatives.split("|") if p.strip()]
    return any(normalise(p) in normalise(answer) for p in parts)


def main():
    pipeline = Pipeline(str(ROOT / "config.yaml"))
    rows = list(csv.DictReader((ROOT / "eval/questions.csv").open(encoding="utf-8")))
    out = []
    stats = Counter()
    for row in rows:
        result = pipeline.answer(row["question"], row["user_type"], row["service_code"] or None)
        answer = result["answer"]
        expected = [x.strip() for x in row["expected_facts"].split(";") if x.strip()]
        passed = (not expected and not result["found"]) or (
            bool(result["found"]) and all(fact_match(answer, f) for f in expected)
        )
        retrieval_hit = all(
            any(fact_match(" ".join(c.get("text", "") for c in result["chunks"]), f) for c in result["chunks"])
            for f in expected
        ) if expected else True
        out.append({
            **row,
            "answer": answer,
            "retrieved_chunk_ids": ";".join(c["chunk_id"] for c in result["chunks"]),
            "cited_pages": ";".join(f"{s['manual']} p.{s['page']}" for s in result["sources"]),
            "auto_pass": "yes" if passed else "no",
            "retrieval_hit": "yes" if retrieval_hit else "no",
            "manual_grade": "",
        })
        stats["total"] += 1
        stats["pass"] += int(passed)
        stats[f"lang_{row['language']}_total"] += 1
        stats[f"lang_{row['language']}_pass"] += int(passed)
        stats[f"user_{row['user_type']}_total"] += 1
        stats[f"user_{row['user_type']}_pass"] += int(passed)
        stats[f"ans_{row['answerable']}_total"] += 1
        stats[f"ans_{row['answerable']}_pass"] += int(passed)

    result_path = ROOT / "eval/results.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)

    print(f"Overall accuracy: {stats['pass'] / stats['total']:.1%}")
    for lang in ("en", "ta", "tanglish"):
        total = stats[f"lang_{lang}_total"]
        if total: print(f"By language {lang}: {stats[f'lang_{lang}_pass']/total:.1%}")
    for user in ("citizen", "operator"):
        total = stats[f"user_{user}_total"]
        if total: print(f"By user type {user}: {stats[f'user_{user}_pass']/total:.1%}")
    for kind in ("yes", "no"):
        total = stats[f"ans_{kind}_total"]
        if total: print(f"Answerable={kind}: {stats[f'ans_{kind}_pass']/total:.1%}")
    print("Retrieval hit rate:", sum(r["retrieval_hit"] == "yes" for r in out) / len(out))
    print("Wrote", result_path)


if __name__ == "__main__":
    main()
