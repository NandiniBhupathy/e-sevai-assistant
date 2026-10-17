"""Service-filtered hybrid retrieval using dense search, BM25 and RRF."""
from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(dense, bm25, rrf_k: int = 60):
    scores = defaultdict(float)
    for rank, (idx, _) in enumerate(dense, 1):
        scores[idx] += 1.0 / (rrf_k + rank)
    for rank, (idx, _) in enumerate(bm25, 1):
        scores[idx] += 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(index, query: str, original_query: str, service_code: str | None,
             is_generic: bool, config, top_k: int | None = None):
    """Retrieve from the requested service plus general chunks."""
    dense_q = index.dense_search(query, int(config.raw["retrieval"]["dense_top_k"]))
    if original_query != query:
        dense_orig = index.dense_search(original_query, int(config.raw["retrieval"]["dense_top_k"]))
        dense_q = dense_q + dense_orig
        # Keep best dense score per index.
        best = {}
        for idx, score in dense_q:
            best[idx] = max(best.get(idx, float("-inf")), score)
        dense_q = sorted(best.items(), key=lambda x: x[1], reverse=True)

    bm25 = index.bm25_search(original_query, int(config.raw["retrieval"]["bm25_top_k"]))
    allowed = set()
    for i, chunk in enumerate(index.chunks):
        if is_generic:
            allowed.add(i)
        elif chunk["section"] == "general" or chunk["service_code"] == service_code:
            allowed.add(i)

    dense = [(i, s) for i, s in dense_q if i in allowed]
    bm = [(i, s) for i, s in bm25 if i in allowed]
    fused = reciprocal_rank_fusion(dense, bm, int(config.raw["retrieval"]["rrf_k"]))
    limit = top_k or int(config.raw["retrieval"]["final_top_k"])

    best_dense = max((s for _, s in dense), default=0.0)
    threshold = float(config.raw["retrieval"]["similarity_threshold"])
    if best_dense < threshold:
        return [], best_dense

    results = []
    for rank, (idx, fusion_score) in enumerate(fused[:limit], 1):
        item = dict(index.chunks[idx])
        item["fusion_rank"] = rank
        item["fusion_score"] = fusion_score
        results.append(item)
    return results, best_dense
