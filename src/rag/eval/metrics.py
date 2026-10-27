"""Retrieval evaluation metrics: Recall@k / MRR@k / nDCG@k / MAP@k.

Input conventions:
- gold_pids: set (or list) of relevant pids
- ranked_pids: ordered list of pids returned by retrieval

Relevance is binary (relevant / not relevant).
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def _ranked(ranked_pids: Iterable[str], k: int) -> list[str]:
    return list(ranked_pids)[:k]


def recall_at_k(gold_pids: Iterable[str], ranked_pids: Iterable[str], k: int) -> float:
    """Recall: fraction of relevant docs found in top-k."""
    gold = set(gold_pids)
    if not gold:
        return 0.0
    return len(gold & set(_ranked(ranked_pids, k))) / len(gold)


def mrr_at_k(gold_pids: Iterable[str], ranked_pids: Iterable[str], k: int) -> float:
    """MRR@k: reciprocal rank of the first relevant document."""
    gold = set(gold_pids)
    for i, pid in enumerate(_ranked(ranked_pids, k), start=1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold_pids: Iterable[str], ranked_pids: Iterable[str], k: int) -> float:
    """nDCG@k: binary relevance; ideal ranking places all relevant docs first."""
    gold = set(gold_pids)
    dcg = 0.0
    for i, pid in enumerate(_ranked(ranked_pids, k), start=1):
        if pid in gold:
            dcg += 1.0 / math.log2(i + 1)

    ideal_n = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(gold_pids: Iterable[str], ranked_pids: Iterable[str], k: int) -> float:
    """Average Precision@k: accumulate precision at relevant ranks, divide by min(k, #relevant)."""
    gold = set(gold_pids)
    if not gold:
        return 0.0
    hits = 0
    ap = 0.0
    for i, pid in enumerate(_ranked(ranked_pids, k), start=1):
        if pid in gold:
            hits += 1
            ap += hits / i
    return ap / min(k, len(gold))


def map_at_k(
    all_gold: Iterable[Iterable[str]],
    all_ranked: Iterable[Iterable[str]],
    k: int,
) -> float:
    """MAP@k: mean of AP@k over all queries."""
    aps = [ap_at_k(g, r, k) for g, r in zip(all_gold, all_ranked)]
    return sum(aps) / len(aps) if aps else 0.0


def evaluate(
    all_gold: Iterable[Iterable[str]],
    all_ranked: Iterable[Iterable[str]],
    k_values: Iterable[int] = (10, 20, 50),
    rank_k: int = 20,
) -> dict[str, float]:
    """Compute several metrics at once, returning {'recall@10': ..., 'mrr@20': ..., ...}.

    all_gold / all_ranked are per-query lists; rank_k controls MRR/nDCG/MAP.
    """
    all_gold = list(all_gold)
    all_ranked = list(all_ranked)
    out: dict[str, float] = {}

    for k in k_values:
        recalls = [recall_at_k(g, r, k) for g, r in zip(all_gold, all_ranked)]
        out[f"recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0

    mrr = [mrr_at_k(g, r, rank_k) for g, r in zip(all_gold, all_ranked)]
    out[f"mrr@{rank_k}"] = sum(mrr) / len(mrr) if mrr else 0.0

    ndcg = [ndcg_at_k(g, r, rank_k) for g, r in zip(all_gold, all_ranked)]
    out[f"ndcg@{rank_k}"] = sum(ndcg) / len(ndcg) if ndcg else 0.0

    out[f"map@{rank_k}"] = map_at_k(all_gold, all_ranked, rank_k)
    return out
