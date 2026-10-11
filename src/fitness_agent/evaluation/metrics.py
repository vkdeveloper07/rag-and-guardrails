"""Retrieval metrics for a ranked list of chunk IDs."""

import math


def _top_k_ids(retrieved_ids: list[str], k: int | None = None) -> list[str]:
    if k is None:
        return list(retrieved_ids)
    if k <= 0:
        return []
    return retrieved_ids[:k]


def _binary_relevant_set(relevant: dict[str, int]) -> set[str]:
    return {chunk_id for chunk_id, grade in relevant.items() if grade and grade > 0}


def _deduped_hits(top_ids: list[str], relevant_set: set[str]) -> list[str]:
    seen: set[str] = set()
    hits: list[str] = []
    for chunk_id in top_ids:
        if chunk_id in relevant_set and chunk_id not in seen:
            seen.add(chunk_id)
            hits.append(chunk_id)
    return hits


def precision_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Relevant results in the top-K divided by K."""
    top_ids = _top_k_ids(retrieved_ids, k)
    if not top_ids:
        return 0.0

    relevant_set = _binary_relevant_set(relevant)
    hits = len(_deduped_hits(top_ids, relevant_set))
    return hits / len(top_ids)


def recall_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Relevant results found in the top-K divided by all relevant results."""
    top_ids = _top_k_ids(retrieved_ids, k)
    relevant_set = _binary_relevant_set(relevant)
    if not relevant_set:
        return 0.0

    hits = len(_deduped_hits(top_ids, relevant_set))
    return hits / len(relevant_set)


def f1_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Harmonic mean of precision and recall."""
    precision = precision_at_k(retrieved_ids, relevant, k)
    recall = recall_at_k(retrieved_ids, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mrr(retrieved_ids: list[str], relevant: dict[str, int]) -> float:
    """Mean reciprocal rank of the first relevant result in the list."""
    relevant_set = _binary_relevant_set(relevant)
    if not relevant_set:
        return 0.0

    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_set:
            return 1.0 / index
    return 0.0


def average_precision(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Average precision over the ranked list, truncated at K if supplied."""
    top_ids = _top_k_ids(retrieved_ids, k)
    relevant_set = _binary_relevant_set(relevant)
    if not relevant_set or not top_ids:
        return 0.0

    hits = 0
    precision_sum = 0.0
    seen: set[str] = set()

    for rank, chunk_id in enumerate(top_ids, start=1):
        if chunk_id in relevant_set and chunk_id not in seen:
            hits += 1
            precision_sum += hits / rank
            seen.add(chunk_id)

    return precision_sum / len(relevant_set)


def map_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Mean average precision on the top-K results for one question."""
    return average_precision(retrieved_ids, relevant, k)


def ndcg_at_k(retrieved_ids: list[str], relevant: dict[str, int], k: int | None = None) -> float:
    """Normalized discounted cumulative gain using graded relevance values."""
    top_ids = _top_k_ids(retrieved_ids, k)
    if not top_ids:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(top_ids, start=1):
        grade = relevant.get(chunk_id, 0)
        if grade > 0:
            dcg += grade / math.log2(rank + 1)

    ideal_grades = sorted(relevant.values(), reverse=True)[: len(top_ids)]
    idcg = 0.0
    for rank, grade in enumerate(ideal_grades, start=1):
        if grade > 0:
            idcg += grade / math.log2(rank + 1)

    if idcg == 0:
        return 0.0
    return dcg / idcg


# Convenience aliases for the common metric names.
precision = precision_at_k
recall = recall_at_k
f1 = f1_at_k

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "mrr",
    "average_precision",
    "map_at_k",
    "ndcg_at_k",
    "precision",
    "recall",
    "f1",
]
