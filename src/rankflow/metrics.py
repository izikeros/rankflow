"""Retrieval evaluation metrics computed per-step from rank data."""

from __future__ import annotations

import contextlib

import numpy as np


def _relevant_set(
    relevant_chunks: list[str | int],
    chunk_labels: list[str],
) -> set[int]:
    """Convert relevant chunk identifiers to a set of column indices."""
    indices = set()
    for item in relevant_chunks:
        if isinstance(item, int):
            indices.add(item)
        else:
            with contextlib.suppress(ValueError):
                indices.add(chunk_labels.index(item))
    return indices


def _ranked_indices_at_step(ranks_row: np.ndarray) -> np.ndarray:
    """Return chunk indices sorted by rank (ascending) for a single step."""
    return np.argsort(ranks_row)


def precision_at_k(
    ranks_row: np.ndarray,
    relevant_indices: set[int],
    k: int,
) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = set(_ranked_indices_at_step(ranks_row)[:k])
    return len(top_k & relevant_indices) / k


def recall_at_k(
    ranks_row: np.ndarray,
    relevant_indices: set[int],
    k: int,
) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_indices or k <= 0:
        return 0.0
    top_k = set(_ranked_indices_at_step(ranks_row)[:k])
    return len(top_k & relevant_indices) / len(relevant_indices)


def mean_reciprocal_rank(
    ranks_row: np.ndarray,
    relevant_indices: set[int],
) -> float:
    """1 / rank of the first relevant document (0-based ranks converted to 1-based)."""
    if not relevant_indices:
        return 0.0
    sorted_indices = _ranked_indices_at_step(ranks_row)
    for position, idx in enumerate(sorted_indices, start=1):
        if idx in relevant_indices:
            return 1.0 / position
    return 0.0


def average_precision(
    ranks_row: np.ndarray,
    relevant_indices: set[int],
) -> float:
    """Average precision for a single step."""
    if not relevant_indices:
        return 0.0
    sorted_indices = _ranked_indices_at_step(ranks_row)
    hits = 0
    sum_precisions = 0.0
    for position, idx in enumerate(sorted_indices, start=1):
        if idx in relevant_indices:
            hits += 1
            sum_precisions += hits / position
    return sum_precisions / len(relevant_indices)


def _dcg_at_k(
    ranks_row: np.ndarray,
    relevance_scores: dict[int, float],
    k: int,
) -> float:
    """Discounted cumulative gain at k."""
    sorted_indices = _ranked_indices_at_step(ranks_row)[:k]
    dcg = 0.0
    for position, idx in enumerate(sorted_indices, start=1):
        rel = relevance_scores.get(idx, 0.0)
        dcg += (2**rel - 1) / np.log2(position + 1)
    return dcg


def ndcg_at_k(
    ranks_row: np.ndarray,
    relevance_scores: dict[int, float],
    k: int,
) -> float:
    """Normalized discounted cumulative gain at k."""
    if not relevance_scores or k <= 0:
        return 0.0
    dcg = _dcg_at_k(ranks_row, relevance_scores, k)
    # Ideal ranking: sort by relevance descending
    ideal_sorted = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for position, rel in enumerate(ideal_sorted, start=1):
        idcg += (2**rel - 1) / np.log2(position + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_metrics_per_step(
    ranks: np.ndarray,
    chunk_labels: list[str],
    relevant_chunks: list[str | int],
    k: int = 5,
    relevance_grades: dict[str | int, float] | None = None,
) -> list[dict[str, float]]:
    """Compute retrieval metrics for each step.

    Returns a list of dicts (one per step) with keys:
    precision_at_k, recall_at_k, mrr, map, ndcg_at_k
    """
    rel_indices = _relevant_set(relevant_chunks, chunk_labels)

    # Build relevance scores dict (index -> grade)
    if relevance_grades is not None:
        rel_scores: dict[int, float] = {}
        for key, grade in relevance_grades.items():
            if isinstance(key, int):
                rel_scores[key] = float(grade)
            else:
                with contextlib.suppress(ValueError):
                    rel_scores[chunk_labels.index(key)] = float(grade)
    else:
        rel_scores = dict.fromkeys(rel_indices, 1.0)

    n_steps = ranks.shape[0]
    effective_k = min(k, ranks.shape[1])
    results = []
    for step in range(n_steps):
        row = ranks[step]
        results.append(
            {
                "precision_at_k": precision_at_k(row, rel_indices, effective_k),
                "recall_at_k": recall_at_k(row, rel_indices, effective_k),
                "mrr": mean_reciprocal_rank(row, rel_indices),
                "map": average_precision(row, rel_indices),
                "ndcg_at_k": ndcg_at_k(row, rel_scores, effective_k),
            }
        )
    return results
