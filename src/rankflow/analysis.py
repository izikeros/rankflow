"""Rank change analysis and top-K filtering utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np


def compute_rank_deltas(ranks: np.ndarray) -> np.ndarray:
    """Compute rank changes between consecutive steps.

    Returns an array of shape (n_steps - 1, n_chunks) where positive values
    mean the chunk moved down (worse rank) and negative means moved up (better).
    """
    return np.diff(ranks, axis=0)


def compute_summary(
    ranks: np.ndarray,
    chunk_labels: list[str],
) -> list[dict]:
    """Per-chunk summary: initial rank, final rank, max gain, max loss, total displacement.

    Returns a list of dicts (one per chunk).
    """
    deltas = compute_rank_deltas(ranks)
    n_chunks = ranks.shape[1]
    summaries = []
    for i in range(n_chunks):
        col_deltas = deltas[:, i]
        summaries.append(
            {
                "chunk": chunk_labels[i],
                "initial_rank": int(ranks[0, i]),
                "final_rank": int(ranks[-1, i]),
                "rank_change": int(ranks[-1, i] - ranks[0, i]),
                "max_gain": int(-np.min(col_deltas)) if len(col_deltas) > 0 else 0,
                "max_loss": int(np.max(col_deltas)) if len(col_deltas) > 0 else 0,
                "total_displacement": int(np.sum(np.abs(col_deltas))),
            }
        )
    return summaries


def filter_top_k(
    ranks: np.ndarray,
    chunk_labels: list[str],
    k: int,
    mode: Literal["any", "initial", "final"] = "any",
) -> tuple[np.ndarray, list[str], list[int]]:
    """Filter to only chunks appearing in top-K.

    Args:
        ranks: (n_steps, n_chunks) array.
        chunk_labels: labels for chunks.
        k: top-K threshold.
        mode: "any" = appears in top-K at any step,
              "initial" = top-K at first step,
              "final" = top-K at last step.

    Returns:
        Filtered ranks, filtered labels, original indices of kept chunks.
    """
    ranks.shape[1]

    if mode == "initial":
        keep = set(np.argsort(ranks[0])[:k])
    elif mode == "final":
        keep = set(np.argsort(ranks[-1])[:k])
    else:  # "any"
        keep = set()
        for step in range(ranks.shape[0]):
            keep |= set(np.argsort(ranks[step])[:k])

    keep_sorted = sorted(keep)
    filtered_ranks = ranks[:, keep_sorted]
    filtered_labels = [chunk_labels[i] for i in keep_sorted]
    return filtered_ranks, filtered_labels, keep_sorted
