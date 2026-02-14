"""Export utilities for RankFlow data."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def ranks_to_dict(
    ranks: np.ndarray,
    step_labels: list[str],
    chunk_labels: list[str],
    metrics: list[dict[str, float]] | None = None,
    summary: list[dict] | None = None,
) -> dict[str, Any]:
    """Convert rankflow data to a nested dictionary."""
    result: dict[str, Any] = {
        "step_labels": step_labels,
        "chunk_labels": chunk_labels,
        "ranks": ranks.tolist(),
    }
    if metrics is not None:
        result["metrics_per_step"] = metrics
    if summary is not None:
        result["summary"] = summary
    return result


def ranks_to_dataframe(
    ranks: np.ndarray,
    step_labels: list[str],
    chunk_labels: list[str],
):
    """Convert ranks array back to a pandas DataFrame."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for to_dataframe(). "
            "Install it with: pip install rankflow[pandas]"
        ) from exc
    return pd.DataFrame(ranks, index=step_labels, columns=chunk_labels)


def ranks_to_json(
    path: str,
    ranks: np.ndarray,
    step_labels: list[str],
    chunk_labels: list[str],
    metrics: list[dict[str, float]] | None = None,
    summary: list[dict] | None = None,
) -> None:
    """Export rankflow data to a JSON file."""
    data = ranks_to_dict(ranks, step_labels, chunk_labels, metrics, summary)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
