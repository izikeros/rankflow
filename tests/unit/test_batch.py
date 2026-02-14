import numpy as np
import pytest

from rankflow.batch import BatchRankFlow
from rankflow.core import RankFlow


def _make_rf(relevant=None):
    ranks = np.array([
        [0, 1, 2, 3],
        [1, 0, 3, 2],
    ])
    return RankFlow(
        ranks=ranks,
        step_labels=["BM25", "Cross-encoder"],
        chunk_labels=["A", "B", "C", "D"],
        relevant_chunks=relevant,
    )


def test_batch_empty():
    with pytest.raises(ValueError):
        BatchRankFlow([])


def test_aggregate_metrics_no_relevant():
    rf = _make_rf()
    batch = BatchRankFlow([rf, rf])
    result = batch.aggregate_metrics(k=2)
    assert result["per_step"] == []


def test_aggregate_metrics():
    rf1 = _make_rf(relevant=["A", "B"])
    rf2 = _make_rf(relevant=["A", "C"])
    batch = BatchRankFlow([rf1, rf2])
    result = batch.aggregate_metrics(k=2)
    assert len(result["per_step"]) == 2
    assert "precision_at_k_mean" in result["per_step"][0]
    assert "precision_at_k_std" in result["per_step"][0]
    assert "precision_at_k_mean" in result["mean"]
