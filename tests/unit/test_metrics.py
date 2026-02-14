import numpy as np

from rankflow.metrics import (
    average_precision,
    compute_metrics_per_step,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _simple_ranks():
    # Step 0: doc order [0,1,2,3,4] (ranks = positions)
    # Step 1: doc order [2,0,1,4,3]
    return np.array([
        [0, 1, 2, 3, 4],
        [1, 2, 0, 4, 3],
    ])


def test_precision_at_k_perfect():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0, 1}  # top 2 are relevant
    assert precision_at_k(ranks, relevant, k=2) == 1.0


def test_precision_at_k_partial():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0, 4}  # doc 0 is rank 0, doc 4 is rank 4
    assert precision_at_k(ranks, relevant, k=2) == 0.5


def test_recall_at_k():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0, 1, 2}
    assert recall_at_k(ranks, relevant, k=2) == 2 / 3


def test_recall_at_k_all():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0, 1}
    assert recall_at_k(ranks, relevant, k=5) == 1.0


def test_mrr():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {2}  # doc 2 is at position 3 (0-indexed rank 2)
    assert mean_reciprocal_rank(ranks, relevant) == 1 / 3


def test_mrr_first_is_relevant():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0}
    assert mean_reciprocal_rank(ranks, relevant) == 1.0


def test_average_precision():
    ranks = np.array([0, 1, 2, 3, 4])
    relevant = {0, 2}  # positions 1 and 3
    # AP = (1/1 + 2/3) / 2
    expected = (1.0 + 2 / 3) / 2
    assert abs(average_precision(ranks, relevant) - expected) < 1e-10


def test_ndcg_at_k_perfect():
    ranks = np.array([0, 1, 2, 3, 4])
    # Binary relevance: docs 0,1 relevant
    rel_scores = {0: 1.0, 1: 1.0}
    # top-2 are docs 0,1 which are the relevant ones -> perfect NDCG
    assert abs(ndcg_at_k(ranks, rel_scores, k=2) - 1.0) < 1e-10


def test_ndcg_at_k_graded():
    ranks = np.array([0, 1, 2, 3, 4])
    rel_scores = {0: 3.0, 1: 2.0, 2: 1.0}
    val = ndcg_at_k(ranks, rel_scores, k=3)
    assert 0.0 < val <= 1.0


def test_compute_metrics_per_step():
    ranks = _simple_ranks()
    chunk_labels = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C"]
    results = compute_metrics_per_step(ranks, chunk_labels, relevant, k=3)
    assert len(results) == 2
    for step_m in results:
        assert "precision_at_k" in step_m
        assert "recall_at_k" in step_m
        assert "mrr" in step_m
        assert "map" in step_m
        assert "ndcg_at_k" in step_m


def test_empty_relevant():
    ranks = np.array([0, 1, 2])
    assert precision_at_k(ranks, set(), k=2) == 0.0
    assert recall_at_k(ranks, set(), k=2) == 0.0
    assert mean_reciprocal_rank(ranks, set()) == 0.0
    assert average_precision(ranks, set()) == 0.0


def test_k_zero():
    ranks = np.array([0, 1, 2])
    assert precision_at_k(ranks, {0}, k=0) == 0.0
    assert recall_at_k(ranks, {0}, k=0) == 0.0
