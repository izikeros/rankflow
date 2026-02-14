import numpy as np

from rankflow.analysis import compute_rank_deltas, compute_summary, filter_top_k


def _sample_ranks():
    return np.array([
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [3, 1, 0, 2],
    ])


def test_compute_rank_deltas_shape():
    ranks = _sample_ranks()
    deltas = compute_rank_deltas(ranks)
    assert deltas.shape == (2, 4)


def test_compute_rank_deltas_values():
    ranks = _sample_ranks()
    deltas = compute_rank_deltas(ranks)
    # Step 0->1: doc0 goes 0->2 (+2), doc1 goes 1->0 (-1)
    assert deltas[0, 0] == 2
    assert deltas[0, 1] == -1


def test_compute_summary():
    ranks = _sample_ranks()
    labels = ["A", "B", "C", "D"]
    summary = compute_summary(ranks, labels)
    assert len(summary) == 4
    assert summary[0]["chunk"] == "A"
    assert summary[0]["initial_rank"] == 0
    assert summary[0]["final_rank"] == 3
    assert summary[0]["rank_change"] == 3


def test_filter_top_k_any():
    ranks = np.array([
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
    ])
    labels = ["A", "B", "C", "D", "E"]
    _filtered_ranks, filtered_labels, _kept = filter_top_k(ranks, labels, k=2, mode="any")
    # Top-2 at step 0: A, B; top-2 at step 1: E, D => keep A, B, D, E
    assert len(filtered_labels) == 4
    assert "C" not in filtered_labels


def test_filter_top_k_initial():
    ranks = np.array([
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
    ])
    labels = ["A", "B", "C", "D", "E"]
    _, filtered_labels, _ = filter_top_k(ranks, labels, k=2, mode="initial")
    assert filtered_labels == ["A", "B"]


def test_filter_top_k_final():
    ranks = np.array([
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
    ])
    labels = ["A", "B", "C", "D", "E"]
    _, filtered_labels, _ = filter_top_k(ranks, labels, k=2, mode="final")
    assert filtered_labels == ["D", "E"]
