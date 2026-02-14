import numpy as np

from rankflow.batch import BatchRankFlow
from rankflow.core import RankFlow


def _make_rf(seed=42, relevant=None):
    rng = np.random.RandomState(seed)
    n_chunks = 10
    n_steps = 3
    ranks = np.zeros((n_steps, n_chunks), dtype=int)
    for s in range(n_steps):
        ranks[s] = rng.permutation(n_chunks)
    return RankFlow(
        ranks=ranks,
        step_labels=["BM25", "Semantic", "Cross-encoder"],
        chunk_labels=[f"Doc_{i}" for i in range(n_chunks)],
        relevant_chunks=relevant or ["Doc_0", "Doc_1", "Doc_2"],
    )


def _make_batch(n=10):
    return BatchRankFlow(
        [_make_rf(seed=i, relevant=["Doc_0", "Doc_1"]) for i in range(n)]
    )


def test_win_loss_analysis():
    batch = _make_batch(20)
    results = batch.win_loss_analysis(metric="ndcg_at_k", k=5)
    assert len(results) == 2  # 3 steps -> 2 transitions
    for r in results:
        assert "transition" in r
        assert "wins" in r
        assert "losses" in r
        assert "ties" in r
        assert r["wins"] + r["losses"] + r["ties"] == 20


def test_failure_cases():
    batch = _make_batch(20)
    # threshold=999 means delta < 999, which is always true -> all returned
    failures = batch.failure_cases(metric="ndcg_at_k", k=5, threshold=999)
    assert len(failures) == 20
    for f in failures:
        assert "query_index" in f
        assert "delta" in f


def test_failure_cases_strict_threshold():
    batch = _make_batch(20)
    # threshold=-999: no query can have delta < -999
    failures = batch.failure_cases(metric="ndcg_at_k", k=5, threshold=-999)
    assert len(failures) == 0


def test_win_loss_no_metrics():
    rf = RankFlow(ranks=np.array([[0, 1], [1, 0]]))
    batch = BatchRankFlow([rf])
    results = batch.win_loss_analysis()
    assert results == []


def test_aggregate_metrics_consistency():
    batch = _make_batch(10)
    agg = batch.aggregate_metrics(k=5)
    assert "per_step" in agg
    assert len(agg["per_step"]) == 3
    for step_agg in agg["per_step"]:
        assert "ndcg_at_k_mean" in step_agg
        assert "ndcg_at_k_std" in step_agg
        assert 0 <= step_agg["ndcg_at_k_mean"] <= 1
