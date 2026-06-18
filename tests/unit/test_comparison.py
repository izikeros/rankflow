"""Tests for experiment comparison."""

from __future__ import annotations

import numpy as np

from rankflow.comparison import ComparisonReport, compare_experiments
from rankflow.core import RankFlow
from rankflow.experiments import Experiment


def _make_rf(ranks, query_label="q0"):
    rf = RankFlow(
        ranks=np.array(ranks),
        step_labels=["BM25", "Reranker"],
        chunk_labels=["doc_a", "doc_b", "doc_c"],
        relevant_chunks=["doc_a"],
        relevance_grades={"doc_a": 2},
    )
    rf.query_label = query_label
    return rf


def _baseline_exp():
    return Experiment(
        name="baseline",
        config={"retriever": "bm25", "top_k": 100},
        rankflows=[
            _make_rf([[2, 0, 1], [1, 0, 2]], "q0"),
            _make_rf([[1, 0, 2], [0, 1, 2]], "q1"),
        ],
    )


def _challenger_exp():
    return Experiment(
        name="challenger",
        config={"retriever": "bm25", "top_k": 50, "reranker": "cross-encoder"},
        rankflows=[
            _make_rf([[2, 0, 1], [0, 1, 2]], "q0"),  # doc_a promoted to rank 0
            _make_rf([[1, 0, 2], [0, 1, 2]], "q1"),  # same as baseline
        ],
    )


class TestCompareExperiments:
    def test_basic_comparison(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        assert isinstance(report, ComparisonReport)
        assert report.baseline_name == "baseline"
        assert report.challenger_name == "challenger"
        assert len(report.per_query) == 2

    def test_config_diff(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        assert "top_k" in report.config_diff
        assert report.config_diff["top_k"]["baseline"] == 100
        assert report.config_diff["top_k"]["challenger"] == 50
        assert "reranker" in report.config_diff

    def test_wins_losses(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        # q0: challenger should be better (doc_a at rank 0 vs rank 1)
        # q1: same results -> tie
        assert report.wins >= 0
        assert report.losses >= 0
        assert report.wins + report.losses + report.ties == 2

    def test_metric_deltas_have_pvalue(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        assert "ndcg_at_k" in report.metric_deltas
        assert "p_value" in report.metric_deltas["ndcg_at_k"]
        assert "delta" in report.metric_deltas["ndcg_at_k"]

    def test_regression_queries(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        regressions = report.regression_queries("ndcg_at_k")
        assert isinstance(regressions, list)

    def test_improved_queries(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        improved = report.improved_queries("ndcg_at_k")
        assert isinstance(improved, list)

    def test_win_rate(self):
        report = compare_experiments(_baseline_exp(), _challenger_exp(), k=3)

        assert 0.0 <= report.win_rate <= 1.0

    def test_same_experiment(self):
        exp = _baseline_exp()
        report = compare_experiments(exp, exp, k=3)

        assert report.wins == 0
        assert report.losses == 0
        assert report.ties == 2
        for _m, data in report.metric_deltas.items():
            assert abs(data["delta"]) < 1e-9
