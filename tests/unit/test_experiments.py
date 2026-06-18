"""Tests for Experiment and ExperimentStore."""

from __future__ import annotations

import numpy as np
import pytest

from rankflow.core import RankFlow
from rankflow.experiments import Experiment, ExperimentStore


def _make_rf(query_label: str = "q0", relevant: bool = True):
    rf = RankFlow(
        ranks=np.array([[0, 1, 2], [2, 0, 1]]),
        step_labels=["BM25", "Reranker"],
        chunk_labels=["doc_a", "doc_b", "doc_c"],
        relevant_chunks=["doc_a"] if relevant else None,
        scores=np.array([[0.9, 0.7, 0.3], [0.5, 0.95, 0.8]]),
        pipeline_config={"retriever": "bm25", "top_k": 10},
    )
    rf.query_label = query_label
    return rf


class TestExperiment:
    def test_create(self):
        exp = Experiment(
            name="test-exp",
            config={"retriever": "bm25"},
            rankflows=[_make_rf("q0"), _make_rf("q1")],
            tags=["test"],
        )
        assert exp.n_queries == 2
        assert exp.timestamp  # auto-generated

    def test_to_dict_roundtrip(self):
        exp = Experiment(
            name="roundtrip",
            config={"retriever": "bm25", "reranker": "cross-encoder"},
            rankflows=[_make_rf("q0"), _make_rf("q1")],
            tags=["baseline"],
            description="test roundtrip",
        )
        data = exp.to_dict()
        loaded = Experiment.from_dict(data)

        assert loaded.name == exp.name
        assert loaded.config == exp.config
        assert loaded.tags == exp.tags
        assert loaded.n_queries == 2
        assert loaded.rankflows[0].query_label == "q0"
        np.testing.assert_array_equal(
            loaded.rankflows[0].ranks, exp.rankflows[0].ranks
        )

    def test_metrics_summary(self):
        exp = Experiment(
            name="with-metrics",
            rankflows=[_make_rf("q0"), _make_rf("q1")],
        )
        summary = exp.metrics_summary(k=3)
        assert "ndcg_at_k_mean" in summary
        assert summary["ndcg_at_k_mean"] >= 0

    def test_metrics_summary_no_relevant(self):
        exp = Experiment(
            name="no-rel",
            rankflows=[_make_rf("q0", relevant=False)],
        )
        summary = exp.metrics_summary(k=3)
        assert summary == {}

    def test_pipeline_config_preserved(self):
        exp = Experiment(
            name="config-test",
            rankflows=[_make_rf("q0")],
        )
        data = exp.to_dict()
        loaded = Experiment.from_dict(data)
        assert loaded.rankflows[0].pipeline_config == {"retriever": "bm25", "top_k": 10}


class TestExperimentStore:
    def test_save_and_load(self, tmp_path):
        store = ExperimentStore(tmp_path / "store")
        exp = Experiment(
            name="my-exp",
            config={"k": 10},
            rankflows=[_make_rf("q0")],
            tags=["v1"],
        )
        store.save(exp)
        assert store.exists("my-exp")

        loaded = store.load("my-exp")
        assert loaded.name == "my-exp"
        assert loaded.n_queries == 1

    def test_list(self, tmp_path):
        store = ExperimentStore(tmp_path / "store")
        store.save(Experiment(name="a", tags=["x"]))
        store.save(Experiment(name="b", tags=["y"]))
        store.save(Experiment(name="c", tags=["x", "y"]))

        all_exps = store.list()
        assert len(all_exps) == 3

        x_exps = store.list(tag="x")
        assert len(x_exps) == 2

    def test_delete(self, tmp_path):
        store = ExperimentStore(tmp_path / "store")
        store.save(Experiment(name="to-delete"))
        assert store.exists("to-delete")
        store.delete("to-delete")
        assert not store.exists("to-delete")

    def test_load_nonexistent(self, tmp_path):
        store = ExperimentStore(tmp_path / "store")
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("nope")

    def test_creates_directory(self, tmp_path):
        store_path = tmp_path / "deep" / "nested" / "store"
        ExperimentStore(store_path)
        assert store_path.is_dir()
