"""Tests for rankflow adapters (TREC, JSON, ranx, RAGAS)."""

from __future__ import annotations

import json
import textwrap
from unittest.mock import MagicMock

import numpy as np
import pytest

from rankflow.core import RankFlow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_rf():
    """A minimal 2-step, 3-chunk RankFlow for export/import tests."""
    return RankFlow(
        ranks=np.array([[0, 1, 2], [1, 0, 2]]),
        step_labels=["BM25", "Reranker"],
        chunk_labels=["doc_a", "doc_b", "doc_c"],
        relevant_chunks=["doc_a", "doc_c"],
        relevance_grades={"doc_a": 2, "doc_c": 1},
        scores=np.array([[0.9, 0.7, 0.3], [0.5, 0.95, 0.1]]),
    )


@pytest.fixture
def trec_run_dir(tmp_path):
    """Create two TREC run files (two steps) and a qrels file."""
    # BM25 run
    bm25 = tmp_path / "bm25.run"
    bm25.write_text(
        textwrap.dedent("""\
        q1 Q0 doc_a 0 0.9 bm25
        q1 Q0 doc_b 1 0.7 bm25
        q1 Q0 doc_c 2 0.3 bm25
        q2 Q0 doc_b 0 0.8 bm25
        q2 Q0 doc_d 1 0.6 bm25
    """)
    )

    # Reranker run
    reranker = tmp_path / "reranker.run"
    reranker.write_text(
        textwrap.dedent("""\
        q1 Q0 doc_b 0 0.95 reranker
        q1 Q0 doc_a 1 0.5 reranker
        q1 Q0 doc_c 2 0.1 reranker
        q2 Q0 doc_d 0 0.9 reranker
        q2 Q0 doc_b 1 0.7 reranker
    """)
    )

    # Qrels
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        textwrap.dedent("""\
        q1 0 doc_a 2
        q1 0 doc_c 1
        q2 0 doc_d 1
    """)
    )

    return tmp_path


# ---------------------------------------------------------------------------
# TREC adapter
# ---------------------------------------------------------------------------


class TestTRECAdapter:
    def test_load_single_query(self, trec_run_dir):
        rf = RankFlow.from_trec_run(
            [trec_run_dir / "bm25.run", trec_run_dir / "reranker.run"],
            qrels_path=trec_run_dir / "qrels.txt",
            query_id="q1",
        )
        assert isinstance(rf, RankFlow)
        assert rf.ranks.shape[0] == 2  # two steps
        assert "doc_a" in rf.chunk_labels
        assert rf.relevant_chunks is not None
        assert "doc_a" in rf.relevant_chunks

    def test_load_multi_query_returns_batch(self, trec_run_dir):
        from rankflow.batch import BatchRankFlow

        result = RankFlow.from_trec_run(
            [trec_run_dir / "bm25.run", trec_run_dir / "reranker.run"],
            qrels_path=trec_run_dir / "qrels.txt",
        )
        assert isinstance(result, BatchRankFlow)
        assert len(result.rankflows) == 2

    def test_load_without_qrels(self, trec_run_dir):
        rf = RankFlow.from_trec_run(
            [trec_run_dir / "bm25.run"],
            query_id="q1",
        )
        assert isinstance(rf, RankFlow)
        assert rf.relevant_chunks is None

    def test_save_trec_run(self, simple_rf, tmp_path):
        out = tmp_path / "output.run"
        simple_rf.to_trec_run(str(out), run_id="test", step_index=-1, query_id="q1")
        content = out.read_text().strip().split("\n")
        assert len(content) == 3
        assert "test" in content[0]
        assert "q1" in content[0]

    def test_roundtrip_trec(self, simple_rf, tmp_path):
        out = tmp_path / "step0.run"
        simple_rf.to_trec_run(str(out), run_id="s0", step_index=0, query_id="q1")
        out2 = tmp_path / "step1.run"
        simple_rf.to_trec_run(str(out2), run_id="s1", step_index=1, query_id="q1")

        loaded = RankFlow.from_trec_run([out, out2], query_id="q1")
        assert isinstance(loaded, RankFlow)
        assert loaded.ranks.shape[0] == 2
        assert loaded.ranks.shape[1] == 3

    def test_load_trec_qrels(self, trec_run_dir):
        from rankflow.adapters.trec import load_trec_qrels

        qrels = load_trec_qrels(trec_run_dir / "qrels.txt")
        assert "q1" in qrels
        assert qrels["q1"]["doc_a"] == 2
        assert qrels["q2"]["doc_d"] == 1


# ---------------------------------------------------------------------------
# JSON adapter
# ---------------------------------------------------------------------------


class TestJSONAdapter:
    def test_save_and_load(self, simple_rf, tmp_path):
        out = tmp_path / "rankflow.json"
        simple_rf.to_rankflow_json(str(out), query="test query")

        loaded = RankFlow.from_rankflow_json(str(out))
        assert isinstance(loaded, RankFlow)
        assert loaded.ranks.shape == simple_rf.ranks.shape
        np.testing.assert_array_equal(loaded.ranks, simple_rf.ranks)
        assert loaded.step_labels == simple_rf.step_labels
        assert loaded.chunk_labels == simple_rf.chunk_labels

    def test_save_includes_scores(self, simple_rf, tmp_path):
        out = tmp_path / "rankflow.json"
        simple_rf.to_rankflow_json(str(out))

        with open(out) as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert len(data["steps"]) == 2
        # Check scores are present
        first_chunk = data["steps"][0]["chunks"][0]
        assert "score" in first_chunk

    def test_save_includes_relevance(self, simple_rf, tmp_path):
        out = tmp_path / "rankflow.json"
        simple_rf.to_rankflow_json(str(out))

        with open(out) as f:
            data = json.load(f)
        assert "relevant" in data
        relevant_ids = {r["id"] for r in data["relevant"]}
        assert "doc_a" in relevant_ids
        assert "doc_c" in relevant_ids

    def test_roundtrip_preserves_relevance(self, simple_rf, tmp_path):
        out = tmp_path / "rankflow.json"
        simple_rf.to_rankflow_json(str(out))
        loaded = RankFlow.from_rankflow_json(str(out))
        assert loaded.relevant_chunks is not None
        assert "doc_a" in loaded.relevant_chunks
        assert loaded.relevance_grades is not None
        assert loaded.relevance_grades["doc_a"] == 2

    def test_no_scores(self, tmp_path):
        rf = RankFlow(
            ranks=np.array([[0, 1], [1, 0]]),
            chunk_labels=["a", "b"],
        )
        out = tmp_path / "no_scores.json"
        rf.to_rankflow_json(str(out))
        loaded = RankFlow.from_rankflow_json(str(out))
        assert loaded.scores is None

    def test_query_field(self, simple_rf, tmp_path):
        out = tmp_path / "q.json"
        simple_rf.to_rankflow_json(str(out), query="what is RAG?")
        with open(out) as f:
            data = json.load(f)
        assert data["query"] == "what is RAG?"


# ---------------------------------------------------------------------------
# ranx adapter (mocked -- ranx may not be installed)
# ---------------------------------------------------------------------------


class TestRanxAdapter:
    def test_from_ranx_mocked(self):
        """Test adapter logic with mocked ranx objects."""
        from unittest.mock import patch

        from rankflow.adapters.ranx_adapter import from_ranx

        run1 = MagicMock()
        run1.to_dict.return_value = {
            "q1": {"doc_a": 0.9, "doc_b": 0.7, "doc_c": 0.3},
        }
        run1.name = "BM25"

        run2 = MagicMock()
        run2.to_dict.return_value = {
            "q1": {"doc_b": 0.95, "doc_a": 0.5, "doc_c": 0.1},
        }
        run2.name = "Reranker"

        qrels_mock = MagicMock()
        qrels_mock.to_dict.return_value = {
            "q1": {"doc_a": 2, "doc_c": 1},
        }

        with patch("rankflow.adapters.ranx_adapter._ensure_ranx"):
            rf = from_ranx([run1, run2], qrels=qrels_mock, query_id="q1")
        assert isinstance(rf, RankFlow)
        assert rf.ranks.shape[0] == 2
        assert rf.relevant_chunks is not None
        assert "doc_a" in rf.relevant_chunks

    def test_from_ranx_empty_raises(self):
        from rankflow.adapters.ranx_adapter import from_ranx

        with pytest.raises(ValueError, match="At least one"):
            from_ranx([])


# ---------------------------------------------------------------------------
# RAGAS adapter (mocked -- ragas may not be installed)
# ---------------------------------------------------------------------------


class TestRagasAdapter:
    def test_from_ragas_dict_samples(self):
        """Test with dict-based samples (no ragas import needed)."""
        from rankflow.adapters.ragas_adapter import from_ragas

        samples = [
            {
                "user_input": "What is RAG?",
                "retrieved_contexts": [
                    "RAG combines retrieval and generation.",
                    "LLMs generate text from prompts.",
                    "Vector databases store embeddings.",
                ],
                "reference_contexts": [
                    "RAG combines retrieval and generation.",
                ],
            },
            {
                "user_input": "How does BM25 work?",
                "retrieved_contexts": [
                    "BM25 is a ranking function.",
                    "TF-IDF measures term importance.",
                ],
                "reference_contexts": [
                    "BM25 is a ranking function.",
                ],
            },
        ]

        from rankflow.batch import BatchRankFlow

        result = from_ragas(samples)
        assert isinstance(result, BatchRankFlow)
        assert len(result.rankflows) == 2

    def test_from_ragas_single_sample(self):
        from rankflow.adapters.ragas_adapter import from_ragas

        samples = [
            {
                "user_input": "query",
                "retrieved_contexts": ["ctx1", "ctx2", "ctx3"],
                "reference_contexts": ["ctx1"],
            },
        ]
        result = from_ragas(samples)
        assert isinstance(result, RankFlow)
        assert result.ranks.shape == (1, 3)
        assert result.relevant_chunks is not None

    def test_from_ragas_empty_raises(self):
        from rankflow.adapters.ragas_adapter import from_ragas

        with pytest.raises(ValueError, match="No valid samples"):
            from_ragas([{"user_input": "q", "retrieved_contexts": []}])

    def test_from_ragas_no_reference(self):
        from rankflow.adapters.ragas_adapter import from_ragas

        samples = [
            {
                "user_input": "query",
                "retrieved_contexts": ["a", "b"],
            },
        ]
        result = from_ragas(samples)
        assert isinstance(result, RankFlow)
        assert result.relevant_chunks is None
