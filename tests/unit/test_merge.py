import numpy as np
import pytest

from rankflow.merge import MergeRankFlow, PipelineStep


def _bm25_step():
    return PipelineStep(
        name="BM25",
        ranks=np.arange(10),
        chunk_labels=[f"doc_{i}" for i in range(10)],
    )


def _vector_step():
    return PipelineStep(
        name="Vector",
        ranks=np.arange(10),
        chunk_labels=[f"doc_{i + 5}" for i in range(10)],  # overlap: doc_5..doc_9
    )


def _merge_step():
    return PipelineStep(
        name="RRF Merge",
        ranks=np.arange(15),
        chunk_labels=[f"doc_{i}" for i in range(15)],
        parents=["BM25", "Vector"],
    )


def _pipeline():
    return MergeRankFlow(steps=[_bm25_step(), _vector_step(), _merge_step()])


def test_pipeline_step_1d_validation():
    with pytest.raises(ValueError, match="1D"):
        PipelineStep("bad", ranks=np.array([[1, 2]]), chunk_labels=["a", "b"])


def test_pipeline_step_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        PipelineStep("bad", ranks=np.array([1, 2, 3]), chunk_labels=["a", "b"])


def test_top_k_set():
    step = _bm25_step()
    top3 = step.top_k_set(3)
    assert top3 == {"doc_0", "doc_1", "doc_2"}


def test_merge_rankflow_roots():
    p = _pipeline()
    assert set(p.root_steps) == {"BM25", "Vector"}


def test_merge_rankflow_merge_steps():
    p = _pipeline()
    assert p.merge_steps == ["RRF Merge"]


def test_invalid_parent_ref():
    with pytest.raises(ValueError, match="unknown parent"):
        MergeRankFlow(
            steps=[
                PipelineStep(
                    "A",
                    ranks=np.array([0, 1]),
                    chunk_labels=["x", "y"],
                    parents=["nonexistent"],
                ),
            ]
        )


def test_overlap_analysis():
    p = _pipeline()
    result = p.overlap_analysis(k=10)
    assert len(result) == 1
    r = result[0]
    assert r["merge_step"] == "RRF Merge"
    # BM25 has doc_0..doc_9, Vector has doc_5..doc_14
    # Shared at top-10: doc_5..doc_9 = 5 docs
    assert r["shared_count"] == 5
    assert r["exclusive_counts"]["BM25"] == 5  # doc_0..doc_4
    assert r["exclusive_counts"]["Vector"] == 5  # doc_10..doc_14
    assert r["total_unique"] == 15


def test_rank_correlation():
    p = _pipeline()
    result = p.rank_correlation()
    assert len(result) == 1
    r = result[0]
    assert r["parent_pair"] == ("BM25", "Vector")
    assert r["common_docs"] == 5
    # Both rank the shared docs (doc_5..doc_9) as positions 5-9 (BM25) and 0-4 (Vector)
    # These are inversely ordered so rho should be negative or low
    assert r["spearman_rho"] is not None


def test_empty_pipeline():
    with pytest.raises(ValueError):
        MergeRankFlow(steps=[])
