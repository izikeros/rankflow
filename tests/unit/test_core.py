import numpy as np
import pandas as pd
import pytest

from rankflow.core import RankFlow


def test_init_from_numpy():
    ranks = np.array([[0, 1], [1, 0]])
    rf = RankFlow(ranks=ranks)
    assert rf.step_labels == ["Step 0", "Step 1"]
    assert rf.chunk_labels == ["Chunk 0", "Chunk 1"]


def test_init_from_dataframe():
    df = pd.DataFrame({"A": [1, 2], "B": [2, 1]}, index=["s1", "s2"])
    rf = RankFlow(df=df)
    assert rf.chunk_labels == ["A", "B"]
    assert rf.step_labels == ["s1", "s2"]


def test_init_no_data():
    with pytest.raises(ValueError):
        RankFlow()


def test_metrics_without_relevant():
    rf = RankFlow(ranks=np.array([[0, 1], [1, 0]]))
    assert rf.metrics() is None


def test_metrics_with_relevant():
    rf = RankFlow(
        ranks=np.array([[0, 1, 2], [2, 0, 1]]),
        chunk_labels=["A", "B", "C"],
        relevant_chunks=["A"],
    )
    m = rf.metrics(k=2)
    assert m is not None
    assert len(m) == 2
    assert (
        m[0]["precision_at_k"] == 0.5
    )  # A is at rank 0, top-2 = {A, B}, 1 relevant / 2


def test_summary():
    rf = RankFlow(
        ranks=np.array([[0, 1, 2], [2, 0, 1]]),
        chunk_labels=["A", "B", "C"],
    )
    s = rf.summary()
    assert len(s) == 3
    assert s[0]["chunk"] == "A"


def test_to_dict():
    rf = RankFlow(ranks=np.array([[0, 1], [1, 0]]))
    d = rf.to_dict()
    assert "ranks" in d
    assert "step_labels" in d


def test_to_dataframe():
    rf = RankFlow(
        ranks=np.array([[0, 1], [1, 0]]),
        step_labels=["s1", "s2"],
        chunk_labels=["A", "B"],
    )
    df = rf.to_dataframe()
    assert list(df.columns) == ["A", "B"]


def test_config_override():
    rf = RankFlow(ranks=np.array([[0, 1], [1, 0]]), title="Custom", line_width=5)
    assert rf.config.title == "Custom"
    assert rf.config.line_width == 5


def test_relevant_with_grades():
    rf = RankFlow(
        ranks=np.array([[0, 1, 2], [2, 0, 1]]),
        chunk_labels=["A", "B", "C"],
        relevant_chunks=["A", "B"],
        relevance_grades={"A": 3, "B": 1},
    )
    m = rf.metrics(k=3)
    assert m is not None
    # NDCG should be valid
    assert 0.0 <= m[0]["ndcg_at_k"] <= 1.0


def test_init_invalid_ranks_ellipsis():
    with pytest.raises(ValueError, match="2D array"):
        RankFlow(ranks=...)


def test_init_invalid_ranks_1d():
    with pytest.raises(ValueError, match="2D array"):
        RankFlow(ranks=np.array([1, 2, 3]))


def test_nan_absent_mask():
    ranks = np.array([[0, 1, np.nan], [1, np.nan, 0]], dtype=float)
    rf = RankFlow(ranks=ranks, chunk_labels=["A", "B", "C"])
    assert rf._absent_mask is not None
    assert rf._absent_mask[0, 2] is np.True_
    assert rf._absent_mask[1, 1] is np.True_


# --- chunk_properties tests ---


def test_chunk_properties_stored():
    rf = RankFlow(
        ranks=np.array([[1, 2], [2, 1]]),
        chunk_labels=["id_1", "id_2"],
        chunk_properties={"title": ["Doc A", "Doc B"]},
    )
    assert rf._chunk_properties == {"title": ["Doc A", "Doc B"]}


def test_chunk_properties_none_by_default():
    rf = RankFlow(ranks=np.array([[1, 2], [2, 1]]))
    assert rf._chunk_properties is None


def test_chunk_properties_length_mismatch():
    with pytest.raises(ValueError, match=r"chunk_properties.*length"):
        RankFlow(
            ranks=np.array([[1, 2], [2, 1]]),
            chunk_labels=["id_1", "id_2"],
            chunk_properties={"title": ["Doc A"]},
        )


def test_get_labels_default_fallback():
    rf = RankFlow(
        ranks=np.array([[1, 2], [2, 1]]),
        chunk_labels=["id_1", "id_2"],
    )
    assert rf._get_left_labels() == ["id_1", "id_2"]
    assert rf._get_right_labels() == ["id_1", "id_2"]


def test_get_labels_with_properties():
    rf = RankFlow(
        ranks=np.array([[1, 2], [2, 1]]),
        chunk_labels=["id_1", "id_2"],
        chunk_properties={
            "title": ["Planning Materiality", "Risk Assessment"],
            "short": ["PM", "RA"],
        },
        left_label_key="short",
        right_label_key="title",
    )
    assert rf._get_left_labels() == ["PM", "RA"]
    assert rf._get_right_labels() == ["Planning Materiality", "Risk Assessment"]


def test_get_labels_invalid_key_falls_back():
    rf = RankFlow(
        ranks=np.array([[1, 2], [2, 1]]),
        chunk_labels=["id_1", "id_2"],
        chunk_properties={"title": ["Doc A", "Doc B"]},
        left_label_key="nonexistent",
    )
    assert rf._get_left_labels() == ["id_1", "id_2"]


def test_get_labels_no_properties_with_key():
    rf = RankFlow(
        ranks=np.array([[1, 2], [2, 1]]),
        chunk_labels=["id_1", "id_2"],
        left_label_key="title",
    )
    assert rf._get_left_labels() == ["id_1", "id_2"]
