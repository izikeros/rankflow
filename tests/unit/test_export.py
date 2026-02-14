import json
import os
import tempfile

import numpy as np

from rankflow.export import ranks_to_dataframe, ranks_to_dict, ranks_to_json


def _sample():
    ranks = np.array([[0, 1, 2], [2, 0, 1]])
    step_labels = ["Step 0", "Step 1"]
    chunk_labels = ["A", "B", "C"]
    return ranks, step_labels, chunk_labels


def test_ranks_to_dict():
    ranks, steps, chunks = _sample()
    d = ranks_to_dict(ranks, steps, chunks)
    assert d["step_labels"] == steps
    assert d["chunk_labels"] == chunks
    assert d["ranks"] == ranks.tolist()


def test_ranks_to_dict_with_metrics():
    ranks, steps, chunks = _sample()
    metrics = [{"p": 0.5}, {"p": 0.8}]
    d = ranks_to_dict(ranks, steps, chunks, metrics=metrics)
    assert d["metrics_per_step"] == metrics


def test_ranks_to_dataframe():
    ranks, steps, chunks = _sample()
    df = ranks_to_dataframe(ranks, steps, chunks)
    assert list(df.columns) == chunks
    assert list(df.index) == steps
    assert df.shape == (2, 3)


def test_ranks_to_json():
    ranks, steps, chunks = _sample()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        ranks_to_json(path, ranks, steps, chunks)
        with open(path) as f:
            data = json.load(f)
        assert data["step_labels"] == steps
        assert data["chunk_labels"] == chunks
    finally:
        os.unlink(path)
