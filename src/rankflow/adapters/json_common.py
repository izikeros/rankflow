"""RankFlow's own JSON interchange format.

Schema (v1.0):
{
  "version": "1.0",
  "query": "optional query text",
  "steps": [
    {
      "name": "BM25",
      "chunks": [
        {"id": "doc_1", "rank": 0, "score": 0.95, "source": "text"},
        ...
      ]
    },
    ...
  ],
  "relevant": [
    {"id": "doc_1", "grade": 2},
    ...
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"


def save_rankflow_json(
    rf,
    path: str | Path,
    query: str | None = None,
) -> None:
    """Export a RankFlow instance to the RankFlow JSON schema.

    Args:
        rf: RankFlow instance.
        path: Output file path.
        query: Optional query text to include.
    """
    steps = []
    for step_idx, step_label in enumerate(rf.step_labels):
        chunks = []
        sorted_indices = np.argsort(rf.ranks[step_idx])
        for chunk_idx in sorted_indices:
            entry: dict = {
                "id": rf.chunk_labels[chunk_idx],
                "rank": int(rf.ranks[step_idx, chunk_idx]),
            }
            if rf.scores is not None:
                entry["score"] = float(rf.scores[step_idx, chunk_idx])
            if rf._source_labels and chunk_idx in rf._source_labels:
                entry["source"] = rf._source_labels[chunk_idx]
            chunks.append(entry)
        steps.append({"name": step_label, "chunks": chunks})

    data: dict = {"version": SCHEMA_VERSION, "steps": steps}
    if query is not None:
        data["query"] = query

    # Relevance
    if rf.relevant_chunks:
        relevant = []
        grades = rf.relevance_grades or {}
        for chunk in rf.relevant_chunks:
            entry = {"id": str(chunk)}
            grade = grades.get(chunk)
            if grade is not None:
                entry["grade"] = grade
            relevant.append(entry)
        data["relevant"] = relevant

    if rf.pipeline_config:
        data["pipeline_config"] = rf.pipeline_config

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_rankflow_json(path: str | Path):
    """Load a RankFlow instance from the RankFlow JSON schema.

    Returns:
        RankFlow instance.
    """
    from rankflow.core import RankFlow

    with open(path) as f:
        data = json.load(f)

    steps = data["steps"]
    n_steps = len(steps)

    # Collect all unique chunk IDs in order of first appearance
    all_chunks: list[str] = []
    seen: set[str] = set()
    for step in steps:
        for chunk in step["chunks"]:
            cid = chunk["id"]
            if cid not in seen:
                all_chunks.append(cid)
                seen.add(cid)

    n_chunks = len(all_chunks)
    chunk_to_idx = {c: i for i, c in enumerate(all_chunks)}

    ranks = np.full((n_steps, n_chunks), n_chunks, dtype=float)
    scores = np.full((n_steps, n_chunks), 0.0)
    has_scores = False
    source_labels: dict[str, str] = {}

    step_labels = []
    for step_idx, step in enumerate(steps):
        step_labels.append(step["name"])
        for chunk in step["chunks"]:
            idx = chunk_to_idx[chunk["id"]]
            ranks[step_idx, idx] = chunk["rank"]
            if "score" in chunk:
                scores[step_idx, idx] = chunk["score"]
                has_scores = True
            if "source" in chunk:
                source_labels[chunk["id"]] = chunk["source"]

    # Relevance
    relevant_chunks = None
    relevance_grades = None
    if "relevant" in data:
        relevant_chunks = [r["id"] for r in data["relevant"]]
        relevance_grades = {
            r["id"]: r["grade"] for r in data["relevant"] if "grade" in r
        }

    rf = RankFlow(
        ranks=ranks,
        step_labels=step_labels,
        chunk_labels=all_chunks,
        relevant_chunks=relevant_chunks,
        relevance_grades=relevance_grades or None,
        scores=scores if has_scores else None,
        source_labels=source_labels or None,
        pipeline_config=data.get("pipeline_config"),
    )
    return rf
