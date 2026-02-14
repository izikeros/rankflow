"""Adapter for the ranx IR evaluation library.

ranx uses Run and Qrels objects that wrap dict[str, dict[str, float]]
structures: {query_id: {doc_id: score/relevance}}.

Requires: pip install rankflow[ranx]
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _ensure_ranx():
    try:
        import ranx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ranx is required for this adapter. "
            "Install it with: pip install rankflow[ranx]"
        ) from exc


def _collect_ranx_docs(qid: str, run_dicts: list[dict]) -> list[str]:
    """Collect unique doc IDs for a query across all runs, ordered by score."""
    all_docs: list[str] = []
    seen: set[str] = set()
    for rd in run_dicts:
        if qid not in rd:
            continue
        for docid in sorted(rd[qid].keys(), key=lambda d: -rd[qid][d]):
            if docid not in seen:
                all_docs.append(docid)
                seen.add(docid)
    return all_docs


def _build_ranx_matrices(
    qid: str,
    all_docs: list[str],
    run_dicts: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Build ranks and scores matrices from ranx run dicts for a query."""
    n_steps = len(run_dicts)
    n_chunks = len(all_docs)
    doc_to_idx = {d: i for i, d in enumerate(all_docs)}

    ranks = np.full((n_steps, n_chunks), n_chunks, dtype=float)
    scores = np.full((n_steps, n_chunks), 0.0)

    for step_idx, rd in enumerate(run_dicts):
        if qid not in rd:
            ranks[step_idx, :] = np.nan
            continue
        qid_data = rd[qid]
        sorted_docs = sorted(qid_data.keys(), key=lambda d: -qid_data[d])
        for rank_pos, docid in enumerate(sorted_docs):
            if docid in doc_to_idx:
                ranks[step_idx, doc_to_idx[docid]] = rank_pos
                scores[step_idx, doc_to_idx[docid]] = qid_data[docid]

    return ranks, scores


def _extract_ranx_relevance(
    qid: str,
    all_docs: list[str],
    qrels_dict: dict[str, dict[str, float]] | None,
) -> tuple[list[str] | None, dict[str, float] | None]:
    """Extract relevance from qrels dict for one query."""
    if not qrels_dict or qid not in qrels_dict:
        return None, None
    relevant_chunks = [d for d in all_docs if qrels_dict[qid].get(d, 0) > 0]
    relevance_grades = {
        d: float(qrels_dict[qid][d])
        for d in all_docs
        if d in qrels_dict[qid] and qrels_dict[qid][d] > 0
    }
    return relevant_chunks, relevance_grades


def from_ranx(
    runs: list[Any],
    qrels: Any | None = None,
    step_labels: list[str] | None = None,
    query_id: str | None = None,
):
    """Create RankFlow or BatchRankFlow from ranx Run objects.

    Args:
        runs: List of ranx.Run objects, one per retrieval step.
        qrels: Optional ranx.Qrels object for relevance judgments.
        step_labels: Optional names for each step. Defaults to Run names.
        query_id: If set, return a single RankFlow for that query.
            If None and multiple queries exist, return a BatchRankFlow.

    Returns:
        RankFlow (single query) or BatchRankFlow (multi-query).
    """
    if not runs:
        raise ValueError("At least one ranx.Run is required.")

    _ensure_ranx()

    from rankflow.batch import BatchRankFlow
    from rankflow.core import RankFlow

    run_dicts = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in runs]
    labels = step_labels or [getattr(r, "name", f"Step {i}") for i, r in enumerate(runs)]

    qrels_dict: dict[str, dict[str, float]] | None = None
    if qrels is not None:
        qrels_dict = qrels.to_dict() if hasattr(qrels, "to_dict") else dict(qrels)

    all_qids: set[str] = set()
    for rd in run_dicts:
        all_qids.update(rd.keys())
    if query_id is not None:
        all_qids = {query_id}

    rankflows: list[RankFlow] = []
    for qid in sorted(all_qids):
        all_docs = _collect_ranx_docs(qid, run_dicts)
        if not all_docs:
            continue

        ranks, scores = _build_ranx_matrices(qid, all_docs, run_dicts)
        relevant_chunks, relevance_grades = _extract_ranx_relevance(qid, all_docs, qrels_dict)

        rf = RankFlow(
            ranks=ranks,
            step_labels=labels,
            chunk_labels=all_docs,
            relevant_chunks=relevant_chunks,
            relevance_grades=relevance_grades,
            scores=scores,
        )
        rf.query_label = qid
        rankflows.append(rf)

    if len(rankflows) == 1:
        return rankflows[0]
    return BatchRankFlow(rankflows)


def to_ranx_run(rf, step_index: int = -1, query_id: str = "q0") -> Any:
    """Export a RankFlow step as a ranx.Run object."""
    _ensure_ranx()
    from ranx import Run

    scores_row = rf.scores[step_index] if rf.scores is not None else -rf.ranks[step_index].astype(float)
    doc_scores = {rf.chunk_labels[i]: float(scores_row[i]) for i in range(len(rf.chunk_labels))}
    return Run({query_id: doc_scores})
