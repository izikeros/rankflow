"""TREC run and qrels format adapter.

TREC Run format (whitespace-separated):
    qid Q0 docno rank score run_id

TREC Qrels format (whitespace-separated):
    qid iter docno relevance
    (iter is typically 0 and ignored)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_trec_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """Load a TREC qrels file.

    Returns:
        Nested dict: qid -> {docno: relevance_grade}
    """
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _iter, docno, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docno] = rel
    return qrels


def _parse_run_file(path: Path) -> tuple[dict[str, list[tuple[str, int, float]]], str]:
    """Parse a single TREC run file.

    Returns:
        (run_data: {qid: [(docno, rank, score)]}, run_id)
    """
    run_data: dict[str, list[tuple[str, int, float]]] = {}
    run_id = ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _q0, docno = parts[0], parts[1], parts[2]
            rank, score, rid = int(parts[3]), float(parts[4]), parts[5]
            run_data.setdefault(qid, []).append((docno, rank, score))
            if not run_id:
                run_id = rid
    return run_data, run_id or path.stem


def _collect_docs_for_query(
    qid: str, step_data: list[dict[str, list[tuple[str, int, float]]]]
) -> list[str]:
    """Collect all unique doc IDs across all steps for a given query."""
    all_docs: list[str] = []
    seen: set[str] = set()
    for sd in step_data:
        if qid not in sd:
            continue
        for docno, _rank, _score in sorted(sd[qid], key=lambda x: x[1]):
            if docno not in seen:
                all_docs.append(docno)
                seen.add(docno)
    return all_docs


def _build_rankflow_for_query(
    qid: str,
    all_docs: list[str],
    step_data: list[dict[str, list[tuple[str, int, float]]]],
    step_labels: list[str],
    qrels: dict[str, dict[str, int]] | None,
):
    """Build a single RankFlow instance for one query."""
    from rankflow.core import RankFlow

    n_steps = len(step_data)
    n_chunks = len(all_docs)
    doc_to_idx = {d: i for i, d in enumerate(all_docs)}

    ranks = np.full((n_steps, n_chunks), n_chunks, dtype=float)
    scores = np.full((n_steps, n_chunks), 0.0)

    for step_idx, sd in enumerate(step_data):
        if qid not in sd:
            ranks[step_idx, :] = np.nan
            continue
        for docno, rank, score in sd[qid]:
            if docno in doc_to_idx:
                ranks[step_idx, doc_to_idx[docno]] = rank
                scores[step_idx, doc_to_idx[docno]] = score

    relevant_chunks, relevance_grades = _extract_relevance(qid, all_docs, qrels)

    rf = RankFlow(
        ranks=ranks,
        step_labels=step_labels,
        chunk_labels=all_docs,
        relevant_chunks=relevant_chunks,
        relevance_grades=relevance_grades,
        scores=scores,
    )
    rf.query_label = qid
    return rf


def _extract_relevance(
    qid: str, all_docs: list[str], qrels: dict[str, dict[str, int]] | None
) -> tuple[list[str] | None, dict[str, float] | None]:
    """Extract relevance info from qrels for a query."""
    if not qrels or qid not in qrels:
        return None, None
    relevant_chunks = [d for d in all_docs if qrels[qid].get(d, 0) > 0]
    relevance_grades = {
        d: float(qrels[qid][d])
        for d in all_docs
        if d in qrels[qid] and qrels[qid][d] > 0
    }
    return relevant_chunks, relevance_grades


def load_trec_run(
    paths: str | Path | list[str | Path],
    qrels_path: str | Path | None = None,
    query_id: str | None = None,
):
    """Load one or more TREC run files into RankFlow or BatchRankFlow.

    Args:
        paths: One or more TREC run file paths. Multiple files are treated
            as multiple retrieval steps (e.g., BM25.run, reranker.run).
        qrels_path: Optional TREC qrels file for relevance judgments.
        query_id: If set, return a single RankFlow for that query.
            If None and multiple queries exist, return a BatchRankFlow.

    Returns:
        RankFlow (single query) or BatchRankFlow (multi-query).
    """
    from rankflow.batch import BatchRankFlow

    if isinstance(paths, (str, Path)):
        paths = [paths]
    paths = [Path(p) for p in paths]

    qrels = load_trec_qrels(qrels_path) if qrels_path else None

    step_data: list[dict[str, list[tuple[str, int, float]]]] = []
    step_labels: list[str] = []
    for path in paths:
        run_data, run_label = _parse_run_file(path)
        step_data.append(run_data)
        step_labels.append(run_label)

    all_qids: set[str] = set()
    for sd in step_data:
        all_qids.update(sd.keys())
    if query_id is not None:
        all_qids = {query_id}

    rankflows = []
    for qid in sorted(all_qids):
        all_docs = _collect_docs_for_query(qid, step_data)
        if not all_docs:
            continue
        rf = _build_rankflow_for_query(qid, all_docs, step_data, step_labels, qrels)
        rankflows.append(rf)

    if len(rankflows) == 1:
        return rankflows[0]
    return BatchRankFlow(rankflows)


def save_trec_run(
    rf,
    path: str | Path,
    run_id: str = "rankflow",
    step_index: int = -1,
    query_id: str = "q0",
) -> None:
    """Export a RankFlow step as a TREC run file."""
    ranks_row = rf.ranks[step_index]
    scores_row = (
        rf.scores[step_index] if rf.scores is not None else np.zeros_like(ranks_row)
    )
    sorted_indices = np.argsort(ranks_row)

    with open(path, "w") as f:
        for rank_pos, chunk_idx in enumerate(sorted_indices):
            docno = rf.chunk_labels[chunk_idx]
            score = float(scores_row[chunk_idx])
            f.write(f"{query_id} Q0 {docno} {rank_pos} {score} {run_id}\n")
