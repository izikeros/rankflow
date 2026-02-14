"""Adapter for the RAGAS evaluation framework.

RAGAS uses SingleTurnSample / EvaluationDataset with fields:
- user_input: query string
- retrieved_contexts: list[str] of retrieved chunk texts
- reference_contexts: list[str] of expected relevant chunk texts
- response: LLM answer

Requires: pip install rankflow[ragas]
"""

from __future__ import annotations

from typing import Any


def from_ragas(
    dataset: Any,
    step_label: str = "Retrieval",
) -> Any:
    """Create a BatchRankFlow from a RAGAS EvaluationDataset.

    Since RAGAS tracks only one retrieval step (the retrieved_contexts),
    and ground truth (reference_contexts), this creates single-step
    RankFlow instances. Each sample becomes one RankFlow in the batch.

    For multi-step analysis, call this once per step with different datasets
    representing different pipeline stages.

    Args:
        dataset: A RAGAS EvaluationDataset or list of SingleTurnSample.
        step_label: Label for the retrieval step.

    Returns:
        BatchRankFlow with one RankFlow per sample.
    """
    import numpy as np

    from rankflow.batch import BatchRankFlow
    from rankflow.core import RankFlow

    # Handle both EvaluationDataset and list of samples
    samples = dataset
    if hasattr(dataset, "samples"):
        samples = dataset.samples
    elif hasattr(dataset, "__iter__"):
        samples = list(dataset)
    else:
        raise TypeError(
            f"Expected RAGAS EvaluationDataset or list of samples, got {type(dataset)}"
        )

    rankflows: list[RankFlow] = []
    for i, sample in enumerate(samples):
        # Extract retrieved contexts (these are the ranked results)
        retrieved = _get_attr(sample, "retrieved_contexts", [])
        if not retrieved:
            continue

        # Use text content as chunk labels (truncated for readability)
        chunk_labels = [_truncate(ctx, 60) for ctx in retrieved]
        n_chunks = len(chunk_labels)
        ranks = np.arange(n_chunks).reshape(1, -1)

        # Ground truth from reference_contexts
        reference = _get_attr(sample, "reference_contexts", [])
        relevant_chunks = None
        if reference:
            ref_set = set(reference)
            relevant_chunks = [
                label
                for label, full_text in zip(chunk_labels, retrieved, strict=False)
                if full_text in ref_set
            ]
            # If exact match fails, try substring matching
            if not relevant_chunks:
                relevant_chunks = [
                    label
                    for label, full_text in zip(chunk_labels, retrieved, strict=False)
                    if any(ref in full_text or full_text in ref for ref in reference)
                ]

        rf = RankFlow(
            ranks=ranks,
            step_labels=[step_label],
            chunk_labels=chunk_labels,
            relevant_chunks=relevant_chunks if relevant_chunks else None,
        )
        query = _get_attr(sample, "user_input", f"query_{i}")
        rf.query_label = query
        rankflows.append(rf)

    if not rankflows:
        raise ValueError("No valid samples found in RAGAS dataset.")

    if len(rankflows) == 1:
        return rankflows[0]
    return BatchRankFlow(rankflows)


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for use as a label."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
