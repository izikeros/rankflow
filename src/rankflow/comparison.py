"""Experiment comparison utilities for retrieval tuning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ComparisonReport:
    """Result of comparing two experiments.

    Attributes:
        baseline_name: Name of the baseline experiment.
        challenger_name: Name of the challenger experiment.
        k: top-K used for metric computation.
        config_diff: Dict of config keys that differ between experiments.
        metric_deltas: Per-metric mean improvement and p-value.
        per_query: List of per-query metric comparisons.
        wins: Number of queries where challenger is better.
        losses: Number of queries where challenger is worse.
        ties: Number of queries with no change.
    """

    baseline_name: str
    challenger_name: str
    k: int
    config_diff: dict[str, dict[str, Any]] = field(default_factory=dict)
    metric_deltas: dict[str, dict[str, float]] = field(default_factory=dict)
    per_query: list[dict[str, Any]] = field(default_factory=list)
    wins: int = 0
    losses: int = 0
    ties: int = 0

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.ties
        return self.wins / total if total else 0.0

    def regression_queries(self, metric: str = "ndcg_at_k") -> list[dict[str, Any]]:
        """Return queries where the challenger was worse than baseline."""
        return [q for q in self.per_query if q.get(f"{metric}_delta", 0) < 0]

    def improved_queries(self, metric: str = "ndcg_at_k") -> list[dict[str, Any]]:
        """Return queries where the challenger improved over baseline."""
        return [q for q in self.per_query if q.get(f"{metric}_delta", 0) > 0]


def _diff_configs(a: dict, b: dict) -> dict[str, dict[str, Any]]:
    """Find keys that differ between two config dicts."""
    all_keys = set(a.keys()) | set(b.keys())
    diff = {}
    for key in sorted(all_keys):
        va = a.get(key)
        vb = b.get(key)
        if va != vb:
            diff[key] = {"baseline": va, "challenger": vb}
    return diff


def _paired_ttest(a_values: list[float], b_values: list[float]) -> float:
    """Paired t-test p-value. Returns 1.0 if test cannot be computed."""
    n = len(a_values)
    if n < 2:
        return 1.0
    diffs = [bv - av for av, bv in zip(a_values, b_values, strict=True)]
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)
    if std_d == 0:
        return 0.0 if mean_d != 0 else 1.0
    t_stat = mean_d / (std_d / np.sqrt(n))
    # Approximate two-tailed p-value using normal distribution for large n
    # For small n this is an approximation but avoids scipy dependency
    p = float(2 * (1 - _norm_cdf(abs(t_stat))))
    return max(p, 0.0)


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF (no scipy required)."""
    import math

    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def compare_experiments(
    baseline,
    challenger,
    k: int = 10,
    primary_metric: str = "ndcg_at_k",
    tolerance: float = 1e-6,
) -> ComparisonReport:
    """Compare two Experiment objects.

    Args:
        baseline: Experiment instance (the reference).
        challenger: Experiment instance (the one being tested).
        k: top-K for metric computation.
        primary_metric: Metric used for win/loss counting.
        tolerance: Minimum delta to count as win/loss.

    Returns:
        ComparisonReport with metric deltas, per-query diffs, and
        statistical significance.
    """
    config_diff = _diff_configs(
        baseline.config or {}, challenger.config or {}
    )

    # Build query label -> index mapping for matching
    base_by_label = {}
    for i, rf in enumerate(baseline.rankflows):
        label = getattr(rf, "query_label", None) or f"query_{i}"
        base_by_label[label] = i

    chal_by_label = {}
    for i, rf in enumerate(challenger.rankflows):
        label = getattr(rf, "query_label", None) or f"query_{i}"
        chal_by_label[label] = i

    # Match queries present in both experiments
    common_labels = sorted(set(base_by_label.keys()) & set(chal_by_label.keys()))

    metric_keys = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"]
    base_values: dict[str, list[float]] = {m: [] for m in metric_keys}
    chal_values: dict[str, list[float]] = {m: [] for m in metric_keys}

    per_query: list[dict[str, Any]] = []
    wins = losses = ties = 0

    for label in common_labels:
        b_rf = baseline.rankflows[base_by_label[label]]
        c_rf = challenger.rankflows[chal_by_label[label]]

        b_metrics = b_rf.metrics(k=k)
        c_metrics = c_rf.metrics(k=k)

        # Use last-step metrics as the final comparison point
        b_final = b_metrics[-1] if b_metrics else {}
        c_final = c_metrics[-1] if c_metrics else {}

        q_row: dict[str, Any] = {"query_label": label}
        for m in metric_keys:
            bv = b_final.get(m, 0.0)
            cv = c_final.get(m, 0.0)
            q_row[f"{m}_baseline"] = bv
            q_row[f"{m}_challenger"] = cv
            q_row[f"{m}_delta"] = cv - bv
            base_values[m].append(bv)
            chal_values[m].append(cv)

        delta = q_row.get(f"{primary_metric}_delta", 0.0)
        if delta > tolerance:
            wins += 1
        elif delta < -tolerance:
            losses += 1
        else:
            ties += 1

        per_query.append(q_row)

    # Compute aggregate deltas + p-values
    metric_deltas: dict[str, dict[str, float]] = {}
    for m in metric_keys:
        bv = base_values[m]
        cv = chal_values[m]
        if bv:
            mean_delta = float(np.mean(cv)) - float(np.mean(bv))
            p_value = _paired_ttest(bv, cv)
            metric_deltas[m] = {
                "baseline_mean": float(np.mean(bv)),
                "challenger_mean": float(np.mean(cv)),
                "delta": mean_delta,
                "p_value": p_value,
            }

    return ComparisonReport(
        baseline_name=baseline.name,
        challenger_name=challenger.name,
        k=k,
        config_diff=config_diff,
        metric_deltas=metric_deltas,
        per_query=per_query,
        wins=wins,
        losses=losses,
        ties=ties,
    )
