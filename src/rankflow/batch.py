"""Batch analysis across multiple queries."""

from __future__ import annotations

from typing import Any

import numpy as np


class BatchRankFlow:
    """Aggregate metrics and visualizations across multiple RankFlow instances."""

    def __init__(self, rankflows: list):
        """Initialize with a list of RankFlow objects.

        All RankFlow objects should have relevant_chunks set for metrics to work.
        """
        if not rankflows:
            raise ValueError("At least one RankFlow object is required.")
        self.rankflows = rankflows

    def aggregate_metrics(self, k: int = 5) -> dict[str, Any]:
        """Compute mean and std of metrics across all queries.

        Returns a dict with:
        - per_step: list of dicts with mean/std for each metric at each step
        - mean: overall mean across all steps and queries
        """
        all_metrics = []
        for rf in self.rankflows:
            m = rf.metrics(k=k)
            if m is not None:
                all_metrics.append(m)

        if not all_metrics:
            return {"per_step": [], "mean": {}}

        # All should have the same number of steps
        n_steps = len(all_metrics[0])
        metric_keys = list(all_metrics[0][0].keys())

        per_step = []
        for step in range(n_steps):
            step_agg = {}
            for key in metric_keys:
                values = [m[step][key] for m in all_metrics if step < len(m)]
                step_agg[f"{key}_mean"] = float(np.mean(values))
                step_agg[f"{key}_std"] = float(np.std(values))
            per_step.append(step_agg)

        # Overall mean
        overall = {}
        for key in metric_keys:
            all_values = []
            for m in all_metrics:
                for step in m:
                    all_values.append(step[key])
            overall[f"{key}_mean"] = float(np.mean(all_values))
            overall[f"{key}_std"] = float(np.std(all_values))

        return {"per_step": per_step, "mean": overall}

    def plot(self, k: int = 5, metric: str = "ndcg_at_k") -> Any:
        """Plot aggregated metric evolution across steps as box plots.

        Args:
            k: top-K for metrics computation.
            metric: which metric to plot.

        Returns:
            matplotlib figure and axes.
        """
        import matplotlib.pyplot as plt

        all_metrics = []
        for rf in self.rankflows:
            m = rf.metrics(k=k)
            if m is not None:
                all_metrics.append(m)

        if not all_metrics:
            raise ValueError(
                "No metrics available. Ensure relevant_chunks is set on RankFlow objects."
            )

        n_steps = len(all_metrics[0])
        step_labels = self.rankflows[0].step_labels[:n_steps]

        # Collect values per step
        data_per_step = []
        for step in range(n_steps):
            values = [m[step][metric] for m in all_metrics if step < len(m)]
            data_per_step.append(values)

        fig, ax = plt.subplots(figsize=(max(5, 1.5 * n_steps), 5))
        bp = ax.boxplot(data_per_step, labels=step_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} across queries per step")
        ax.grid(axis="y", alpha=0.3)
        return fig, ax

    def plot_metric_evolution(self, k: int = 5) -> Any:
        """Plot mean metric values with error bars across steps.

        Returns:
            matplotlib figure and axes.
        """
        import matplotlib.pyplot as plt

        agg = self.aggregate_metrics(k=k)
        per_step = agg["per_step"]
        if not per_step:
            raise ValueError("No aggregated metrics available.")

        n_steps = len(per_step)
        step_labels = self.rankflows[0].step_labels[:n_steps]
        metric_keys = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"]

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_steps), 5))
        x = np.arange(n_steps)

        for key in metric_keys:
            means = [per_step[s][f"{key}_mean"] for s in range(n_steps)]
            stds = [per_step[s][f"{key}_std"] for s in range(n_steps)]
            ax.errorbar(x, means, yerr=stds, marker="o", capsize=4, label=key)

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=15)
        ax.set_ylabel("Metric value")
        ax.set_title("Metric evolution across retrieval steps")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        return fig, ax

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_all_metrics(self, k: int) -> list[list[dict[str, float]]]:
        """Collect per-step metrics for all queries that have relevant_chunks."""
        result = []
        for rf in self.rankflows:
            m = rf.metrics(k=k)
            if m is not None:
                result.append(m)
        return result

    # ------------------------------------------------------------------
    # Per-step metric dashboard
    # ------------------------------------------------------------------

    def plot_dashboard(self, k: int = 10) -> Any:
        """Multi-panel figure showing all key metrics as box plots per step.

        Returns (fig, axes_array).
        """
        import matplotlib.pyplot as plt

        all_metrics = self._collect_all_metrics(k)
        if not all_metrics:
            raise ValueError("No metrics available.")

        metric_keys = ["precision_at_k", "recall_at_k", "mrr", "ndcg_at_k", "map"]
        metric_titles = ["Precision@K", "Recall@K", "MRR", "NDCG@K", "MAP"]
        n_steps = len(all_metrics[0])
        step_labels = self.rankflows[0].step_labels[:n_steps]

        n_metrics = len(metric_keys)
        ncols = 3
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (key, title) in enumerate(
            zip(metric_keys, metric_titles, strict=True)
        ):
            ax = axes_flat[idx]
            data_per_step = []
            for step in range(n_steps):
                values = [m[step][key] for m in all_metrics if step < len(m)]
                data_per_step.append(values)

            bp = ax.boxplot(data_per_step, labels=step_labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
            # Mean line overlay
            means = [np.mean(v) for v in data_per_step]
            ax.plot(
                range(1, n_steps + 1),
                means,
                "r--o",
                markersize=4,
                linewidth=1,
                label="mean",
            )
            ax.set_title(title, fontsize=11)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7)

        # Hide unused subplots
        for idx in range(n_metrics, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"Retrieval metrics dashboard (K={k})", fontsize=14)
        plt.tight_layout()
        return fig, axes

    # ------------------------------------------------------------------
    # Win / loss / tie analysis
    # ------------------------------------------------------------------

    def win_loss_analysis(
        self,
        metric: str = "ndcg_at_k",
        k: int = 10,
        tolerance: float = 1e-6,
    ) -> list[dict[str, Any]]:
        """Count wins/losses/ties per step transition.

        Returns a list of dicts (one per transition) with keys:
        transition, wins, losses, ties, win_pct, loss_pct.
        """
        all_metrics = self._collect_all_metrics(k)
        if not all_metrics:
            return []

        n_steps = len(all_metrics[0])
        step_labels = self.rankflows[0].step_labels[:n_steps]
        results = []
        for s in range(n_steps - 1):
            wins = losses = ties = 0
            for m in all_metrics:
                if s + 1 >= len(m):
                    continue
                delta = m[s + 1][metric] - m[s][metric]
                if delta > tolerance:
                    wins += 1
                elif delta < -tolerance:
                    losses += 1
                else:
                    ties += 1
            total = wins + losses + ties
            results.append(
                {
                    "transition": f"{step_labels[s]} -> {step_labels[s + 1]}",
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "win_pct": wins / total * 100 if total else 0,
                    "loss_pct": losses / total * 100 if total else 0,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Query difficulty segmentation
    # ------------------------------------------------------------------

    def plot_by_difficulty(
        self,
        k: int = 10,
        metric: str = "ndcg_at_k",
        buckets: int = 3,
    ) -> Any:
        """Segment queries by difficulty and show metric evolution per bucket.

        Difficulty is determined by the final-step metric value.
        Returns (fig, axes).
        """
        import matplotlib.pyplot as plt

        all_metrics = self._collect_all_metrics(k)
        if not all_metrics:
            raise ValueError("No metrics available.")

        n_steps = len(all_metrics[0])
        step_labels = self.rankflows[0].step_labels[:n_steps]

        # Score each query by final-step metric
        final_scores = [m[-1][metric] for m in all_metrics]
        sorted_indices = np.argsort(final_scores)
        bucket_size = len(sorted_indices) // buckets
        bucket_names = (
            ["Hard", "Medium", "Easy"]
            if buckets == 3
            else [f"Bucket {i + 1}" for i in range(buckets)]
        )

        fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_steps), 5))
        x = np.arange(n_steps)
        colors = ["red", "orange", "green", "blue", "purple"]

        for b in range(buckets):
            start = b * bucket_size
            end = (b + 1) * bucket_size if b < buckets - 1 else len(sorted_indices)
            bucket_indices = sorted_indices[start:end]
            bucket_metrics = [all_metrics[i] for i in bucket_indices]

            means = []
            stds = []
            for s in range(n_steps):
                vals = [m[s][metric] for m in bucket_metrics if s < len(m)]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            label = bucket_names[b] if b < len(bucket_names) else f"Bucket {b + 1}"
            color = colors[b % len(colors)]
            ax.errorbar(
                x,
                means,
                yerr=stds,
                marker="o",
                capsize=4,
                label=f"{label} (n={len(bucket_indices)})",
                color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, rotation=15)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by query difficulty (K={k})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        return fig, ax

    # ------------------------------------------------------------------
    # Metric improvement heatmap
    # ------------------------------------------------------------------

    def plot_improvement_heatmap(
        self,
        metric: str = "ndcg_at_k",
        k: int = 10,
    ) -> Any:
        """Heatmap: rows=queries, cols=step transitions, color=metric delta.

        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        all_metrics = self._collect_all_metrics(k)
        if not all_metrics:
            raise ValueError("No metrics available.")

        n_steps = len(all_metrics[0])
        n_queries = len(all_metrics)
        step_labels = self.rankflows[0].step_labels[:n_steps]

        # Build delta matrix (n_queries x n_transitions)
        n_transitions = n_steps - 1
        delta_matrix = np.zeros((n_queries, n_transitions))
        for q in range(n_queries):
            for s in range(n_transitions):
                delta_matrix[q, s] = (
                    all_metrics[q][s + 1][metric] - all_metrics[q][s][metric]
                )

        # Sort by overall improvement (sum of deltas)
        sort_order = np.argsort(delta_matrix.sum(axis=1))
        delta_matrix = delta_matrix[sort_order]

        transition_labels = [
            f"{step_labels[s]}\n->\n{step_labels[s + 1]}" for s in range(n_transitions)
        ]

        fig_height = max(4, n_queries * 0.15)
        fig, ax = plt.subplots(figsize=(max(5, 2 * n_transitions), fig_height))

        vmax = max(abs(delta_matrix.min()), abs(delta_matrix.max()), 0.01)
        im = ax.imshow(
            delta_matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax
        )
        ax.set_xticks(range(n_transitions))
        ax.set_xticklabels(transition_labels, fontsize=8)
        ax.set_ylabel("Queries (sorted by total improvement)")
        ax.set_title(f"{metric} change per step transition (K={k})")
        fig.colorbar(im, ax=ax, label=f"Δ {metric}")
        plt.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Failure case identification
    # ------------------------------------------------------------------

    def failure_cases(
        self,
        metric: str = "ndcg_at_k",
        k: int = 10,
        threshold: float = -0.1,
    ) -> list[dict[str, Any]]:
        """Find queries where the pipeline degraded performance.

        Returns a list of dicts with query_index, initial_value,
        final_value, delta for queries where final - initial < threshold.
        """
        all_metrics = self._collect_all_metrics(k)
        failures = []
        for idx, m in enumerate(all_metrics):
            initial = m[0][metric]
            final = m[-1][metric]
            delta = final - initial
            if delta < threshold:
                rf = self.rankflows[idx]
                label = getattr(rf, "query_label", None) or f"query_{idx}"
                failures.append(
                    {
                        "query_index": idx,
                        "query_label": label,
                        "initial_value": initial,
                        "final_value": final,
                        "delta": delta,
                    }
                )
        return failures
