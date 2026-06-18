"""DAG-aware rank flow for hybrid search pipelines with branching/merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BRANCH_LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class PipelineStep:
    """A single step in a retrieval pipeline DAG.

    Args:
        name: Human-readable name (e.g., "BM25", "RRF Merge").
        ranks: 1D array of ranks for each chunk at this step.
            Length may differ between steps (e.g., 100 for BM25, 200 for merge).
        chunk_labels: Labels for the chunks ranked in this step.
        parents: Names of parent steps. Empty for root steps.
    """

    name: str
    ranks: np.ndarray = field(repr=False)
    chunk_labels: list[str] = field(repr=False)
    parents: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.ranks = np.asarray(self.ranks)
        if self.ranks.ndim != 1:
            raise ValueError(
                f"PipelineStep '{self.name}' ranks must be 1D, got shape {self.ranks.shape}"
            )
        if len(self.ranks) != len(self.chunk_labels):
            raise ValueError(
                f"PipelineStep '{self.name}': ranks length ({len(self.ranks)}) "
                f"!= chunk_labels length ({len(self.chunk_labels)})"
            )

    @property
    def top_k_labels(self) -> list[str]:
        """Return chunk labels sorted by rank (ascending)."""
        order = np.argsort(self.ranks)
        return [self.chunk_labels[i] for i in order]

    def top_k_set(self, k: int) -> set[str]:
        """Set of chunk labels in the top-K."""
        return set(self.top_k_labels[:k])


class MergeRankFlow:
    """Visualize and analyze a DAG-structured retrieval pipeline.

    Models hybrid search flows where multiple branches (e.g., BM25, vector)
    merge via reciprocal rank fusion or other combination strategies.
    """

    def __init__(self, steps: list[PipelineStep]):
        if not steps:
            raise ValueError("At least one PipelineStep is required.")
        self.steps = {s.name: s for s in steps}
        self._step_order = [s.name for s in steps]
        self._validate_dag()

    def _validate_dag(self):
        """Ensure all parent references are valid."""
        names = set(self.steps.keys())
        for step in self.steps.values():
            for parent in step.parents:
                if parent not in names:
                    raise ValueError(
                        f"Step '{step.name}' references unknown parent '{parent}'"
                    )

    @property
    def root_steps(self) -> list[str]:
        """Steps with no parents (entry points)."""
        return [name for name, step in self.steps.items() if not step.parents]

    @property
    def merge_steps(self) -> list[str]:
        """Steps with multiple parents (merge points)."""
        return [name for name, step in self.steps.items() if len(step.parents) > 1]

    # ------------------------------------------------------------------
    # Overlap analysis
    # ------------------------------------------------------------------

    def overlap_analysis(self, k: int = 10) -> list[dict[str, Any]]:
        """Analyze doc overlap between branches at each merge point.

        For each merge step, computes overlap between its parent branches.
        Returns a list of dicts with: merge_step, parents, shared, exclusive
        per parent, total_unique.
        """
        results = []
        for merge_name in self.merge_steps:
            merge_step = self.steps[merge_name]
            parent_sets = {}
            for pname in merge_step.parents:
                parent_step = self.steps[pname]
                parent_sets[pname] = parent_step.top_k_set(k)

            parent_names = list(parent_sets.keys())
            all_sets = list(parent_sets.values())

            # Shared across all parents
            shared = set.intersection(*all_sets) if all_sets else set()

            # Exclusive to each parent
            exclusive = {}
            for pname, pset in parent_sets.items():
                others = (
                    set.union(*(s for n, s in parent_sets.items() if n != pname))
                    if len(parent_sets) > 1
                    else set()
                )
                exclusive[pname] = pset - others

            total_unique = set.union(*all_sets) if all_sets else set()

            results.append(
                {
                    "merge_step": merge_name,
                    "parents": parent_names,
                    "shared": shared,
                    "shared_count": len(shared),
                    "exclusive": dict(exclusive),
                    "exclusive_counts": {p: len(ex) for p, ex in exclusive.items()},
                    "total_unique": len(total_unique),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Rank correlation between branches
    # ------------------------------------------------------------------

    def rank_correlation(self) -> list[dict[str, Any]]:
        """Compute Spearman rank correlation between branches at merge points.

        For docs appearing in both parent branches, compute how similarly
        they are ranked. High correlation = redundant searches.

        Returns a list of dicts with: merge_step, parent_pair,
        common_docs, spearman_rho.
        """
        results = []
        for merge_name in self.merge_steps:
            merge_step = self.steps[merge_name]
            parents = merge_step.parents
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    p1 = self.steps[parents[i]]
                    p2 = self.steps[parents[j]]
                    # Find common docs
                    set1 = set(p1.chunk_labels)
                    set2 = set(p2.chunk_labels)
                    common = set1 & set2
                    if len(common) < 2:
                        results.append(
                            {
                                "merge_step": merge_name,
                                "parent_pair": (parents[i], parents[j]),
                                "common_docs": len(common),
                                "spearman_rho": None,
                            }
                        )
                        continue

                    # Build rank vectors for common docs
                    label_to_rank1 = dict(zip(p1.chunk_labels, p1.ranks, strict=True))
                    label_to_rank2 = dict(zip(p2.chunk_labels, p2.ranks, strict=True))
                    common_sorted = sorted(common)
                    r1 = np.array(
                        [label_to_rank1[c] for c in common_sorted], dtype=float
                    )
                    r2 = np.array(
                        [label_to_rank2[c] for c in common_sorted], dtype=float
                    )

                    # Spearman rho (rank correlation)
                    n = len(common_sorted)
                    d_sq = np.sum((r1 - r2) ** 2)
                    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))

                    results.append(
                        {
                            "merge_step": merge_name,
                            "parent_pair": (parents[i], parents[j]),
                            "common_docs": n,
                            "spearman_rho": float(rho),
                        }
                    )
        return results

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, top_k: int = 15, relevant_chunks: list[str] | None = None) -> Any:
        """Plot the DAG-structured pipeline as a multi-column rankflow.

        Branches are shown as parallel columns on the left, converging
        at merge points on the right. Only top-K docs per step are shown
        as individual lines; the rest as density bands.

        Returns (fig, axes).
        """
        import matplotlib.pyplot as plt

        levels = self._compute_levels()
        max_level = max(levels.values())
        step_x = self._compute_step_positions(levels)

        fig_width = max(8, 3 * (max_level + 1))
        fig_height = max(6, 0.3 * top_k + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.invert_yaxis()

        # Draw vertical step lines and labels
        for name, x in step_x.items():
            ax.axvline(x=x, color=(0.3, 0.3, 0.3), linestyle="--", linewidth=0.5)
            ax.text(x, -1.5, name, fontsize=10, ha="center", fontweight="bold")

        chunk_color = self._assign_chunk_colors(top_k)
        legend_handles = self._draw_step_lines(
            ax, step_x, chunk_color, top_k, relevant_chunks
        )
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

        ax.set_xlim(-0.5, max_level + 0.8)
        ax.set_ylabel("Rank")
        ax.set_title("Pipeline flow: branch & merge")
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        return fig, ax

    def _compute_step_positions(self, levels: dict[str, int]) -> dict[str, float]:
        """Assign x positions based on topological levels."""
        step_x: dict[str, float] = {}
        for name, level in levels.items():
            step_x[name] = float(level)
        return step_x

    def _assign_chunk_colors(self, top_k: int) -> dict[str, str]:
        """Assign consistent colors to chunks across steps."""
        colors = [
            "blue",
            "green",
            "red",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "magenta",
            "lime",
            "teal",
            "navy",
        ]
        chunk_color: dict[str, str] = {}
        color_idx = 0
        for step_name in self._step_order:
            step = self.steps[step_name]
            for label in step.top_k_labels[:top_k]:
                if label not in chunk_color:
                    chunk_color[label] = colors[color_idx % len(colors)]
                    color_idx += 1
        return chunk_color

    def _draw_step_lines(self, ax, step_x, chunk_color, top_k, relevant_chunks):
        """Draw doc dots and connecting lines for each step.

        Returns a list of legend handles for branch line styles.
        """
        import matplotlib.lines as mlines

        # Build a stable parent -> linestyle mapping across all merge points
        parent_style: dict[str, str] = {}
        style_idx = 0
        for sn in self._step_order:
            s = self.steps[sn]
            if len(s.parents) > 1:
                for pname in s.parents:
                    if pname not in parent_style:
                        parent_style[pname] = BRANCH_LINESTYLES[
                            style_idx % len(BRANCH_LINESTYLES)
                        ]
                        style_idx += 1

        # Build legend handles from the stable mapping
        legend_handles: list = []
        for pname, ls in parent_style.items():
            handle = mlines.Line2D(
                [],
                [],
                color="gray",
                linestyle=ls,
                linewidth=2,
                label=f"from {pname}",
            )
            legend_handles.append(handle)

        for step_name in self._step_order:
            step = self.steps[step_name]
            x_pos = step_x[step_name]
            label_to_rank = dict(zip(step.chunk_labels, step.ranks, strict=True))
            top_labels = step.top_k_labels[:top_k]

            for label in top_labels:
                rank = label_to_rank[label]
                color = chunk_color.get(label, "gray")
                is_relevant = relevant_chunks is not None and label in relevant_chunks
                alpha = 0.9 if is_relevant else 0.4
                lw = 3 if is_relevant else 1

                ax.plot(x_pos, rank, "o", color=color, markersize=4, alpha=alpha)
                self._draw_parent_connections(
                    ax,
                    step,
                    label,
                    rank,
                    x_pos,
                    step_x,
                    color,
                    alpha,
                    lw,
                    top_k,
                    parent_style,
                )

            # Label top docs at leaf steps (no children)
            is_leaf = not any(
                step_name in self.steps[s].parents for s in self._step_order
            )
            if is_leaf:
                for label in top_labels:
                    ax.text(
                        x_pos + 0.1,
                        label_to_rank[label],
                        label,
                        fontsize=7,
                        ha="left",
                        va="center",
                        alpha=0.7,
                    )

        return legend_handles

    def _draw_parent_connections(
        self,
        ax,
        step,
        label,
        rank,
        x_pos,
        step_x,
        color,
        alpha,
        lw,
        top_k,
        parent_style,
    ):
        """Draw lines from parent steps to the current step for a given doc.

        Each parent branch gets a distinct line style so that lines from
        different sources (e.g., text search vs. vector search) are
        visually distinguishable at merge points.
        """
        for parent_name in step.parents:
            ls = parent_style.get(parent_name, "-")
            parent = self.steps[parent_name]
            parent_label_to_rank = dict(
                zip(parent.chunk_labels, parent.ranks, strict=True)
            )
            if label in parent_label_to_rank:
                parent_rank = parent_label_to_rank[label]
                if parent_rank < top_k:
                    ax.plot(
                        [step_x[parent_name], x_pos],
                        [parent_rank, rank],
                        color=color,
                        alpha=alpha,
                        linewidth=lw,
                        linestyle=ls,
                        solid_capstyle="round",
                    )
            else:
                ax.plot(
                    [x_pos - 0.3, x_pos],
                    [rank, rank],
                    color=color,
                    alpha=alpha * 0.5,
                    linewidth=lw,
                    linestyle=":",
                    solid_capstyle="round",
                )

    def _compute_levels(self) -> dict[str, int]:
        """Topological sort to assign level (x-position) to each step."""
        levels: dict[str, int] = {}
        visited: set[str] = set()

        def visit(name: str) -> int:
            if name in levels:
                return levels[name]
            if name in visited:
                raise ValueError(f"Cycle detected involving step '{name}'")
            visited.add(name)
            step = self.steps[name]
            if not step.parents:
                levels[name] = 0
            else:
                parent_levels = [visit(p) for p in step.parents]
                levels[name] = max(parent_levels) + 1
            return levels[name]

        for name in self._step_order:
            visit(name)
        return levels
