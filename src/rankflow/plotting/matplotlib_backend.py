"""Matplotlib implementation of the RankFlow plot."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rankflow.config import PlotConfig
from rankflow.plotting.base import PlotBackend


class MatplotlibBackend(PlotBackend):
    def render(
        self,
        ranks: np.ndarray,
        step_labels: list[str],
        chunk_labels: list[str],
        config: PlotConfig,
        relevant_indices: set[int] | None = None,
        relevance_grades: dict[int, float] | None = None,
        scores: np.ndarray | None = None,
        step_metrics: list[dict[str, float]] | None = None,
        deltas: np.ndarray | None = None,
        absent_mask: np.ndarray | None = None,
    ) -> Any:
        fig, axs = self._initialize_plot(ranks, config)

        if config.score_mode == "dual" and scores is not None:
            ax_scores = axs.twinx()
            ax_scores.set_ylabel("Score", fontsize=config.step_label_font_size)
        else:
            ax_scores = None

        self._plot_rank_evolution(
            axs,
            ranks,
            chunk_labels,
            config,
            relevant_indices,
            relevance_grades,
            scores,
            ax_scores,
            absent_mask,
        )
        self._add_step_lines_and_labels(axs, ranks, step_labels, config)
        self._add_chunk_labels(axs, ranks, chunk_labels, step_labels, config)
        self._add_rank_text(axs, ranks, step_labels, chunk_labels, config, absent_mask)

        if config.show_deltas and deltas is not None:
            self._add_delta_annotations(axs, ranks, deltas, config)

        if config.show_metrics and step_metrics is not None:
            self._add_metrics_annotations(axs, ranks, step_labels, step_metrics, config)

        self._finalize_plot(axs, step_labels, config)
        return fig, axs

    def _initialize_plot(self, ranks, config):
        user_fig_size = config.fig_size
        if user_fig_size is None:
            n_docs = ranks.shape[1]
            n_steps = ranks.shape[0]
            x_size = max(5, 1.5 * n_steps)
            y_size = max(5, 0.5 * n_docs)
            fig_size = (x_size, y_size)
        else:
            fig_size = user_fig_size

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=fig_size)
        axs.invert_yaxis()
        return fig, axs

    def _plot_rank_evolution(
        self,
        axs,
        ranks,
        chunk_labels,
        config,
        relevant_indices,
        relevance_grades,
        scores,
        ax_scores,
        absent_mask,
    ):
        n_chunks = len(chunk_labels)
        colors = config.colors
        n_colors = len(colors)
        has_relevance = relevant_indices is not None

        # Build color map from relevance grades
        grade_colors = None
        if has_relevance and relevance_grades:
            try:
                import matplotlib.cm as cm
                cmap = cm.get_cmap(config.relevance_colormap)
                max_grade = max(relevance_grades.values()) if relevance_grades else 1
                min_grade = min(relevance_grades.values()) if relevance_grades else 0
                grade_range = max_grade - min_grade if max_grade != min_grade else 1
                grade_colors = {}
                for idx, grade in relevance_grades.items():
                    normalized = (grade - min_grade) / grade_range
                    grade_colors[idx] = cmap(normalized)
            except Exception:
                grade_colors = None

        for i in range(n_chunks):
            # Determine line style based on relevance
            if has_relevance:
                is_relevant = i in relevant_indices
                alpha = config.relevant_line_alpha if is_relevant else config.irrelevant_line_alpha
                lw = config.line_width * (
                    config.relevant_line_width_multiplier if is_relevant else config.irrelevant_line_width_multiplier
                )
            else:
                alpha = 0.7
                lw = config.line_width

            # Determine color
            if grade_colors and i in grade_colors:
                color = grade_colors[i]
            else:
                color = colors[i % n_colors]

            # Handle absent docs (NaN mask)
            y_data = ranks[:, i].astype(float)
            if absent_mask is not None:
                absent_col = absent_mask[:, i]
                # Plot segments between present points
                segments = self._split_segments(y_data, absent_col)
                for seg_x, seg_y, is_absent in segments:
                    ls = config.absent_line_style if is_absent else "-"
                    a = config.absent_line_alpha if is_absent else alpha
                    axs.plot(seg_x, seg_y, color=color, alpha=a, linewidth=lw, linestyle=ls, solid_capstyle="round")
            else:
                axs.plot(y_data, color=color, alpha=alpha, linewidth=lw, solid_capstyle="round")

            # Score overlay
            if scores is not None and config.score_mode in ("scores", "dual"):
                target_ax = ax_scores if ax_scores is not None else axs
                target_ax.plot(
                    scores[:, i],
                    color=color,
                    alpha=alpha * 0.6,
                    linewidth=max(1, lw * 0.3),
                    linestyle=":",
                    solid_capstyle="round",
                )

    def _split_segments(self, y_data, absent_col):
        """Split a line into segments of present and absent points."""
        n = len(y_data)
        segments = []
        i = 0
        while i < n:
            is_absent = bool(absent_col[i])
            start = i
            while i < n and bool(absent_col[i]) == is_absent:
                i += 1
            # Extend by one to connect segments
            end = min(i + 1, n)
            if start > 0:
                start -= 1
            seg_x = list(range(start, end))
            seg_y = [y_data[j] for j in seg_x]
            segments.append((seg_x, seg_y, is_absent))
        return segments

    def _add_step_lines_and_labels(self, axs, ranks, step_labels, config):
        n_steps = len(step_labels)
        for i in range(n_steps):
            axs.axvline(
                x=i,
                color=config.vertical_line_color,
                linestyle="--",
                linewidth=config.vertical_line_width,
            )
            step_text_y = 1.1 * np.nanmax(ranks)
            axs.text(
                i,
                step_text_y,
                step_labels[i],
                fontsize=config.step_label_font_size,
                ha="center",
            )

    def _add_chunk_labels(self, axs, ranks, chunk_labels, step_labels, config):
        n_chunks = len(chunk_labels)
        n_steps = len(step_labels)
        for i in range(n_chunks):
            axs.text(
                -config.x_axis_limit_offset,
                ranks[0, i],
                chunk_labels[i],
                fontsize=config.chunk_label_font_size,
                ha="right",
            )
            axs.text(
                n_steps - 1 + config.x_axis_limit_offset,
                ranks[-1, i],
                chunk_labels[i],
                fontsize=config.chunk_label_font_size,
                ha="left",
            )

    def _add_rank_text(self, axs, ranks, step_labels, chunk_labels, config, absent_mask):
        n_steps = len(step_labels)
        n_chunks = len(chunk_labels)
        for i in range(n_steps):
            for j in range(n_chunks):
                if absent_mask is not None and absent_mask[i, j]:
                    continue
                axs.text(
                    i + config.x_offset,
                    ranks[i, j],
                    f"{ranks[i, j]}",
                    fontsize=config.rank_text_font_size,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "pad": config.text_pad,
                        "boxstyle": "circle",
                        "alpha": config.text_alpha,
                    },
                    ha="center",
                    va="center",
                    color="black",
                )

    def _add_delta_annotations(self, axs, ranks, deltas, config):
        n_steps = deltas.shape[0]
        n_chunks = deltas.shape[1]
        for i in range(n_steps):
            for j in range(n_chunks):
                d = int(deltas[i, j])
                if d == 0:
                    continue
                x_pos = i + 0.5
                y_pos = (ranks[i, j] + ranks[i + 1, j]) / 2.0
                color = "green" if d < 0 else "red"
                label = f"{d:+d}"
                axs.text(
                    x_pos,
                    y_pos,
                    label,
                    fontsize=config.delta_font_size,
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    alpha=0.8,
                )

    def _add_metrics_annotations(self, axs, ranks, step_labels, step_metrics, config):
        n_steps = len(step_labels)
        y_base = np.nanmax(ranks) + 1.5
        for i in range(n_steps):
            m = step_metrics[i]
            text_lines = [
                f"P@K={m['precision_at_k']:.2f}",
                f"R@K={m['recall_at_k']:.2f}",
                f"MRR={m['mrr']:.2f}",
                f"NDCG={m['ndcg_at_k']:.2f}",
            ]
            text = "\n".join(text_lines)
            axs.text(
                i,
                y_base,
                text,
                fontsize=config.metrics_font_size,
                ha="center",
                va="top",
                bbox={
                    "facecolor": "lightyellow",
                    "edgecolor": "gray",
                    "pad": 3,
                    "alpha": 0.85,
                },
                family="monospace",
            )

    def _finalize_plot(self, axs, step_labels, config):
        n_steps = len(step_labels)
        axs.set_title(
            config.title,
            fontsize=config.title_font_size,
            pad=config.title_pad,
        )
        axs.text(
            -config.x_axis_limit_offset,
            -0.75,
            "Initial\nranking",
            fontsize=config.initial_final_ranking_font_size,
            ha="right",
        )
        axs.text(
            n_steps - 1 + config.x_axis_limit_offset,
            -0.75,
            "Final\nranking",
            fontsize=config.initial_final_ranking_font_size,
            ha="left",
        )
        axs.set_xlim(
            -config.x_axis_limit_offset,
            n_steps - 1 + config.x_axis_limit_offset,
        )
        axs.set_xticks([])
        axs.set_yticks([])
        for spine in axs.spines.values():
            spine.set_visible(False)
        plt.figtext(
            0.5,
            0.02,
            config.caption,
            ha="center",
            fontsize=config.caption_font_size,
        )
        plt.axis("tight")
        plt.subplots_adjust(bottom=0.15)

    # ------------------------------------------------------------------
    # Density mode rendering
    # ------------------------------------------------------------------

    def render_density(  # noqa: C901
        self,
        ranks: np.ndarray,
        step_labels: list[str],
        chunk_labels: list[str],
        config: PlotConfig,
        relevant_indices: set[int] | None = None,
        source_labels: dict[int, str] | None = None,
        focus_k: int | None = None,
    ) -> Any:
        """Render a density-band plot suitable for 100+ documents.

        Shows individual lines for top-K and relevant docs,
        and shaded density bands for the rest.
        """
        fig, axs = self._initialize_plot(ranks, config)
        n_steps, n_chunks = ranks.shape
        k = focus_k or config.density_focus_k

        # Determine which docs get individual lines
        focus_set: set[int] = set()
        if relevant_indices:
            focus_set |= relevant_indices
        # Add top-K at final step
        final_top_k = set(np.argsort(ranks[-1])[:k])
        focus_set |= final_top_k

        background_indices = [i for i in range(n_chunks) if i not in focus_set]

        # Draw density bands for background docs
        if background_indices:
            bg_ranks = ranks[:, background_indices]
            for step in range(n_steps):
                sorted_r = np.sort(bg_ranks[step])
                n_bg = len(sorted_r)
                # Quartile bands
                q25 = sorted_r[max(0, int(n_bg * 0.25))]
                q75 = sorted_r[min(n_bg - 1, int(n_bg * 0.75))]
                q10 = sorted_r[max(0, int(n_bg * 0.1))]
                q90 = sorted_r[min(n_bg - 1, int(n_bg * 0.9))]
                if step < n_steps - 1:
                    next_sorted = np.sort(bg_ranks[step + 1])
                    nq25 = next_sorted[max(0, int(n_bg * 0.25))]
                    nq75 = next_sorted[min(n_bg - 1, int(n_bg * 0.75))]
                    nq10 = next_sorted[max(0, int(n_bg * 0.1))]
                    nq90 = next_sorted[min(n_bg - 1, int(n_bg * 0.9))]
                    # Inner band (25-75%)
                    axs.fill(
                        [step, step + 1, step + 1, step],
                        [q25, nq25, nq75, q75],
                        color=config.density_band_color,
                        alpha=config.density_band_alpha * 2,
                    )
                    # Outer band (10-90%)
                    axs.fill(
                        [step, step + 1, step + 1, step],
                        [q10, nq10, nq90, q90],
                        color=config.density_band_color,
                        alpha=config.density_band_alpha,
                    )

        # Draw focus lines
        colors = config.colors
        n_colors = len(colors)
        for idx, i in enumerate(sorted(focus_set)):
            is_relevant = relevant_indices is not None and i in relevant_indices
            alpha = 0.9 if is_relevant else 0.6
            lw = config.line_width * (0.8 if is_relevant else 0.4)
            color = colors[idx % n_colors]

            # Source marker
            marker = None
            if source_labels and i in source_labels:
                src = source_labels[i]
                marker = config.source_markers.get(src, "o")
                if src in config.source_colors:
                    color = config.source_colors[src]

            y_data = ranks[:, i].astype(float)
            axs.plot(y_data, color=color, alpha=alpha, linewidth=lw, solid_capstyle="round")

            if marker:
                for s in range(n_steps):
                    axs.plot(s, y_data[s], marker=marker, color=color, markersize=6, alpha=alpha)

            # Label at the end
            axs.text(
                n_steps - 1 + config.x_axis_limit_offset,
                ranks[-1, i],
                chunk_labels[i],
                fontsize=config.chunk_label_font_size,
                ha="left",
                alpha=alpha,
            )

        self._add_step_lines_and_labels(axs, ranks, step_labels, config)

        # Legend for density bands
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=config.density_band_color, alpha=config.density_band_alpha * 2,
                  label="Rank 25-75th pctl"),
            Patch(facecolor=config.density_band_color, alpha=config.density_band_alpha,
                  label="Rank 10-90th pctl"),
        ]
        if source_labels:
            for src, marker in config.source_markers.items():
                c = config.source_colors.get(src, "black")
                legend_elements.append(
                    plt.Line2D([0], [0], marker=marker, color=c, linestyle="",
                               markersize=6, label=f"Source: {src}")
                )
        axs.legend(handles=legend_elements, loc="upper right", fontsize=8)

        self._finalize_plot(axs, step_labels, config)
        return fig, axs
