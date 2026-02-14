"""Interactive Plotly implementation of the RankFlow plot."""

from __future__ import annotations

from typing import Any

import numpy as np

from rankflow.config import PlotConfig
from rankflow.plotting.base import PlotBackend


def _ensure_plotly():
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "plotly is required for the interactive backend. "
            "Install it with: pip install rankflow[interactive]"
        ) from exc


class PlotlyBackend(PlotBackend):
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
        _ensure_plotly()
        import plotly.graph_objects as go

        n_steps, n_chunks = ranks.shape
        colors = config.colors
        n_colors = len(colors)
        has_relevance = relevant_indices is not None

        fig = go.Figure()

        for i in range(n_chunks):
            is_relevant = has_relevance and i in relevant_indices
            alpha = 1.0 if (not has_relevance or is_relevant) else 0.2
            lw = config.line_width * 0.3 if not has_relevance else (
                config.line_width * 0.4 if is_relevant else config.line_width * 0.15
            )
            color = colors[i % n_colors]

            y_data = ranks[:, i].astype(float).tolist()

            # Build hover text
            hover_texts = []
            for step_idx in range(n_steps):
                parts = [
                    f"<b>{chunk_labels[i]}</b>",
                    f"Step: {step_labels[step_idx]}",
                    f"Rank: {int(ranks[step_idx, i])}",
                ]
                if scores is not None:
                    parts.append(f"Score: {scores[step_idx, i]:.4f}")
                if deltas is not None and step_idx > 0:
                    d = int(deltas[step_idx - 1, i])
                    parts.append(f"Delta: {d:+d}")
                if step_metrics is not None:
                    m = step_metrics[step_idx]
                    parts.append(f"P@K: {m['precision_at_k']:.2f}")
                    parts.append(f"R@K: {m['recall_at_k']:.2f}")
                hover_texts.append("<br>".join(parts))

            dash = None
            if absent_mask is not None and np.any(absent_mask[:, i]):
                dash = "dash"

            fig.add_trace(
                go.Scatter(
                    x=list(range(n_steps)),
                    y=y_data,
                    mode="lines+markers",
                    name=chunk_labels[i],
                    line={"color": color, "width": lw, "dash": dash},
                    opacity=alpha,
                    hovertext=hover_texts,
                    hoverinfo="text",
                    marker={"size": 8},
                )
            )

        # Step labels as x-axis ticks
        fig.update_layout(
            title={"text": config.title, "font": {"size": int(config.title_font_size)}},
            xaxis={
                "tickmode": "array",
                "tickvals": list(range(n_steps)),
                "ticktext": step_labels,
            },
            yaxis={
                "title": "Rank",
                "autorange": "reversed",
            },
            hovermode="closest",
            showlegend=True,
            template="plotly_white",
        )

        if config.score_mode == "dual" and scores is not None:
            for i in range(n_chunks):
                color = colors[i % n_colors]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(n_steps)),
                        y=scores[:, i].tolist(),
                        mode="lines",
                        name=f"{chunk_labels[i]} (score)",
                        line={"color": color, "width": 1, "dash": "dot"},
                        opacity=0.5,
                        yaxis="y2",
                        showlegend=False,
                    )
                )
            fig.update_layout(
                yaxis2={
                    "title": "Score",
                    "overlaying": "y",
                    "side": "right",
                },
            )

        return fig
