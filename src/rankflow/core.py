"""Core RankFlow class -- orchestrates data, metrics, plotting, and export."""

from __future__ import annotations

import contextlib
from typing import Any, Literal

import numpy as np

from rankflow.analysis import compute_rank_deltas, compute_summary, filter_top_k
from rankflow.config import PlotConfig
from rankflow.export import ranks_to_dataframe, ranks_to_dict, ranks_to_json
from rankflow.metrics import _relevant_set, compute_metrics_per_step


class RankFlow:
    """Visualize and analyze rank evolution across retrieval steps.

    Args:
        ranks: (n_steps, n_chunks) numpy array of integer ranks.
        step_labels: Labels for each step (row).
        chunk_labels: Labels for each chunk/document (column).
        df: pandas DataFrame alternative to ranks/step_labels/chunk_labels.
        relevant_chunks: List of chunk labels or column indices that are
            known to be relevant (ground truth).
        relevance_grades: Mapping from chunk label/index to a numeric
            relevance grade (higher = more relevant).
        scores: (n_steps, n_chunks) array of raw retrieval scores.
        **kwargs: Override any PlotConfig field.
    """

    def __init__(
        self,
        ranks=None,
        step_labels=None,
        chunk_labels=None,
        df=None,
        relevant_chunks: list[str | int] | None = None,
        relevance_grades: dict[str | int, float] | None = None,
        scores: np.ndarray | None = None,
        source_labels: dict[str | int, str] | None = None,
        pipeline_config: dict[str, Any] | None = None,
        chunk_properties: dict[str, list[str]] | None = None,
        **kwargs,
    ):
        if df is not None:
            ranks = df.to_numpy()
            step_labels = df.index.to_list()
            chunk_labels = df.columns.to_list()
        elif ranks is None:
            raise ValueError("ranks must be provided if not providing dataframe.")

        self.ranks = np.asarray(ranks)
        if self.ranks.ndim != 2:
            raise ValueError(
                f"ranks must be a 2D array with shape (n_steps, n_chunks), "
                f"got shape {self.ranks.shape}"
            )
        self.step_labels = self._default_labels(
            step_labels, self.ranks.shape[0], "Step"
        )
        self.chunk_labels = self._default_labels(
            chunk_labels, self.ranks.shape[1], "Chunk"
        )
        self.config = PlotConfig.from_kwargs(**kwargs)

        # Document label properties
        if chunk_properties is not None:
            for key, values in chunk_properties.items():
                if len(values) != len(self.chunk_labels):
                    raise ValueError(
                        f"chunk_properties['{key}'] length ({len(values)}) "
                        f"must match chunk_labels length ({len(self.chunk_labels)})"
                    )
        self._chunk_properties = chunk_properties

        # Relevance
        self.relevant_chunks = relevant_chunks
        self.relevance_grades = relevance_grades
        self._relevant_indices = (
            _relevant_set(relevant_chunks, self.chunk_labels)
            if relevant_chunks is not None
            else None
        )
        self._relevance_grade_map = self._resolve_grades(relevance_grades)

        # Scores
        self.scores = np.asarray(scores) if scores is not None else None

        # Source provenance
        self._source_labels = self._resolve_labels_map(source_labels)

        # Pipeline config metadata (for experiment tracking)
        self.pipeline_config = dict(pipeline_config) if pipeline_config else None

        # Absent mask: True where rank is NaN
        self._absent_mask: np.ndarray | None = None
        if np.issubdtype(self.ranks.dtype, np.floating) and np.any(
            np.isnan(self.ranks)
        ):
            self._absent_mask = np.isnan(self.ranks)
            max_rank = np.nanmax(self.ranks)
            self.ranks = np.where(self._absent_mask, max_rank + 1, self.ranks)

    @staticmethod
    def _default_labels(labels, n, prefix):
        if labels is None:
            return [f"{prefix} {i}" for i in range(n)]
        return list(labels)

    def _resolve_grades(self, relevance_grades):
        if relevance_grades is None:
            return None
        grade_map: dict[int, float] = {}
        for key, grade in relevance_grades.items():
            if isinstance(key, int):
                grade_map[key] = float(grade)
            else:
                with contextlib.suppress(ValueError):
                    grade_map[self.chunk_labels.index(key)] = float(grade)
        return grade_map

    def _resolve_labels_map(self, source_labels):
        if source_labels is None:
            return None
        result: dict[int, str] = {}
        for key, src in source_labels.items():
            if isinstance(key, int):
                result[key] = src
            else:
                with contextlib.suppress(ValueError):
                    result[self.chunk_labels.index(key)] = src
        return result

    # ------------------------------------------------------------------
    # Label resolution
    # ------------------------------------------------------------------

    def _get_labels_for_key(self, key: str | None) -> list[str]:
        """Resolve a label key to a list of display labels."""
        if key and self._chunk_properties and key in self._chunk_properties:
            return self._chunk_properties[key]
        return self.chunk_labels

    def _get_left_labels(self) -> list[str]:
        return self._get_labels_for_key(self.config.left_label_key)

    def _get_right_labels(self) -> list[str]:
        return self._get_labels_for_key(self.config.right_label_key)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(
        self,
        backend: Literal["matplotlib", "plotly"] = "matplotlib",
        mode: Literal["standard", "density"] = "standard",
    ) -> Any:
        """Render the rankflow plot.

        Args:
            backend: "matplotlib" (default) or "plotly" (requires plotly).
            mode: "standard" (default line plot) or "density" (density bands
                  with focus lines for top-K and relevant docs -- suitable
                  for 100+ documents).

        Returns:
            (fig, axes) for matplotlib or a plotly Figure for plotly.
        """
        # Apply top-K filtering if configured (only in standard mode)
        if mode == "density":
            ranks = self.ranks
            chunk_labels = self.chunk_labels
            kept_indices = list(range(self.ranks.shape[1]))
        else:
            ranks, chunk_labels, kept_indices = self._apply_top_k()

        # Resolve display labels
        full_left = self._get_left_labels()
        full_right = self._get_right_labels()

        # Density mode -- uses a separate render path
        if mode == "density":
            from rankflow.plotting.matplotlib_backend import MatplotlibBackend

            renderer = MatplotlibBackend()
            relevant_idx = self._relevant_indices
            source_map = self._source_labels
            right_labels = full_right if full_right is not self.chunk_labels else None
            return renderer.render_density(
                ranks=ranks,
                step_labels=self.step_labels,
                chunk_labels=chunk_labels,
                config=self.config,
                relevant_indices=relevant_idx,
                source_labels=source_map,
                focus_k=self.config.density_focus_k,
                right_labels=right_labels,
            )

        # Compute per-step metrics if relevant_chunks provided and show_metrics
        step_metrics = None
        if self._relevant_indices is not None and self.config.show_metrics:
            step_metrics = compute_metrics_per_step(
                ranks,
                chunk_labels,
                list(self._remap_relevant(kept_indices)),
                k=self.config.top_k or 5,
                relevance_grades=self._remap_grades(kept_indices),
            )

        # Compute deltas if needed
        deltas = compute_rank_deltas(ranks) if self.config.show_deltas else None

        # Remap relevance indices to filtered space
        relevant_idx = (
            self._remap_relevant(kept_indices) if self._relevant_indices else None
        )
        grade_map = (
            self._remap_grades(kept_indices) if self._relevance_grade_map else None
        )

        # Scores (filtered)
        scores = self.scores[:, kept_indices] if self.scores is not None else None

        # Absent mask (filtered)
        absent_mask = (
            self._absent_mask[:, kept_indices]
            if self._absent_mask is not None
            else None
        )

        if backend == "plotly":
            from rankflow.plotting.plotly_backend import PlotlyBackend

            renderer = PlotlyBackend()
        else:
            from rankflow.plotting.matplotlib_backend import MatplotlibBackend

            renderer = MatplotlibBackend()

        # Filter display labels to match kept_indices
        left_labels = (
            [full_left[i] for i in kept_indices]
            if full_left is not self.chunk_labels
            else None
        )
        right_labels = (
            [full_right[i] for i in kept_indices]
            if full_right is not self.chunk_labels
            else None
        )

        return renderer.render(
            ranks=ranks,
            step_labels=self.step_labels,
            chunk_labels=chunk_labels,
            config=self.config,
            relevant_indices=relevant_idx,
            relevance_grades=grade_map,
            scores=scores,
            step_metrics=step_metrics,
            deltas=deltas,
            absent_mask=absent_mask,
            left_labels=left_labels,
            right_labels=right_labels,
        )

    def iplot(self) -> Any:
        """Shorthand for interactive Plotly plot."""
        return self.plot(backend="plotly")

    # ------------------------------------------------------------------
    # A/B Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare(
        a: RankFlow, b: RankFlow, labels: tuple = ("Pipeline A", "Pipeline B")
    ) -> Any:
        """Side-by-side comparison of two RankFlow instances.

        Returns (fig, (ax_left, ax_right)).
        """
        import matplotlib.pyplot as plt

        from rankflow.plotting.matplotlib_backend import MatplotlibBackend

        backend = MatplotlibBackend()

        fig, (ax_left, ax_right) = plt.subplots(
            1, 2, figsize=(14, max(5, 0.5 * a.ranks.shape[1])), sharey=True
        )

        # Render pipeline A
        ax_left.invert_yaxis()
        _render_on_axes(backend, a, ax_left)
        ax_left.set_title(labels[0], fontsize=a.config.title_font_size)

        # Render pipeline B
        _render_on_axes(backend, b, ax_right)
        ax_right.set_title(labels[1], fontsize=b.config.title_font_size)

        plt.tight_layout()
        return fig, (ax_left, ax_right)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self, k: int = 5) -> list[dict[str, float]] | None:
        """Compute retrieval metrics at each step.

        Returns a list of dicts (one per step) or None if no relevant_chunks.
        """
        if self._relevant_indices is None:
            return None
        return compute_metrics_per_step(
            self.ranks,
            self.chunk_labels,
            self.relevant_chunks,
            k=k,
            relevance_grades=self.relevance_grades,
        )

    def metrics_df(self, k: int = 5):
        """Return metrics as a pandas DataFrame with step_labels as index."""
        m = self.metrics(k=k)
        if m is None:
            return None
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required for metrics_df()") from exc
        return pd.DataFrame(m, index=self.step_labels)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def summary(self) -> list[dict]:
        """Per-chunk rank change summary."""
        return compute_summary(self.ranks, self.chunk_labels)

    def summary_df(self):
        """Summary as a pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas required for summary_df()") from exc
        return pd.DataFrame(self.summary())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        m = self.metrics()
        s = self.summary()
        return ranks_to_dict(self.ranks, self.step_labels, self.chunk_labels, m, s)

    def to_dataframe(self):
        return ranks_to_dataframe(self.ranks, self.step_labels, self.chunk_labels)

    def to_json(self, path: str) -> None:
        m = self.metrics()
        s = self.summary()
        ranks_to_json(path, self.ranks, self.step_labels, self.chunk_labels, m, s)

    def to_rankflow_json(self, path: str, query: str | None = None) -> None:
        """Export to RankFlow's own JSON interchange format."""
        from rankflow.adapters.json_common import save_rankflow_json

        save_rankflow_json(self, path, query=query)

    def to_trec_run(
        self,
        path: str,
        run_id: str = "rankflow",
        step_index: int = -1,
        query_id: str = "q0",
    ) -> None:
        """Export a step as a TREC run file."""
        from rankflow.adapters.trec import save_trec_run

        save_trec_run(
            self, path, run_id=run_id, step_index=step_index, query_id=query_id
        )

    def to_ranx_run(self, step_index: int = -1, query_id: str = "q0"):
        """Export a step as a ranx.Run object (requires ranx)."""
        from rankflow.adapters.ranx_adapter import to_ranx_run

        return to_ranx_run(self, step_index=step_index, query_id=query_id)

    # ------------------------------------------------------------------
    # Import classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_trec_run(cls, paths, qrels_path=None, query_id=None):
        """Load from TREC run file(s). See adapters.trec.load_trec_run."""
        from rankflow.adapters.trec import load_trec_run

        return load_trec_run(paths, qrels_path=qrels_path, query_id=query_id)

    @classmethod
    def from_rankflow_json(cls, path: str):
        """Load from RankFlow JSON schema. See adapters.json_common."""
        from rankflow.adapters.json_common import load_rankflow_json

        return load_rankflow_json(path)

    @classmethod
    def from_ranx(cls, runs, qrels=None, step_labels=None, query_id=None):
        """Load from ranx Run/Qrels objects (requires ranx)."""
        from rankflow.adapters.ranx_adapter import from_ranx

        return from_ranx(runs, qrels=qrels, step_labels=step_labels, query_id=query_id)

    @classmethod
    def from_ragas(cls, dataset, step_label: str = "Retrieval"):
        """Load from RAGAS EvaluationDataset (requires ragas)."""
        from rankflow.adapters.ragas_adapter import from_ragas

        return from_ragas(dataset, step_label=step_label)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_top_k(self):
        """Apply top-K filtering if configured."""
        top_k = self.config.top_k
        if top_k is not None:
            ranks, labels, indices = filter_top_k(
                self.ranks, self.chunk_labels, top_k, self.config.top_k_mode
            )
            return ranks, labels, indices
        return self.ranks, self.chunk_labels, list(range(self.ranks.shape[1]))

    def _remap_relevant(self, kept_indices) -> set[int]:
        """Remap relevant indices to the filtered chunk space."""
        if self._relevant_indices is None:
            return set()
        old_to_new = {old: new for new, old in enumerate(kept_indices)}
        return {old_to_new[i] for i in self._relevant_indices if i in old_to_new}

    def _remap_grades(self, kept_indices) -> dict[int, float] | None:
        """Remap relevance grades to the filtered chunk space."""
        if self._relevance_grade_map is None:
            return None
        old_to_new = {old: new for new, old in enumerate(kept_indices)}
        return {
            old_to_new[i]: g
            for i, g in self._relevance_grade_map.items()
            if i in old_to_new
        }


def _render_on_axes(backend, rf, ax):
    """Render a RankFlow onto a pre-existing matplotlib Axes (for compare)."""
    import matplotlib.pyplot as plt

    # Temporarily monkeypatch plt.subplots to return our existing axes
    original_subplots = plt.subplots
    plt.subplots = lambda **kw: (ax.figure, ax)

    ranks, chunk_labels, kept = rf._apply_top_k()
    relevant_idx = rf._remap_relevant(kept) if rf._relevant_indices else None
    grade_map = rf._remap_grades(kept) if rf._relevance_grade_map else None
    scores = rf.scores[:, kept] if rf.scores is not None else None
    absent_mask = rf._absent_mask[:, kept] if rf._absent_mask is not None else None
    deltas = compute_rank_deltas(ranks) if rf.config.show_deltas else None

    step_metrics = None
    if rf._relevant_indices is not None and rf.config.show_metrics:
        step_metrics = compute_metrics_per_step(
            ranks,
            chunk_labels,
            list(rf._remap_relevant(kept)),
            k=rf.config.top_k or 5,
            relevance_grades=rf._remap_grades(kept),
        )

    full_left = rf._get_left_labels()
    full_right = rf._get_right_labels()
    left_labels = (
        [full_left[i] for i in kept] if full_left is not rf.chunk_labels else None
    )
    right_labels = (
        [full_right[i] for i in kept] if full_right is not rf.chunk_labels else None
    )

    backend.render(
        ranks=ranks,
        step_labels=rf.step_labels,
        chunk_labels=chunk_labels,
        config=rf.config,
        relevant_indices=relevant_idx,
        relevance_grades=grade_map,
        scores=scores,
        step_metrics=step_metrics,
        deltas=deltas,
        absent_mask=absent_mask,
        left_labels=left_labels,
        right_labels=right_labels,
    )

    plt.subplots = original_subplots
