"""Abstract base for plot backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rankflow.config import PlotConfig


class PlotBackend(ABC):
    """Interface that every plotting backend must implement."""

    @abstractmethod
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
        left_labels: list[str] | None = None,
        right_labels: list[str] | None = None,
    ) -> Any:
        """Render the rankflow plot and return the figure/axes or equivalent."""
        ...
