from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_COLORS = [
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
    "maroon",
    "gold",
]


@dataclass
class PlotConfig:
    """Configuration for RankFlow plots."""

    fig_size: tuple[float, float] | None = None
    colors: list[str] = field(default_factory=lambda: list(DEFAULT_COLORS))
    line_width: float = 20
    vertical_line_width: float = 1
    vertical_line_color: tuple[float, float, float] = (0.3, 0.3, 0.3)
    x_offset: float = 0.00
    title: str = "Rank evolution"
    title_font_size: float = 20
    title_pad: float = 20
    step_label_font_size: float = 12
    chunk_label_font_size: float = 10
    caption: str = "Re-ranking step"
    caption_font_size: float = 15
    initial_final_ranking_font_size: float = 12
    rank_text_font_size: float = 10
    x_axis_limit_offset: float = 0.1
    text_pad: float = 0.5
    text_alpha: float = 0.5

    # Highlighting & relevance
    relevant_line_alpha: float = 0.9
    relevant_line_width_multiplier: float = 1.3
    irrelevant_line_alpha: float = 0.15
    irrelevant_line_width_multiplier: float = 0.5
    relevance_colormap: str = "RdYlGn"

    # Metrics display
    show_metrics: bool = False
    metrics_font_size: float = 8

    # Delta annotations
    show_deltas: bool = False
    delta_font_size: float = 8

    # Top-K filtering
    top_k: int | None = None
    top_k_mode: str = "any"  # "any", "initial", "final"

    # Score visualization
    score_mode: str = "ranks"  # "ranks", "scores", "dual"

    # Dropped/added docs
    absent_line_style: str = "--"
    absent_line_alpha: float = 0.15

    # Density plot mode
    density_band_alpha: float = 0.15
    density_band_color: str = "gray"
    density_focus_k: int = 10

    # Source provenance
    source_markers: dict[str, str] = field(
        default_factory=lambda: {
            "text": "s",
            "vector": "^",
            "both": "o",
        }
    )
    source_colors: dict[str, str] = field(
        default_factory=lambda: {
            "text": "blue",
            "vector": "red",
            "both": "purple",
        }
    )

    @classmethod
    def from_kwargs(cls, **kwargs) -> PlotConfig:
        """Create a PlotConfig from keyword arguments, ignoring unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
        return cls(**filtered)

    def to_dict(self) -> dict:
        """Convert config to a plain dictionary."""
        from dataclasses import asdict

        return asdict(self)

    def override(self, **kwargs) -> PlotConfig:
        """Return a new PlotConfig with specified fields overridden."""
        d = self.to_dict()
        d.update(kwargs)
        return PlotConfig.from_kwargs(**d)
