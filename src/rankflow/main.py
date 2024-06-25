# TODO: KS: 2024-06-13: Get rid of numpy dependency
from typing import List, Optional

import matplotlib.pyplot as plt

# Configuration variables
config = {
    "fig_size": (10, 5),
    "colors": [
        "blue",
        "green",
        "red",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "olive",
    ],
    "line_width": 20,
    "vertical_line_width": 1,
    "vertical_line_color": (0.3, 0.3, 0.3),  # dark gray,
    "x_offset": 0.00,
    "title_font_size": 20,
    "step_label_font_size": 12,
    "chunk_label_font_size": 10,
    "caption_font_size": 15,
    "initial_final_ranking_font_size": 12,
    "rank_text_font_size": 10,
    "title_pad": 20,
    "x_axis_limit_offset": 0.1,
    "text_pad": 0.5,
    "text_alpha": 0.5,
}


class RankFlow:
    def __init__(self, ranks, step_labels, chunk_labels, **kwargs):
        self.ranks = ranks
        self.step_labels = self._generate_default_step_labels(step_labels)
        self.chunk_labels = self._generate_default_chunk_labels(chunk_labels)
        self.config = config
        # override default config with user provided config via kwargs
        self.config.update(kwargs)
        self.axs = None

    def plot(self):
        """Main function to execute the plot creation process."""
        self._generate_default_step_labels()
        self._generate_default_chunk_labels()
        self._initialize_plot()
        self._plot_rank_evolution()
        self._add_step_lines_and_labels()
        self._add_chunk_labels()
        self._add_rank_text()
        self._finalize_plot()

    def _initialize_plot(self) -> None:
        """Initialize the plot with a single subplot and set the figure size.

        Sets self.axs to the Axes object for the subplot.
        Operate on inverted y-axis - we expect reading from top to bottom.

        Returns:
            axs: The Axes object for the subplot.
        """
        _, self.axs = plt.subplots(nrows=1, ncols=1, figsize=self.config["fig_size"])
        self.axs.invert_yaxis()

    def _plot_rank_evolution(self) -> None:
        """Plot the rank evolution for each chunk.

        Returns:
            None
        """
        n_chunks = len(self.chunk_labels)
        colors = self.config.get("colors", config["colors"])
        n_colors = len(colors)
        line_width = self.config.get("line_width", config["line_width"])
        for i in range(n_chunks):
            color = colors[i % n_colors]
            self.axs.plot(
                self.ranks[:, i],
                color=color,
                alpha=0.7,
                linewidth=line_width,
                solid_capstyle="round",
            )

    def _add_step_lines_and_labels(self) -> None:
        """
        Draw vertical lines corresponding to the steps and add step labels.

        Args:
            axs: The Axes object for the subplot.
            chunk_labels: A list of strings representing the chunk labels.
            step_labels: A list of strings representing the step labels.
            kwargs: Additional keyword arguments:
                - vertical_line_color: The color of the vertical lines.
                - vertical_line_width: The width of the vertical lines.
                - step_label_font_size: The font size of the step labels.

        """
        vertical_line_color = self.config["vertical_line_color"]
        vertical_line_width = self.config["vertical_line_width"]
        step_label_font_size = self.config["step_label_font_size"]

        n_steps = len(self.step_labels)
        for i in range(1, n_steps + 1):
            self.axs.axvline(
                x=i - 1,
                color=vertical_line_color,
                linestyle="--",
                linewidth=vertical_line_width,
            )
            self.axs.text(
                i - 1,
                len(self.chunk_labels),
                self.step_labels[i - 1],
                fontsize=step_label_font_size,
                ha="center",
            )

    def _add_rank_text(self) -> None:
        """
        Add text with rank value for each step and chunk.

        Args:
            axs: The Axes object for the subplot.
            ranks: A numpy array representing the ranks.
        """
        n_steps = len(self.step_labels)
        n_chunks = len(self.chunk_labels)
        for i in range(n_steps):
            for j in range(n_chunks):
                self.axs.text(
                    i + self.config["x_offset"],
                    self.ranks[i, j],
                    f"{self.ranks[i, j]}",
                    fontsize=self.config["rank_text_font_size"],
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "pad": self.config["text_pad"],
                        "boxstyle": "circle",
                        "alpha": self.config["text_alpha"],
                    },
                    ha="center",
                    va="center",
                    color="black",
                )

    def _add_chunk_labels(self) -> None:
        """
        Add chunk labels at the beginning and end of the plot.

        Args:
            axs: The Axes object for the subplot.
            ranks: A numpy array representing the ranks.
            n_steps: The number of steps.
        """
        n_chunks = len(self.chunk_labels)
        n_steps = len(self.step_labels)
        for i in range(n_chunks):
            self.axs.text(
                -self.config["x_axis_limit_offset"],
                self.ranks[0, i],
                self.chunk_labels[i],
                fontsize=self.config["chunk_label_font_size"],
                ha="right",
            )
            self.axs.text(
                n_steps - 1 + self.config["x_axis_limit_offset"],
                self.ranks[-1, i],
                self.chunk_labels[i],
                fontsize=self.config["chunk_label_font_size"],
                ha="left",
            )

    def _finalize_plot(self) -> None:
        """
        Set titles, labels, and remove borderlines around the plot.

        Args:
            axs: The Axes object for the subplot.
            n_steps: The number of steps.
        """
        n_steps = len(self.step_labels)
        self.axs.set_title(
            "Rank evolution",
            fontsize=self.config["title_font_size"],
            pad=self.config["title_pad"],
        )
        self.axs.text(
            -self.config["x_axis_limit_offset"],
            -0.75,
            "Initial\nranking",
            fontsize=self.config["initial_final_ranking_font_size"],
            ha="right",
        )
        self.axs.text(
            n_steps - 1 + self.config["x_axis_limit_offset"],
            -0.75,
            "Final\nranking",
            fontsize=self.config["initial_final_ranking_font_size"],
            ha="left",
        )
        self.axs.set_xlim(
            -self.config["x_axis_limit_offset"],
            n_steps - 1 + self.config["x_axis_limit_offset"],
        )
        self.axs.set_xticks([])
        self.axs.set_yticks([])
        for spine in self.axs.spines.values():
            spine.set_visible(False)
        plt.figtext(
            0.5,
            0.02,
            "Re-ranking step",
            ha="center",
            fontsize=self.config["caption_font_size"],
        )

        plt.axis("tight")
        # There is a text added to the lower part of the plot that is trimmed at the bottom.
        # Expand the axis that the text will be displayed without trimming
        plt.subplots_adjust(bottom=0.15)

    def _generate_default_chunk_labels(self, chunk_labels: Optional[List[str]] = None):
        if chunk_labels is None:
            self.chunk_labels = [f"Chunk {i}" for i in range(self.ranks.shape[1])]
        else:
            self.chunk_labels = chunk_labels

    def _generate_default_step_labels(self, step_labels: Optional[List[str]] = None):
        if step_labels is None:
            self.step_labels = [f"Step {i}" for i in range(self.ranks.shape[0])]
        else:
            self.step_labels = step_labels
