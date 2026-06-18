"""Lightweight experiment registry for retrieval tuning.

Stores named experiment runs as JSON files on disk. No database required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Experiment:
    """A named experiment run containing pipeline config and results.

    Args:
        name: Unique experiment identifier (e.g., "bm25-baseline").
        config: Pipeline configuration dict (retriever, reranker, params...).
        rankflows: List of RankFlow instances (one per query).
        tags: Optional tags for filtering (e.g., ["baseline", "v1"]).
        description: Optional human-readable description.
        timestamp: Auto-set to current UTC time if not provided.
    """

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    rankflows: list = field(default_factory=list, repr=False)
    tags: list[str] = field(default_factory=list)
    description: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    @property
    def n_queries(self) -> int:
        return len(self.rankflows)

    def metrics_summary(self, k: int = 10) -> dict[str, float]:
        """Compute headline metrics across all queries."""
        from rankflow.batch import BatchRankFlow

        if not self.rankflows:
            return {}
        batch = BatchRankFlow(self.rankflows)
        agg = batch.aggregate_metrics(k=k)
        return agg.get("mean", {})

    def as_batch(self):
        """Return a BatchRankFlow for this experiment's queries."""
        from rankflow.batch import BatchRankFlow

        return BatchRankFlow(self.rankflows)

    def to_dict(self) -> dict[str, Any]:
        """Serialize experiment to a JSON-compatible dict."""
        queries = []
        for rf in self.rankflows:
            q: dict[str, Any] = {
                "ranks": rf.ranks.tolist(),
                "step_labels": rf.step_labels,
                "chunk_labels": rf.chunk_labels,
            }
            if rf.relevant_chunks is not None:
                q["relevant_chunks"] = rf.relevant_chunks
            if rf.relevance_grades is not None:
                q["relevance_grades"] = {
                    str(k): v for k, v in rf.relevance_grades.items()
                }
            if rf.scores is not None:
                q["scores"] = rf.scores.tolist()
            if rf.pipeline_config is not None:
                q["pipeline_config"] = rf.pipeline_config
            label = getattr(rf, "query_label", None)
            if label is not None:
                q["query_label"] = label
            queries.append(q)

        return {
            "name": self.name,
            "config": self.config,
            "tags": self.tags,
            "description": self.description,
            "timestamp": self.timestamp,
            "n_queries": self.n_queries,
            "queries": queries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """Deserialize experiment from a dict."""
        from rankflow.core import RankFlow

        rankflows = []
        for q in data.get("queries", []):
            scores = np.array(q["scores"]) if "scores" in q else None
            rel_grades = None
            if "relevance_grades" in q:
                rel_grades = {k: float(v) for k, v in q["relevance_grades"].items()}

            rf = RankFlow(
                ranks=np.array(q["ranks"]),
                step_labels=q["step_labels"],
                chunk_labels=q["chunk_labels"],
                relevant_chunks=q.get("relevant_chunks"),
                relevance_grades=rel_grades,
                scores=scores,
                pipeline_config=q.get("pipeline_config"),
            )
            if "query_label" in q:
                rf.query_label = q["query_label"]
            rankflows.append(rf)

        return cls(
            name=data["name"],
            config=data.get("config", {}),
            rankflows=rankflows,
            tags=data.get("tags", []),
            description=data.get("description", ""),
            timestamp=data.get("timestamp", ""),
        )


class ExperimentStore:
    """File-based experiment store.

    Each experiment is saved as a single JSON file in the store directory.

    Usage::

        store = ExperimentStore("./experiments")
        store.save(experiment)
        loaded = store.load("bm25-baseline")
        all_names = store.list()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _exp_path(self, name: str) -> Path:
        safe_name = name.replace("/", "_").replace("\\", "_")
        return self.path / f"{safe_name}.json"

    def save(self, experiment: Experiment) -> Path:
        """Save an experiment to disk. Returns the file path."""
        out = self._exp_path(experiment.name)
        data = experiment.to_dict()
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return out

    def load(self, name: str) -> Experiment:
        """Load an experiment by name."""
        p = self._exp_path(name)
        if not p.exists():
            raise FileNotFoundError(f"Experiment '{name}' not found at {p}")
        with open(p) as f:
            data = json.load(f)
        return Experiment.from_dict(data)

    def list(self, tag: str | None = None) -> list[dict[str, Any]]:
        """List all experiments. Optionally filter by tag.

        Returns a list of summary dicts (name, timestamp, tags, n_queries, config).
        Does not load full query data for performance.
        """
        results = []
        for p in sorted(self.path.glob("*.json")):
            with open(p) as f:
                data = json.load(f)
            if tag and tag not in data.get("tags", []):
                continue
            results.append(
                {
                    "name": data["name"],
                    "timestamp": data.get("timestamp", ""),
                    "tags": data.get("tags", []),
                    "description": data.get("description", ""),
                    "n_queries": data.get("n_queries", 0),
                    "config": data.get("config", {}),
                }
            )
        return results

    def delete(self, name: str) -> None:
        """Delete an experiment by name."""
        p = self._exp_path(name)
        if p.exists():
            p.unlink()

    def exists(self, name: str) -> bool:
        return self._exp_path(name).exists()
