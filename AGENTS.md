# AGENTS.md

## Project Overview

**rankflow** is a Python library for visualizing and evaluating rank evolution across retrieval/re-ranking steps, primarily used in RAG (Retrieval Augmented Generation) pipelines.

## Repository Structure

```
src/rankflow/
├── __init__.py              # Public API: RankFlow, BatchRankFlow, PlotConfig
├── _version.py              # Single source of version (e.g. "0.2.0")
├── core.py                  # RankFlow class - main orchestrator
├── main.py                  # Backward-compat shim (re-exports RankFlow)
├── config.py                # PlotConfig dataclass
├── metrics.py               # Retrieval metrics (P@K, R@K, MRR, NDCG, MAP)
├── analysis.py              # Rank deltas, summary, top-K filtering
├── batch.py                 # BatchRankFlow for multi-query aggregation
├── export.py                # to_dict, to_dataframe, to_json utilities
└── plotting/
    ├── __init__.py
    ├── base.py              # PlotBackend ABC
    ├── matplotlib_backend.py # Default matplotlib renderer
    └── plotly_backend.py    # Optional interactive plotly renderer
tests/
├── __init__.py
├── e2e/
│   └── test_main.py         # End-to-end visual tests
└── unit/
    ├── test_metrics.py
    ├── test_analysis.py
    ├── test_config.py
    ├── test_export.py
    └── test_batch.py
```

## Key Commands

```bash
# Run unit tests
make test

# Run end-to-end tests (opens matplotlib windows)
make test-e2e

# Lint
make lint

# Format
make format

# Type check
make type
```

## Conventions

- Python >=3.9 compatibility required
- Formatter: black (88 char line length)
- Linter: ruff (see ruff.toml for rule selection)
- Import sorting: isort
- Type hints used throughout; Optional types for nullable params
- Dependencies: matplotlib + numpy are core; pandas and plotly are optional extras
- Tests use pytest; e2e tests are in tests/e2e/, unit tests in tests/unit/
- Version is stored in `src/rankflow/_version.py` AND `pyproject.toml` (keep in sync)
- Build system: PDM

## Architecture Notes

- `PlotConfig` (dataclass) holds all visual configuration -- never use a mutable global dict.
- `PlotBackend` (ABC) is the interface for renderers. New backends (e.g. SVG) can be added by implementing this interface.
- `RankFlow.plot(backend="matplotlib"|"plotly")` selects the renderer at call time.
- Metrics are pure functions in `metrics.py` -- stateless, easily testable.
- `BatchRankFlow` aggregates across multiple `RankFlow` instances for multi-query evaluation.
- All new features (highlighting, top-K, deltas, scores) are opt-in via constructor parameters or PlotConfig fields. The default `RankFlow(df=df).plot()` call remains backward-compatible.
