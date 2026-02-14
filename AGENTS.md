# AI Agent Instructions

## Project Overview

**rankflow** is a Python library for visualizing and evaluating rank evolution across retrieval/re-ranking steps, primarily used in RAG (Retrieval Augmented Generation) pipelines.

## Repository Structure

```
src/rankflow/
├── __init__.py              # Public API: RankFlow, BatchRankFlow, PlotConfig
├── _version.py              # Single source of version
├── core.py                  # RankFlow class - main orchestrator
├── config.py                # PlotConfig dataclass
├── metrics.py               # Retrieval metrics (P@K, R@K, MRR, NDCG, MAP)
├── analysis.py              # Rank deltas, summary, top-K filtering
├── batch.py                 # BatchRankFlow for multi-query aggregation
├── export.py                # to_dict, to_dataframe, to_json utilities
├── merge.py                 # MergeRankFlow for pipeline comparison
├── adapters/                # Format adapters (TREC, RAGAS, ranx)
└── plotting/
    ├── base.py              # PlotBackend ABC
    ├── matplotlib_backend.py # Default matplotlib renderer
    └── plotly_backend.py    # Optional interactive plotly renderer
tests/
├── e2e/                     # End-to-end visual tests
└── unit/                    # Unit tests
docs/                        # MkDocs documentation
notebooks/                   # Tutorial notebooks
```

## Development Commands

```bash
make dev          # Set up development environment
make test         # Run unit tests
make test-e2e     # Run end-to-end tests
make test-cov     # Run tests with coverage
make lint         # Check code style
make format       # Auto-format code
make type-check   # Run type checker
make security     # Run security checks
make docs         # Build documentation
make serve-docs   # Serve docs locally
make commit       # Interactive conventional commit
```

## Tooling Stack

- **Package Manager**: uv
- **Build Backend**: Hatchling
- **Linter/Formatter**: Ruff
- **Type Checker**: ty (Astral)
- **Testing**: pytest + pytest-cov
- **Security**: bandit + pip-audit
- **Documentation**: mkdocs-material

## Code Conventions

- Python >=3.10 required
- Line length: 88 characters
- Type hints: Required for all public functions
- Docstrings: Google style
- Import sorting: Handled by Ruff (isort rules)

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: bug fix
docs: documentation changes
refactor: code refactoring
test: add/update tests
chore: maintenance tasks
perf: performance improvements
```

**Important**: Never add `Co-authored-by` lines to commit messages.

## Architecture Notes

- `PlotConfig` (dataclass) holds all visual configuration
- `PlotBackend` (ABC) is the interface for renderers
- `RankFlow.plot(backend="matplotlib"|"plotly")` selects the renderer
- Metrics are pure functions in `metrics.py` - stateless, easily testable
- `BatchRankFlow` aggregates across multiple `RankFlow` instances
- All new features are opt-in via constructor parameters

## Testing Requirements

- Maintain >80% code coverage
- Run `make test-cov` before submitting PR
- E2E tests are in tests/e2e/, unit tests in tests/unit/

## Release Process

Releases are automated via GitHub Actions when a version tag is pushed:

```bash
make release-patch  # 0.1.0 → 0.1.1
make release-minor  # 0.1.0 → 0.2.0
make release-major  # 0.1.0 → 1.0.0
```
