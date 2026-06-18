# Changelog

All notable changes to this project will be documented in this file.

## [0.2.1] - 2026-06-18

### Bug Fixes

- Suppress false-positive bandit subprocess warnings in cli
- Use raw string for regex pattern in pytest.raises match (RUF043)

### Documentation

- Update README and add experiment/UI notebooks

### Features

- Add chunk label properties and pipeline_config metadata
- Add experiment tracking and comparison
- Add CLI and Streamlit web UI
- Distinguish merge branches with line styles and legend

### Styling

- Apply ruff formatting

## [0.2.0] - 2026-02-14

### Build

- Migrate from PDM to uv/hatchling
- Add pre-commit, update ruff.toml and cliff.toml

### CI

- Add GitHub Actions workflows

### Documentation

- Add tutorial notebook series and demo
- Add mkdocs documentation with Material theme
- Regenerate CHANGELOG.md with git-cliff

### Features

- Add BatchRankFlow and MergeRankFlow
- Add adapters for TREC, JSON, ranx, and RAGAS formats

### Miscellaneous

- Bump version to 0.2.0 and update project metadata

### Refactor

- Extract config, metrics, analysis, and export modules
- Extract core RankFlow and plotting backend
- Update Makefile and add supporting files

### Testing

- Add unit tests for all new modules

## [0.1.4] - 2024-07-14

### Documentation

- Update changelog
- Add link to article to readme

### Miscellaneous

- Bump-up version

## [0.1.3] - 2024-07-01

### Documentation

- Add changelog
- Update changelog
- Use GitHub raw graphics links
- Add call for action for starring

### Miscellaneous

- Add git-cliff config
- Remove extra targets from Makefile

### Bump

- Bump-up version

## [0.1.2] - 2024-06-26

### Bug Fixes

- Fix automatic label generation

### Build

- Add optional dependencies
- Bump-up version
- Update pdm.lock

### Documentation

- Add more examples to README

### Features

- Accept dataframe as input
- Add automatic fig_size selection

### Styling

- Add comments, reformat

### Testing

- Add more tests

## [0.1.1] - 2024-06-25

### Miscellaneous

- Add heading image
- Update readme
- Remove
- Update project description and versioning

## [0.1.0] - 2024-06-25

### Miscellaneous

- Initial commit


