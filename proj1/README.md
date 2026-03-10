# Advanced Analytics Assignment 1

## Overview

This project contains the first assignment for Advanced Analytics in Business.

## Project Structure

```
proj1/
├── data/           # Raw and processed data files
├── notebooks/      # Jupyter notebooks for analysis/workflows
├── plots/          # Generated plots and visualizations
├── report/         # Final report and documentation
├── src/proj1/      # Main Python package
├── pyproject.toml  # Project configuration and dependencies
├── uv.lock         # Lock file for `uv`
└── README.md
```

## Setup

Run `uv` commands from the folder of the project (i.e. '/proj1') 

This project uses `uv` for dependency management.

1. Install `uv` (Linux/MacOS): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Or/and upgrade to make sure you use the latest version: `uv self update`
3. Install dependencies: `uv sync`
4. Activate environment: `source .venv/bin/activate` (or equivalent for your shell)

## Usage

- Run a notebook: `uv run jupyter notebook`
- Format/lint code: `uv run ruff check`
- Type check code: `uv run ty check`

## Development

- Add dependencies: `uv add <package>`
- Add dev dependencies: `uv add --dev <package>`
