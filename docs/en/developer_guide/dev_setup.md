# Development Environment Setup

This document describes common commands for documentation generation, development image building, and pre-commit checks used in local MindIE SD development.

## Documentation Generation

Documentation build dependencies are listed in `docs/requirements-docs.txt`. Use the following commands to generate HTML documentation locally:

```bash
python -m pip install -r docs/requirements-docs.txt
# Build both Chinese and English
python docs/build_docs.py
# Build Chinese only
SPHINX_LANGUAGE=zh sphinx-build -b html -c docs docs/zh docs/_build/zh/html
# Build English only
SPHINX_LANGUAGE=en sphinx-build -b html -c docs docs/en docs/_build/en/html
```

Local preview method:

```bash
python -m http.server 8080 --directory docs/_build
```

<http://localhost:8080> → auto-redirects to the Chinese version

## Development Image Build

The repository provides a development image definition file `docker/Dockerfile_910b_aarch64.ubuntu` based on the Atlas 800I A2 inference server AArch64 environment. To build the image locally:

```bash
docker build --network=host -f docker/Dockerfile_910b_aarch64.ubuntu -t mindiesd:910b-aarch64-head .
```

## Lint and Pre-Commit Checks

Lint-related dependencies are listed in `requirements-lint.txt`. Before starting local development for the first time, it is recommended to install and enable `pre-commit`:

```bash
python -m pip install -r requirements-lint.txt
pre-commit install
pre-commit run --all-files
```

`pre-commit install` writes the repository hooks to `.git/hooks/pre-commit`. After installation, subsequent `git commit` commands will automatically run the checks enabled by default in the current repository.

To explicitly run Markdown document checks, additionally execute:

```bash
pre-commit run markdownlint --all-files --hook-stage manual
```

Use `git commit --no-verify` only when you explicitly need to bypass checks.
