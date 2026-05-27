# Developer Tooling and Docs

This document describes the common local commands for documentation generation, development image builds, and pre-commit checks in MindIE SD.

## Documentation Build

Documentation build dependencies are listed in `docs/requirements-docs.txt`. To generate the local HTML site:

```bash
python -m pip install -r docs/requirements-docs.txt
sphinx-build -b html docs docs/_build/html
```

The generated site is written to `docs/_build/html/`, and the homepage file is `docs/_build/html/index.html`.

## Development Image Build

The repository provides a 910B aarch64 development image definition in `docker/Dockerfile_910b_aarch64.ubuntu`. Build it locally with:

```bash
docker build --network=host -f docker/Dockerfile_910b_aarch64.ubuntu -t mindiesd:910b-aarch64-head .
```

## Lint and Pre-Commit Checks

Lint-related dependencies are listed in `requirements-lint.txt`. Before the first local commit, install and enable `pre-commit`:

```bash
python -m pip install -r requirements-lint.txt
pre-commit install
pre-commit run --all-files
```

`pre-commit install` writes the repository hook to `.git/hooks/pre-commit`. After that, later `git commit` commands automatically run the configured checks.

To run Markdown checks explicitly:

```bash
pre-commit run markdownlint --all-files --hook-stage manual
```

Use `git commit --no-verify` only when you intentionally need to bypass the hook.
