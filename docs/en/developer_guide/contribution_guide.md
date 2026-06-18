# Contribution Guide

## Prerequisites

1. Fork the repository to your personal account and clone it locally:

   ```bash
   git clone https://github.com/<your-username>/MindIE-SD.git
   ```

2. Complete the source build following [build_guide.md](build_guide.md).

3. Install lint dependencies and pre-commit hooks:

   ```bash
   pip install -r requirements-lint.txt
   pre-commit install
   ```

   For detailed pre-commit instructions, see the "Lint and Pre-Commit Checks" section in [dev_setup.md](dev_setup.md).

## Coding Standards

### Ruff Configuration

Ruff configuration comes from the `[tool.ruff]` section of `pyproject.toml`:

| Config Item | Value | Description |
| -------- | ----- | ------ |
| Line length | 100 | Maximum characters per line |
| Python version | py310 | Target version |
| Docstring formatting | true | Format code blocks in docstrings |

Enabled Lint rule groups:

| Rule Group | Meaning |
| -------- | ------ |
| E | pycodestyle errors (blank lines, indentation, etc.) |
| F | Pyflakes (unused imports, undefined names, etc.) |
| I | isort import ordering |
| N | pep8-naming naming conventions |
| W | pycodestyle warnings |
| UP | pyupgrade syntax modernization |
| B | flake8-bugbear common errors |
| C4 | flake8-comprehensions comprehension optimization |
| SIM | flake8-simplify simplification suggestions |
| G | flake8-logging-format logging format |

### Pre-commit Hooks

Hooks installed by `pre-commit install` run automatically on every `git commit`:

| Hook | Behavior |
| ------ | ------ |
| `ruff-check` | Auto-fix Python lint issues |
| `ruff-format` | Auto-format Python code |
| `pylint` | Python code quality check |
| `bandit` | Python security vulnerability scan |
| `codespell` | Comment/docstring spell check (skips code files) |
| `typos` | Code identifier spell check |
| `clang-format` | C++ code formatting |
| `markdownlint` | Markdown format check (manual mode) |
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Add newline at end of file |
| `check-yaml` | YAML format check |
| `check-added-large-files` | Block files larger than 50MB |
| `check-merge-conflict` | Detect unresolved merge conflicts |
| `detect-private-key` | Detect hardcoded keys |

### Python Coding Conventions

**File header template**: All `.py` files must include the Mulan PSL v2 license header.

**Import order** (enforced by Ruff I rule):

1. Standard library (e.g., `contextlib`, `dataclasses`, `typing`)
2. Third-party libraries (e.g., `torch`)
3. Local modules (e.g., `from .compiliation_config import ...`)

Sorted alphabetically within each group.

**Blank lines**: 2 blank lines before module-level function/class definitions, 1 blank line before methods within a class.

**Quotes**: Use double quotes consistently.

**Type annotations**: Targeting Python 3.10, use `X | Y` union syntax (PEP 604). Function signatures should include parameter and return type annotations where possible.

**Logging**: Use the standard `logging` module. Avoid `.format()` or f-strings in log strings; prefer `%`-style arguments:

```python
logger = logging.getLogger(__name__)
logger.debug("...")
logger.warning("...")
```

### C/C++

Use **clang-format** for code formatting. Configuration is in `.clang-format`.

### Markdown

Use **markdownlint** for document format checking. See [dev_setup.md](dev_setup.md) for manual execution.

Core rules:

| Rule | Description |
| ------ | ------ |
| MD040 | Fenced code blocks must specify a language tag |
| MD009 | No trailing whitespace |
| MD012 | No consecutive blank lines |
| MD031 | Blank lines before and after fenced code blocks |

Code block language selection:

| Content Type | Language Tag |
| ---------- | ---------- |
| Python code | `python` |
| Shell commands | `bash` |
| Directory/file tree | `text` |
| Terminal output/logs | `text` |

### Spell and Name Checking

**codespell** + **typos** perform spell checking on code and documentation.

## Pre-Commit Checks

Ensure all pre-commit checks pass before committing:

```bash
pre-commit run --all-files
```

To temporarily bypass (emergency only):

```bash
git commit --no-verify
```

## Running Tests

### CPU-Friendly Unit Tests

```bash
bash tests/run_UT_test.sh
```

### Full NPU Tests

```bash
bash tests/run_test.sh --all
```

Optional parameters: `--cpu_only`, `--npu_only`.

For detailed test instructions, see [test.md](test.md).

## Branch and Commit Conventions

- Branch naming: `feat/<feature-name>`, `fix/<issue>`, `docs/<topic>`, `refactor/<scope>`
- Commit message: Concise first line (max 72 characters), optional detailed description paragraph

## Pull Request Process

1. Create a new branch from `main`.
2. Complete your changes and ensure pre-commit + tests pass.
3. Submit a PR, link related Issues (if any), and describe the changes and motivation.
4. Reviewers provide technical review feedback.
5. Approvers confirm that the changes are ready for merging.
6. Branch keepers or authorized maintainers merge the changes into protected branches.

For role definitions and decision-making processes, see [../community/governance.md](../community/governance.md).

## Documentation Sync Requirements

- User-visible changes must update corresponding documents in both `docs/zh/` and `docs/en/`.
- Significant changes must update the version notes in `release_note.md`.
- Major API or behavioral changes should submit an RFC in advance.

## Getting Started

Newcomers can start by browsing all open Issues in the repository and selecting tasks matching their skill level. Read [repo_structure.md](repo_structure.md) to understand the code layout, which helps with quick onboarding.

## Coding Best Practices

- Avoid duplicate code; extract repeated segments exceeding 5 lines into shared functions.
- Minimize NPU-CPU synchronization operations (e.g., `.item()`, `.cpu()`); prefer vectorized approaches.
- Runtime checks during model forward passes should be cached as member variables to avoid per-layer recomputation.
- Prefer pure functions; avoid modifying parameters in-place.
- Split files exceeding 2000 lines.
- Place utility functions at the bottom of files and core data structures at the top.
- Never use `pickle.loads()` / `pickle.load()` to deserialize untrusted data.
- When adding features or adapting to new hardware, avoid modifying existing code when possible; prefer creating new files and independent modules.
- In multi-branch conditionals, ensure the default/common path is listed first.
