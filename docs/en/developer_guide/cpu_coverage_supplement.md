# CPU coverage supplement

This report accompanies [Issue #269](https://gitcode.com/Ascend/MindIE-SD/issues/269). It adds 26 CPU unit tests for log configuration, SoC detection, and EPLB state management.

## Scope

- `EnvVar`: component overrides, disabled logging, control characters, and rotation limits.
- `get_npu_device`: supported SoC ranges, unknown SoCs, driver errors, and cached queries.
- `TaskPayload`: type validation, required fields, and independent default dictionaries.
- `ExpertLoadCollector`: accumulation, read/reset, and task notification.
- `DynamicDispatcher`: expert maps, rank readiness, and consumption of pending weights and layouts.

Only hardware boundaries are substituted: CPU `bincount` represents NPU token counting, ordinary CPU tensors represent pinned allocation, and ready/not-ready results represent the cross-rank `all_reduce`. The production state-management code remains intact. These tests verify CPU contracts; they do not validate NPU kernels or multi-device EP communication.

## Coverage results

Both runs use the repository `.coveragerc`, with `source=mindiesd` and `branch=True`, and the same `tests/UT tests/utils tests/eplb` selection. The denominator includes the whole `mindiesd` package, including modules that are not executed.

The table shows coverage's combined statement/branch percentage: `(covered statements + covered branches) / (statements + branches)`.

| Module | Baseline | Supplement |
| --- | ---: | ---: |
| `utils/env.py` | 78.31% | 100% |
| `utils/get_platform.py` | 43.18% | 100% |
| `eplb/task_payload.py` | 75.00% | 100% |
| `eplb/collector.py` | 0% | 100% |
| `eplb/dispatcher.py` | 0% | 100% |
| Total | 16.78% | 18.26% |

Total statement coverage rises from 21.07% to 22.64%. The denominators remain 8862 statements and 2888 branches. The baseline has 217 passed and 5 skipped tests; the supplement has 243 passed and 5 skipped tests.

## Reproduce

Run from the repository root after installing the repository and `requirements-test.txt` dependencies.

```bash
export ASCEND_RT_VISIBLE_DEVICES=-1
export MINDIE_TEST_MODE=CPU
export MINDIE_LOG_TO_FILE=false
export MINDIE_LOG_TO_STDOUT=false
export COVERAGE_FILE=/path/to/task/coverage.data
python -m coverage run --rcfile .coveragerc -m pytest tests/UT tests/utils tests/eplb \
    --junitxml=/path/to/task/tests.xml
python -m coverage json --rcfile .coveragerc -o /path/to/task/coverage.json
python -m coverage report --rcfile .coveragerc
```

Save separate baseline and supplement reports. Use the same environment, command, configuration, and production source set. This CPU selection is not a complete NPU regression.

## Environment and limits

- Source baseline: `44fbd543133ea8ced34b63c7579e78776c236108`, `dev`.
- openEuler, aarch64, Python 3.11.6, torch/torch_npu 2.9.0, and CANN 9.1.0.
- The host has one 910B3; NPU visibility is disabled for these CPU tests.
- Five existing `TestEplbScheduler` tests are unconditionally skipped by `skipIf(True, ...)`. They are not counted as passed.
- One existing warning reports missing Triton-Ascend. Full NPU regression and multi-device EPLB comparisons have not run.
- An initial import with NPU visibility enabled failed because `libPTAExtensionOPS.so` was absent. That failure is retained and is not the coverage baseline.

## Static checks

Applicable pre-commit hooks for changed files, manual Markdownlint, strict Sphinx builds for both languages, and diff checks passed. Native operator hooks have no applicable files and are not counted as passed.

Full-repository pre-commit exits with code 1 for both the baseline and supplement in isolated copies. Both have the same 11 failed hook categories: trailing whitespace, EOF fixer, ruff check, ruff format, codespell, pylint, bandit, typos, profiling-collect wiring selftest, clang-format, and clang-tidy. The clang-tidy executable is missing. This comparison checks failed hook sets, not equality of every diagnostic.

Automatic changes from full-repository checks remain in the check copies. The delivery source retains the focused test and documentation changes.
