# Testing

The MindIE SD test suite supports running with or without Ascend NPU hardware. CPU-compatible tests can be executed even without NPU hardware.

## Test Environment Description

Tests are strictly split by whether they need a device, and live in two directories:

| Type | Description | Requires NPU | Directory |
|------|------|:---:|------|
| CPU-compatible tests (UT) | Configuration parsing, utility functions, quantization parameter validation, compilation logic, etc. | No | `tests/UT/` |
| NPU-dependent tests (operator cases) | Custom operator accuracy, Flash Attention, model tensor operations, etc. | Yes | `tests/ops/` |

`tests/conftest.py` enforces this boundary: cases under `tests/UT/` must not touch the NPU, and cases under `tests/ops/` are skipped when no NPU is present. Only cases that genuinely need the device belong in `tests/ops/`; everything else goes to `tests/UT/` and is executed by CI when a pull request is submitted.

## Unified Test Entry Point

Every test run enters through the single script `tests/scripts/run_suites.py`; only the command line differs. The script launches pytest in separate subprocesses per mode (`CPU` / `NPU`) and then merges the coverage results.

```bash
python -m pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
python -m pip install -r requirements-test.txt
```

### Common Commands

| Scenario | Command |
|------|------|
| Full run (CPU + NPU) | `python tests/scripts/run_suites.py` |
| CPU-compatible tests only (no NPU) | `python tests/scripts/run_suites.py --cpu_only` |
| NPU-dependent tests only (NPU required) | `python tests/scripts/run_suites.py --npu_only` |
| Specific files or directories | `python tests/scripts/run_suites.py tests/UT/layers` |
| Run only what the current changes affect | `python tests/scripts/run_suites.py --incremental` |
| Preview the incremental selection first | `python tests/scripts/run_suites.py --incremental --dry-run` |

### Options

| Option | Description |
|------|------|
| no mode flag | Runs the CPU and NPU suites in turn |
| `--cpu_only` | Runs only `tests/UT/` (CPU mode) |
| `--npu_only` | Runs only `tests/ops/` (NPU mode); verifies `torch.npu.is_available()` first |
| `--all` | Explicitly requests a full run; same as passing no mode flag |
| `test_files ...` | Positional: files or directories under `tests/` |
| `--incremental` | Picks the affected tests from the git changes and the dependency graph |
| `--base-branch` | Baseline branch for incremental analysis, default `dev` |
| `--no-staged` / `--no-unstaged` | Ignore staged / unstaged changes in the incremental analysis |
| `--dry-run` | Prints the incremental selection without running it (requires `--incremental`) |

### Artifacts and Coverage Gate

Artifacts go to the directory named by `MINDIE_TEST_RESULTS`, or to `mindiesd-test-results/` under the system temp directory when it is unset. Each run creates a `run-*` subdirectory containing:

- `cpu.xml` / `npu.xml`: JUnit reports
- `combined.coverage`, `coverage.json`: merged coverage data
- `file-coverage.json`: per-file coverage results

On a full CPU run (no explicit test selection), `tests/scripts/check_file_coverage.py` verifies that every Python file under `mindiesd/` reaches at least 80% branch coverage and returns a non-zero exit code otherwise.

### CI-Only Entry Point

One shell wrapper remains, `tests/run_UT_test.sh`, and it serves the CI CPU pipeline only: it picks a Python interpreter and calls `run_suites.py` with `--cpu_only tests/UT`. The pipeline deletes that script's final line and then judges the result with `grep OK`, so do not use it for local runs — call `run_suites.py` directly.

```bash
bash tests/run_UT_test.sh   # CI only; same as run_suites.py --cpu_only tests/UT
```

## LA Operator Accuracy Testing

This section describes the method for self-testing the accuracy of LA operators in the MindIE SD repository.

1. To switch the installed version, first uninstall the current MindIE SD:

   ```bash
   python -m pip uninstall mindiesd
   ```

2. Modify the `tests/ops/plugin/la_acc_prof.py` file, select Option 1 or Option 2, and load the `test_la.csv` or `enumerated_cases.csv` file to test the accuracy of LA operators under the specified shapes.

   - `tests/ops/plugin/test_la.csv`: Contains input shapes for commonly used SD models.
   - `enumerated_cases.csv`: Contains various enumerated shapes.

3. After making the modifications, run the following command from the repository root:

   ```bash
   python tests/ops/plugin/la_acc_prof.py
   ```

Upon successful execution, a result file is generated in the repository directory, recording the similarity between LA and FAScore. This can be used to inspect the operator's accuracy performance under the target shapes.

## Test Writing Conventions

### Test Organization Principles

Tests are stored in separate directories according to whether they need a device:

| Source Module | CPU Test Directory | NPU Test Directory |
|----------|-------------|-------------|
| `mindiesd/layers/` | `tests/UT/layers/` | `tests/ops/layers/` |
| `mindiesd/cache/` | `tests/UT/cache/` | — |
| `mindiesd/compilation/` | `tests/UT/compilation/` | — |
| `mindiesd/quantization/` | `tests/UT/quantization/` | `tests/ops/quantization/` |

When adding new code, check whether the corresponding test directory already has test files, and append to existing files or create new ones to cover the new functionality. There is a single criterion: a case belongs in `tests/ops/` only when it actually calls NPU operators or device APIs; otherwise it belongs in `tests/UT/`. Operator cases must also match the device model — a case that only holds on a specific device is marked for that device, and must not be reused to make another model pass.

### Test Performance Conventions

- If a single test file takes more than 500 seconds to run, split it into multiple files
- Prefer reusing existing device/model loading to reduce repeated initialization overhead
- CPU-friendly tests can be verified before submitting with `python tests/scripts/run_suites.py --cpu_only`
