# Testing

The MindIE SD test suite supports running with or without Ascend NPU hardware. CPU-compatible tests can be executed even without NPU hardware.

## Test Environment Description

Tests are divided into two categories:

| Type | Description | Requires NPU |
|------|------|:---:|
| CPU-compatible tests | Configuration parsing, utility functions, quantization parameter validation, compilation logic, etc. | No |
| NPU-dependent tests | Custom operator accuracy, Flash Attention, model tensor operations, etc. | Yes |

## Test Entry Points

### Entry Point 1: CPU-Friendly Unit Tests (Recommended for users without NPU)

`run_UT_test.sh` always runs in CPU mode, suitable for development environments without NPU hardware.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
bash tests/run_UT_test.sh
```

Default artifacts are generated in the `tests/UT/` directory, including:

- `run_UT.log`
- `final.xml`
- `coverage.xml`
- `htmlcov/`

The `tests/scripts/check_coverage.py` script in the repository is used in CI to verify the coverage gate for newly added Python files.

### Entry Point 2: Full Tests (Supports three modes)

`run_test.sh` controls the test scope via parameters and supports the following three modes:

**1. Full tests (default)**

Run all CPU-compatible tests and NPU-dependent tests:

```bash
cd tests/
bash run_test.sh --all
```

When no parameters are passed, full tests are executed by default:

```bash
cd tests/
bash run_test.sh
```

**2. Run CPU-compatible tests only (no NPU hardware required)**

```bash
cd tests/
bash run_test.sh --cpu_only
```

**3. Run NPU-dependent tests only (NPU hardware required)**

```bash
cd tests/
bash run_test.sh --npu_only
```

## LA Operator Accuracy Testing

This section describes the method for self-testing the accuracy of LA operators in the MindIE SD repository.

1. To switch the installed version, first uninstall the current MindIE SD:

   ```bash
   python -m pip uninstall mindiesd
   ```

2. Modify the `tests/plugin/la_acc_prof.py` file, select Option 1 or Option 2, and load the `test_la.csv` or `enumerated_cases.csv` file to test the accuracy of LA operators under the specified shapes.

   - `./tests/plugin/test_la.csv`: Contains input shapes for commonly used SD models.
   - `enumerated_cases.csv`: Contains various enumerated shapes.

3. After making the modifications, run the following command:

   ```bash
   cd tests
   python plugin/la_acc_prof.py
   ```

Upon successful execution, a result file is generated in the repository directory, recording the similarity between LA and FAScore. This can be used to inspect the operator's accuracy performance under the target shapes.

## Test Writing Conventions

### Test Organization Principles

Test directories under `tests/` correspond to the `mindiesd/` module hierarchy:

| Source Module | Corresponding Test Directory |
|----------|-------------|
| `mindiesd/layers/` | `tests/layers/` |
| `mindiesd/cache/` | `tests/cache/` |
| `mindiesd/compilation/` | `tests/compilation/` |
| `mindiesd/quantization/` | `tests/quantization/` |

When adding new code, check whether the corresponding test directory already has test files, and append to existing files or create new ones to cover the new functionality.

### Test Performance Conventions

- If a single test file takes more than 500 seconds to run, split it into multiple files
- Prefer reusing existing device/model loading to reduce repeated initialization overhead
- CPU-friendly tests should preferably be verified using `bash tests/run_UT_test.sh`
