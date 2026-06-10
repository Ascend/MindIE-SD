# Test Guide

The MindIE SD test suite supports running with or without Ascend NPU hardware. CPU-compatible tests can be executed even when no NPU hardware is available.

## Test environment overview

Tests fall into two categories:

| Type | Description | NPU required |
|------|-------------|:---:|
| CPU-compatible | Configuration parsing, utility functions, quantization parameter validation, compilation logic, etc. | No |
| NPU-dependent | Custom operator accuracy, Flash Attention, tensor operations on device, etc. | Yes |

## Test entry points

### Option 1: CPU-friendly unit tests (recommended for users without NPU)

`run_UT_test.sh` always runs in CPU mode, making it suitable for development environments without NPU hardware.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt
bash tests/run_UT_test.sh
```

Artifacts are generated under `tests/UT/`, including:

- `run_UT.log`
- `final.xml`
- `coverage.xml`
- `htmlcov/`

The repository also provides `tests/scripts/check_coverage.py` for CI coverage gating on newly added Python files.

### Option 2: Full tests (three modes)

`run_test.sh` accepts a flag to control the test scope. Three modes are available:

**1. All tests (default)**

Run both CPU-compatible and NPU-dependent tests:

```bash
cd tests/
bash run_test.sh --all
```

When no flag is given, the default is to run all tests:

```bash
cd tests/
bash run_test.sh
```

**2. CPU-compatible tests only (no NPU hardware required)**

```bash
cd tests/
bash run_test.sh --cpu_only
```

**3. NPU-dependent tests only (requires NPU hardware)**

```bash
cd tests/
bash run_test.sh --npu_only
```

## LA Operator Accuracy Test

This section describes how to run LA operator accuracy verification in the MindIE SD repository.

1. If needed, uninstall the currently installed MindIE SD package first:

   ```bash
   pip uninstall mindiesd
   ```

2. Update `tests/plugin/la_acc_prof.py`, choose Option 1 or Option 2, and load either `test_la.csv` or `enumerated_cases.csv` to verify LA accuracy under the required shapes.

   - `./tests/plugin/test_la.csv`: common input shapes used by SD models
   - `enumerated_cases.csv`: enumerated shape combinations

3. Run the script:

   ```bash
   cd tests
   python plugin/la_acc_prof.py
   ```

After the run, result files are generated in the repository root and can be used to inspect similarity between LA and FAScore outputs.

## Common Exceptions

When using MindIE SD for inference, users are responsible for the safety of model files such as weights, configuration files, and model code. Common exceptions include:

- If default model configuration values are changed during initialization, interfaces may be affected; excessively large weights or configuration values may trigger out-of-memory errors such as `RuntimeError: NPU out of memory. Tried to allocate xxx GiB.`.
- Large tensor shapes during inference may also trigger similar out-of-memory errors.
- Invalid input or environment mismatch may raise exceptions that should be handled by upper-layer services.

| Exception Type | Description |
| -- | -- |
| ZeroDivisionError | Division by zero. |
| ValueError | Invalid parameter value. |
