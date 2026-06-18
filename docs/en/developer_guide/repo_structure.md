# Repository Structure

## Top-Level Directory

| Directory | Purpose |
| ------ | ------ |
| `mindiesd/` | Core Python package with layer operators, compilation, cache, parallelism, and quantization modules |
| `csrc/` | C/C++ source code (AscendC / triton operator implementations) |
| `tests/` | Test suite organized to mirror the mindiesd/ hierarchy |
| `examples/` | Model inference examples (dummy_run, cache, service, wan) |
| `docs/` | Sphinx documentation source (bilingual: Chinese and English) |
| `docker/` | Development container and image definitions |
| `build/` | Build artifacts |
| `benchmarks/` | Performance benchmark data |
| `pre-commit/` | Supplementary pre-commit config (pyproject.toml, typos.toml) |

## mindiesd/ Key Modules

| Module | Description |
| ------ | ------ |
| `layers/` | External acceleration APIs (attn, moe, quant, and other layer implementations) |
| `kernel/` | Ascend high-performance kernel wrappers |
| `cache/` | Compute-via-storage (Attention Cache, DiT Cache) |
| `parallelism/` | Multi-card parallelism (CFG, USP, EPLB) |
| `quantization/` | Automatic quantization enablement |
| `compilation/` | torch.compile custom fusion passes |

## tests/ Test Architecture

| Subdirectory | Test Target |
| -------- | ---------- |
| `tests/cache/` | Cache module |
| `tests/compilation/` | Compilation module (including pattern tests) |
| `tests/layers/` | Layer module (including flash_attn tests) |
| `tests/plugin/` | Custom plugin operator accuracy |
| `tests/quantization/` | Quantization module |
| `tests/eplb/` | Dynamic expert load balancing |
| `tests/UT/` | CPU-friendly unit tests |
| `tests/scripts/` | Test helper scripts (coverage checking, etc.) |

Test entry points are documented in [test.md](test.md).

## docs/ Documentation Architecture

| Path | Description |
| ------ | ------ |
| `docs/zh/` | Chinese documentation |
| `docs/en/` | English documentation |
| `docs/zh/developer_guide/` | Developer guide (this document's directory) |
| `docs/zh/features/` | Acceleration feature descriptions |
| `docs/zh/community/` | Community governance |

## Key Configuration Files

| File | Description |
| ------ | ------ |
| `setup.py` / `pyproject.toml` | Python package build and distribution |
| `version.py` | Version number definition |
| `requirements.txt` | Core dependencies |
| `requirements-test.txt` | Test dependencies |
| `requirements-lint.txt` | Lint dependencies |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |
| `.clang-format` | C++ code formatting |
| `.coveragerc` | Coverage configuration |
| `.readthedocs.yaml` | Read the Docs build |
