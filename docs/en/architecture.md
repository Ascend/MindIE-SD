# Architecture

## Design goals

MindIE SD focuses on accelerating multimodal generation workloads on Ascend hardware, with an emphasis on diffusion-based models and related operator-heavy pipelines.

The project is designed so that acceleration features can be used independently or stacked together. This allows users to combine cache, quantization, parallelism, and compilation features based on their model and deployment requirements.

## Core capabilities

- **Layer interfaces**: expose Ascend-optimized attention, normalization, quantization, and related primitives to Python users.
- **Kernel implementations**: provide Ascend-oriented custom operators and fused kernels for multimodal generation.
- **Compilation passes**: use FX graph rewriting and `torch.compile` integration to replace compatible operators automatically.
- **Quantization**: add Ascend-oriented quantization and sparse quantization entrypoints.
- **Cache acceleration**: support multiple cache granularities including DiT block and attention cache paths.
- **Parallel execution**: provide multi-card execution strategies such as CFG and sequence parallel variants.

## High-level structure

MindIE SD consists of:

- `mindiesd/`: Python package entrypoints and user-facing modules
- `csrc/`: custom operators and kernel sources
- `examples/`: model and service examples
- `tests/`: unit and functional validation
- `docs/`: project documentation

The Python package integrates with external ecosystems such as diffusers and related model suites, while the lower-level kernel and compilation layers provide Ascend-specific acceleration beneath those APIs.
