# MindIE SD

MindIE SD is an Ascend-focused inference acceleration toolkit for Stable Diffusion and related multimodal generation scenarios.

## What it provides

- Ascend-friendly kernels and fused operators for diffusion workloads
- Quantization and sparsity features tailored for Ascend NPUs
- Cache-based acceleration strategies for DiT and attention workloads
- Parallel execution strategies for multi-card deployments
- Compile-time graph rewriting to replace generic operators with Ascend-oriented implementations

## Getting started

For environment preparation and end-to-end inference walkthroughs, use the English guides below:

- [Quick Start](quick_start.md)
- [Installation Guide](installing_guide.md)

## Documentation map

- [Quick Start](quick_start.md)
- [Installation Guide](installing_guide.md)
- [User Guide](menu_user_manual.md)
- [Architecture](architecture.md)
- [Environment Variable Guide](environment_variable_configuration.md)
- English feature docs:
  - [Supported matrix](features/supported_matrix.md)
  - [Cache](features/cache.md)
  - [Compilation](features/compilation.md)
  - [Dynamic EPLB](features/DyEPLB.md)
  - [Graphics memory optimization](features/graphics_memory_optimization.md)
  - [Compute optimization](features/others.md)
  - [Parallelism](features/parallelism.md)
  - [Sparse quantization](features/sparse_quantization.md)
- Appendix:
  - [Environment variables](appendix/environment_variable.md)
  - [File and directory permissions](appendix/file_directory_permissions_description.md)

## Repository resources

- [Project repository](https://gitcode.com/Ascend/MindIE-SD)
- [Contribution guide](../../contributing.md)
- [Governance](community/governance.md)
