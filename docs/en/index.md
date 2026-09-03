# MindIE SD

MindIE SD is an Ascend-focused multimodal acceleration suite that works with diffusers and other model suites to provide Ascend-optimized key operators and fused operators, compilation acceleration, compute-via-storage, quantization/sparse algorithms, and multi-card parallelism capabilities, enabling fast migration of multimodal generation models to Ascend for acceleration, suitable for production-grade inference workflows.

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quick_start
```

```{toctree}
:maxdepth: 2
:caption: Acceleration Features

architecture
features/sparse
features/quantization
features/core_layers
features/fused_moe
features/compilation
features/parallelism
features/usp
features/fa_power_cap
features/cache
features/cpu_offload
features/share_memory
features/DyEPLB
```

```{toctree}
:maxdepth: 2
:caption: Developer Guide

developer_guide/build_guide
developer_guide/test
developer_guide/dev_setup
developer_guide/repo_structure
developer_guide/pattern_dev_guide
developer_guide/benchmark_and_profiling
```

```{toctree}
:maxdepth: 1
:caption: Appendix

features/supported_matrix
```
