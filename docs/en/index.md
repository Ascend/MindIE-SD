# MindIE SD

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:12:59.179Z pushedAt=2026-06-08T09:21:30.660Z -->

MindIE SD is a multimodal acceleration suite for Ascend. It integrates with model libraries such as Diffusers to provide Ascend affinity and fused operators, compilation acceleration, DiTCache, quantization/sparse algorithms, and multi-card parallelism. These capabilities enable fast migration and accelerated inference of multimodal generative models, making it suitable for production-grade inference workflows.

```{toctree}
:maxdepth: 2
:caption: quick start

installation
quick_start
```

```{toctree}
:maxdepth: 2
:caption: acceleration feature

architecture
features/sparse
features/quantization
features/core_layers
features/compilation
features/parallelism
features/cache
features/cpu_offload
features/share_memory
features/DyEPLB
```

```{toctree}
:maxdepth: 2
:caption: development guide

developer_guide/build_guide
developer_guide/test
<!-- developer_guide/dev_setup -->
<!-- developer_guide/repo_structure -->
<!-- developer_guide/contribution_guide -->
<!-- developer_guide/pattern_dev_guide -->
<!-- developer_guide/benchmark_and_profiling -->
```

```{toctree}
:maxdepth: 1
:caption: appendix

features/supported_matrix
```

```{toctree}
:maxdepth: 1
:caption: community

community/governance
```
