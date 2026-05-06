# MindIE SD

MindIE SD 是面向昇腾的多模态加速系列套件，配合 diffusers 等模型套件提供昇腾亲和的关键算子和融合算子、编译加速、以存代算、量化/稀疏算法及多卡并行能力，实现对多模态生成模型的快速迁移和昇腾加速，适用于生产级推理工作流。

```{toctree}
:maxdepth: 2
:caption: 快速开始

installation
quick_start
```

```{toctree}
:maxdepth: 2
:caption: 加速特性

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
:caption: 开发者指南

developer_guide/build_guide
developer_guide/test
developer_guide/tooling
```

```{toctree}
:maxdepth: 1
:caption: 附录

features/supported_matrix
```

```{toctree}
:maxdepth: 1
:caption: 社区

community/governance
```
