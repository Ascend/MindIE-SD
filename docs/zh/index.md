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

## 技术报告

<ul>
  <li><a href="../tech_report/RotateAttention.pdf" target="_blank">RotateAttention 论文</a></li>
  <li><a href="../tech_report/RotateAttention Poster.pdf" target="_blank">RotateAttention ECCV 2026 海报</a></li>
  <li><a href="../tech_report/RainFusion2.0.pdf" target="_blank">RainFusion2.0 技术报告</a></li>
</ul>

```{toctree}
:maxdepth: 2
:caption: 开发者指南

developer_guide/build_guide
developer_guide/test
developer_guide/dev_setup
developer_guide/repo_structure
developer_guide/pattern_dev_guide
developer_guide/benchmark_and_profiling
```

```{toctree}
:maxdepth: 2
:caption: 附录

features/supported_matrix
appendix/log
appendix/error_code
```
