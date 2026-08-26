# 算子优化技能地图

> 目的：算子开发/优化任务按场景路由到对应 skill，**优先引用外部技能库
> `cannbot-skills`（线上 <https://gitcode.com/cann/cannbot-skills>；本地使用时
> 自行 `git clone` 到任意目录，勿依赖固定本机路径），本文件不重复其内容**，
> 只列场景 → skill 映射与 MindIE-SD 特有补充。
>
> 外部技能库不可用（未克隆/无网络）时：跳过外部引用，仅使用 §2 本仓库补充经验。
>
> 适用：新增/优化算子（triton / Ascend C / Catlass / PyPTO / TileLang）、
> 算子性能/精度问题、图编译与模型级推理优化。

---

## 1. 场景路由总表

### 1.1 Triton 算子（本仓库 fusion pattern 的 replacement kernel 用这条链）

| 场景 | 引用 skill（cannbot-skills/ops/） | 说明（不展开内容） |
| --- | --- | --- |
| 从 PyTorch 代码提取算子任务 | `triton-task-extractor` | 构建标准化任务文件（单/多 case） |
| 算法草图设计 | `triton-op-designer` | UnifiedSketch DSL，指导代码生成 |
| 代码生成/迭代修复 | `triton-op-coding` | 生成含 ModelNew 的内核代码 |
| **性能优化（核心）** | `triton-latency-optimizer` | 31 个优化点严格顺序扫描、单点验证；`references/Index.md` 索引 |
| 瓶颈不明/宣称硬件极限前 | `triton-simulator-optimizer` | msprof op simulator 流水诊断，只采集+诊断 |
| 精度对齐（MERE/MARE 失败） | `triton-precision-debug` | 五阶隔离法定位 ULP 级差异 |
| 功能/性能验证 | `triton-op-verifier` | verify.py + benchmark.py |
| 算子特定经验 | `references/operators/adain.md`、`swiglu-quant.md`、`permute-layout-transform.md`、`general-insights.md`、`dimension-merge-large-block.md` | AdaIN/SwiGLU 量化/布局变换等已沉淀案例 |

> 本次 AdaLN/SwiGLU/gate 三算子直接命中：`avoid_scalar_lowering.md`
> （i64 向量算术降级）、`vector_core_partition.md`（多行并行/grid 匹配）、
> `discrete_memory_access.md`（标量 gather 不触发离散访存）、
> `multibuffer-and-double-buffering.md`（实测本平台无收益）、
> `operators/swiglu-quant.md`（launch 缓存/多行/host 特化）。

### 1.2 其他算子 DSL（选型：需要非 triton 实现时）

| 场景 | 引用 skill | 说明 |
| --- | --- | --- |
| Ascend C 算子 | `ascendc-tiling-design` / `ascendc-perf-optimize` / `ascendc-performance-best-practices` / `ascendc-precision-debug` / `ascendc-runtime-debug` / `ascendc-crash-debug` / `ascendc-env-check` / `ascendc-code-review` | tiling→优化→精度→运行时→崩溃→环境→检视全链 |
| Catlass | `catlass-op-design` / `catlass-op-develop` / `catlass-op-perf-tune` | 组件选型→生成→TileShape/DispatchPolicy 调优 |
| PyPTO | `pypto-op-design` / `pypto-op-develop` / `pypto-op-perf-tune` / `pypto-precision-debug` 等 | 迭代式设计→编码→自动调优 |
| TileLang | `tilelang-op-design` / `tilelang-op-develop` / `tilelang-perf-optimization` | design.md → 实现 → 性能劣化模式检查 |
| 架构知识 | `npu-arch` | 芯片型号/代际/archXX 特性 |
| 直调转自定义算子 | `ascendc-direct-invoke-to-registry-invoke`、`ascendc-registry-invoke-template` | `<<<>>>` ↔ registry 工程 |
| torch.ops 对接 | `torch-ascendc-op-extension` | TORCH_LIBRARY 接入 PyTorch |

### 1.3 性能采集与工具

| 场景 | 引用 skill | 说明 |
| --- | --- | --- |
| 算子级 profiling | `ops-profiling` | msprof 算子级瓶颈 + kernel-level 对比 |
| 自定义算子 vs 标杆 | `torch-ops-profiler` | torch_npu.profiler JSONL 用例 + 性能报告 |
| 无 NPU 仿真 | `ops-simulator` | 精度/性能仿真、流水分析 |
| 故障/日志工具 | `tools/asys-toolkit` / `tools/msaicerr-toolkit` / `tools/msnpureport-toolkit` | asys 收集 / AI Core Error 分析 / Device 日志导出 |

### 1.4 模型级推理优化（cannbot model/、graph/）

| 场景 | 引用 skill | 说明 |
| --- | --- | --- |
| 融合算子替换 | `model-infer-fusion` | 识别可替换 torch_npu 融合算子的模式 |
| torch.compile 图模式 | `model-infer-graph-mode` | npugraph_ex 图模式适配 |
| 性能分解 | `model-infer-perf-breakdown` / `model-infer-profiling` | 端到端耗时分解、profiling |
| 权重预取 | `model-infer-prefetch` | npu_prefetch 缓解 matmul 访存等待 |
| SuperKernel | `model-infer-superkernel` | 算子二进制融合 |
| 多流 | `model-infer-multi-stream` | 模块/算子 DAG 并行 |
| custom_op 入图 | `graph/torch-custom-ops-guide` | torch.library.custom_op 全流程 |
| npugraph_ex 诊断 | `graph/torch-npugraph-ex-*`（compile-error / dfx-triage / performance / runtime-error / knowledge / template） | 编译期/运行期/性能诊断 |
| GE 融合 pass | `graph/ge-fusion-pass-skill` | GE PatternFusionPass 开发/验证 |

---

## 2. 本仓库补充（cannbot 未覆盖的 MindIE-SD 特有经验）

以下为本仓库已沉淀、cannbot 无对应内容的部分，**只列入口不重复正文**：

| 场景 | 本仓库入口 |
| --- | --- |
| **register_replacement pattern 命中≠收益**：kernel diff → 逐 pass AB → R1-R5 根因目录（含"负收益先查 kernel 形态"教训） | `compilation-dev/references/benefit-rootcause.md` |
| **kernel diff 方法论**：kernel_details.csv 聚合对比、L2-flush bench 必须放计时区外、warm/cold 双档测量 | `compilation-dev/references/benefit-rootcause.md` §2 + `dev-workflow/references/rework-lessons.md` #26 |
| **模型级验证闭环**（不只看单测）：compute-precision 图验证、叠加 AB、远端 NPU 部署流程 | `dev-workflow/references/rework-lessons.md`、`dummy-run-dev/` |
| MiniMax-H3 算子上下文（npu_swiglu 语义、表 [3,D] L2 驻留、真实图形态） | `dummy-run-dev/references/minimax-h3-notes.md` §10 |
| CP/多卡通信掩盖（HcclAlltoAllV 缺陷、pad+等分绕过） | `parallelism-strategy/` |
| MindIE-SD 编译后端（default/Inductor 与 aclgraph 批量下发）与 Copy 消减 | `compilation-dev/`（copy-elimination-guide）+ `aclgraph-dev/` |

---

## 3. 本次算子优化（2026-08）验证过的组合路径

```text
模型 kernel diff 发现瓶颈（本仓库方法）
  → triton-latency-optimizer: i32 索引（avoid_scalar_lowering）
    → vector_core_partition: 3 行/program（UB 上限内）
    → 形态级优化：gather 融合（表小 L2 驻留时吸收 index_select）
    → swiglu 免 cat（cat 大张量是隐藏成本）
  → 每步 bench（warm/cold 双档）+ 模型级 AB + kernel diff 确认
结果：AdaLN +0.64ms → -0.45ms；SwiGLU -0.35 → 免 cat 后总 -1.86ms；
     gate 融合新增 -0.4ms；总 25.96 → 24.10ms（-7.2%）
```

> 经验要点（详见 `compilation-dev/references/benefit-rootcause.md` §3 R1-R5）：负收益先查 kernel 形态
> （流量冗余/融合边界/i64 降级/并行度），不要归咎 triton 本身；隔离
> bench 热缓存值会高估，必须以模型内 profile device 时间为准。

## 维护与更新

当外部 cannbot-skills 结构或本仓补充经验变化时，按 dev-workflow 的复盘流程更新本文件。
