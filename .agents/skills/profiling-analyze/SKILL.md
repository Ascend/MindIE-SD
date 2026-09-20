---
name: profiling-analyze
compatibility: Python 3.10+（脚本仅用标准库）；输入为 profiling 数据（kernel_details.csv / trace_view.json）
description: 对 profiling 数据（kernel_details.csv / trace_view.json / step_trace_time.csv）做统一分析：
             5 层递进管道（analyze_trace.py）+ baseline/优化后 kernel diff（compare_traces.py），输出
             瓶颈诊断、算子执行序、可融合候选与 P0-P2 方向建议。输入统一来自 profiling-collect。
             只要用户有 profiling
             产出并问"为什么慢/瓶颈在哪/优化前后差多少/往哪个方向"，都用本技能；采集走
             profiling-collect，方案选档走 dit-perf-opt，实现/并行/基准走
             pattern-dev/dit-parallel-opt/benchmark-dev——本技能只做诊断与方向。
             由 model-auto-optimization 的 S1（融合分析）/S3（并行）阶段调用，亦由 dev-workflow 的分析阶段指引加载。
             融合机会候选只做**识别**；候选的**边界与收益判定**（能不能融成一个单元、融多大、值不值）交
             `fusion-scope-analyze`（本技能的辅助技能）——本技能不判融合边界。
---

# 性能分析

基于昇腾 NPU profiling 数据，实现 5 层递进分析管道。

## 数据源

Profiling 数据由 profiling-collect skill 在远端 NPU 环境采集产出（已剔除 warmup）。
也可来自 dummy-run 的粗粒度时序。

| 数据文件 | 格式 | 说明 |
|---------|------|------|
| `kernel_details.csv` | CANN Profiler CSV | 每行一个 NPU 算子：Name, Start Time(us), Duration(us), Wait Time(us) |
| `trace_view.json` | Chrome Trace JSON | Host 端 + Device 端事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总：Computing, Communication, Free, Bubble |
| `communication.json` | JSON | 通信算子详情（若开启） |
| 单元利用率档（`op_summary_*.csv`，PipeUtilization） | CANN Profiler CSV | `*_vec_ratio` / `*_mac_ratio` / `*_mte2_ratio` / `*_mte3_ratio` / `cube_utilization(%)`；**融合判型（`fusion-scope-analyze`）的必需输入是这四族 ratio**，缺列或全 `N/A` 即判该次采集不合格（口径与门禁单点见 `profiling-collect/scripts/check_output.py`）。`memory_bound` 是**可算字段**（`mte2_ratio / max(mac_ratio, vec_ratio)`），真实导出常不含该列，**不列为必需列**、按公式现算 |

> **同源提醒（不是两份独立证据）**：`kernel_details.csv` 与 `op_summary*.csv` 是**同一批 task 行的两种表头** ——
> 前者由 `generate_view()` 从 `OP_SUMMARY` 生成（库内 `_kernel_view_parser.py`）⇒ **两文件行数相同、时间列同单位**。
> **不得**把它们当成两份证据去"交叉验证"（那只会在同一份数据上自证 ✗）。
> 另：**"导出报成功但没有新文件"**是本环境的已知陷阱（报成功的那一层把异常吞了），
> 断言与绕过写法见 `../profiling-collect/SKILL.md`「导出类操作一律『无新文件即失败』」。

## 读表口径与可复现性（三条实测，引用数据前必读）

**① 「每层 / 每步耗时」必须声明口径：跨流 ΣDuration 还是纯计算忙碌**
`kernel_details` 的 `Duration` 是**逐 task** 的 ⇒ 把**所有流**的 Duration 相加时，
**被重叠的通信流会被算第二遍** ✗。本仓一条实测：ΣDuration 口径与纯计算忙碌口径
**不相等**（差额 = 被重叠的通信量级；两者绝对值与占比出库 `{run_results_dir}/archive/`）。
⇒ 报数时**写明口径**，并用 `step_trace_time` 的 `Computing / Communication / Overlapped / Free`
四元组把两个口径**对上**（占比重叠读法见
`../../dit-parallel-opt/references/parallel-plan-attribution-method.md` §2；
"设备省了 e2e 没省"的方向性判据见 `../../perf-gate/references/measurement-discipline.md` §10）。

**② 逐 kernel 计时不可复现，聚合量与计数可复现**
同一配置、两次独立采集（不同 device）实测：**算子名字族的调用次数完全一致**、
Top-kernel 集合相同、**聚合量差落在噪声内**；但**通信类与单个集合通信核的耗时差异显著**
（单个集合通信核差异可达接近一倍）✗。
⇒ **分层引用**：**调用次数 / Top 集合 / 聚合量与占比可信**；**单核毫秒数不可在 ±10% 内引用**、
**跨设备单核差异可达接近一倍**（绝对偏差出库 `{run_results_dir}/archive/`）
⇒ **禁止用单核毫秒数定位单卡 / 单设备问题**；
跨设备比较只用**调用次数与占比**。
⚠️ 该分层是**本仓实测上界、不是普适常数**（仅一次两设备对照）⇒ **每次换卡 / 换版本都要重测**。

**③ 列语义必须实测自证，不能照抄二手字段说明**
近名列并存，且**同一位次在不同表里的列名不同**。本仓实测（三张 kernel 表逐列取**去重值集合**后比对）：

| 列名 | 出现在 | 实际装的是 | 等价关系（**去重值集合实测**） |
|---|---|---|---|
| `Type` | `kernel_details`（48 列布局） | **算子类型** | **≡ `OP Type`**：各 60 个值、集合完全相等（`Slice`/`ViewCopy`/`Cast`/…/`HcclLaunchAicpuKernel`/`hcom_*`） |
| `OP Type` | `op_summary` / `op_statistic` / `communication_statistic` | **算子类型** | ≡ `Type`（同上，60/60 相等） |
| `Accelerator Core` | `kernel_details` | **执行单元 / 核类型** | **≡ `Task Type`**：各 6 个值、集合完全相等（`AI_VECTOR_CORE` / `MIX_AIV` / `AI_CPU` / `COMMUNICATION` / `AI_CORE` / `MIX_AIC`） |
| `Task Type` | `op_summary` | **执行单元 / 核类型** | ≡ `Accelerator Core`；⚠️ **`kernel_details` 里没有这个列名** |
| `Core Type` | `op_statistic`（按算子聚合的表） | **执行单元 / 核类型**（聚合口径） | **值域同族但不等价**：只有 5 个值、**缺 `COMMUNICATION`** ⇒ 拿它统计通信会**整类漏掉** ✗ |
| `kernel_type` | `task_time` | **更宽的 task 类型枚举** | **不等价**：16 个值，另含 `MEMCPY_ASYNC` / `NOTIFY_RECORD` / `DAVID_EVENT_*` 等非计算核 |
| `Input/Output Data Types` | `kernel_details` | **数据类型**（与"算子类型"无关） | 易混名，勿当算子类型用 |

⇒ **读表第一步是「列语义自证」**：对候选列取**去重值集合**判它到底是什么，**报告写明据以判定的证据**；
**不要**引用别人给的字段摘要。另：**三张 kernel 表列数相同（48）但列名不同**
（`Type`/`Name`/`Accelerator Core` vs `OP Type`/`Op Name`/`Task Type`）——
这正是"**同一批 task 行、两种表头布局**"的实测确认 ⇒ **不能按"第 N 列"或照抄列名取数** ✗。

**④ 一次便宜的交叉核对（自证解析正确）**：trace 里的 `Communication` 应与 `hcom_*` 行的
**ΣDuration 应精确相等**（不等即说明表与 trace 已不同源或解析有变）
⇒ 可当作"表与 trace 同源、解析正确"的一次核对 ✓。

## 分析管道

```text
Layer 0: 预处理（warmup 验证）
    ↓
Layer 1: 阶段分离（DiT vs VAE）
    ↓
Layer 2: 算子分类占比（FA / MatMul / Vector / Comm，分阶段给出）
    ↓
Layer 3: 三层递进分析（Host Bound → 通信掩盖 → 融合机会，分阶段给出）
    ↓
Layer 4: 算子明细（占比 >1%）
    ↓
Layer 5: 优化建议（P0-P2 优先级 + 引用 docs/zh/features 对应节）
```

---

### Layer 0: 预处理 — Warmup 验证

确认 profiling 数据已剔除 warmup 步。若检测到 warmup 特征（首步耗时异常偏高、编译 kernel 集中出现），标注 `WARMUP_NOT_STRIPPED`。

### Layer 1: 阶段分离 — DiT vs VAE

将 kernel 按名称/类别聚合到两个阶段：

| 阶段 | 识别特征 | 典型算子 |
|------|---------|---------|
| **DiT (Transformer)** | attention_forward, MatMul, LayerNorm, RoPE | FlashAttention, Linear, RMSNorm |
| **VAE** | Conv2D, GroupNorm, Upsample | Conv2D, ResBlock |

输出：

```text
DiT: xx ms (xx%)  |  VAE: xx ms (xx%)
```

### Layer 2: 算子分类占比（分阶段）

对每个阶段按四类聚合。**仅显示占比 >1% 的类别**，低于此阈值归入"其他"。

> 聚合规则与各算子 NPU 已知问题以 `references/operator-catalog.md` 为唯一真相源，下表为速查快照；改动先改 reference，再同步本表。

| 分类 | 包含算子 |
|------|---------|
| **FA** | FlashAttention, SDPA, attention_forward, fused_attn_score |
| **MatMul** | Linear, MatMul, GEMM, DequantGEMM |
| **Vector** | 激活函数 (GELU/SiLU/ReLU), Norm (LayerNorm/RMSNorm), element-wise (Mul/Add/Div) |
| **Comm** | HCCL: all_gather, all_reduce, reduce_scatter, broadcast |

输出格式：

```text
### DiT 算子分布          ### VAE 算子分布
| FA      | xx% |         | MatMul  | xx% |
| MatMul  | xx% |         | Vector  | xx% |
| Vector  | xx% |         | Comm    | —   |
| Comm    | xx% |
```

### Layer 3: 三层递进分析（分阶段）

对每个阶段独立做三层分析：

> 三层判断的启发式明细以 `references/heuristics.md` 为唯一真相源（正文保留判断主链）；通信掩盖相关方案见 dit-parallel-opt（本仓），而非外部 hccl-test。

#### Layer 3a: Host Bound 分析

多指标核算体系：同时维护以下指标（参照 ascend-profiling-anomaly）：

| 指标 | 含义 |
|------|------|
| `wall_ms` | 阶段从 start 到 end 的总经过时间 |
| `busy_union_ms` | 设备计算区间合并（去重叠后的真实计算时间） |
| `kernel_sum_ms` | 各 kernel 耗时累加（含并行重叠部分） |
| `bubble_ms` | `wall_ms - busy_union_ms` = 设备空闲时间 |

```text
Underfeed = Service Time - Device Busy Union
Host Bound % = Underfeed / Service Time × 100
```

关键指标：underfeed_ratio, prelaunch_gap, tail_gap, internal_bubble_total, largest_internal_bubble

Anomaly 标签（参照 ascend-profiling-anomaly）：

| 标签 | 触发条件 |
|------|---------|
| `DEVICE_IDLE_GAP_HEAVY` | underfeed_ratio >= 0.30 |
| `PRELAUNCH_GAP_HEAVY` | prelaunch_gap >= max(1ms, 10% step) |
| `TAIL_GAP_HEAVY` | tail_gap >= max(1ms, 10% step) |
| `INTERNAL_BUBBLE_HEAVY` | largest_internal_bubble >= max(1ms, 10% step) |
| `HOST_ORIGINATED_RISK` | 高 underfeed + 周期性 bubble + host event 证据 |

#### 快捷判别：先排除 torch.compile 重编译，再归因 kernel

当 `wall_ms / kernel_sum_ms >> 10`（kernel 总耗时只占墙钟个位数百分比）、`Wait Time` 合计接近
wall、且出现**单个超大设备空闲间隙**（如整个墙钟里几乎全程空闲）时，优先怀疑 **Dynamo guard 失败导致
每次调用重编译**，而不是 kernel 慢。典型根因：算子层 forward 内就地修改模块状态（如把 bias 从
bf16 改 fp32）使 guard 不稳定。

```shell
# 确认重编译与 guard 失败原因（比 trace 分析更直接）
TORCH_LOGS=recompiles python {infer}.py --compile ... 2>&1 | grep -E "Recompiling|guard failure"
# 输出形如: tensor '..._buffers['bias']' dtype mismatch. expected BFloat16, actual Float
```

重编译一次 ≈ Dynamo trace + Inductor codegen + triton JIT（秒级开销），会让 compile 比 eager 慢
一到两个数量级。修复（forward 用局部变量、不 mutate 模块状态）后 compile 恢复应有的收益。
详见 pattern-dev/references/pattern-dev-notes.md §4（模块状态就地变更类问题）与 pattern-dev/references/mismatch-catalog.md（7 类 mismatch）。

#### Layer 3b: 通信掩盖分析（多卡）

```text
Exposed Ratio = 未与计算重叠的通信耗时 / 通信总耗时
```

- Exposed Ratio > 30%：显著不可掩盖 → 检查 RSP 通信流水线
- Exposed Ratio < 10%：通信良好掩盖

通信算子参考表（参照 hccl-test skill，Ascend agent-skills）：

| 通信算子 | 推荐度 | 适用场景 |
|---------|:--:|------|
| AllReduce | 推荐 | TP reduce-scale 梯度/数据聚合 |
| AllGather | 推荐 | 序列并行结果收集 |
| AlltoAll | 条件 | Ulysses USP 注意力头重组 |
| Broadcast | 可选 | 权重/配置广播 |

> 完整 HCCL 测试和带宽数据见 hccl-test（Ascend agent-skills）。

#### Layer 3c: 融合机会分析（候选识别，不做边界与收益判定）

从执行序导出**融合机会候选**，逐条给出「模式名 + 区域坐标（起止算子）+ 相似度分级」，再**交棒**
`../fusion-scope-analyze/SKILL.md`——由它按融合单元构造规则切边界、用计算单元利用率判型并估收益
（本技能只出候选，不判「能不能融、融多大、值不值」）。**交棒必须带候选清单**（每条含区域坐标）；
交付件的列契约与门禁见 `../fusion-scope-analyze/scripts/check_fusion_scope.py`。

> 候选族与识别规则的单点已迁至 `../fusion-scope-analyze/references/fusion-unit-method.md` §3；
> 编译侧开关名与启用方式见 `docs/zh/features/compilation.md` §Pattern 融合（开关真源）。
> 「预期收益量级」属业内启发式（非本仓实测、非承诺），只用于排序取舍；本技能不再保留该列。

当没有精确匹配的模式时，标注相似度：**high / medium / low**

- **high**: kernel 序列模式、source location、TP context 高度一致
- **medium**: 部分特征匹配但缺少关键证据
- **low**: 仅 kernel 名称接近，语义结构和上下文不匹配

### Layer 4: 算子明细

列出占比 >1% 的单一算子（按耗时降序）。**低于 1% 的算子不列出**：

| 算子名 | 耗时(ms) | 占比 | 类型 | 所属阶段 |
|-------|---------|------|------|:--:|
| flash_attn_score | xx | xx% | FA | DiT |
| npu_linear | xx | xx% | MatMul | DiT |

### Layer 5: 优化建议

每条建议固定格式：优先级 | 发现 | 优化方向 | 引用

分析仅给出**优化方向**，具体方案（API/算法/参数选择）由 dit-perf-opt 确定
（特性真源 `docs/zh/features/*`；支持状态 `framework-integration/references/framework-support-matrix.md`）。

建议触发规则：

| Layer 2/3 发现 | 阈值 | 优化方向 | 引用 |
|---------|:--:|------|------|
| DiT, MatMul 占比高 | >50% | MatMul 量化 | docs/zh/features/quantization.md §Linear量化 |
| DiT, FA 占比高 | >30% | Attention 优化（量化+稀疏） | docs/zh/features/quantization.md §FA量化 + sparse.md |
| DiT, Vector 占比高 | >20% | 编译融合 | docs/zh/features/compilation.md §Pattern 融合 |
| DiT, Comm exposed | >30% | 通信掩盖 | docs/zh/features/parallelism.md |
| VAE, MatMul 占比高 | >30% | ACLGraph 加速 | docs/zh/features/compilation.md §ACLGraph 加速 |
| VAE, Conv2D 连续 | — | VAE 融合（通用） | 需自行实现 |
| Host Bound 高 | >20% | re-profile with with_stack=true | — |
| MindIE-SD Pattern 命中 | — | 开启对应 CompilationConfig 开关 | docs/zh/features/compilation.md §Pattern 融合 |

优先级规则：

- **P0** — MindIE-SD Pattern 命中，有开关可直接启用
- **P1** — 算子分类触发建议，有 `docs/zh/features/*` 对应方向
- **P2** — 通用融合建议或数据质量建议，需自行实现/验证

> 建议结构同样遵循分阶段原则：DiT 和 VAE 各自的建议分开输出。

## 分析脚本

- `scripts/analyze_trace.py` — 5 层递进分析，输出 `profiling_report.md` + `model_architecture_report.md`
- `scripts/compare_traces.py` — 两次 run 的算子级对比，标注 REGRESSION/improvement

## Reference Files

- `references/capability-matrix.md` — 加载时机: 确定分析路径和可用 profiler 工具时
- `references/operator-catalog.md` — 加载时机: 识别具体算子对应的 NPU 实现和已知问题时
- `references/heuristics.md` — 加载时机: 判断优化方向时
- `references/performance-analysis-methodology.md` — 加载时机: 多卡统计口径（固定 rank0/p50）、收益三层证据与同窗口同卡组对比基准时
- `references/analysis-flow.md` — 加载时机: 需要端到端分析流程时
- `references/eager-vs-compile-report.md` — 加载时机: 需要出 **compile vs eager 收益对比的双报表**（聚合口径、收益分母、站点→kernel 归属、未实现行填充规则、fail-closed 记帐）时——自 `dummy-run` 下沉，**该口径的单点在本文件**

## 维护与更新

当发现新的瓶颈类型、算子耗时分析方法更新、CANN profiler 输出格式变更或性能诊断工具升级时，
按 dev-workflow 的复盘流程更新本 skill。

- **更新触发条件**：CANN profiler **列名 / 列语义**变化（`Type` vs `Accelerator Core` 这类同名列的
  归属会随版本变）、`step_trace_time` 四元组口径变化、或 §「读表口径与可复现性」三条的
  读数（重叠计入的量级、跨设备可复现性分层、列语义归属）不再成立时。
- **复核方法**：换版本 / 换采集档后重跑三项最小核对 —— ① 同一份产物分别用「跨流 ΣDuration」与
  「纯计算忙碌」出数，确认两者**不相等**且差额与"被重叠的通信"量级一致（相等才该怀疑口径已变）；
  ② 同配置**两次独立采集**比调用次数与总量，确认"计数一致、单核漂移"的分层仍成立；
  ③ 对 `Type` / `Accelerator Core` 各取**去重值集合**，确认哪些值属于哪一列。
- **失效信号**：若某次采集里「trace 的 `Communication`」与「`hcom_*` 行 ΣDuration」**不再相等**，
  说明表与 trace 的对应关系或解析已变 —— 先修解析，再引用任何通信读数。
