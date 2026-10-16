---
name: performance-analysis
description: 针对真实 NPU 设备的 profiling 数据（trace.json / kernel_details.csv），
             分析模型性能分布，定位算子级耗时与显存热点，输出瓶颈诊断和改进建议。
             当用户有实际 profiling 产出、需要理解"模型为什么慢"或查找具体瓶颈时使用此 skill。
             即使用户只提到"这个模型为什么慢"或"帮我看看瓶颈在哪"，也应触发。
             由 dev-workflow 的分析阶段触发。
---

# 性能分析

基于昇腾 NPU profiling 数据，实现 5 层递进分析管道。

## 数据源

Profiling 数据由 profiling-collection skill 在远端 NPU 设备上采集产出（已剔除 warmup）。
也可来自 model-verification 的粗粒度时序或 performance-evaluation 的 msmodeling 分析。

| 数据文件 | 格式 | 说明 |
|---------|------|------|
| `kernel_details.csv` | CANN Profiler CSV | 每行一个 NPU 算子：Name, Start Time(us), Duration(us), Wait Time(us) |
| `trace_view.json` | Chrome Trace JSON | Host 端 + Device 端事件时间线 |
| `step_trace_time.csv` | CANN Profiler CSV | Step 级汇总：Computing, Communication, Free, Bubble |
| `communication.json` | JSON | 通信算子详情（若开启） |

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
Layer 5: 优化建议（P0-P2 优先级 + 引用 mindiesd-features.md）
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

对每个阶段按四类聚合。**仅显示占比 >1% 的类别**，低于此阈值归入"其他"。| 分类 | 包含算子 |
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

**Layer 3a: Host Bound 分析**

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

**Layer 3b: 通信掩盖分析（多卡）**

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

**Layer 3c: 融合机会分析**

优先检查 MindIE-SD 编译 Pattern（有开关可直接启用）：

| 优先级 | 融合模式 | 对应开关 | 识别规则 |
|:--:|---------|---------|---------|
| 1 | RMSNorm | `enable_rms_norm` | RMSNorm + 相邻 MatMul |
| 2 | RoPE | `enable_rope` | RoPE kernel 连续出现 |
| 3 | AdaLayerNorm | `enable_adalayernorm` | AdaLN + 相邻 kernel |
| 4 | fastGELU | `enable_fast_gelu` | MatMul → Add → GELU 连续 |
| 5 | Mul+Add | `enable_mul_add` | Mul → Add 连续 |

补充建议（业内通用，需自行实现）：

| 融合模式 | 识别规则 | 预期收益 |
|---------|---------|---------|
| MatMul + BiasAdd + GELU | MatMul → Add → GELU | ~25-30% |
| Scale + Softmax + MatMul | Mul → Softmax → MatMul | ~20-25% |
| Element-wise 链 (≥3) | 3+ 连续 Mul/Add/Div | ~15-20% |
| FlashAttention + MatMul | Attn → proj MatMul | ~5-10% |
| Conv2D + GroupNorm | CNN → GN（VAE 专有） | ~10-15% |

> 当无明显精确匹配的融合模式时，标注相似度：**high / medium / low**
>
> - **high**: kernel 序列模式、source location、TP context 高度一致
> - **medium**: 部分特征匹配但缺少关键证据
> - **low**: 仅 kernel 名称接近，语义结构和上下文不匹配

### Layer 4: 算子明细

列出占比 >1% 的单一算子（按耗时降序）。**低于 1% 的算子不列出**：

| 算子名 | 耗时(ms) | 占比 | 类型 | 所属阶段 |
|-------|---------|------|------|:--:|
| flash_attn_score | xx | xx% | FA | DiT |
| npu_linear | xx | xx% | MatMul | DiT |

### Layer 5: 优化建议

每条建议固定格式：优先级 | 发现 | 优化方向 | 引用

分析仅给出**优化方向**，具体方案（API/算法/参数选择）由 performance-optimization 确定。

建议触发规则：

| Layer 2/3 发现 | 阈值 | 优化方向 | 引用 |
|---------|:--:|------|------|
| DiT, MatMul 占比高 | >50% | MatMul 量化 | mindiesd-features.md §MatMul量化 |
| DiT, FA 占比高 | >30% | Attention 优化（量化+稀疏） | mindiesd-features.md §Attention优化 |
| DiT, Vector 占比高 | >20% | 编译融合 | mindiesd-features.md §编译路径优化 |
| DiT, Comm exposed | >30% | 通信掩盖 | mindiesd-features.md §通信掩盖 |
| VAE, MatMul 占比高 | >30% | ACLGraph 加速 | mindiesd-features.md §编译路径优化 |
| VAE, Conv2D 连续 | — | VAE 融合（通用） | 需自行实现 |
| Host Bound 高 | >20% | re-profile with with_stack=true | — |
| MindIE-SD Pattern 命中 | — | 开启对应 CompilationConfig 开关 | mindiesd-features.md §编译路径优化 |

优先级规则：

- **P0** — MindIE-SD Pattern 命中，有开关可直接启用
- **P1** — 算子分类触发建议，有 mindiesd-features.md 对应方向
- **P2** — 通用融合建议或数据质量建议，需自行实现/验证

> 建议结构同样遵循分阶段原则：DiT 和 VAE 各自的建议分开输出。

## 分析脚本

- `scripts/analyze_trace.py` — 5 层递进分析，输出 `profiling_report.md` + `model_architecture_report.md`
- `scripts/compare_traces.py` — 两次 run 的算子级对比，标注 REGRESSION/improvement

## Reference Files

- 📊 `references/capability-matrix.md` — 加载时机: 确定分析路径和可用 profiler 工具时
- 📋 `references/operator-catalog.md` — 加载时机: 识别具体算子对应的 NPU 实现和已知问题时
- 🔧 `references/heuristics.md` — 加载时机: 判断优化方向时
- 📝 `references/analysis_workflow.md` — 加载时机: 需要端到端分析流程时

## 维护与更新

当发现新的瓶颈类型、算子耗时分析方法更新、CANN profiler 输出格式变更或性能诊断工具升级时，
按 dev-workflow 的复盘流程更新本 skill。
