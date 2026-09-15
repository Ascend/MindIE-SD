# NPU 优化启发式

具体 API 和算法名见 `docs/zh/features/*`（特性真源）。

> ⚠️ 本文件的「预期收益」与「触发阈值」栏均为**启发式量级指引（非本仓实测值、非承诺）**（含 % 与 × 倍形式）：只用于排序与取舍，不得当作某组合的预期收益引用。

## 编译路径选择

| 条件 | 选择 |
|------|------|
| 静态 shape / 大 batch，需减少 host launch 开销 | **aclgraph** 批量下发（见 `aclgraph-dev`） |
| 模型使用标准 Norm 层 (如 FLUX) | **default** (MindieSDBackend, pattern 全部命中) |
| Pattern 命中但 Copy 膨胀 | 修复 pattern / 混合模式（见 pattern-dev Phase 7） |
| 模型未支持 MindieSDBackend | eager baseline → 标记为"待编译器适配" |
| VAE 部分不稳定 | compiled transformer + eager VAE (混合模式) |

> 后端事实：本仓只有 default（Inductor）与 aclgraph（批量下发）两条路径；
> torchair_ge / npugraph_ex 在本仓未实现，不采用。

## 融合机会判断

### 优先：MindIE-SD Pattern（有开关可直接启用）

| 优先级 | 融合模式 | 对应开关 | 检查条件 | 收益 |
|:--:|---------|---------|---------|------|
| 1 | RMSNorm | `CompilationConfig.fusion_patterns.enable_rms_norm` | transformer 前向路径 | 减少 kernel launch |
| 2 | RoPE | `CompilationConfig.fusion_patterns.enable_rope` | 每层 attention 前后 | 减少 kernel launch |
| 3 | AdaLayerNorm | `CompilationConfig.fusion_patterns.enable_adalayernorm` | DiT 类模型 | 减少同步 |
| 4 | fastGELU | `CompilationConfig.fusion_patterns.enable_fast_gelu` | FFN 激活路径 | 减少中间显存 |
| 5 | Mul+Add | `CompilationConfig.fusion_patterns.enable_mul_add` | element-wise 操作 | 减少 kernel launch |

### 补充：业内通用融合（需自行实现，标注预期收益）

| 融合模式 | 识别规则 | 预期收益 | 适用阶段 |
|---------|---------|---------|:--:|
| MatMul + BiasAdd + GELU | MatMul → Add → GELU 连续 | ~25-30% | DiT |
| Scale + Softmax + MatMul | Mul(scale) → Softmax → MatMul | ~20-25% | DiT |
| Element-wise 链 (≥3) | 连续 3+ element-wise 算子 (Add/Mul/Div/Sub) | ~15-20% | DiT / VAE |
| FlashAttention + MatMul (proj) | Attention → MatMul 连续 | ~5-10% | DiT |
| Conv2D + GroupNorm | Conv2D → GroupNorm 连续 | ~10-15% | VAE |

## Attention 优化选择

Attention 自身不可融合——优化手段为 FA 量化和稀疏注意力。

| 优先级 | 策略 | 适用条件 | 预期收益 |
|-------|------|---------|---------|
| 1 | FA 量化 (FP8) | 910B，head_dim 兼容 Q/K/V 布局 | 显存带宽降低 |
| 2 | 稀疏 rf_v2 | 图像/视频模型 | 1.5–1.8× 端到端加速 |
| 3 | 稀疏 ada_bsa | rf_v2 不兼容时 | 灵活调节 |

> 详细接口、约束见 `docs/zh/features/quantization.md` §FA量化 与 `docs/zh/features/sparse.md`；
> 支持矩阵见 framework-integration/references/framework-support-matrix.md。

## 显存优化优先级

| 优先级 | 策略 | 适用条件 | 预期收益 |
|-------|------|---------|---------|
| 1 | CPU offload | 单卡显存不足 | 峰值降低 60-70% |
| 2 | TP（张量并行） | 单机多卡，hidden_size 大 | 单卡显存随卡数线性降 |
| 3 | Activation checkpoint | 激活值占比高 | 显存换计算时间 |
| 4 | MatMul 量化 (MXFP4/FP8) | 精度容忍 | 权重显存减半 |
| 5 | 分辨率/帧数降低 | 可接受质量折衷 | 线性降低 |

## 优化建议触发规则

分析给出**优化方向**（非具体算法）。具体方案由 dit-perf-opt 从 `docs/zh/features/*` 选取。

| Layer 2/3 发现 | 阈值 | 优化方向 | 引用 |
|---------|:--:|------|------|
| DiT, MatMul 占比高 | >50% | MatMul 量化 | docs/zh/features/quantization.md §Linear量化 |
| DiT, FA 占比高 | >30% | Attention 优化（量化+稀疏） | docs/zh/features/quantization.md §FA量化 + sparse.md |
| DiT, Vector 占比高 | >20% | 编译融合 | docs/zh/features/compilation.md §Pattern 融合 |
| DiT, Comm exposed | >30% | 通信掩盖 | docs/zh/features/parallelism.md |
| VAE, MatMul 占比高 | >30% | ACLGraph 加速 | docs/zh/features/compilation.md §ACLGraph 加速 |
| Host Bound 高 | >20% | re-profile with with_stack=true | — |
| MindIE-SD Pattern 命中 | — | 开启 CompilationConfig 开关 | 标注开关名 |

优先级规则：P0 = MindIE-SD Pattern 命中 → P1 = 算子分类触发 → P2 = 通用融合/数据质量

## Host/Kernel 与数据搬运归因（prep/稀疏类开销判定）

- **host vs kernel 判定**：用 `step_trace_time.csv` 的 `Computing / Communication / Free / Stage`——`Free`≈0 且步墙钟≈`Stage` ⇒ device-bound，host 间隙不是瓶颈，别再往 host 方向找。
- **per-op（per-name）时长归因**：按 `kernel_details.csv` 的 kernel 名聚合时长，区分「计算类」与「数据搬运/prep 类」（rearrange/cat/transpose/cast/mean 等）；"看似 mask 贵"常是搬运贵——如稀疏 mask 选择逻辑（topk/softmax/阈值/保护）仅 ms 级，而每层每步全尺寸 rearrange/pool 搬运占主体且随层数线性放大（开启方式与读数见 framework-integration `cache-dit-enablement.md` §3.3）。
- **集合通信归因：「暴露」与「占用」不是一回事**：按算子名里的 communicator id 分成 `(族, gid)`，
  再用「`次数 ≈ 层数 × 每层调用数 × 步数`」对账把每族认到具体并行组（除不尽 ⇒ 有同族额外通信：
  逐层 offload 的权重 gather / TP / 编码器 / VAE）；同族时间区间 `union/sum ≈ 1` 说明该耗时是
  **传输时间**而非依赖等待。注意 kernel 级总和是**占用**（跨流并发会多计），**暴露**只认
  `step_trace_time.csv` 的 `Communication(Not Overlapped)`。
  方法与脚本：`../../dit-parallel-opt/references/parallel-plan-attribution-method.md` +
  `../../dit-parallel-opt/scripts/collective_attribution.py`；掩盖上限与达成率：
  `../../dit-parallel-opt/references/comm-masking-method.md` +
  `../../dit-parallel-opt/scripts/mask_bound_calc.py`。
- **单调用微基准 vs 模型步级矛盾核对**：同几何（S/txt_len/latent/sparsity/block_size/head）对拍 dense vs 目标路径；核对①路径一致（eager 公共 API 的外部 mask vs 融合算子 op 内 mask+BSA 是两条路，口径不可混）②覆盖步数（start_step 前缀 dense）③内容相关 mask 需真实输入（随机输入 pooled 相似度可能 realized≈dense）。
- **外部 mask（eager）路径 op 级慢于 dense 是可能的真实现象，不是配置错**：该路径把开销放在**每层每步的全尺寸 rearrange/pool 搬运**上（随层数线性放大），而融合 op 把 mask+BSA 收进单个 kernel——**先定路径归属，再谈收益**；融合 op 未接线时其价值只能靠 op 级证据固化，端到端组合为待决项。
- op 级微基准姿势：**先 `import mindiesd` 再建 NPU 张量**（自定义算子注册前置），warm+稳态取 ms。

## 维护与更新

当优化启发式或决策表变化时，按 dev-workflow 的复盘流程更新本文件。
