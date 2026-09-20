# 优化维度详解

具体 API 和算法名见 `docs/zh/features/*`（特性真源），本文档仅描述通用原理和决策逻辑。

## 1. 编译路径优化

通过 `MindieSDBackend()` 启用 Pattern 融合和 ACLGraph 静态图捕获。
融合作用于 Norm/激活/元素级操作，不作用于 MatMul 和 Attention 本身（这是**编译侧 pattern 的替换范围**）。
若要以 **MatMul 为首算子**构造融合单元（Cube 首 + 向后包裹 Vec），按
`../../fusion-scope-analyze/references/fusion-unit-method.md` 规则 4 判定——候选族、锚点与终止条件的
单点在该文件，本文件不复制判据。

| 方向 | 方法 | 预期收益 | 风险 |
|------|------|---------|------|
| 开启 MindieSDBackend | 确保 pattern 注册完整，无 eager fallback | 加速显著（量级指引，非承诺） | 依赖算子兼容性 |
| 混合模式 | transformer compiled + VAE eager | 编译稳定性 | 部分加速 |
| JIT 预热 | 增加 warmup 步数（≥5步） | 首次推理耗时降低 | 无 |

> 融合开关控制、ACLGraph 细节见 `docs/zh/features/compilation.md`（§Pattern 融合 / §ACLGraph 加速）。

## 2. Attention 优化

Attention 本身**不可通过算子融合加速**（FA 及其变种是**硬锚点**：不吸收、不跨越；判据见
`../../fusion-scope-analyze/references/fusion-unit-method.md` 规则 2）。优化手段为：

- **FA 量化**: Q/K/V FP8 块量化，降低注意力显存带宽
- **稀疏注意力**: 跳过低相关 Token 对，减少有效计算量

| 方向 | 瓶颈指标 | 选择指南 |
|------|---------|---------|
| FA 量化 (FP8) | Attention 显存带宽瓶颈 | head_dim 兼容时优先 |
| 稀疏 rf_v2 | Attention 占比 >30%，视频/图像 | 视频 sparsity=0.8，图像 sparsity=0.6 |
| 稀疏 ada_bsa | rf_v2 模型不兼容时 | 备选 |

> 接口与硬件约束见 `docs/zh/features/quantization.md` §FA量化 + `docs/zh/features/sparse.md`；
> 模型支持矩阵见 `framework-integration/references/framework-support-matrix.md`。

## 3. MatMul 量化

MatMul 本身的性能瓶颈通过低比特量化解决。

| 精度 | 适用组件 | 选择指南 |
|------|---------|---------|
| MXFP8 (W8A8) | Transformer 权重+激活 | 通用首选 |
| MXFP4 (W4A4) | Transformer 权重+激活 | 更高压缩比，需精度验证 |
| FP8/INT8 (W8A8 系列) | Transformer 权重+激活 | 无 MX 格式硬件时的备选 |
| W8A16 / W4A16 | 仅权重 | 激活保持 FP16，兼容性最好 |

> 接口、算法名、硬件约束见 `docs/zh/features/quantization.md` §Linear量化。

## 4. 显存优化

| 策略 | 说明 | 典型收益 |
|------|------|---------|
| CPU offload | 异步流水线，计算与权重搬运并行 | 峰值显著下降（Wan2.2 实测，本组合观测） |
| 张量并行 (TP) | 按行/按列切分权重到多卡 | 单卡显存随卡数线性降低 |
| Activation checkpoint | 重计算换显存 | 激活值显存降低 |
| 层数裁剪 | 仅 dummy run，减小 num_layers | 参数量线性降低 |

> 接口和参数见 `docs/zh/features/cpu_offload.md`（CPU offload）与 `docs/zh/features/parallelism.md`
> §Tensor Parallel（TP）。

## 5. 并行策略（多卡）

| 卡数 | 策略 | 适用条件 |
|------|------|---------|
| 2 | TP=2 或 CFG parallel | TP 需 hidden_size 可切分; CFG 需 guidance_scale > 1 |
| 4 | USp=4 或 TP=2 + CFG | USp 需 head_num 被并行度整除 |
| ≥4，长序列 | RSP | 序列长度 >> head_dim 时通信可掩盖 |

> 通信方式、代码示例见 `docs/zh/features/parallelism.md`（TP / RSP / Ulysses / USP / CFG 各节）。

## 6. 缓存加速（以存代算）

扩散模型相邻时间步存在冗余计算，通过缓存中间结果跳过。

| 方案 | 粒度 | 优先条件 |
|------|------|---------|
| DiTCache | block 级 | 通用首选 |
| AttentionCache | Attention 级 | Attention 占比高时更优 |
| 时间步优化 | 步级 | 辅助，与其他方案互补 |

> 接口见 `docs/zh/features/cache.md`（DiTCache / AttentionCache / 时间步优化）。

## 7. 去冗余（把运行时的重复计算固化掉）

与 §6「以存代算」并列的一族：**不改数值语义地删掉运行时重复劳动**。识别顺序 = 先看"这段计算
是不是每次都一样"，再看"能不能在更早的时刻算一次"。

| 子类 | 识别信号（怎么发现它） | 固化位置 | 判据 / 现场取数 |
|------|----------------------|----------|------------------|
| **静态低秩适配器离线合并**（LoRA→权重） | ①适配器在**服务启动时**加载、请求侧 scale 恒定（不随请求变）②逐算子清单里该链的**下发实例数占比高**而 device 耗时占比低（§`../../perf-gate/references/measurement-discipline.md` §10.5 归因）③该链是**线性**的（可写成 `W' = W + (α/r)·B·A`） | 权重（离线合并） | 精度存活判据与三形态选择见 `../../quantization-dev/references/lora-merge-and-precision.md`；**运行时链成本结构见 `lora-adapter-cost.md`** |
| **重复的整段解码 / 冗余广播计算** | 多 rank 上同一张量指纹逐位相同（说明每卡都在算同一份） | 分片 + 集合通信 | 见 `../../vae-opt/references/independence-proof.md`（"指纹相同"只说明当前是冗余，不说明可切） |
| **每层重复的几何/常量构造** | 逐算子清单里大量小切片/索引类算子、其输入只依赖**模型常量与拓扑** | 初始化期缓存（一次性） | 判据 = 关缓存后产物是否**逐字节相同**；注意"可省的只有断言/校验套件"这一类要单独量 |
| **恒成立的钳位 / 边界分支** | 由构造保证的范围约束（如量化 scale 使取值天然有界） | 算子内部 | 见 `../../operator-dev/references/vertical-fusion-notes.md`（clamp 冗余的证明方式） |

**纪律**：

- **去冗余与融合是两种手段、可能争同一处收益**：同一段链上"离线固化"和"运行时融合"往往只能选
  一个，**收益不可相加**（判定法见 `../../fusion-scope-analyze/references/fusion-candidate-identification.md`）；
- **固化会改变权重来源**（把外部训练权重折进权重）⇒ 归组见
  `../../model-auto-optimization/references/overview-report.md` §1 的优化类型枚举，质量门禁按有损档走；
- **上限 = 该链的 wall 差**：不得宣称超过"整条链消失"能省下的量（`lora-adapter-cost.md`）；
- **一次性缺陷不得固化成结构**：绕行一律可一键关闭。

## 选择决策树

```text
瓶颈定位（来自 profiling-analyze 三表）
├─ 算力瓶颈 (MatMul+Attention > 60%)
│   ├─ MatMul 为主 → docs/zh/features/quantization.md §Linear量化
│   └─ Attention 为主 → docs/zh/features/quantization.md §FA量化 + docs/zh/features/sparse.md
├─ 显存瓶颈 (峰值 ≈ 物理显存)
│   └─ → docs/zh/features/cpu_offload.md（+ parallelism.md §Tensor Parallel）
├─ 通信瓶颈 (Comm exposed 达 profiling-analyze 触发阈值——判据单点在其 Layer 5 表)
│   └─ → docs/zh/features/parallelism.md
├─ 编译未触发 (eager fallback)
│   └─ → docs/zh/features/compilation.md
└─ 冗余计算 (相邻步相似 latent)
    └─ → docs/zh/features/cache.md
```

## 维护与更新

当优化维度决策树变化时，按 dev-workflow 的复盘流程更新本文件。
