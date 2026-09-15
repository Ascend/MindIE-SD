# 优化维度详解

具体 API 和算法名见 `docs/zh/features/*`（特性真源），本文档仅描述通用原理和决策逻辑。

## 1. 编译路径优化

通过 `MindieSDBackend()` 启用 Pattern 融合和 ACLGraph 静态图捕获。
融合作用于 Norm/激活/元素级操作，不作用于 MatMul 和 Attention 本身。

| 方向 | 方法 | 预期收益 | 风险 |
|------|------|---------|------|
| 开启 MindieSDBackend | 确保 pattern 注册完整，无 eager fallback | 加速显著（量级指引，非承诺） | 依赖算子兼容性 |
| 混合模式 | transformer compiled + VAE eager | 编译稳定性 | 部分加速 |
| JIT 预热 | 增加 warmup 步数（≥5步） | 首次推理耗时降低 | 无 |

> 融合开关控制、ACLGraph 细节见 `docs/zh/features/compilation.md`（§Pattern 融合 / §ACLGraph 加速）。

## 2. Attention 优化

Attention 本身**不可通过算子融合加速**。优化手段为：

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
