# NPU 优化启发式

具体 API 和算法名见 performance-optimization/references/mindiesd-features.md（唯一真相源）。

## 编译路径选择

| 条件 | 选择 |
|------|------|
| Copy 膨胀 >50% 且 模型含 3D attention | **torchair_ge** (绕过 aot_autograd functionalization) |
| 模型使用标准 Norm 层 (如 FLUX) | **default** (MindieSDBackend, pattern 全部命中) |
| 模型使用 FP32LayerNorm (如 Wan2.2) | **torchair_ge** (pattern 不匹配, Copy 膨胀) |
| 模型未支持 MindieSDBackend | eager baseline → 标记为"待编译器适配" |
| VAE 部分不稳定 | compiled transformer + eager VAE (混合模式) |
| 实验/对比原生 npugraph_ex | **npugraph_ex** — torch 2.9 上与 default 等价 (仍走 aot_autograd) |

> 完整 4-backend 对比见 `compilation-dev/references/backend-comparison.md`

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

> 详细接口、约束、支持矩阵见 mindiesd-features.md §Attention 优化。

## 显存优化优先级

| 优先级 | 策略 | 适用条件 | 预期收益 |
|-------|------|---------|---------|
| 1 | CPU offload | 单卡显存不足 | 峰值降低 60-70% |
| 2 | TP（张量并行） | 单机多卡，hidden_size 大 | 单卡显存随卡数线性降 |
| 3 | Activation checkpoint | 激活值占比高 | 显存换计算时间 |
| 4 | MatMul 量化 (MXFP4/FP8) | 精度容忍 | 权重显存减半 |
| 5 | 分辨率/帧数降低 | 可接受质量折衷 | 线性降低 |

## 优化建议触发规则

分析给出**优化方向**（非具体算法）。具体方案由 performance-optimization 从 mindiesd-features.md 选取。

| Layer 2/3 发现 | 阈值 | 优化方向 | 引用 |
|---------|:--:|------|------|
| DiT, MatMul 占比高 | >50% | MatMul 量化 | mindiesd-features.md §MatMul量化 |
| DiT, FA 占比高 | >30% | Attention 优化（量化+稀疏） | mindiesd-features.md §Attention优化 |
| DiT, Vector 占比高 | >20% | 编译融合 | mindiesd-features.md §编译路径优化 |
| DiT, Comm exposed | >30% | 通信掩盖 | mindiesd-features.md §通信掩盖 |
| VAE, MatMul 占比高 | >30% | ACLGraph 加速 | mindiesd-features.md §编译路径优化 |
| Host Bound 高 | >20% | re-profile with with_stack=true | — |
| MindIE-SD Pattern 命中 | — | 开启 CompilationConfig 开关 | 标注开关名 |

优先级规则：P0 = MindIE-SD Pattern 命中 → P1 = 算子分类触发 → P2 = 通用融合/数据质量
