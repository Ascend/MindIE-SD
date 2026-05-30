# MindIE-SD 优化特性映射

> 瓶颈 → MindIE-SD 方案的唯一映射表。版本升级时只需更新此文件。
>
> **最后同步**: 2026-05-10 18:21 UTC (由 scripts/refresh_features.py 自动生成)
>
> **源文件** (当内容可疑时查阅):
>
> - [quantization.md](../../../../docs/zh/features/quantization.md)
> - [sparse.md](../../../../docs/zh/features/sparse.md)
> - [parallelism.md](../../../../docs/zh/features/parallelism.md)
> - [cpu_offload.md](../../../../docs/zh/features/cpu_offload.md)
> - [compilation.md](../../../../docs/zh/features/compilation.md)
> - [cache.md](../../../../docs/zh/features/cache.md)
> - [supported_matrix.md](../../../../docs/zh/features/supported_matrix.md)

## MatMul 量化

MatMul 本身不通过融合优化，而是通过低比特量化减少显存带宽和计算量。

| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |
|---------|---------------|------|---------|---------|
| 仅权重量化需求 | W8A16 | `quantize(model, "quant_desc_w8a16_0.json")` | Atlas 800I A2 | — |
| 仅权重量化需求 | W4A16 | `quantize(model, "quant_desc_w4a16_0.json")` | Atlas 800I A2 | — |
| 仅权重量化需求 | W4A16_AWQ | `quantize(model, "quant_desc_w4a16_awq_0.json")` | Atlas 800I A2 | — |
| 仅权重量化需求 | W8A16_GPTQ | `quantize(model, "quant_desc_w8a16_gptq_0.json")` | Atlas 800I A2 | — |
| 仅权重量化需求 | W4A16_GPTQ | `quantize(model, "quant_desc_w4a16_gptq_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50% | W8A8（INT8 权重激活量化） | `quantize(model, "quant_desc_w8a8_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，时间步动态 | W8A8_TIMESTEP（INT8 权重激活量化） | `quantize(model, "quant_desc_w8a8_timestep_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，动态量化 | W8A8_DYNAMIC（INT8 权重激活量化） | `quantize(model, "quant_desc_w8a8_dynamic_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50% | W8A8_PER_CHANNEL（INT8 权重激活量化） | `quantize(model, "quant_desc_w8a8_per_channel_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50% | W8A8_PER_TENSOR（INT8 权重激活量化） | `quantize(model, "quant_desc_w8a8_per_tensor_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50% | W8A8_MXFP8（MX 格式） | `quantize(model, "quant_desc_w8a8_mxfp8_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，高压缩，动态量化 | W4A4_DYNAMIC | `quantize(model, "quant_desc_w4a4_dynamic_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，高压缩 | W4A4_MXFP4_SVD | `quantize(model, "quant_desc_w4a4_mxfp4_svd_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，高压缩 | W4A4_MXFP4_DUALSCALE | `quantize(model, "quant_desc_w4a4_mxfp4_dualscale_0.json")` | Atlas 800I A2 | — |
| MatMul 占比 >50%，高压缩，动态量化 | W4A4_MXFP4_DYNAMIC | `quantize(model, "quant_desc_w4a4_mxfp4_dynamic_0.json")` | Atlas 800I A2 | — |

> 量化描述符和权重文件由 msmodelslim 工具预导出。详见 [quantization.md](../../../../docs/zh/features/quantization.md)。

## Attention 优化

Attention 本身不可通过算子融合加速。优化手段为 FA 量化（FP8块量化 Q/K/V）和稀疏注意力。

| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |
|---------|---------------|------|---------|---------|
| Attention 占比 >30%，头间显存带宽瓶颈 | FA 量化 (FP8) | `quantize(model, ...)` 自动注入 `FP8RotateQuantFA` | **仅** Atlas 800I A2 | Q/K/V 布局支持 BNSD/BSND |

> 详见 [quantization.md](../../../../docs/zh/features/quantization.md) §FA量化。

| Attention 占比 >30%，视频模型 | 稀疏 rf_v2 (RainFusion2.0) | `sparse_attention(q,k,v, sparse_type="rf_v2", sparsity=0.8, latent_shape_q=[t,h,w])` | Atlas 800I A2 | 需 `latent_shape_q/k` |
| Attention 占比 >30%，图像模型 | 稀疏 rf_v2 | `sparse_attention(q,k,v, sparse_type="rf_v2", sparsity=0.6)` | Atlas 800I A2 | — |
| Attention 占比 >30%，rf_v2 不兼容 | 稀疏 ada_bsa | `sparse_attention(q,k,v, sparse_type="ada_bsa", cdf_threshold=...)` | Atlas 800I A2 | — |

> rf_v2 80% 稀疏率下端到端加速 1.5–1.8×。图像 sparsity 建议 0.6 起步，视频 0.8 起步。
> 详见 [sparse.md](../../../../docs/zh/features/sparse.md)。

## 编译路径优化（融合 + 图捕获）

通过 `torch.compile(backend=MindieSDBackend())` 触发，同时启用 Pattern 融合和 ACLGraph 加速。

| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 |
|---------|---------------|------|---------|
| 未触发 MindieSDBackend（eager fallback） | 启用编译后端 | `torch.compile(model, backend=MindieSDBackend())` | — |
| Norm 层大量小 kernel launch | RMSNorm 融合 | `CompilationConfig.fusion_patterns.enable_rms_norm = True` | — |
| RoPE 独立 kernel 开销 | RoPE 融合 | `CompilationConfig.fusion_patterns.enable_rope = True` | — |
| AdaLayerNorm 独立调度 | AdaLayerNorm 融合 | `CompilationConfig.fusion_patterns.enable_adalayernorm = True` | — |
| GELU/SiLU 激活独立 kernel | fastGELU 融合 | `CompilationConfig.fusion_patterns.enable_fast_gelu = True` | — |
| element-wise Mul+Add 开销 | Mul+Add 融合 | `CompilationConfig.fusion_patterns.enable_mul_add = True` | — |
| 每步动态图调度开销 | ACLGraph 静态图捕获 | 自动启用（`MindieSDBackend()` 内） | — |

> Pattern 融合作用于 Norm/激活/元素级操作，不作用于 MatMul 和 Attention 本身。
> 首次推理有 JIT 编译开销（最多 8 次尝试）。Benchmark 时需 ≥5 步 warmup 排除编译耗时。
> 详见 [compilation.md](../../../../docs/zh/features/compilation.md)。

## 显存优化

| 瓶颈指标 | MindIE-SD 方案 | 接口 | 硬件约束 | 模型约束 |
|---------|---------------|------|---------|---------|
| 峰值≈物理显存，block 数多 | 异步 CPU Offload | `enable_offload(model, blocks, min_reserved_blocks_count=2)` | — | 需指定 blocks 列表 |
| 峰值≈物理显存，hidden_size 大 | 张量并行 (TP) | 按行/按列切分权重 | 单机多卡 HCCS | TP degree ≤ 卡数 |
| 激活值占比高 | Activation Checkpoint | PyTorch 原生 | — | 换计算时间 |

> 详见 [cpu_offload.md](../../../../docs/zh/features/cpu_offload.md) 和 [parallelism.md](../../../../docs/zh/features/parallelism.md)。

## 通信掩盖

多卡场景中不可避免的通信开销可通过计算掩盖。

| 场景 | MindIE-SD 方案 | 原理 | 硬件约束 |
|------|---------------|------|---------|
| 序列较长，hidden_size 大 | 张量并行 (TP) | 按行/按列切分权重，减少单卡显存 | 单机多卡 HCCS |
| 序列较长，head_dim 大 | 环状序列并行 (RSP) | P2P 环形传递 KV，计算耗时掩盖通信 | 同机 NPU HCCS |
| head 数多，AlltoAll 带宽充裕 | Ulysses 序列并行 (USP) | AlltoAll 在头维度重组，通信量恒定 | 并行度需整除 head_num |
| CFG > 1 | CFG 并行 | 正负样本分卡并行，通信量极小 | ≥ 2 卡 |

> 详见 [parallelism.md](../../../../docs/zh/features/parallelism.md)。

## 缓存加速（以存代算）

扩散模型相邻时间步存在冗余计算，通过缓存中间结果跳过重复计算。

| 场景 | MindIE-SD 方案 | 接口 | 适用条件 |
|------|---------------|------|---------|
| block 数多 | DiTCache | `CacheConfig(method="dit_block_cache", ...)` + `CacheAgent` | 通用 |
| Attention 占比高 | AttentionCache | `CacheConfig(method="attention_cache", ...)` + `CacheAgent` | Attention 密集型 |
| 辅助任何缓存方案 | 时间步优化 | 减少/跳过扩散步数 | 需质量容忍 |

> DiTCache 优先尝试，AttentionCache 备选。
> 详见 [cache.md](../../../../docs/zh/features/cache.md)。

## 模型/硬件支持矩阵（速查）

| 模型 | 并行 | 稀疏FA | 量化 | Cache | 融合算子 |
|------|:----:|:-----:|:----:|:-----:|:-------:|
| FLUX.1-dev | — | — | — | — | — |
| Wan2.2 | — | — | — | — | — |
| HunyuanVideo-1.5 | — | — | — | — | — |
| Qwen-Image | — | — | — | — | — |

> 完整矩阵见 [supported_matrix.md](../../../../docs/zh/features/supported_matrix.md)。

---

## 维护说明

- **更新触发**: MindIE-SD 发版新增/废弃算法时

- **更新方式**: 运行 `python scripts/refresh_features.py --docs-dir <path>`

- **手动更新**: 运行脚本后可在输出文件中手动补充表格行
