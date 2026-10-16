# 算子分类与瓶颈判断

具体 API 和算法名见 performance-optimization/references/mindiesd-features.md（唯一真相源）。

## 阶段划分

DiT（Transformer）和 VAE 通过 kernel 名称/类别分离：

| 阶段 | 识别特征 | 包含算子类别 |
|------|---------|------------|
| **DiT** | attention_forward, MatMul, LayerNorm, RoPE, GELU, SiLU | FA, MatMul, Vector |
| **VAE** | Conv2D, GroupNorm, Upsample, ResBlock | MatMul, Vector, Conv2D |

## 算子四类聚合规则

| 分类 | 聚合规则 |
|------|---------|
| **FA** | 名称含 `attn`, `FlashAttention`, `SDPA`, `fused_attn` |
| **MatMul** | 名称含 `MatMul`, `Linear`, `GEMM`, `DequantGEMM` |
| **Vector** | 名称含 `GELU`, `SiLU`, `ReLU`, `Norm`, `Add`, `Mul`, `Div`, `Reshape` |
| **Comm** | task_type = HCCL，名称含 `all_gather`, `all_reduce`, `reduce_scatter`, `broadcast` |

## 阶段 × 分类 瓶颈矩阵

| 阶段 | FA 占比高 | MatMul 占比高 | Vector 占比高 | Comm 多 |
|------|:--:|:--:|:--:|:--:|
| **DiT** | FA 量化+稀疏 | MatMul 量化 | 编译融合 | RSP 掩盖 |
| **VAE** | 通常无 | ACLGraph | 编译优化 | 通常无 |

## NPU 已知问题速查

| 问题 | 算子 | 硬件 | 影响 |
|------|------|------|------|
| GE error 4294967295 | Conv2D (ResBlock) | 910B | VAE 部分不可用 |
| `expandable_segments:True` 误判 OOM | allocator | 910B | 移除此配置后恢复 |
| triton vs triton-ascend 包名混淆 | triton | 全部 | `import triton` 成功但 0 active drivers |
