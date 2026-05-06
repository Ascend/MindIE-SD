# 稀疏

## 稀疏注意力概述

注意力机制是 DiT 类模型的核心计算瓶颈。在推理过程中，Q 和 K 的注意力分数矩阵中有大量冗余计算——部分 Token 对之间的相关性极低，对生成结果几乎没有贡献。稀疏注意力的基本思想是：跳过这些不重要的计算，只保留关键 Token 对之间的注意力交互，从而减少计算量，降低推理延迟。

实现稀疏注意力的核心挑战有两个：

1. **如何判断哪些计算可以跳过**——即稀疏掩码的生成方式。
2. **如何让跳过计算带来真实的硬件加速**——即稀疏模式是否与硬件计算单元对齐。

## 技术特点

本仓库通过 `sparse_attention` 接口提供以下稀疏方案。

### rf_v2（RainFusion2.0）

RainFusion2.0 是一种在线自适应的块稀疏注意力方案，通过以下三项技术解决上述挑战：

**块代表 Token 预测**（解决"如何判断"）

不计算完整的注意力分数矩阵来生成稀疏掩码，而是将 Q/K 按空间形状分块，取每块的均值作为代表 Token，通过代表 Token 间的相似度预测稀疏掩码，大幅降低掩码预测开销。

**空时感知 Token 重排**（解决"如何对齐"）

视频中相邻帧的相同空间位置的 Token 高度相似，但按光栅顺序展平后相距很远，破坏了分块自相似性。RainFusion2.0 将 Token 按 `[t, h, w]` 三维窗口重排，使块内 Token 更相似，提升稀疏掩码的命中率和硬件效率。

**首帧 Sink 机制**

视频生成模型中，首帧 Token 对最终生成质量有决定性影响（类似 LLM 中的 attention sink 现象）。RainFusion2.0 强制首帧参与全注意力计算，在 80% 稀疏率下保持生成质量基本无损。

以上三项技术共同使 RainFusion2.0 在昇腾 NPU 上达到 80% 稀疏率下 1.5–1.8× 端到端加速，生成质量指标与全注意力基本持平。

详细技术说明请参见 [RainFusion2.0 技术报告](../../tech_report/RainFusion2.0.pdf)。

### ada_bsa（自适应块稀疏）

通过 CDF 阈值动态估计稀疏块集合，适用于需要灵活调节稀疏粒度的场景。

**推荐方案：**

- **优先使用 rf_v2（RainFusion2.0）**：端到端加速 1.5–1.8×，质量基本无损，覆盖图像和视频模型。
- **ada_bsa 备选**：当 rf_v2 不满足模型兼容性要求时尝试。
- **默认 sparsity 建议**：图像任务从 0.6 起步，视频任务从 0.8 起步，根据生成质量微调。

## 接口说明

稀疏注意力通过 `sparse_attention` 接口对外提供，完整参数说明请参见 [core_layers.md 中的 sparse_attention 章节](core_layers.md#sparse_attention)。

基本调用方式如下：

```python
from mindiesd import sparse_attention

out = sparse_attention(q, k, v, head_num=24, input_layout="BNSD", sparse_type="rf_v2", sparsity=0.8)
```

### 常用参数速查

| 参数 | 必选 | 说明 |
|------|------|------|
| `sparse_type` | 否 | 稀疏策略：`None`（全注意力）、`"rf_v2"`、`"ada_bsa"` |
| `sparsity` | 否 | 稀疏率，取值范围 `[0, 1]`，`0` 表示不稀疏 |
| `txt_len` | 否 | 文本 Token 长度，仅在 `sparse_type="rf_v2"` 时生效 |
| `latent_shape_q` | 否 | Query 潜空间形状 `[t, h, w]`，仅在 `sparse_type="rf_v2"` 时生效 |
| `latent_shape_k` | 否 | Key 潜空间形状 `[t, h, w]`，仅在 `sparse_type="rf_v2"` 时生效 |
| `keep_sink` | 否 | 是否保留 Sink Token，仅在 `sparse_type="ada_bsa"` 时生效 |
| `keep_recent` | 否 | 是否保留 Recent Token，仅在 `sparse_type="ada_bsa"` 时生效 |
| `cdf_threshold` | 否 | CDF 阈值，仅在 `sparse_type="ada_bsa"` 时生效 |

### 使用示例

图像模型：

```python
import torch
from mindiesd import sparse_attention

q = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
k = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
v = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)

out = sparse_attention(q, k, v, head_num=24, input_layout="BNSD", sparse_type="rf_v2", sparsity=0.6)
```

视频模型：

```python
out = sparse_attention(
    q, k, v,
    head_num=24,
    input_layout="BNSD",
    sparse_type="rf_v2",
    sparsity=0.8,
    latent_shape_q=[t, h, w],
    latent_shape_k=[t, h, w],
)
```

### 注意事项

- 稀疏率的取值需要在加速比和生成质量间权衡。参考实验数据：`sparsity=0.8` 时端到端加速 1.5–1.8×，质量指标与全注意力基本持平；建议图像任务从 0.6 开始调试，视频任务从 0.8 开始调试。
- `block_size` 参数当前仅支持 128。
- 本接口仅提供前向推理，不支持反向梯度计算。
