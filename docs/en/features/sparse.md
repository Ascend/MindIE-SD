# Sparse

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:15:36.322Z pushedAt=2026-06-08T09:04:08.644Z -->

## Sparse Attention Overview

Attention is the primary computational bottleneck in DiT-like models. During inference, the attention score matrix between Q and K contains substantial redundancy—many token pairs have negligible correlation and contribute little to the final output. Sparse attention addresses this by skipping such unimportant computations and retaining only critical token–token interactions, thereby reducing both computation and inference latency.

The two core challenges in implementing sparse attention are:

1. **Determining what to skip** – how the sparse mask is generated.

2. **Ensuring hardware speedups** – whether the sparse pattern aligns with hardware compute units.

## Technical Features

This repository provides the following sparsity schemes through the `sparse_attention` interface.

### RainFusion2.0 (`rf_v2`)

RainFusion2.0 is an online adaptive block-sparse attention scheme that addresses the above challenges through the following three techniques:

**Block Representative Token Prediction** (Addressing "What to Skip")

Instead of computing the full attention score matrix to generate a sparse mask, Q/K are partitioned into blocks according to their spatial shape, and the mean of each block is taken as a representative token. The sparse mask is then predicted based on the similarity between these representative tokens, significantly reducing the overhead of mask prediction.

**Spatiotemporal-Aware Token Reordering** (Addressing "How to Align")

In video, tokens at the same spatial position across adjacent frames are highly similar, but when flattened in raster order, they become far apart—breaking the block-wise self-similarity. RainFusion2.0 reorganizes tokens into 3D windows of shape `[t, h, w]`, making tokens within each block more similar and improving both sparse mask hit rates and hardware efficiency.

**First-Frame Sink Mechanism**

In video generation models, the first-frame token plays a decisive role in final output quality, similar to the attention sink phenomenon in LLMs. RainFusion2.0 enforces full attention computation involving the first frame, preserving generation quality with negligible degradation even at an 80% sparsity ratio.

These three techniques together enable RainFusion2.0 to achieve 1.5–1.8× end-to-end speedup on Ascend NPU at 80% sparsity, while maintaining generation quality comparable to full attention.

For details, see the [RainFusion2.0 Technical Report](../../tech_report/RainFusion2.0.pdf).

### Adaptive Block Sparse Attention (`ada_bsa`)

Dynamically estimate sparse block sets via a CDF threshold, ideal for scenarios requiring flexible sparsity granularity.

**Recommended Solution:**

- **`rf_v2` (preferred)**: Provides end-to-end acceleration of 1.5–1.8× with virtually no quality loss, covering both image and video models.

- **`ada_bsa` (alternative)**: Use only when `rf_v2` is incompatible with the model.

- **Default sparsity recommendation**: Start from 0.6 for image tasks and 0.8 for video tasks, then fine-tune based on generation quality.

## Interface Description

Sparse Attention is provided externally through the `sparse_attention` interface. For complete parameter descriptions, see [sparse_attention section in core_layers.md](core_layers.md#sparse_attention).

The basic calling method is as follows:

```python
from mindiesd import sparse_attention

out = sparse_attention(q, k, v, head_num=24, input_layout="BNSD", sparse_type="rf_v2", sparsity=0.8)
```

### Quick Parameter Reference

| Parameter | Required (Yes/No) | Description |
|------|------|------|
| `sparse_type` | No | Sparse strategy: `None` (full attention), `"rf_v2"`, or `"ada_bsa"` |
| `sparsity` | No | Sparsity ratio, value range `[0, 1]`, `0` means no sparsity |
| `txt_len` | No | Text token length, effective only when `sparse_type="rf_v2"` |
| `latent_shape_q` | No | Query latent space shape `[t, h, w]`, effective only when `sparse_type="rf_v2"` |
| `latent_shape_k` | No | Key latent space shape `[t, h, w]`, effective only when `sparse_type="rf_v2"` |
| `keep_sink` | No | Whether to retain sink token, effective only when `sparse_type="ada_bsa"` |
| `keep_recent` | No | Whether to retain recent token, effective only when `sparse_type="ada_bsa"` |
| `cdf_threshold` | No | CDF threshold, effective only when `sparse_type="ada_bsa"` |

### Usage Example

Image model:

```python
import torch
from mindiesd import sparse_attention

q = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
k = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
v = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)

out = sparse_attention(q, k, v, head_num=24, input_layout="BNSD", sparse_type="rf_v2", sparsity=0.6)
```

Video model:

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

### Precautions

- The sparsity rate controls the trade-off between speedup and generation quality. Based on experimental results, `sparsity=0.8` yields an end-to-end speedup of 1.5–1.8× with quality comparable to full attention. For image tasks, start tuning from `0.6`; for video tasks, start from `0.8`.

- The `block_size` parameter currently only supports `128`.

- This interface supports inference only and does not compute backward gradients.
