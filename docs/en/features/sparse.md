# Sparse

## Sparse Attention Overview

The attention mechanism is the core computational bottleneck in DiT-class models. During inference, the Q-K attention score matrix contains significant redundant computation --- some token pairs have extremely low correlation and contribute almost nothing to the output. The basic idea of sparse attention is to skip these unimportant computations, retaining only the attention interactions between critical token pairs, thereby reducing computation and inference latency.

There are two core challenges in implementing sparse attention:

1. **How to determine which computations can be skipped** --- i.e., the sparse mask generation method.
2. **How to achieve real hardware acceleration from skipping computations** --- i.e., whether the sparse pattern aligns with hardware compute units.

## Technical Features

This repository provides the following sparse approaches through the `sparse_attention` API.

### rf_v2 (RainFusion2.0)

RainFusion2.0 is an online adaptive block-sparse attention scheme that addresses the above challenges through three techniques:

**Block Representative Token Prediction** (Solving "How To Determine")

Instead of computing the full attention score matrix to generate sparse masks, Q and K are partitioned into blocks by spatial shape, and the mean of each block is taken as a representative token. Sparse masks are predicted through similarity between representative tokens, significantly reducing mask prediction overhead.

**Spatio-Temporal Token Reordering** (Solving "How To Align")

Tokens at the same spatial position in adjacent video frames are highly similar, but become far apart after raster-scan flattening, breaking block self-similarity. RainFusion2.0 reorders tokens by `[t, h, w]` 3D windows, making tokens within blocks more similar and improving sparse mask hit rate and hardware efficiency.

**First-Frame Sink Mechanism**

In video generation models, first-frame tokens have a decisive impact on final generation quality (similar to the attention sink phenomenon in LLMs). RainFusion2.0 forces the first frame to participate in full attention computation, maintaining near-lossless generation quality at 80% sparsity.

These three techniques together enable RainFusion2.0 to achieve 1.5--1.8x end-to-end acceleration at 80% sparsity on Ascend NPU, with generation quality metrics nearly matching full attention.

For detailed technical description, see the [RainFusion2.0 Technical Report](../../tech_report/RainFusion2.0.pdf).

<!-- npu="950" id1 -->
### rf_v3 (RainFusion3.0 / BSA)

A block sparse attention solution for 950PR&950DT products. Mask generation and spatio-temporal reordering reuse rf_v2's strategy (representative token prediction, spatio-temporal aware reordering, and first-frame protection). On the operator side, block sparse computation is performed by the `aclnnBlockSparseAttention` kernel family, and the kernel version is probed as V3 -> V2 -> V1 at load time with fallback. The `precision` parameter supports four execution precisions:

- `"bf16"` (default): no quantization, plain BF16 sparse attention.
- `"mix"`: EagleQBSA mixed precision (block-wise INT8 for Q/K plus per-channel FP8 for V).
- `"fp8"`: apply Hadamard rotation to Q/K first to spread out outliers, then block-quantize Q/K/V to FP8 (requires V2 or later kernels).
- `"mxfp4"`: takes the MXFP4 path of `aclnnBlockSparseAttentionV3` -- Q/K are Hadamard-rotated and then quantized to FP4 E2M1 data with E8M0 scales (grouped by 32 elements along the head dimension for Q/K and along the sequence dimension for V). `mxfp4_dst_type_max` selects the range policy: `0` for OCP (quantMode=2, floor), a positive value (6 to 12 in practice) for CX (quantMode=3, ceil, with 7.25 recommended for range calibration). Supported only on 950PR&950DT products.

For API parameters and usage examples, see the [sparse_attention section in core_layers.md](core_layers.md#sparse_attention).
<!-- end id1 -->

### ada_bsa (Adaptive Block Sparse Attention)

Dynamically estimates the sparse block set through CDF thresholding, suitable for scenarios requiring flexible sparsity granularity.

**Recommended approach:**

- **Prefer rf_v2 (RainFusion2.0)**: 1.5--1.8x end-to-end acceleration, near-lossless quality, covering image and video models.
<!-- npu="950" id2 -->
- **Use rf_v3 on 950PR&950DT products**: rf_v2 automatically routes to rf_v3 on 950PR&950DT products; select `precision="fp8"` or `"mxfp4"` when quantized acceleration is needed.
<!-- end id2 -->
- **ada_bsa as fallback**: Try when rf_v2 does not meet model compatibility requirements.
- **Default sparsity suggestion**: Start from 0.6 for image tasks, 0.8 for video tasks, fine-tune based on generation quality.

## API Reference

Sparse attention is exposed through the `sparse_attention` API. For complete parameter descriptions, see the [sparse_attention section in Core Acceleration API](core_layers.md#sparse_attention).

Basic usage:

```python
from mindiesd import sparse_attention

out = sparse_attention(q, k, v, head_num=24, input_layout="BNSD", sparse_type="rf_v2", sparsity=0.8)
```

### Quick Parameter Reference

| Parameter | Required | Description |
| ------ | ------ | ------ |
| `sparse_type` | No | Sparse strategy: `None` (full attention), `"rf_v2"`, `"ada_bsa"` |
| `sparsity` | No | Sparsity rate, range `[0, 1]`, `0` means no sparsification |
| `txt_len` | No | Text token length, only effective when `sparse_type="rf_v2"` |
| `latent_shape_q` | No | Query latent space shape `[t, h, w]`, only effective when `sparse_type="rf_v2"` |
| `latent_shape_k` | No | Key latent space shape `[t, h, w]`, only effective when `sparse_type="rf_v2"` |
| `keep_sink` | No | Whether to keep sink tokens, only effective when `sparse_type="ada_bsa"` |
| `keep_recent` | No | Whether to keep recent tokens, only effective when `sparse_type="ada_bsa"` |
| `cdf_threshold` | No | CDF threshold, only effective when `sparse_type="ada_bsa"` |

### Usage Examples

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

### Notes

- The sparsity rate requires balancing acceleration and generation quality. Reference experimental data: 1.5–1.8x end-to-end acceleration at `sparsity=0.8` with quality metrics nearly matching full attention; start debugging at 0.6 for image tasks and 0.8 for video tasks.
- The `block_size` parameter currently only supports 128.
- This API only provides forward inference; backward gradient computation is not supported.
