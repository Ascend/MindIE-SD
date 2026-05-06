# Layers Module API Reference

This document describes the public interfaces exposed by the `layers` module in the `mindiesd` package. All interfaces can be imported directly via `from mindiesd import <interface_name>`.

## Interface Overview

### FA Series

The FA (Flash Attention) series provides Ascend-optimized attention computation capabilities, covering standard attention, variable-length sequence attention, and sparse attention scenarios.

| Interface | Type | Description |
|-----------|------|-------------|
| `attention_forward` | Function | Standard attention forward computation with automatic operator selection |
| `attention_forward_varlen` | Function | Variable-length sequence attention forward computation |
| `sparse_attention` | Function | Sparse attention forward computation, supporting rf_v2 / ada_bsa strategies |

### Fused Operator Series

The fused operator series provides high-performance Ascend fused operators, covering position encoding, normalization, and activation functions.

| Interface | Type | Description |
|-----------|------|-------------|
| `rotary_position_embedding` | Function | Rotary Position Embedding (RoPE) fused operator |
| `RMSNorm` | Class | RMS normalization fused operator |
| `fast_layernorm` | Function | High-performance LayerNorm fused operator |
| `layernorm_scale_shift` | Function | Adaptive LayerNorm (AdaLayerNorm) fused operator |
| `get_activation_layer` | Function | Get an activation function instance (includes NPU-accelerated variants) |

---

# FA Series

## attention_forward

Standard attention forward computation interface, supporting multiple underlying operators (PFA, FASCore, LaserAttention, etc.) and automatic tuning.

```python
from mindiesd import attention_forward
```

### Function Signature

```python
attention_forward(
    query, key, value,
    attn_mask=None,
    scale=None,
    fused=True,
    head_first=False,
    **kwargs
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `torch.Tensor` | Yes | - | Query tensor, 4D, layout `[B,S,N,D]` or `[B,N,S,D]` |
| `key` | `torch.Tensor` | Yes | - | Key tensor, 4D, same layout as `query` |
| `value` | `torch.Tensor` | Yes | - | Value tensor, 4D, same layout as `query` |
| `attn_mask` | `torch.Tensor` | No | `None` | Attention mask |
| `scale` | `float` | No | `None` | Scale factor; defaults to `head_dim ** -0.5` when `None` |
| `fused` | `bool` | No | `True` | Whether to use fused operators; falls back to native computation when `False` |
| `head_first` | `bool` | No | `False` | Whether the head dimension precedes the sequence dimension; `True` means `[B,N,S,D]`, `False` means `[B,S,N,D]` |
| `kwargs.opt_mode` | `str` | No | `"runtime"` | Operator dispatch mode: `"runtime"`, `"static"`, or `"manual"` |
| `kwargs.op_type` | `str` | No | `"fused_attn_score"` | Operator type, only effective when `opt_mode="manual"`; supports `"prompt_flash_attn"`, `"fused_attn_score"`, `"ascend_laser_attention"` |
| `kwargs.layout` | `str` | No | `"BNSD"` | Operator layout, only effective when `opt_mode="manual"`; supports `"BNSD"`, `"BSND"`, `"BSH"` |

### Returns

`torch.Tensor`: Attention computation result, same layout as input.

### Example

```python
import torch
from mindiesd import attention_forward

query = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
key = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
value = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)

out = attention_forward(query, key, value)
```

### Migration Guide

- When migrating from `torch.nn.functional.scaled_dot_product_attention`, adjust the input layout from `[B,N,S,D]` to `[B,S,N,D]` and remove `transpose` operations.
- When migrating from `flash_attn.flash_attn_func`, the input layout is already `[B,S,N,D]` and can be replaced directly.
- This interface provides forward inference only and does not support backward gradient computation. Remove `dropout` and set `requires_grad=False` on input tensors when migrating.

---

## attention_forward_varlen

Variable-length sequence attention forward computation interface, suitable for scenarios where sequence lengths vary within a batch.

```python
from mindiesd import attention_forward_varlen
```

### Function Signature

```python
attention_forward_varlen(
    q, k, v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=None,
    alibi_slopes=None,
    deterministic=None,
    return_attn_probs=None,
    block_table=None
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | `torch.Tensor` | Yes | - | Query tensor, 3D, layout `[T, N, D]` (T is total token count across all sequences) |
| `k` | `torch.Tensor` | Yes | - | Key tensor, 3D, layout `[T, N, D]` |
| `v` | `torch.Tensor` | Yes | - | Value tensor, 3D, layout `[T, N, D]` |
| `cu_seqlens_q` | `list[torch.Tensor]` | Yes | - | Cumulative sequence lengths for query, shape `(batch_size + 1,)`, dtype `torch.int32` |
| `cu_seqlens_k` | `list[torch.Tensor]` | Yes | - | Cumulative sequence lengths for key, shape `(batch_size + 1,)`, dtype `torch.int32` |
| `max_seqlen_q` | `int` | No | `None` | Reserved parameter |
| `max_seqlen_k` | `int` | No | `None` | Reserved parameter |
| `dropout_p` | `float` | No | `0.0` | Dropout probability; currently only `0.0` is supported |
| `softmax_scale` | `float` | No | `None` | Scale factor; defaults to `head_dim ** -0.5` when `None` |
| `causal` | `bool` | No | `False` | Whether to apply causal attention mask |
| `window_size` | `int` | No | `None` | Reserved parameter |
| `softcap` | `float` | No | `None` | Reserved parameter |
| `alibi_slopes` | `torch.Tensor` | No | `None` | Reserved parameter |
| `deterministic` | `bool` | No | `None` | Reserved parameter |
| `return_attn_probs` | `bool` | No | `None` | Reserved parameter |
| `block_table` | `torch.Tensor` | No | `None` | Reserved parameter |

### Returns

`torch.Tensor`: Attention computation result, shape `(total, nheads, headdim)`.

### Example

```python
import torch
from mindiesd import attention_forward_varlen

q = torch.randn(8192, 24, 128, device="npu", dtype=torch.float16)
k = torch.randn(8192, 24, 128, device="npu", dtype=torch.float16)
v = torch.randn(8192, 24, 128, device="npu", dtype=torch.float16)
cu_seqlens_q = torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device="npu")
cu_seqlens_k = torch.tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32, device="npu")

out = attention_forward_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=False)
```

### Migration Guide

- When migrating from `flash_attn.flash_attn_varlen_func`, the interface parameters are largely compatible and can be replaced directly.

---

## sparse_attention

Sparse attention forward computation interface, supporting two sparse strategies: RainFusion (rf_v2) and Adaptive Block Sparse Attention (ada_bsa).

```python
from mindiesd import sparse_attention
```

### Function Signature

```python
sparse_attention(
    q, k, v,
    attn_mask=None,
    scale=None,
    is_causal=False,
    head_num=1,
    input_layout="BNSD",
    inner_precise=0,
    sparse_type=None,
    txt_len=0,
    block_size=128,
    latent_shape_q=None,
    latent_shape_k=None,
    keep_sink=True,
    keep_recent=True,
    cdf_threshold=1.0,
    sparsity=0.0,
    **kwargs
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `q` | `torch.Tensor` | Yes | - | Query tensor, 4D, layout determined by `input_layout` |
| `k` | `torch.Tensor` | Yes | - | Key tensor, 4D, layout determined by `input_layout` |
| `v` | `torch.Tensor` | Yes | - | Value tensor, 4D, layout determined by `input_layout` |
| `attn_mask` | `torch.Tensor` | No | `None` | Attention mask, reserved parameter |
| `scale` | `float` | No | `None` | Scale factor; defaults to `head_dim ** -0.5` when `None` |
| `is_causal` | `bool` | No | `False` | Whether to apply causal attention mask |
| `head_num` | `int` | No | `1` | Number of attention heads |
| `input_layout` | `str` | No | `"BNSD"` | Tensor layout, supports `"BNSD"` or `"BSND"` |
| `inner_precise` | `int` | No | `0` | Compute precision mode: `0` for high precision, `1` for high performance |
| `sparse_type` | `str` | No | `None` | Sparse type: `None`, `"rf_v2"`, or `"ada_bsa"` |
| `txt_len` | `int` | No | `0` | Text sequence length, only effective when `sparse_type="rf_v2"` |
| `block_size` | `int` | No | `128` | Block size; currently only `128` is supported |
| `latent_shape_q` | `list` | No | `None` | Latent shape for query `[t, h, w]`, `t*h*w = qseqlen`, only effective when `sparse_type="rf_v2"` |
| `latent_shape_k` | `list` | No | `None` | Latent shape for key `[t, h, w]`, `t*h*w = kseqlen`, only effective when `sparse_type="rf_v2"` |
| `keep_sink` | `bool` | No | `True` | Whether to retain sink tokens, only effective when `sparse_type="ada_bsa"` |
| `keep_recent` | `bool` | No | `True` | Whether to retain recent tokens, only effective when `sparse_type="ada_bsa"` |
| `cdf_threshold` | `float` | No | `1.0` | CDF threshold, only effective when `sparse_type="ada_bsa"` |
| `sparsity` | `float` | No | `0.0` | Sparsity ratio, range `[0, 1]`; `0` disables sparse algorithm |

### Returns

`torch.Tensor`: Attention computation result, same layout as input.

### Example

```python
import torch
from mindiesd import sparse_attention

q = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
k = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)
v = torch.randn(2, 24, 4096, 128, device="npu", dtype=torch.float16)

out = sparse_attention(
    q, k, v,
    head_num=24,
    input_layout="BNSD",
    sparse_type="ada_bsa",
    sparsity=0.5
)
```

---

# Fused Operator Series

## rotary_position_embedding

Rotary Position Embedding (RoPE) fused operator, injecting positional information into query and key tensors through rotation matrices.

```python
from mindiesd import rotary_position_embedding
```

### Function Signature

```python
rotary_position_embedding(
    x, cos, sin,
    rotated_mode="rotated_half",
    head_first=False,
    fused=True
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `x` | `torch.Tensor` | Yes | - | Query or key tensor, 4D, supports layouts `[B,N,S,D]`, `[B,S,N,D]`, `[S,B,N,D]` |
| `cos` | `torch.Tensor` | Yes | - | Precomputed cosine frequency tensor, 2D `[S,D]` or 4D `[1,1,S,D]`/`[1,S,1,D]`/`[S,1,1,D]` |
| `sin` | `torch.Tensor` | Yes | - | Precomputed sine frequency tensor, same dimensions as `cos` |
| `rotated_mode` | `str` | No | `"rotated_half"` | Rotation mode: `"rotated_half"` for half rotation, `"rotated_interleaved"` for interleaved rotation |
| `head_first` | `bool` | No | `False` | Whether the head dimension precedes the sequence dimension |
| `fused` | `bool` | No | `True` | Whether to use fused operators |

### Returns

`torch.Tensor`: Tensor with rotary position embeddings applied, same shape as input `x`.

### Example

```python
import torch
from mindiesd import rotary_position_embedding

x = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
cos = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)
sin = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)

out = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
```

### Rotation Mode Description

- **rotated_half**: Suitable for models such as OpenSoraPlan and Stable Audio. Splits `x` into front and back halves for rotation.
- **rotated_interleaved**: Suitable for models such as HunyuanDiT, OpenSora, Flux, and CogVideox. Rotates `x` by interleaving adjacent elements.

---

## RMSNorm

RMS normalization fused operator, equivalent to T5LayerNorm. It avoids explicit mean computation and focuses on the root mean square of the input tensor.

```python
from mindiesd import RMSNorm
```

### Class Signature

```python
RMSNorm(hidden_size, eps=1e-6)
```

### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hidden_size` | `int` | Yes | - | Hidden dimension size |
| `eps` | `float` | No | `1e-6` | Numerical stability parameter |

### forward Method

```python
forward(hidden_states, if_fused=True) -> torch.Tensor
```

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `hidden_states` | `torch.Tensor` | Yes | - | Input tensor, dimension range 2~8 |
| `if_fused` | `bool` | No | `True` | Whether to use NPU fused operators |

### Example

```python
import torch
from mindiesd import RMSNorm

norm = RMSNorm(1024, eps=1e-6)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
out = norm(x)
```

---

## fast_layernorm

High-performance LayerNorm fused operator, supporting multiple compute precision modes.

```python
from mindiesd import fast_layernorm
```

### Function Signature

```python
fast_layernorm(
    norm, x,
    impl_mode=0,
    fused=True
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `norm` | `torch.nn.LayerNorm` | Yes | - | PyTorch LayerNorm instance |
| `x` | `torch.Tensor` | Yes | - | Input tensor, 3D, layout `[B,S,H]` |
| `impl_mode` | `int` | No | `0` | Compute mode: `0` high precision, `1` high performance, `2` float16 mode (only available when all inputs are float16) |
| `fused` | `bool` | No | `True` | Whether to use fused operators; falls back to standard `torch.nn.LayerNorm` when `False` |

### Returns

`torch.Tensor`: LayerNorm computation result, same shape as input `x`.

### Example

```python
import torch
import torch.nn as nn
from mindiesd import fast_layernorm

norm = nn.LayerNorm(1024, eps=1e-5)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)

out = fast_layernorm(norm, x, impl_mode=0, fused=True)
```

---

## layernorm_scale_shift

Adaptive LayerNorm (AdaLayerNorm) fused operator, adding adaptive scaling and shifting on top of LayerNorm.

Computation formula: `out = layernorm(x) * (1 + scale) + shift`

```python
from mindiesd import layernorm_scale_shift
```

### Function Signature

```python
layernorm_scale_shift(
    layernorm, x, scale, shift,
    fused=True
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `layernorm` | `torch.nn.LayerNorm` | Yes | - | PyTorch LayerNorm instance |
| `x` | `torch.Tensor` | Yes | - | Input tensor, 3D, layout `[B,S,H]` |
| `scale` | `torch.Tensor` | Yes | - | Adaptive scaling parameter, 2D `[B,H]` or 3D `[B,1,H]` |
| `shift` | `torch.Tensor` | Yes | - | Adaptive shifting parameter, 2D `[B,H]` or 3D `[B,1,H]` |
| `fused` | `bool` | No | `True` | Whether to use fused operators |

### Returns

`torch.Tensor`: AdaLayerNorm computation result, same shape as input `x`.

### Example

```python
import torch
import torch.nn as nn
from mindiesd import layernorm_scale_shift

norm = nn.LayerNorm(1024, eps=1e-5)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
scale = torch.randn(2, 1024, device="npu", dtype=torch.float16)
shift = torch.randn(2, 1024, device="npu", dtype=torch.float16)

out = layernorm_scale_shift(norm, x, scale, shift, fused=True)
```

### Constraints

- The last dimension of `x` must equal the last dimensions of `scale` and `shift`.
- If `scale` or `shift` is a 3D tensor, the second dimension must be 1.

---

## get_activation_layer

Get an activation function instance by name. Some activation functions provide NPU-accelerated variants.

```python
from mindiesd import get_activation_layer
```

### Function Signature

```python
get_activation_layer(act_type: str) -> nn.Module
```

### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `act_type` | `str` | Yes | - | Activation function name, case-insensitive |

### Supported Activation Functions

| Name | Implementation | Description |
|------|---------------|-------------|
| `"swish"` | `nn.SiLU` | Swish activation function |
| `"silu"` | `nn.SiLU` | SiLU activation function (equivalent to swish) |
| `"mish"` | `nn.Mish` | Mish activation function |
| `"gelu"` | `GELU` | Standard GELU |
| `"relu"` | `nn.ReLU` | ReLU activation function |
| `"gelu-tanh"` | `GELU(approximate="tanh")` | tanh-approximated GELU |
| `"gelu-fast"` | `GELU(approximate="fast")` | Fast GELU, accelerated by NPU `npu_fast_gelu` operator |

### Returns

`nn.Module`: Instance of the requested activation function.

### Example

```python
from mindiesd import get_activation_layer

act = get_activation_layer("gelu-fast")
out = act(hidden_states)
```
