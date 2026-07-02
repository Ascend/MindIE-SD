# Core Acceleration APIs

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:14:55.274Z pushedAt=2026-06-08T06:15:16.010Z -->

This document describes the interfaces exposed by the `mindiesd` package through the `layers` module. All interfaces can be directly imported and used via `from mindiesd import <API Name>`.

## Flash Attention (FA) Series

The FA series interfaces provide Ascend affinity attention operations, supporting standard attention, variable-length sequence attention, and sparse attention scenarios.

| API                        | Type     | Function Description                                                               |
| -------------------------- | -------- | ---------------------------------------------------------------------------------- |
| `attention_forward`        | Function | Standard attention forward calculation, supporting automatic operator optimization |
| `attention_forward_varlen` | Function | Variable-length sequence attention forward calculation                             |
| `sparse_attention`         | Function | Sparse attention forward calculation, supporting rf_v2 / ada_bsa sparse strategies |

### `attention_forward`

Standard forward interface for attention computation, supporting multiple underlying operators (PFA, FASCore, LaserAttention, etc.) and automatic operator tuning.

```python
from mindiesd import attention_forward
```

#### Function Signature

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

#### Parameter Description

| Parameter         | Type           | Required | Default Value        | Description                                                                                                                              |
| ----------------- | -------------- | -------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `query`           | `torch.Tensor` | Yes      | -                    | Query tensor, 4D, with layout `[B,S,N,D]` or `[B,N,S,D]`                                                                                 |
| `key`             | `torch.Tensor` | Yes      | -                    | Key tensor, 4D, with the same layout as `query`                                                                                          |
| `value`           | `torch.Tensor` | Yes      | -                    | Value tensor, 4D, with the same layout as `query`                                                                                        |
| `attn_mask`       | `torch.Tensor` | No       | `None`               | Attention mask                                                                                                                           |
| `scale`           | `float`        | No       | `None`               | Scaling factor. If `None`, defaults to `head_dim ** -0.5`                                                                                |
| `fused`           | `bool`         | No       | `True`               | Whether to use the fused operator. If `False`, falls back to native computation.                                                         |
| `head_first`      | `bool`         | No       | `False`              | Whether the head dimension precedes the sequence dimension. If `True`, shape is `[B, N, S, D]`; if `False`, shape is `[B, S, N, D]`      |
| `kwargs.opt_mode` | `str`          | No       | `"runtime"`          | Operator scheduling mode. Supports `"runtime"`, `"static"`, `"manual"`                                                                   |
| `kwargs.op_type`  | `str`          | No       | `"fused_attn_score"` | Operator type. Only effective when `opt_mode="manual"`. Supports `"prompt_flash_attn"`, `"fused_attn_score"`, `"ascend_laser_attention"` |
| `kwargs.layout`   | `str`          | No       | `"BNSD"`             | Operator layout. Only effective when `opt_mode="manual"`. Supports `"BNSD"`, `"BSND"`, `"BSH"`                                           |

#### Return Value

`torch.Tensor`: The result of attention computation, with the same layout as the input.

#### Example

```python
import torch
from mindiesd import attention_forward

query = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
key = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
value = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)

out = attention_forward(query, key, value)
```

#### Migration Guide

- When migrating from `torch.nn.functional.scaled_dot_product_attention`, change input layout from `[B, N, S, D]` to `[B, S, N, D]` and remove the `transpose` operation.

- When migrating from `flash_attn.flash_attn_func`, the input layout is already `[B, S, N, D]` and can be used directly as a drop-in replacement.

- This API only provides forward inference and does not support backward gradient computation. When migrating, remove `dropout` and set the `requires_grad` of the input tensors to `False`.

### `attention_forward_varlen`

Variable-length sequence attention forward computation API, suitable for scenarios where sequence lengths within the same batch are inconsistent.

```python
from mindiesd import attention_forward_varlen
```

#### Function Signature

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

#### Parameter Description

| Parameter           | Type           | Required | Default Value | Description                                                                                      |
| ------------------- | -------------- | -------- | ------------- | ------------------------------------------------------------------------------------------------ |
| `q`                 | `torch.Tensor` | Yes      | -             | Query tensor, 3D, with layout `[T, N, D]` (T is the total number of tokens across all sequences) |
| `k`                 | `torch.Tensor` | Yes      | -             | Key tensor, 3D, with layout `[T, N, D]`                                                          |
| `v`                 | `torch.Tensor` | Yes      | -             | Value tensor, 3D, with layout `[T, N, D]`                                                        |
| `cu_seqlens_q`      | `torch.Tensor` | Yes      | -             | Cumulative lengths of query sequences, 1D tensor, shape `(batch_size + 1,)`, dtype `torch.int32` |
| `cu_seqlens_k`      | `torch.Tensor` | Yes      | -             | Cumulative lengths of key sequences, 1D tensor, shape `(batch_size + 1,)`, dtype `torch.int32`   |
| `max_seqlen_q`      | `int`          | No       | `None`        | Reserved                                                                                         |
| `max_seqlen_k`      | `int`          | No       | `None`        | Reserved                                                                                         |
| `dropout_p`         | `float`        | No       | `0.0`         | Dropout probability, currently only supports `0.0`                                               |
| `softmax_scale`     | `float`        | No       | `None`        | Scaling factor. If `None`, defaults to `head_dim ** -0.5`                                        |
| `causal`            | `bool`         | No       | `False`       | Whether to use causal attention mask                                                             |
| `window_size`       | `int`          | No       | `None`        | Reserved                                                                                         |
| `softcap`           | `float`        | No       | `None`        | Reserved                                                                                         |
| `alibi_slopes`      | `torch.Tensor` | No       | `None`        | Reserved                                                                                         |
| `deterministic`     | `bool`         | No       | `None`        | Reserved                                                                                         |
| `return_attn_probs` | `bool`         | No       | `None`        | Reserved                                                                                         |
| `block_table`       | `torch.Tensor` | No       | `None`        | Reserved                                                                                         |

#### Return Value

`torch.Tensor`: Attention computation result, shape `(total, nheads, headdim)`.

#### Example

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

#### Migration Guide

- When migrating from `flash_attn.flash_attn_varlen_func`, the API parameters are basically the same, allowing for a drop-in replacement.

### `sparse_attention`

Sparse Attention forward computation API, supporting two sparse strategies: RainFusion (rf_v2/rf_v3) and adaptive block sparse attention (ada_bsa).

```python
from mindiesd import sparse_attention
```

#### Function Signature

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

#### Parameter Description

| Parameter        | Type           | Required | Default Value | Description                                                                                                   |
| ---------------- | -------------- | -------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| `q`              | `torch.Tensor` | Yes      | -             | Query tensor, 4D, layout determined by `input_layout`                                                         |
| `k`              | `torch.Tensor` | Yes      | -             | Key tensor, 4D, layout determined by `input_layout`                                                           |
| `v`              | `torch.Tensor` | Yes      | -             | Value tensor, 4D, layout determined by `input_layout`                                                         |
| `attn_mask`      | `torch.Tensor` | No       | `None`        | Reserved, attention mask                                                                                      |
| `scale`          | `float`        | No       | `None`        | Scaling factor. If `None`, defaults to `head_dim ** -0.5`                                                     |
| `is_causal`      | `bool`         | No       | `False`       | Whether to use causal attention mask                                                                          |
| `head_num`       | `int`          | No       | `1`           | Number of attention heads                                                                                     |
| `input_layout`   | `str`          | No       | `"BNSD"`      | Tensor layout, supports `"BNSD"` or `"BSND"`                                                                  |
| `inner_precise`  | `int`          | No       | `0`           | Precision mode, `0` for high precision or `1` for high performance                                            |
| `sparse_type`    | `str`          | No       | `None`        | Sparse type, supports `None`, `"rf_v2"`, `"rf_v3"`, or `"ada_bsa"`                                            |
| `txt_len`        | `int`          | No       | `0`           | Text sequence length, only takes effect when `sparse_type="rf_v2"`                                            |
| `block_size`     | `int`          | No       | `128`         | Block size, currently only supports `128`                                                                     |
| `latent_shape_q` | `list`         | No       | `None`        | Latent space shape of the query `[t, h, w]` (`t*h*w = qseqlen`), only takes effect when `sparse_type="rf_v2"` |
| `latent_shape_k` | `list`         | No       | `None`        | Latent space shape of the key `[t, h, w]` (`t*h*w = kseqlen`), only takes effect when `sparse_type="rf_v2"`   |
| `keep_sink`      | `bool`         | No       | `True`        | Whether to keep sink token, only takes effect when `sparse_type="ada_bsa"`                                    |
| `keep_recent`    | `bool`         | No       | `True`        | Whether to keep recent token, only takes effect when `sparse_type="ada_bsa"`                                  |
| `cdf_threshold`  | `float`        | No       | `1.0`         | CDF threshold, only takes effect when `sparse_type="ada_bsa"`                                                 |
| `sparsity`       | `float`        | No       | `0.0`         | Sparsity rate, value range `[0, 1]`, `0` means no sparse algorithm is used                                    |

#### Return Value

`torch.Tensor`: The result of the attention computation, with the same layout as the input.

#### Example

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

## Fused Operator

The fusion operator series APIs provide Ascend high-performance fusion operators, covering basic computations such as position encoding, normalization, and activation functions.

| API                         | Type     | Function Description                                                     |
| --------------------------- | -------- | ------------------------------------------------------------------------ |
| `rotary_position_embedding` | Function | Rotary position embedding (RoPE) fused operator                          |
| `RMSNorm`                   | Class    | RMS normalization fused operator                                         |
| `fast_layernorm`            | Function | High-performance LayerNorm fused operator                                |
| `layernorm_scale_shift`     | Function | Adaptive LayerNorm (AdaLayerNorm) fused operator                         |
| `get_activation_layer`      | Function | Gets an activation function instance (including NPU accelerated version) |

### `rotary_position_embedding`

RoPE fused operator: Injects positional information into query and key tensors via rotation matrices.

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

#### Parameter Description

| Parameter      | Type           | Required | Default Value    | Description                                                                                         |
| -------------- | -------------- | -------- | ---------------- | --------------------------------------------------------------------------------------------------- |
| `x`            | `torch.Tensor` | Yes      | -                | Query or key tensor, 4D, supports layouts `[B,N,S,D]`, `[B,S,N,D]`, `[S,B,N,D]`                     |
| `cos`          | `torch.Tensor` | Yes      | -                | Precomputed cosine frequency tensor, 2D `[S,D]` or 4D `[1,1,S,D]`/`[1,S,1,D]`/`[S,1,1,D]`           |
| `sin`          | `torch.Tensor` | Yes      | -                | Precomputed sine frequency tensor, dimensions consistent with `cos`                                 |
| `rotated_mode` | `str`          | No       | `"rotated_half"` | Rotation mode: `"rotated_half"` for half rotation, `"rotated_interleaved"` for interleaved rotation |
| `head_first`   | `bool`         | No       | `False`          | Whether the head dimension precedes the sequence dimension                                          |
| `fused`        | `bool`         | No       | `True`           | Whether to use fused operator                                                                       |

#### Return Value

`torch.Tensor`: Tensor with rotary position encoding applied, shape identical to the input `x`.

#### Example

```python
import torch
from mindiesd import rotary_position_embedding

x = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
cos = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)
sin = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)

out = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
```

#### Rotation Mode Description

- **rotated_half**: Applicable to models such as OpenSoraPlan and Stable Audio, splits `x` into front and back halves for rotation.

- **rotated_interleaved**: Applicable to models such as HunyuanDiT, OpenSora, Flux, and CogVideox, rotates `x` by interleaving adjacent elements.

### `RMSNorm`

The RMS normalization fused operator, equivalent to T5LayerNorm, omits mean calculation and focuses solely on the RMS value of the input tensor.

```python
from mindiesd import RMSNorm
```

#### Class Signature

```python
RMSNorm(hidden_size, eps=1e-6)
```

#### Construction Parameter

| Parameter     | Type    | Required | Default Value | Description                   |
| ------------- | ------- | -------- | ------------- | ----------------------------- |
| `hidden_size` | `int`   | Yes      | -             | Hidden layer dimension size   |
| `eps`         | `float` | No       | `1e-6`        | Numerical stability parameter |

#### `forward`

```python
forward(hidden_states, if_fused=True) -> torch.Tensor
```

| Parameter       | Type           | Required | Default Value | Description                                       |
| --------------- | -------------- | -------- | ------------- | ------------------------------------------------- |
| `hidden_states` | `torch.Tensor` | Yes      | -             | Input Tensor, with dimensions ranging from 2 to 8 |
| `if_fused`      | `bool`         | No       | `True`        | Whether to use NPU fused operator                 |

#### Example

```python
import torch
from mindiesd import RMSNorm

norm = RMSNorm(1024, eps=1e-6)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
out = norm(x)
```

### `fast_layernorm`

High-performance LayerNorm fused operator with support for multiple precision modes.

```python
from mindiesd import fast_layernorm
```

#### Function Signature

```python
fast_layernorm(
    norm, x,
    impl_mode=0,
    fused=True
) -> torch.Tensor
```

#### Parameter Description

| Parameter   | Type                 | Required | Default Value | Description                                                                                                                  |
| ----------- | -------------------- | -------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `norm`      | `torch.nn.LayerNorm` | Yes      | -             | PyTorch LayerNorm instance                                                                                                   |
| `x`         | `torch.Tensor`       | Yes      | -             | Input tensor, 3D, layout `[B,S,H]`                                                                                           |
| `impl_mode` | `int`                | No       | `0`           | Computation mode: `0` high precision, `1` high performance, or `2` float16 mode (available only when all inputs are float16) |
| `fused`     | `bool`               | No       | `True`        | Whether to use fused operator, falls back to standard `torch.nn.LayerNorm` when `False`                                      |

#### Return Value

`torch.Tensor`: LayerNorm computation result, with the same shape as the input `x`.

#### Example

```python
import torch
import torch.nn as nn
from mindiesd import fast_layernorm

norm = nn.LayerNorm(1024, eps=1e-5)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)

out = fast_layernorm(norm, x, impl_mode=0, fused=True)
```

### `layernorm_scale_shift`

AdaLayerNorm is a fused operator that adds adaptive scaling and shifting to LayerNorm.

Formula: `out = layernorm(x) * (1 + scale) + shift`

```python
from mindiesd import layernorm_scale_shift
```

#### Function Signature

```python
layernorm_scale_shift(
    layernorm, x, scale, shift,
    fused=True
) -> torch.Tensor
```

#### Parameter Description

| Parameter   | Type                 | Required | Default Value | Description                                             |
| ----------- | -------------------- | -------- | ------------- | ------------------------------------------------------- |
| `layernorm` | `torch.nn.LayerNorm` | Yes      | -             | PyTorch LayerNorm instance                              |
| `x`         | `torch.Tensor`       | Yes      | -             | Input tensor, 3D, layout `[B,S,H]`                      |
| `scale`     | `torch.Tensor`       | Yes      | -             | Adaptive scaling parameter, 2D `[B,H]` or 3D `[B,1,H]`  |
| `shift`     | `torch.Tensor`       | Yes      | -             | Adaptive shifting parameter, 2D `[B,H]` or 3D `[B,1,H]` |
| `fused`     | `bool`               | No       | `True`        | Whether to use fused operator                           |

#### Return Value

`torch.Tensor`: AdaLayerNorm computation result, with the same shape as the input `x`.

#### Example

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

#### Constraints

- The last dimension of `x` must be equal to the last dimension of `scale` and `shift`.

- If `scale` or `shift` is a 3D tensor, the second dimension must be 1 or equal to the second dimension (sequence length) of `x`.

### `get_activation_layer`

Obtains an activation function instance of the specified type. NPU-accelerated versions are available for some activation functions.

```python
from mindiesd import get_activation_layer
```

#### Function Signature

```python
get_activation_layer(act_type: str) -> nn.Module
```

#### Parameter Description

| Parameter  | Type  | Required | Default Value | Description                                       |
| ---------- | ----- | -------- | ------------- | ------------------------------------------------- |
| `act_type` | `str` | Yes      | -             | Name of the activation function, case-insensitive |

#### Supported Activation Functions

| Name          | Corresponding Implementation | Description                                                 |
| ------------- | ---------------------------- | ----------------------------------------------------------- |
| `"swish"`     | `nn.SiLU`                    | Swish activation function                                   |
| `"silu"`      | `nn.SiLU`                    | SiLU activation function (equivalent to swish)              |
| `"mish"`      | `nn.Mish`                    | Mish activation function                                    |
| `"gelu"`      | `GELU`                       | Standard GELU                                               |
| `"relu"`      | `nn.ReLU`                    | ReLU activation function                                    |
| `"gelu-tanh"` | `GELU(approximate="tanh")`   | tanh, an approximation of the GELU                          |
| `"gelu-fast"` | `GELU(approximate="fast")`   | Fast GELU, accelerated using NPU's `npu_fast_gelu` operator |

#### Return Value

`nn.Module`: An instance of the corresponding activation function.

#### Example

```python
from mindiesd import get_activation_layer

act = get_activation_layer("gelu-fast")
out = act(hidden_states)
```

### Linear

A custom linear layer that is consistent with PyTorch's `nn.Linear` in usage, but adds an `op_type` parameter for selecting the underlying operator implementation.

```python
from mindiesd import Linear
```

#### Construction Parameter

| Parameter      | Type          | Required | Default Value | Description                                                                   |
| -------------- | ------------- | -------- | ------------- | ----------------------------------------------------------------------------- |
| `in_features`  | `int`         | Yes      | -             | Input feature dimension                                                       |
| `out_features` | `int`         | Yes      | -             | Output feature dimension                                                      |
| `bias`         | `bool`        | No       | `True`        | Whether to use bias                                                           |
| `device`       | `str`         | No       | `None`        | Weight storage device                                                         |
| `dtype`        | `torch.dtype` | No       | `None`        | Weight data type                                                              |
| `op_type`      | `str`         | No       | `"matmulv2"`  | Operator type, supports `"matmulv2"`, `"batchmatmulv2"`, or `"batchmatmulv3"` |

#### `forward`

```python
forward(input) -> torch.Tensor
```

| Parameter | Type           | Required | Description                                                     |
| --------- | -------------- | -------- | --------------------------------------------------------------- |
| `input`   | `torch.Tensor` | Yes      | Input Tensor, the last dimension must be equal to `in_features` |
