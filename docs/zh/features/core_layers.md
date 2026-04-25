# 核心加速API

本文档描述 `mindiesd` 包通过 `layers` 模块对外暴露的接口。所有接口均可通过 `from mindiesd import <接口名>` 直接导入使用。

## FA 系列

FA（Flash Attention）系列接口提供昇腾亲和的注意力计算能力，涵盖标准注意力、变长序列注意力和稀疏注意力场景。

| 接口名 | 类型 | 功能描述 |
|--------|------|----------|
| `attention_forward` | 函数 | 标准注意力前向计算，支持自动算子寻优 |
| `attention_forward_varlen` | 函数 | 变长序列注意力前向计算 |
| `sparse_attention` | 函数 | 稀疏注意力前向计算，支持 rf_v2 / ada_bsa 稀疏策略 |

### attention_forward

标准注意力前向计算接口，支持多种底层算子（PFA、FASCore、LaserAttention 等）和自动寻优。

```python
from mindiesd import attention_forward
```

#### 函数签名

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

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | `torch.Tensor` | 是 | - | 查询张量，4D，布局为 `[B,S,N,D]` 或 `[B,N,S,D]` |
| `key` | `torch.Tensor` | 是 | - | 键张量，4D，布局与 `query` 一致 |
| `value` | `torch.Tensor` | 是 | - | 值张量，4D，布局与 `query` 一致 |
| `attn_mask` | `torch.Tensor` | 否 | `None` | 注意力掩码 |
| `scale` | `float` | 否 | `None` | 缩放因子，为 `None` 时自动取 `head_dim ** -0.5` |
| `fused` | `bool` | 否 | `True` | 是否使用融合算子，`False` 时回退到原生计算 |
| `head_first` | `bool` | 否 | `False` | 头维度是否在序列维度之前，`True` 表示 `[B,N,S,D]`，`False` 表示 `[B,S,N,D]` |
| `kwargs.opt_mode` | `str` | 否 | `"runtime"` | 算子调度模式，支持 `"runtime"`、`"static"`、`"manual"` |
| `kwargs.op_type` | `str` | 否 | `"fused_attn_score"` | 算子类型，仅在 `opt_mode="manual"` 时生效，支持 `"prompt_flash_attn"`、`"fused_attn_score"`、`"ascend_laser_attention"` |
| `kwargs.layout` | `str` | 否 | `"BNSD"` | 算子布局，仅在 `opt_mode="manual"` 时生效，支持 `"BNSD"`、`"BSND"`、`"BSH"` |

#### 返回值

`torch.Tensor`：注意力计算结果，布局与输入一致。

#### 使用示例

```python
import torch
from mindiesd import attention_forward

query = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
key = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
value = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)

out = attention_forward(query, key, value)
```

#### 迁移指南

- 从 `torch.nn.functional.scaled_dot_product_attention` 迁移时，输入布局需从 `[B,N,S,D]` 调整为 `[B,S,N,D]`，并去掉 `transpose` 操作。
- 从 `flash_attn.flash_attn_func` 迁移时，输入布局已为 `[B,S,N,D]`，可直接替换。
- 本接口仅提供前向推理，不支持反向梯度计算，迁移时需去掉 `dropout` 并将输入张量的 `requires_grad` 设为 `False`。

---

### attention_forward_varlen

变长序列注意力前向计算接口，适用于同一 batch 内序列长度不一致的场景。

```python
from mindiesd import attention_forward_varlen
```

#### 函数签名

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

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `q` | `torch.Tensor` | 是 | - | 查询张量，3D，布局为 `[T, N, D]`（T 为所有序列 token 总数） |
| `k` | `torch.Tensor` | 是 | - | 键张量，3D，布局为 `[T, N, D]` |
| `v` | `torch.Tensor` | 是 | - | 值张量，3D，布局为 `[T, N, D]` |
| `cu_seqlens_q` | `list[torch.Tensor]` | 是 | - | 查询序列的累积长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32` |
| `cu_seqlens_k` | `list[torch.Tensor]` | 是 | - | 键序列的累积长度，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32` |
| `max_seqlen_q` | `int` | 否 | `None` | 预留参数 |
| `max_seqlen_k` | `int` | 否 | `None` | 预留参数 |
| `dropout_p` | `float` | 否 | `0.0` | Dropout 概率，当前仅支持 `0.0` |
| `softmax_scale` | `float` | 否 | `None` | 缩放因子，为 `None` 时自动取 `head_dim ** -0.5` |
| `causal` | `bool` | 否 | `False` | 是否使用因果注意力掩码 |
| `window_size` | `int` | 否 | `None` | 预留参数 |
| `softcap` | `float` | 否 | `None` | 预留参数 |
| `alibi_slopes` | `torch.Tensor` | 否 | `None` | 预留参数 |
| `deterministic` | `bool` | 否 | `None` | 预留参数 |
| `return_attn_probs` | `bool` | 否 | `None` | 预留参数 |
| `block_table` | `torch.Tensor` | 否 | `None` | 预留参数 |

#### 返回值

`torch.Tensor`：注意力计算结果，形状为 `(total, nheads, headdim)`。

#### 使用示例

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

#### 迁移指南

- 从 `flash_attn.flash_attn_varlen_func` 迁移时，接口参数基本一致，可直接替换调用。

---

### sparse_attention

稀疏注意力前向计算接口，支持 RainFusion（rf_v2）和自适应块稀疏（ada_bsa）两种稀疏策略。

```python
from mindiesd import sparse_attention
```

#### 函数签名

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

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `q` | `torch.Tensor` | 是 | - | 查询张量，4D，布局由 `input_layout` 决定 |
| `k` | `torch.Tensor` | 是 | - | 键张量，4D，布局由 `input_layout` 决定 |
| `v` | `torch.Tensor` | 是 | - | 值张量，4D，布局由 `input_layout` 决定 |
| `attn_mask` | `torch.Tensor` | 否 | `None` | 注意力掩码，预留参数 |
| `scale` | `float` | 否 | `None` | 缩放因子，为 `None` 时自动取 `head_dim ** -0.5` |
| `is_causal` | `bool` | 否 | `False` | 是否使用因果注意力掩码 |
| `head_num` | `int` | 否 | `1` | 注意力头数量 |
| `input_layout` | `str` | 否 | `"BNSD"` | 张量布局，支持 `"BNSD"` 或 `"BSND"` |
| `inner_precise` | `int` | 否 | `0` | 计算精度模式，`0` 为高精度，`1` 为高性能 |
| `sparse_type` | `str` | 否 | `None` | 稀疏类型，支持 `None`、`"rf_v2"`、`"ada_bsa"` |
| `txt_len` | `int` | 否 | `0` | 文本序列长度，仅在 `sparse_type="rf_v2"` 时生效 |
| `block_size` | `int` | 否 | `128` | 块大小，当前仅支持 `128` |
| `latent_shape_q` | `list` | 否 | `None` | 查询的潜空间形状 `[t, h, w]`，`t*h*w = qseqlen`，仅在 `sparse_type="rf_v2"` 时生效 |
| `latent_shape_k` | `list` | 否 | `None` | 键的潜空间形状 `[t, h, w]`，`t*h*w = kseqlen`，仅在 `sparse_type="rf_v2"` 时生效 |
| `keep_sink` | `bool` | 否 | `True` | 是否保留 sink token，仅在 `sparse_type="ada_bsa"` 时生效 |
| `keep_recent` | `bool` | 否 | `True` | 是否保留 recent token，仅在 `sparse_type="ada_bsa"` 时生效 |
| `cdf_threshold` | `float` | 否 | `1.0` | CDF 阈值，仅在 `sparse_type="ada_bsa"` 时生效 |
| `sparsity` | `float` | 否 | `0.0` | 稀疏率，取值范围 `[0, 1]`，`0` 表示不使用稀疏算法 |

#### 返回值

`torch.Tensor`：注意力计算结果，布局与输入一致。

#### 使用示例

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

## 融合算子

融合算子系列接口提供昇腾高性能融合算子，涵盖位置编码、归一化和激活函数等基础计算。

| 接口名 | 类型 | 功能描述 |
|--------|------|----------|
| `rotary_position_embedding` | 函数 | 旋转位置编码（RoPE）融合算子 |
| `RMSNorm` | 类 | RMS 归一化融合算子 |
| `fast_layernorm` | 函数 | 高性能 LayerNorm 融合算子 |
| `layernorm_scale_shift` | 函数 | 自适应 LayerNorm（AdaLayerNorm）融合算子 |
| `get_activation_layer` | 函数 | 获取激活函数实例（含 NPU 加速版本） |

### rotary_position_embedding

旋转位置编码（RoPE）融合算子，将位置信息通过旋转矩阵注入到查询和键张量中。

```python
from mindiesd import rotary_position_embedding
```

### 函数签名

```python
rotary_position_embedding(
    x, cos, sin,
    rotated_mode="rotated_half",
    head_first=False,
    fused=True
) -> torch.Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `x` | `torch.Tensor` | 是 | - | 查询或键张量，4D，支持布局 `[B,N,S,D]`、`[B,S,N,D]`、`[S,B,N,D]` |
| `cos` | `torch.Tensor` | 是 | - | 预计算的余弦频率张量，2D `[S,D]` 或 4D `[1,1,S,D]`/`[1,S,1,D]`/`[S,1,1,D]` |
| `sin` | `torch.Tensor` | 是 | - | 预计算的正弦频率张量，维度与 `cos` 一致 |
| `rotated_mode` | `str` | 否 | `"rotated_half"` | 旋转模式：`"rotated_half"` 为半旋转，`"rotated_interleaved"` 为交错旋转 |
| `head_first` | `bool` | 否 | `False` | 头维度是否在序列维度之前 |
| `fused` | `bool` | 否 | `True` | 是否使用融合算子 |

#### 返回值

`torch.Tensor`：应用了旋转位置编码的张量，形状与输入 `x` 一致。

#### 使用示例

```python
import torch
from mindiesd import rotary_position_embedding

x = torch.randn(2, 4096, 24, 128, device="npu", dtype=torch.float16)
cos = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)
sin = torch.randn(1, 4096, 1, 128, device="npu", dtype=torch.float16)

out = rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
```

#### 旋转模式说明

- **rotated_half**：适用于 OpenSoraPlan、Stable Audio 等模型，将 `x` 拆分为前后两半进行旋转。
- **rotated_interleaved**：适用于 HunyuanDiT、OpenSora、Flux、CogVideox 等模型，将 `x` 按相邻元素交错进行旋转。

---

### RMSNorm

RMS 归一化融合算子，等效于 T5LayerNorm，不涉及均值计算，专注于输入张量的根均方值。

```python
from mindiesd import RMSNorm
```

#### 类签名

```python
RMSNorm(hidden_size, eps=1e-6)
```

#### 构造参数

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `hidden_size` | `int` | 是 | - | 隐藏层维度大小 |
| `eps` | `float` | 否 | `1e-6` | 数值稳定性参数 |

#### forward 方法

```python
forward(hidden_states, if_fused=True) -> torch.Tensor
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `hidden_states` | `torch.Tensor` | 是 | - | 输入张量，维度范围为 2~8 |
| `if_fused` | `bool` | 否 | `True` | 是否使用 NPU 融合算子 |

#### 使用示例

```python
import torch
from mindiesd import RMSNorm

norm = RMSNorm(1024, eps=1e-6)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
out = norm(x)
```

---

### fast_layernorm

高性能 LayerNorm 融合算子，支持多种计算精度模式。

```python
from mindiesd import fast_layernorm
```

#### 函数签名

```python
fast_layernorm(
    norm, x,
    impl_mode=0,
    fused=True
) -> torch.Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `norm` | `torch.nn.LayerNorm` | 是 | - | PyTorch LayerNorm 实例 |
| `x` | `torch.Tensor` | 是 | - | 输入张量，3D，布局为 `[B,S,H]` |
| `impl_mode` | `int` | 否 | `0` | 计算模式：`0` 高精度、`1` 高性能、`2` float16 模式（仅当所有输入均为 float16 时可用） |
| `fused` | `bool` | 否 | `True` | 是否使用融合算子，`False` 时回退到标准 `torch.nn.LayerNorm` 计算 |

#### 返回值

`torch.Tensor`：LayerNorm 计算结果，形状与输入 `x` 一致。

#### 使用示例

```python
import torch
import torch.nn as nn
from mindiesd import fast_layernorm

norm = nn.LayerNorm(1024, eps=1e-5)
x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)

out = fast_layernorm(norm, x, impl_mode=0, fused=True)
```

---

### layernorm_scale_shift

自适应 LayerNorm（AdaLayerNorm）融合算子，在 LayerNorm 基础上添加自适应缩放和偏移。

计算公式：`out = layernorm(x) * (1 + scale) + shift`

```python
from mindiesd import layernorm_scale_shift
```

#### 函数签名

```python
layernorm_scale_shift(
    layernorm, x, scale, shift,
    fused=True
) -> torch.Tensor
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `layernorm` | `torch.nn.LayerNorm` | 是 | - | PyTorch LayerNorm 实例 |
| `x` | `torch.Tensor` | 是 | - | 输入张量，3D，布局为 `[B,S,H]` |
| `scale` | `torch.Tensor` | 是 | - | 自适应缩放参数，2D `[B,H]` 或 3D `[B,1,H]` |
| `shift` | `torch.Tensor` | 是 | - | 自适应偏移参数，2D `[B,H]` 或 3D `[B,1,H]` |
| `fused` | `bool` | 否 | `True` | 是否使用融合算子 |

#### 返回值

`torch.Tensor`：AdaLayerNorm 计算结果，形状与输入 `x` 一致。

#### 使用示例

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

#### 约束条件

- `x` 的最后一维必须与 `scale`、`shift` 的最后一维相等。
- 若 `scale` 或 `shift` 为 3D 张量，则第二维必须为 1。

---

### get_activation_layer

获取指定类型的激活函数实例，部分激活函数提供 NPU 加速版本。

```python
from mindiesd import get_activation_layer
```

#### 函数签名

```python
get_activation_layer(act_type: str) -> nn.Module
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `act_type` | `str` | 是 | - | 激活函数名称，不区分大小写 |

#### 支持的激活函数

| 名称 | 对应实现 | 说明 |
|------|----------|------|
| `"swish"` | `nn.SiLU` | Swish 激活函数 |
| `"silu"` | `nn.SiLU` | SiLU 激活函数（与 swish 等价） |
| `"mish"` | `nn.Mish` | Mish 激活函数 |
| `"gelu"` | `GELU` | 标准 GELU |
| `"relu"` | `nn.ReLU` | ReLU 激活函数 |
| `"gelu-tanh"` | `GELU(approximate="tanh")` | tanh 近似 GELU |
| `"gelu-fast"` | `GELU(approximate="fast")` | 快速 GELU，使用 NPU 的 `npu_fast_gelu` 算子加速 |

#### 返回值

`nn.Module`：对应激活函数的实例。

#### 使用示例

```python
from mindiesd import get_activation_layer

act = get_activation_layer("gelu-fast")
out = act(hidden_states)
```
