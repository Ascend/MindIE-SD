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
| `cu_seqlens_q` | `torch.Tensor` | 是 | - | 查询序列的累积长度，1D 张量，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32` |
| `cu_seqlens_k` | `torch.Tensor` | 是 | - | 键序列的累积长度，1D 张量，形状为 `(batch_size + 1,)`，dtype 为 `torch.int32` |
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

稀疏注意力前向计算接口，支持 RainFusion（rf_v2 / rf_v3）和自适应块稀疏（ada_bsa）两种稀疏策略。

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
| `sparse_type` | `str` | 否 | `None` | 稀疏类型，支持 `None`、`"rf_v2"`、`"rf_v3"`、`"ada_bsa"` |
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
| `apply_rotary_pos_emb` | 函数 | 复用 PTA，融合并原地更新 query 和 key 的 RoPE 算子 |
| `add_layer_norm` | 函数 | 残差 Add 与 LayerNorm 融合算子 |
| `add_rms_norm` | 函数 | 残差 Add 与 RMSNorm 融合算子 |
| `RMSNorm` | 类 | RMS 归一化融合算子 |
| `fast_layernorm` | 函数 | 高性能 LayerNorm 融合算子 |
| `layernorm_scale_shift` | 函数 | 自适应 LayerNorm（AdaLayerNorm）融合算子 |
| `get_activation_layer` | 函数 | 获取激活函数实例（含 NPU 加速版本） |

### rotary_position_embedding

旋转位置编码（RoPE）融合算子，将位置信息通过旋转矩阵注入到查询和键张量中。

```python
from mindiesd import rotary_position_embedding
```

#### 函数签名

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

### apply_rotary_pos_emb

将 query 和 key 两路 RoPE 计算融合，并原地更新这两个张量。此接口直接封装 `torch_npu.npu_apply_rotary_pos_emb`，不再单独注册 MindIE-SD C++ 算子。

```python
from mindiesd import apply_rotary_pos_emb
```

也可以从 `mindiesd.layers` 导入同一个函数。

#### 函数签名

```python
apply_rotary_pos_emb(
    query, key, cos, sin,
    layout="BSND",
    rotary_mode="half"
) -> tuple[torch.Tensor, torch.Tensor]
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `query` | `torch.Tensor` | 是 | - | NPU 查询张量，调用后原地覆盖 |
| `key` | `torch.Tensor` | 是 | - | NPU 键张量，调用后原地覆盖；注意力头数可与 query 不同 |
| `cos` | `torch.Tensor` | 是 | - | NPU 余弦缓存，形状需与所选布局兼容 |
| `sin` | `torch.Tensor` | 是 | - | NPU 正弦缓存，形状和数据类型与 `cos` 相同 |
| `layout` | `str` | 否 | `"BSND"` | `BSND`、`SBND`、`BNSD`、`TND` 之一，实际支持情况取决于设备和后端 |
| `rotary_mode` | `str` | 否 | `"half"` | `half`、`interleave`、`quarter` 之一，实际支持情况取决于设备和后端 |

各张量应位于同一 NPU，并使用相同数据类型。支持的浮点类型取决于设备和后端，见下方约束说明。包装层显式向 PTA 传递 `layout`：本接口默认值是 `BSND`，不是 PTA 的 `BSH`；本接口不接受 `BSH`。

`B` 表示批大小，`S` 表示序列长度，`Nq`/`Nk` 表示查询/键的注意力头数，`D` 表示每个头的维度（不是总隐藏维度），`T` 表示打包布局中的 token 数。query 和 key 除头数外的维度一致。不使用批维广播时，缓存形状如下：

| 布局 | query 形状 | key 形状 | `cos` / `sin` 形状 |
|------|------|------|------|
| `BSND` | `[B,S,Nq,D]` | `[B,S,Nk,D]` | `[B,S,1,D]` |
| `SBND` | `[S,B,Nq,D]` | `[S,B,Nk,D]` | `[S,B,1,D]` |
| `BNSD` | `[B,Nq,S,D]` | `[B,Nk,S,D]` | `[B,1,S,D]` |
| `TND` | `[T,Nq,D]` | `[T,Nk,D]` | `[T,1,D]` |

缓存的头维度为 1，沿注意力头广播。Ascend 950 上缓存的批维度也可以为 1；A2/A3 上必须与输入批大小一致。与 `rotary_position_embedding` 不同，本接口不会自动把二维 `[S,D]` 缓存转换成所需形状。

#### 旋转模式

对 query 和 key 分别计算 `x * cos + rotate(x) * sin`。以下操作均沿最后一维进行：

- `half`：将 `x` 等分为两段 `[x1, x2]`，`rotate(x) = [-x2, x1]`。
- `interleave`：将每对相邻元素 `[a, b]` 旋转为 `[-b, a]`。
- `quarter`：将 `x` 等分为四段 `[x1, x2, x3, x4]`，`rotate(x) = [-x2, x1, -x4, x3]`。

`half` 和 `interleave` 要求 `D` 为偶数；`quarter` 要求 `D` 能被 4 整除。缓存的坐标配对方式必须与旋转模式及模型配置一致，同时仍须满足后端的形状和对齐限制。

#### 返回值与原地更新

`tuple[torch.Tensor, torch.Tensor]`：依次返回旋转后的 query 和 key。各输出的形状、数据类型、设备与对应输入相同，并且**与对应输入共享存储，不是独立副本**。

**调用会覆盖传入的 query 和 key。** 指向相同存储的别名或视图也会观察到更新。如果后续还需要未旋转的值，应在调用前克隆 query 和 key，或者把克隆后的张量传给此函数。对同一组张量再次调用，会再次施加 RoPE。

#### 设备与版本约束

本接口**仅支持 NPU**，没有 `fused` 开关，也没有 CPU/原生参考回退路径。安装的 `torch_npu` 必须提供 `npu_apply_rotary_pos_emb`，否则包装层抛出带升级提示的 `RuntimeError`。非法布局或模式名称会抛出 `ParametersInvalid`；张量及设备相关限制交由 PTA/CANN 检查。

下表摘录 [CANN ApplyRotaryPosEmbV2 约束](https://gitcode.com/cann/ops-transformer/blob/master/posembedding/apply_rotary_pos_emb/docs/aclnnApplyRotaryPosEmbV2.md)中 A2/A3 和 950 的支持范围。这些是后端能力，不代表每个 PTA/CANN 版本都已开放所有组合。

| 产品 | 布局 | 旋转模式 | 每头维度 `D` | 数据类型 |
|------|------|------|------|------|
| Atlas A2/A3 训练及推理系列产品 | `BSND`、`TND` | `half` | 64 或 128 | `float16`、`bfloat16`、`float32` |
| Ascend 950PR / 950DT | `BSND`、`SBND`、`BNSD`、`TND` | `half`、`interleave`、`quarter` | 不超过 1024，且满足模式和后端限制 | `float16`、`bfloat16`、`float32` |

上表 A2/A3 范围不包含 Atlas 200I/500 A2 推理产品，引用的 CANN 接口不支持这些产品。其他产品及版本差异请查阅 [PTA 接口文档](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-apply_rotary_pos_emb.md)与安装版本对应的 CANN 文档。通过 Python 包装层的布局/模式校验，不代表可以绕过上述限制。

当前 RoPE 回归用例已在 Ascend 950PR、`torch-npu 2.9.0.post4`、CANN 9.1 上验证；这不代表已覆盖所有设备、版本、每头维度或布局/模式/数据类型组合。

#### 与 rotary_position_embedding 的区别

| 对比项 | `rotary_position_embedding` | `apply_rotary_pos_emb` |
|------|------|------|
| 输入与输出 | 一个 `x`，返回一个结果张量 | 同时输入 query 和 key，返回两个结果张量 |
| 输入修改 | 返回新结果，不覆盖 `x` | 原地覆盖 query/key，输出与输入共享存储 |
| PTA 融合接口 | `torch_npu.npu_rotary_mul` | `torch_npu.npu_apply_rotary_pos_emb` |
| 模式参数 | `rotated_mode`：`rotated_half`、`rotated_interleaved` | `rotary_mode`：`half`、`interleave`、`quarter` |
| 布局处理 | 四维输入配合 `head_first`；自动调整支持的二维缓存 | 显式 `layout`，包括三维 `TND`；不自动调整缓存形状 |
| 参考实现 | 可通过 `fused=False` 使用 | 无，仅 NPU |

两者不能直接相互替换。迁移时除 RoPE 数学公式外，还必须核对输入是否修改、缓存形状、模式名称和设备支持范围。

#### 使用示例

示例使用 `BSND`、FP16 和 `D=64`。实际模型应使用其位置编码配置生成的余弦/正弦缓存。

```python
import torch
import torch_npu
from mindiesd import apply_rotary_pos_emb

query = torch.randn(2, 8, 6, 64, device="npu", dtype=torch.float16)
key = torch.randn(2, 8, 2, 64, device="npu", dtype=torch.float16)
angles = torch.randn(2, 8, 1, 32, device="npu", dtype=torch.float16)
angles = torch.cat((angles, angles), dim=-1)
cos, sin = angles.cos(), angles.sin()

query_out, key_out = apply_rotary_pos_emb(
    query, key, cos, sin, layout="BSND", rotary_mode="half"
)
assert query_out.data_ptr() == query.data_ptr()
assert key_out.data_ptr() == key.data_ptr()
```

---

### add_layer_norm

融合残差加法与 LayerNorm，并同时返回归一化结果和残差加法结果。

```python
from mindiesd import add_layer_norm
```

#### 函数签名

```python
add_layer_norm(
    x, residual, weight, bias,
    eps=1e-5,
    fused=True
) -> tuple[torch.Tensor, torch.Tensor]
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `x` | `torch.Tensor` | 是 | - | 输入张量，维度范围为 2~8 |
| `residual` | `torch.Tensor` | 是 | - | 残差张量，形状、数据类型和设备与 `x` 一致 |
| `weight` | `torch.Tensor` | 是 | - | 一维权重，长度等于 `x` 的最后一维 |
| `bias` | `torch.Tensor` | 是 | - | 一维偏置，形状与 `weight` 一致 |
| `eps` | `float` | 否 | `1e-5` | 有限正数，用于保证数值稳定性 |
| `fused` | `bool` | 否 | `True` | `True` 时使用 NPU 融合算子；`False` 时使用 PyTorch 参考实现 |

`x`、`residual`、`weight` 和 `bias` 支持 `float16`、`bfloat16` 和 `float32`。融合模式仅支持 NPU 张量。

#### 返回值

`tuple[torch.Tensor, torch.Tensor]`：依次返回归一化结果和 `x + residual`，两个张量的形状、数据类型和设备均与 `x` 一致。

#### 使用示例

```python
import torch
from mindiesd import add_layer_norm

x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
residual = torch.randn_like(x)
weight = torch.ones(1024, device="npu", dtype=torch.float16)
bias = torch.zeros(1024, device="npu", dtype=torch.float16)
out, residual_out = add_layer_norm(x, residual, weight, bias)
```

---

### add_rms_norm

融合残差加法与 RMSNorm，并同时返回归一化结果和残差加法结果。

```python
from mindiesd import add_rms_norm
```

#### 函数签名

```python
add_rms_norm(
    x, residual, weight,
    eps=1e-6,
    fused=True
) -> tuple[torch.Tensor, torch.Tensor]
```

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `x` | `torch.Tensor` | 是 | - | 输入张量，维度范围为 2~8 |
| `residual` | `torch.Tensor` | 是 | - | 残差张量，形状、数据类型和设备与 `x` 一致 |
| `weight` | `torch.Tensor` | 是 | - | 一维权重，长度等于 `x` 的最后一维 |
| `eps` | `float` | 否 | `1e-6` | 有限正数，用于保证数值稳定性 |
| `fused` | `bool` | 否 | `True` | `True` 时使用 NPU 融合算子；`False` 时使用 PyTorch 参考实现 |

`x`、`residual` 和 `weight` 支持 `float16`、`bfloat16` 和 `float32`。融合模式仅支持 NPU 张量。

#### 返回值

`tuple[torch.Tensor, torch.Tensor]`：依次返回归一化结果和 `x + residual`，两个张量的形状、数据类型和设备均与 `x` 一致。

#### 使用示例

```python
import torch
from mindiesd import add_rms_norm

x = torch.randn(2, 4096, 1024, device="npu", dtype=torch.float16)
residual = torch.randn_like(x)
weight = torch.ones(1024, device="npu", dtype=torch.float16)
out, residual_out = add_rms_norm(x, residual, weight)
```

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
- 若 `scale` 或 `shift` 为 3D 张量，则第二维必须为 1 或与 `x` 的第二维（序列长度）相等。

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

---

### Linear

自定义线性层，与 PyTorch 的 `nn.Linear` 用法一致，但新增 `op_type` 参数用于选择底层算子实现。

```python
from mindiesd import Linear
```

#### 构造参数

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `in_features` | `int` | 是 | - | 输入特征维度 |
| `out_features` | `int` | 是 | - | 输出特征维度 |
| `bias` | `bool` | 否 | `True` | 是否使用偏置 |
| `device` | `str` | 否 | `None` | 权重存储设备 |
| `dtype` | `torch.dtype` | 否 | `None` | 权重数据类型 |
| `op_type` | `str` | 否 | `"matmulv2"` | 算子类型，支持 `"matmulv2"`、`"batchmatmulv2"`、`"batchmatmulv3"` |

#### forward

```python
forward(input) -> torch.Tensor
```

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `input` | `torch.Tensor` | 是 | 输入张量，最后一维须等于 `in_features` |
