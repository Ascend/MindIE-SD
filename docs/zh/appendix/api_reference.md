# API参考

## def rotary_position_embedding

**函数功能**

旋转位置编码技术可以提升DiT模型在处理序列数据时的性能和效率，其具体使用方法请参见[RoPE](../features/Acceleration_api.md#rope)。

**函数原型**

```python
def rotary_position_embedding(x: torch.Tensor,
                              cos: torch.Tensor,
                              sin: torch.Tensor,
                              rotated_mode: str = "rotated_half",
                              head_first: bool = False,
                              fused: bool = True) -> torch.Tensor:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|x|输入|torch.Tensor|应用旋转嵌入的q或k张量，shape要求输入为4维，一般为[B, N, S, D]或[B, S, N, D]或[S, B, N, D]。x可表示为 [x_0, x_1,..., x_d/2-1, x_d/2, x_d/2+1,..., x_d-1]。|
|cos|输入|torch.Tensor|预计算的复指数cos频率张量。shape要求输入为2维或4维，一般为[S,D]或[1, 1, S, D]或[1, S, 1, D]或[S, 1, 1, D]。|
|sin|输入|torch.Tensor|预计算的复指数sin频率张量。shape要求输入为2维或4维，一般为[S,D]或[1, 1, S, D]或[1, S, 1, D]或[S, 1, 1, D]。|
|rotated_mode|输入|str|旋转模式：支持rotated_half和rotated_interleaved两种模式。<ul><li>rotated_half：对半旋转，将x旋转为[-x_d/2, -x_d/2+1,..., -x_d-1, x_0, x_1,..., x_d/2-1]。</li><li>rotated_interleaved：相邻旋转，将x旋转为[-x_1, x_0, -x_3, x_2,..., -x_d-1, x_d-2]。</li></ul>|
|head_first|输入|bool|当x的layout中，head_dim在seqlen前面时，设置为True，否则设置为False。|
|fused|输入|bool|是否开启融合操作。<ul><li>True：选择高性能的RoPE融合算子。</li><li>False：使用原始计算公式。</li></ul>|

<br>

**返回值说明**

返回经旋转嵌入修改后的q张量和k张量。

## def attention_forward

**函数功能**

注意力计算模块，集成多种计算模式，包含原始计算模式、多种近似优化，用于搜索最优的计算公式，其具体使用方法请参见[attention_forward](../features/Acceleration_api.md#attention_forward)。

**函数原型**

```python
def attention_forward(query, key, value, attn_mask=None, scale=None, fused=True, **kwargs):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|query|输入|torch.Tensor|注意力计算公式的q输入，输入格式必须为(batch, seq_len, num_heads, head_dim)。|
|key|输入|torch.Tensor|注意力计算公式的k输入，输入格式必须为(batch, seq_len, num_heads, head_dim)。|
|value|输入|torch.Tensor|注意力计算公式的v输入，输入格式必须为(batch, seq_len, num_heads, head_dim)。|
|attn_mask|输入|torch.Tensor|注意力掩码。|
|scale|输入|float|输入缩放。|
|fused|输入|bool|是否开启融合操作。<ul><li>True：可选择融合算子。</li><li>False：使用原始计算公式，即缩放点积注意力（scaled dot-product attention）。</li></ul>|
|kwargs|输入|-|其他参数，包含以下三个可选项：<ul><li>opt_mode：str类型，支持runtime、static和manual三种模式，默认为runtime。<br>- runtime：在运行时动态搜索最佳融合算子，仅第一次搜索会消耗时间。<br>- static：通过静态表获取最佳融合算子。<br>- manual：手动设置融合算子类型。</li><li>op_type：str类型，表示融合算子类型，支持prompt_flash_attn、fused_attn_score和ascend_laser_attention。</li><li>layout：str类型，表示注意力机制布局方式，仅当opt_mode参数设置为manual时生效，支持BNSD、BSND和BSH。</li></ul>|

<br>

**返回值说明**

返回搜索后的最优注意力计算公式。

>[!NOTE]说明
>
>- 接口的输入shape为(batch, seq_len, num_heads, head_dim)，输出shape为(batch, seq_len, num_heads, head_dim)。
>- manual模式下设置算子的layout只会影响内部算子执行的layout，接口的输入shape和输出shape依然为(batch, seq_len, num_heads, head_dim)。

## def attention_forward_varlen

**函数功能**

不等长场景的注意力计算模块，其具体使用方法请参见[attention_forward_varlen](../features/Acceleration_api.md#attention_forward_varlen)。

**函数原型**

```python
def attention_forward_varlen(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: list[torch.Tensor],
        cu_seqlens_k: list[torch.Tensor],
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Optional[int] = None,
        softcap: Optional[float] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: Optional[bool] = None,
        return_attn_probs: Optional[bool] = None,
        block_table: Optional[torch.Tensor] = None,
):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|q|输入|torch.Tensor|查询(Query)张量。输入格式必须为(total_q, num_heads, head_dim)。total_q 是 batch 中所有query序列长度的总和（即“packed”格式，无填充）。num_heads 是注意力头数，head_dim 是每个头的维度。|
|k|输入|torch.Tensor|键(Key)张量。输入格式必须为(total_k, num_heads, head_dim)。total_k 是 batch 中所有key序列长度的总和（即“packed”格式，无填充）。num_heads 是注意力头数，head_dim 是每个头的维度。|
|v|输入|torch.Tensor|值(Value)张量。输入格式必须为(total_v, num_heads, head_dim)。total_v 是 batch 中所有value序列长度的总和（即“packed”格式，无填充）。num_heads 是注意力头数，head_dim 是每个头的维度。|
|cu_seqlens_q|输入|list[torch.Tensor]|查询序列的累积长度，用于将 packed 的 q 张量分割成独立序列。|
|cu_seqlens_k|输入|list[torch.Tensor]|键/值序列的累积长度。用于索引 k 和 v。|
|max_seqlen_q|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|max_seqlen_k|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|dropout_p|输入|float|表示数据需要忽略的概率，默认值为0.0，推理阶段建议保持默认值即可。|
|softmax_scale|输入|float|表示对QKT的缩放系数，若为None，则会根据head_dim ** -0.5进行缩放|
|causal|输入|bool|<ul><li>causal=true时，算子会传入下三角形式的atten mask；</li><li>causal=false时，算子不会传入atten mask。</li></ul>|
|window_size|输入|-|仅接口参数与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|softcap|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|alibi_slopes|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|deterministic|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|return_attn_probs|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|
|block_table|输入|-|该参数仅支持与flash_attn_varlen_func保持一致，npu无需配置该参数，也不支持该参数。|

**返回值说明**

返回不等长场景的注意力机制的输出张量。

## class CacheAgent

### 类说明

模型迭代推理中间特征缓存管理模块，其具体使用方法请参见[Cache](../features/cache.md)。

**成员**

|成员名称|描述|
|--|--|
|`__init__`|类初始化函数。|
|`apply`|缓存管理函数。|

### def \_\_init__

**函数功能**

类初始化函数。

**函数原型**

```python
def __init__(self):
```

**参数说明**

无

**返回值说明**

无

### def apply

**函数功能**

缓存管理函数。

**函数原型**

```python
def apply(self, function: callable, *args, **kwargs):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|function|输入|callable|缓存算法函数。|
|args|输入|-|缓存算法函数的位置参数。|
|kwargs|输入|-|缓存算法函数的关键字参数。|

**返回值说明**

返回应用缓存算法函数后的结果。

## class CacheConfig

**类说明**

缓存算法参数配置，其具体使用方法请参见[Cache](../features/cache.md)。

**成员**

|成员名称|类型|描述|
|--|--|--|
|method|str|缓存算法名称，支持attention_cache和dit_block_cache两种。|
|blocks_count|int|DiT模型的时空注意力Block个数。|
|steps_count|int|DiT模型的迭代步数。|
|step_start|int|开始缓存的迭代步数，默认值为0。|
|step_interval|int|缓存数据的间隔步数，0 < 'step_interval' < 'steps_count'，默认值为1。|
|step_end|int|结束缓存的迭代步数，0 < 'step_start' < 'step_end' < 'steps_count'，默认值为10000。|
|block_start|int|开始缓存的Block层数，0 < 'block_start' < 'blocks_count'，默认值为0。|
|block_end|int|结束缓存的Block层数，0 < 'block_start' < 'block_end' < 'blocks_count'，默认值为10000。|

## class RMSNorm

### 类说明

RMSNorm（Root Mean Square Normalization，均方根归一化）是一种归一化方法，与LayerNorm类似，但它不计算输入张量的均值，而是直接基于输入张量的平方均值（Root Mean Square，RMS）进行归一化操作。这种方法在某些深度学习模型中（如：T5）被广泛使用，因为它可以减少计算复杂度并提升性能。该模块的主要功能是对输入张量hidden\_states进行归一化，并通过可学习的权重参数weight对归一化后的结果进行缩放，其具体使用方法请参见[Norm](../features/Acceleration_api.md#norm)。

**成员**

|成员名称|描述|
|--|--|
|__init__|类初始化函数。|
|weight|类成员变量，可学习的权重参数，形状为(hidden_size,)，初始值为全1的张量。用于对归一化后的结果进行缩放。|
|variance_epsilon|类成员变量，归一化过程中使用的平滑因子，防止数值不稳定。|
|forward|根据设备类型（是否支持NPU）选择不同的实现方式，对输入张量hidden_states进行RMS归一化操作。|

### def \_\_init__

**函数功能**

类初始化函数。

**函数原型**

```python
def __init__(self, hidden_size, eps=1e-6):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|hidden_size|输入|torch.Tensor|输入张量的最后一维大小（即隐藏层的维度）。用于初始化归一化权重参数weight的形状。|
|eps|输入|float|一个小的常数，用于防止除零错误。默认值为1e-6。|

**返回值说明**

无

### def forward

**函数功能**

根据参数if_fused选择不同的实现方式，对输入张量hidden_states进行RMS归一化操作。

**函数原型**

```python
def forward(self, hidden_states, if_fused=True):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|hidden_states|输入|torch.Tensor|输入张量，形状通常为(batch_size, sequence_length, hidden_size)或类似的多维张量。最后一维的大小必须等于hidden_size。|
|if_fused|输入|bool|是否使用融合算子，默认为True。|

**返回值说明**

归一化后的张量，形状与输入张量hidden_states相同。

## class Linear

### 类说明

Linear是一个自定义的线性层，类似于PyTorch中的torch.nn.Linear，初始化参数新增op_type。实现了矩阵乘法的核心功能，并支持不同的操作类型（op_type）以及对输入张量的处理，其具体使用方法请参见[Linear](../features/Acceleration_api.md#linear)。

**成员**

|成员名称|描述|
|--|--|
|__init__|类初始化函数。|
|forward|根据不同的op_type设置，执行线性变换。|

### def \_\_init__

**函数功能**

类初始化函数。

**函数原型**

```python
def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, op_type="matmulv2") -> None:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|in_features|输入|int|输入特征的维度（即输入张量的最后一维大小）。|
|out_features|输入|int|输出特征的维度（即输出张量的最后一维大小）。|
|bias|输入|bool|是否启用偏置项。<ul><li>True：添加一个可训练的偏置参数；</li><li>False：不添加偏置。</li></ul>|
|device|输入|str|指定权重和偏置参数存储的设备（如："cpu"或"npu"）。|
|dtype|输入|str|指定权重和偏置参数的数据类型（如：torch.float32、torch.bfloat16或torch.float16）。|
|op_type|输入|str|可选值有："matmulv2"、"batchmatmulv2"和"batchmatmulv3"；默认值为"matmulv2"。|

**返回值说明**

无

### def forward

**函数功能**

根据不同的op_type设置，执行线性变换。

**函数原型**

```python
def forward(self, input: Tensor) -> Tensor:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|input|输入|torch.Tensor|输入张量，形状通常为(batch_size, sequence_length, hidden_size)或类似的多维张量。最后一维的大小必须等于in_features。|

**返回值说明**

线性变换后的结果。

## def get_activation_layer

**函数功能**

从字符串中获取激活函数的辅助函数，其具体方法请参见[activation](../features/Acceleration_api.md#activation)。

**函数原型**

```python
def get_activation_layer(act_type: str) -> nn.Module:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|act_type|输入|str|激活函数的名称。|

**返回值说明**

返回激活函数。

## def quantize

**函数功能**

传入浮点模型，通过导出的量化权重和对应描述符，将量化权重进行处理并转化为量化模型。

**函数原型**

```python
def quantize(model, quant_des_path, timestep_config=None, dtype=torch.bfloat16):
```

**参数说明**

|参数名|输入|类型|说明|
|--|--|--|--|
|model|输入|float|该参数需要为nn.Module。|
|quant_des_path|输入|str|通过工具导出的权重描述符全路径。|
|timestep_config|可选输入|-|当使用时间步量化算法时需要输入，该参数需要为[class TimestepPolicyConfig](#class-timesteppolicyconfig)类型。|
|dtype|可选输入|torch.float16/torch.bfloat16|可选输入，默认为torch.bfloat16。|

**返回值说明**

量化后的模型。

## class QuantFA

### 类说明

QuantFA是承载了量化FA的layer，其中包含了量化FA的相关权重，使用[def quantize](#def-quantize)量化使用时，需要确保attention类中有heads和inner dim的属性即可自动生成，在模型推理时需要更换FA的推理逻辑，其具体使用方法请参见[forward](#forward)。

### def \_\_init__

**函数功能**

类初始化函数。

**函数原型**

```python
def __init__(self, ori_head_num, ori_inner_dim, prefix, quant_weights=None, dtype=torch.bfloat16):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|ori_head_num|输入|int|原始的attention heads。|
|ori_inner_dim|输入|int|原始的inner dimension。|
|prefix|输入|str|该功能模块（FA）对应层的前缀名称。|
|quant_weights|输入|safetensors.torch.SafeTensorsFile|量化权重。|
|dtype|输入|float|可选输入，需要为torch.float16/torch.bfloat16类型，默认torch.bfloat16。|

### forward

**函数功能**

类初始化函数。

**函数原型**

```python
def forward(self, query, key, value, seq_len_list):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|query|输入|float16或bfloat16|query的激活值，layout支持TND。|
|key|输入|float16或bfloat16|key的激活值，layout支持TND。|
|value|输入|float16或bfloat16|value的激活值，layout支持TND。|
|seq_len_list|输入|list[int]|seq_len_list为各batch上seq_len之和，shape为[batch size]。|

## class TimestepManager

### 类说明

管理时间步信息的类，用于捕获和获取当前时间步信息，其具体使用方法请参见[量化](../features/sparse_quantization.md#linear量化)。

### set_timestep_idx

**函数功能**

设置当前迭代时间步。

**函数原型**

```python
def set_timestep_idx(cls, cur_timestep: int) -> None:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|cur_timestep|输入|int|当前迭代的时间步，传入的时间步范围和量化权重中的范围不一致会报错。|

### get_timestep_idx

**函数功能**

获取当前迭代时间步。

**函数原型**

```python
def get_timestep_idx(cls) -> Optional[int]:
```

**返回值说明**

返回当前迭代的时间步。

### set_timestep_idx_max

**函数功能**

设置最大时间步。

**函数原型**

```python
def set_timestep_idx_max(cls, t_idx: int) -> None:
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|t_idx|输入|int|设置的最大时间步，当前时间步不能超过最大时间步。|

### get_timestep_idx_max

**函数功能**

类初始化函数。

**函数原型**

```python
def get_timestep_idx_max(cls) -> Optional[int]:
```

**返回值说明**

返回设置的最大时间步。

## class TimestepPolicyConfig

### 类说明

与时间步量化算法配套的配置类，用于配置时间步量化算法的策略，其具体使用方法请参见[def register](#def-register)。

### def \_\_init__

**函数功能**

类初始化函数。

**函数原型**

```python
def __init__(self):
```

### def register

**函数功能**

用来注册时间步对应的策略。

**函数原型**

```python
def register(self, step_range, strategy):
```

参数说明

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|step_range|输入|range、list或int|对应strategy的时间步，表示一定范围的时间步或某个时间步采用的策略，如果未设置所有时间步，则采用默认策略。|
|strategy|输入|static或dynamic|对应输入时间步的策略，时间步的默认策略是dynamic。|

### get_strategy

**函数功能**

返回时间步对应的策略。

**函数原型**

```python
def get_strategy(self, step):
```

**参数说明**

|参数名|输入/输出|类型|说明|
|--|--|--|--|
|step|输入|int|指定的时间步。|

**返回值说明**

返回时间步对应的策略。
