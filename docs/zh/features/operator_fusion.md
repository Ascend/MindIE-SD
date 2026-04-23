# 算子融合

## RoPE融合算子

RoPE（Rotary Position Embedding）融合算子是一种旋转位置编码技术，提升DiT模型在处理序列数据时的性能和效率。算子位置如下：

![](../figures/%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88-image-1.png)

- 旋转位置编码：以旋转矩阵的方式在q、k中注入位置信息，使得attention计算时能够感受到token的位置关系，在各大模型中被广泛应用，是一种高效的位置编码方式。
- 旋转编码：通过旋转操作将位置信息编码到每个token的嵌入向量中，确保模型能够捕捉到序列中元素的相对位置信息，而不依赖于绝对位置。
- 维度保持：旋转操作在每个维度上独立进行，有助于模型在不同的特征维度上捕获位置信息。
- 计算效率：不需要额外的参数来编码位置信息，而是通过数学计算实现，效率高。
- 在使用该算子时，原始代码一般会调用rotary-embedding-torch库里的apply_rotary_emb接口，在使用mindiesd接口进行优化时，可以替换为rotary_position_embedding方法。

- 原始代码：

    ```python
    class Attention(nn.Module):
    def __init__(self, xxx):
    # 省略
    def forward(self, hidden_states, freqs_cis_img):
        # 省略
        # 对query进行旋转位置编码处理，apply_rotary_emb为原始代码中的方法
        query = apply_rotary_emb(query, freqs_cis_img)
    ```

- 调用接口优化后的代码：

    ```python
    from mindiesd import rotary_position_embedding

    class Attention(nn.Module):
        def __init__(self, xxx):
            # 省略
        def forward(self, hidden_states, freqs_cis_img):
            # 省略
            cos, sin = freqs_cis_img
            cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
            query = rotary_position_embedding(query, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
            key = rotary_position_embedding(key, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
    ```

## RMSNorm融合算子

RMSNorm（Root Mean Square Normalization）融合算子是一种归一化方法，不涉及均值计算，而是专注于输入张量的根均方值，减少计算开销。

RMSNorm在模型中的位置多出现于DiTBlock中q k v linear之后，FA之前，位置示意图如下：

![](../figures/%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88-image-2.png)

在使用mindiesd接口进行优化时，可以使用`RMSNorm`方法

- 原始代码

    ```python
    norm_q = RMSNorm(dim_head, eps=eps)
    query = norm_q(query)
    ```

- 调用class RMSNorm优化后的代码

    ```python
    from mindiesd import RMSNorm
    norm_q = RMSNorm(dim_head, eps=eps)
    query = norm_q(query)
    ```

## Attention_forward

支持选择底层的算子类型（PFA、FASCore、LaserAttention等），支持自动寻找最优性能算子，自动寻优支持cache缓存，相同输入时，自动使用之前cache的结果，也支持指定算子类型。主要用在DiTBlock中的attention模块，包含SelfAttention和CrossAttention场景。
Attention接口支持自动寻优功能，主要是在运行时自动统计客户场景算子耗时，使用耗时最短的Attention后端。流程主要分为两部分：

- 用户执行推理warmup（当第一次接收新的格式的时候会自动执行，可配置开关关闭自动调优，使用静态dispatch方案），解析用户输入的shape（B N D Q_Seqlen K_Seqlen），dtype信息，运行测试代码，获取最优OP以及Format(BNSD/BSND/BSH等)，并cache结果，根据结果选择attention算子后端，执行推理。
- 用户稳态运行业务，此时解析用户输入的shape，dtype信息，使用cache结果配置后端，执行推理。整个业务场景中，当有新的shape，dtype输入时才进行在线性能测试，获取最优结果，获取的最优结果会cache存储，后续调用时可根据缓存直接调用。

在使用MindIE SD接口进行优化时，可以使用attention_forward接口：

- 从torch.nn.functional.scaled_dot_product_attention迁移
- 原始代码

    ```python
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    hidden_states = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    ```

- 调用接口优化后的代码

    ```python
    from mindiesd import attention_forward
    # q,k,v shape is batch, seq_len, num_heads, head_dim
    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, attn.heads, head_dim)
    value = value.view(batch_size, -1, attn.heads, head_dim)
    # the input shape of attention_forward = (batch, seq_len, num_heads, head_dim)
    # the output of attention_forward = (batch, seq_len, num_heads, head_dim)
    hidden_states = attention_forward(query, key, value, attn_mask=attention_mask)
    hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
    ```

- 从flash_attention.flash_attn_func迁移
- 原始代码

    ```python
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    out = flash_attention.flash_attn_func(q, k, v)
    ```

- 调用接口优化后的代码

    ```python
    from mindiesd import attention_forward
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype)
    out = attention_forward(q, k, v)
    ```

    >[!NOTE]说明
    >- 注意attention_forward接口的输入shape为(batch, seq_len, num_heads, head_dim)，输出shape为(batch, seq_len, num_heads, head_dim)。
    >- attention_forward接口仅提供前向推理功能，不提供反向梯度计算，因此迁移时需要去掉dropout，并将输入tensor梯度设置为False。
- 从flash_attn .flash_attn_varlen_func迁移，不使能causal时

- 原始代码

    ```python
    out = flash_attn_varlen_func( q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False)
    ```

- 调用接口优化后的代码

    ```python
    from mindiesd import attention_forward_varlen
    out = attention_forward_varlen( q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, softmax_scale=None, causal=False)
    ```

- 从flash_attn .flash_attn_varlen_func迁移，使能causal时
- 原始代码

    ```python
    out = flash_attn_varlen_func( q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=True)
    ```

- 调用接口优化后的代码

    ```python
    from mindiesd import attention_forward_varlen
    out = attention_forward_varlen( q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, softmax_scale=None, causal=True)
    ```
