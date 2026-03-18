> 本章节主要介绍`计算优化`相关加速特性
> - [CV并行](#cv并行)
> - [多实例](#多实例)
> - [CFG融合](#cfg融合)
> - [算子融合](#算子融合)

## CV并行
通过调整，使得多流间Cube和Vector计算可以并行，减少冲突，提升计算效率。

![](../figures/CV%E5%B9%B6%E8%A1%8C.png)
---
## 多实例
单实例场景下，GPU上的任务是串行执行，当存在多个模型时，后续模型会等待前置模型推理完成，这会导致硬件资源处于浪费状态。因此引入了多实例的优化方法，其原理是让单个GPU同时运行多个模型实例，来更好的利用硬件资源，从而提高总体服务吞吐量。

下图展示了双实例的示例：

(1) ①中两个模型串行执行，task1必须等待task0执行完成，而双实例②，task0和task1并行计算，提升了整体的吞吐量；

(2) 在双实例③中，相对②，两个模型还复用了权重，既降低了额外的内存需求，又提升了吞吐量

![](../figures/%E5%A4%9A%E5%AE%9E%E4%BE%8B.png)

---
## CFG融合
CFG（Classifier-Free Guidance）通过联合正样本计算和负样本计算来提升模型生成的质量。在传统实现中，正样本和负向本需要分别进行完整的前向传播计算，由于这两个计算在模型中大部分计算路径相同，所以导致了大量计算冗余。CFG融合通过将正负样本拼接，减少算子调用次数和重复计算，提升推理速度。示意图如下所示：

![](../figures/CFG%E8%9E%8D%E5%90%88.png)

---
## 算子融合
(1) RoPE融合算子：是一种旋转位置编码技术，提升DiT模型在处理序列数据时的性能和效率。算子位置如下：

![](../figures/%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88-image-1.png)

- 旋转位置编码（Rotary Position Embedding, RoPE）：以旋转矩阵的方式在q、k中注入位置信息，使得attention计算时能够感受到token的位置关系，在各大模型中被广泛应用，是一种高效的位置编码方式。 
  - 旋转编码：通过旋转操作将位置信息编码到每个token的嵌入向量中，确保模型能够捕捉到序列中元素的相对位置信息，而不依赖于绝对位置。 
  - 维度保持：旋转操作在每个维度上独立进行，有助于模型在不同的特征维度上捕获位置信息。 
  - 计算效率：不需要额外的参数来编码位置信息，而是通过数学计算实现，效率高。
- 在使用该算子时，原始代码一般会调用rotary-embedding-torch库里的apply_rotary_emb接口，在使用mindiesd接口进行优化时，可以替换为`rotary_position_embedding`方法。
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
            cos, sin = cos.to(x.device), sin.to(x.device)
            query = rotary_position_embedding(query, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
            key = rotary_position_embedding(key, cos, sin, rotated_mode="rotated_half", head_first=False, fused=True)
        ```

(2) RMSNorm融合算子(Root Mean Square Normalization)：是一种归一化方法，不涉及均值计算，而是专注于输入张量的根均方值，减少计算开销。

- RMSNorm在模型中的位置多出现于DiTBlock中q k v linear之后，FA之前，位置示意图如下：

	![](../figures/%E7%AE%97%E5%AD%90%E8%9E%8D%E5%90%88-image-2.png)
- 在使用mindiesd接口进行优化时，可以使用`RMSNorm`方法
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
    
 
 
(3) Attention\_forward：支持选择底层的算子类型（PFA、FASCore、LaserAttention等），支持自动寻找最优性能算子，自动寻优支持cache缓存，相同输入时，自动使用之前cache的结果，也支持指定算子类型。主要用在DiTBlock中的attention模块，包含SelfAttention和CrossAttention场景。
- Attention接口支持自动寻优功能，主要是在运行时自动统计客户场景算子耗时，使用耗时最短的Attention后端。流程主要分为两部分：
	- 客户执行推理warmup\(当第一次接收新的格式的时候会自动执行，可配置开关关闭自动调优，使用静态dispatch方案\)，解析用户输入的shape\(B N D Q\_Seqlen K\_Seqlen\)，dtype信息，运行测试代码，获取最优OP以及Format\(BNSD/BSND/BSH等\)，并cache结果，根据结果选择attention算子后端，执行推理；
	- 客户稳态运行业务，此时解析用户输入的shape，dtype信息，使用cache结果配置后端，执行推理。整个业务场景中，当有新的shape，dtype输入时才进行在线性能测试，获取最优结果，获取的最优结果会cache存储，后续调用时可根据缓存直接调用。

- 在使用mindiesd接口进行优化时，可以使用`attention_forward`接口
	- 从torch.nn.functional.scaled\_dot\_product\_attention迁移
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
  - 从flash\_attention.flash\_attn\_func迁移
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
        >**说明**
        >- 注意attention\_forward接口的输入shape为\(batch, seq\_len, num\_heads, head\_dim\)，输出shape为\(batch, seq\_len, num\_heads, head\_dim\)
        >- attention\_forward接口仅提供前向推理功能，不提供反向梯度计算，因此迁移时需要去掉dropout，并将输入tensor梯度设置为False。
  - 从flash\_attn .flash\_attn\_varlen\_func迁移，不使能causal时
  	- 原始代码
    	```python
        out = flash_attn_varlen_func( q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=False)
        ```
    - 调用接口优化后的代码
    	```python
        from mindiesd import attention_forward_varlen
        out = attention_forward_varlen( q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, softmax_scale=None, causal=False)
        ```
  - 从flash\_attn .flash\_attn\_varlen\_func迁移，使能causal时
    - 原始代码
    	```python
        out = flash_attn_varlen_func( q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=0.0, softmax_scale=None, causal=True)
        ```
   	- 调用接口优化后的代码
    	```python
        from mindiesd import attention_forward_varlen
        out = attention_forward_varlen( q, k, v, cu_seqlens_q, cu_seqlens_k, dropout_p=0.0, softmax_scale=None, causal=True)
        ```