# 动态专家负载均衡

## 通用原理

随着视觉生成模型向 DiT 架构演进，引入 MoE 机制以突破 Scaling Law 已成为行业共识。然而，DiT-MoE 庞大的参数规模迫使我们采用专家并行（EP）策略。与 LLM 不同，视觉数据的强空间局部性极易诱发特定专家过载，导致严重的计算负载不均。更进一步，扩散模型的去噪过程中专家激活分布呈现出显著的时序动态变化，这意味着传统的静态负载均衡策略在面对这种时空双重异构性时彻底失效。

![](../figures/DyEPLB-image-1.png)

## 技术特点

本方案通过负载信息动态调整 Rank 上的专家权重以达到专家负载均衡，实现模型推理加速。方案具备以下特点：

- **无侵入式设计**：全局同步点检查和权重更新位置可根据模型具体实现自行选择。
- **异步流水线处理**：算法计算和专家权重拼接使用额外的线程和进程处理，最小化对主推理流程的影响。
- **三种 EP 模式**：A2A（标准 all-to-all）、AG（all-gather）、EX（可控模式），通过 `mode` 参数选择。
- **与 CPU 卸载的互斥提醒**：涉及 H2D 数据传输，与 [CPU 卸载](cpu_offload.md) 同时使用时可能存在带宽争抢，需自行调整执行时机。

## 接口和使用

### 推荐方案

- **A2A 模式**：标准 all-to-all EP，通信均衡，推荐通用场景使用。
- **AG 模式**：all-gather EP，需要额外进行变换矩阵与专家 scores 的 matmul，适合需全局同步的场景。
- **EX 模式**：可控模式，通过 `max_move` 限制专家布局改变规模，适合与 offload 共存时降低峰值显存。

### 适配流程

> [!NOTE]说明
> 为了最小程度的减少对主推理的影响，将算法和专家权重的拼接使用额外的线程和进程来处理。

1. 启动 EPLB 算法进程。启动参数如下：

   | 参数 | 默认值 | 说明 |
   |------|--------|------|
   | `world_size` | 必填 | EP 数 |
   | `expert_num` | 必填 | 全局专家数量 |
   | `block_num` | 必填 | MoE 层数 |
   | `max_move` | — | EX 模式下最大移动专家数量 |
   | `redundant` | — | 冗余专家数 |
   | `mode` | 必填 | A2A / AG / EX |
   | `auth_key` | `secret_key` | 默认读取环境变量 `EPLB_AUTH_KEY` |

   ```shell
   python -m mindiesd.eplb.eplb_scheduler \
       --world_size 2 \
       --host localhost \
       --port 50001 \
       --mode A2A
   ```

2. 引入负载采集器和调度器，初始化后启动 worker 线程。

   ```python
   from mindiesd.eplb.dispatcher import DynamicDispatcher
   from mindiesd.eplb.collector import ExpertLoadCollector
   from mindiesd.eplb.task_manager import construct_expert_info_transfer_pool

   model.init()

   model.moe_module.block.expert_load_collector = ExpertLoadCollector(expert_num, lb_interval)
   model.moe_module.block.dispatcher = DynamicDispatcher(expert_num, weight1, weight2, rank_in_group, ep_size)

   if eplb_enabled:
       construct_expert_info_transfer_pool(
           module=model, rank_in_group=rank_in_group, device=device,
           ip=host, port=port, auth_key=auth_key
       )

   model.forward()
   ```

3. AG 模式下需额外进行变换矩阵乘法。

   ```python
   if EP_AG and self.dispatcher.update_flag:
       expert_trans_tensor = self.dispatcher.get_expert_trans_tensor()
       trans_scores = torch.matmul(scores, expert_trans_tensor)
   ```

4. 在 MoE 前向的 `npu_moe_init_routing` 之后、`npu_grouped_matmul_finalize_routing` 之前接入负载采集和权重替换。

   ```python
   expanded_tokens, expanded_row_idx, expanded_indices = torch_npu.npu_moe_init_routing(
       tokens, row_idx, indices, tokens.shape[0])

   self.expert_load_collector.collect_expert_load(expanded_indices)
   self.dispatcher.check_consistency()

   if self.dispatcher.update_flag:
       weight1, weight2, local_expert_num, device_indices_map, \
           local_expert_indices_map, local_expert_list = \
           self.dispatcher.update_module_weight_and_map()
       self.weight1 = weight1
       self.weight2 = weight2
       self.local_expert_num = local_expert_num

   tokens = torch_npu.npu_grouped_matmul_finalize_routing()
    ```

### 类说明

#### ExpertLoadCollector

```python
from mindiesd.eplb.collector import ExpertLoadCollector
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `expert_num` | `int` | 是 | - | 全局专家数 |
| `lb_interval` | `int` | 否 | `1` | EPLB 间隔步数 |

#### DynamicDispatcher

```python
from mindiesd.eplb.dispatcher import DynamicDispatcher
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `expert_num` | `int` | 是 | - | 全局专家数 |
| `weight1` | `Tensor` | 是 | - | UP 权重 |
| `weight2` | `Tensor` | 是 | - | DOWN 权重 |
| `rank_in_group` | `int` | 是 | - | EP 通信组组内编号 |
| `ep_size` | `int` | 是 | - | EP 数 |

#### construct_expert_info_transfer_pool

```python
from mindiesd.eplb.task_manager import construct_expert_info_transfer_pool
```

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| `module` | `Module` | 是 | - | 初始化后的 model |
| `rank_in_group` | `int` | 是 | - | EP 通信组组内编号 |
| `device` | `int` | 是 | - | rank 对应的 device 编号 |
| `ip` | `str` | 是 | - | 与服务端 ip 一致 |
| `port` | `int` | 是 | - | 与服务端 port 一致 |
| `auth_key` | `str` | 否 | `secret_key` | 默认读取环境变量 `EPLB_AUTH_KEY` |
