# Dynamic Expert Load Balancing

## General Principles

As visual generation models evolve toward the DiT architecture, introducing MoE mechanisms to break through Scaling Law has become an industry consensus. However, the massive parameter scale of DiT-MoE forces us to adopt Expert Parallelism (EP) strategies. Unlike LLMs, the strong spatial locality of visual data easily induces overload on specific experts, leading to severe computational load imbalance. Furthermore, the expert activation distribution during the diffusion model's denoising process exhibits significant temporal dynamics, meaning traditional static load balancing strategies completely fail in the face of this spatiotemporal dual heterogeneity.

![](../../figures/dyeplb_image_1.png)

## Technical Features

This solution dynamically adjusts expert weights on Ranks based on load information to achieve expert load balancing and model inference acceleration. The solution has the following features:

- **Non-intrusive design**: Global synchronization point checks and weight update positions can be chosen according to the specific model implementation.
- **Asynchronous pipeline processing**: Algorithm computation and expert weight concatenation use additional threads and processes to minimize impact on the main inference flow.
- **Three EP modes**: A2A (standard all-to-all), AG (all-gather), EX (controllable mode), selected via the `mode` parameter.
- **Mutual exclusion reminder with CPU offload**: Involves H2D data transfer, so bandwidth contention may occur when used simultaneously with [CPU offload](cpu_offload.md); you need to adjust execution timing yourself.

## Interface and Usage

### Recommended Approaches

- **A2A mode**: Standard all-to-all EP with balanced communication, recommended for general scenarios.
- **AG mode**: all-gather EP, requires additional matmul of transformation matrix and expert scores, suitable for scenarios that require global synchronization.
- **EX mode**: Controllable mode, limits the scale of expert placement changes via `max_move`, suitable for reducing peak memory when coexisting with offload.

### Integration Process

> [!NOTE] Note
> To minimize the impact on the main inference flow, the algorithm and expert weight concatenation are processed using additional threads and processes.

1. Start the EPLB algorithm process. Startup parameters are as follows:

   | Parameter | Default | Description |
   |------|--------|------|
   | `world_size` | Required | Number of EPs |
   | `expert_num` | Required | Number of global experts |
   | `block_num` | Required | Number of MoE layers |
   | `max_move` | — | Maximum number of experts to move in EX mode |
   | `redundant` | — | Number of redundant experts |
   | `mode` | Required | A2A / AG / EX |
   | `auth_key` | `secret_key` | Reads the `EPLB_AUTH_KEY` environment variable by default |

   ```shell
   python -m mindiesd.eplb.eplb_scheduler \
       --world_size 2 \
       --host localhost \
       --port 50001 \
       --mode A2A
   ```

2. Import the load collector and dispatcher, initialize them, and start the worker thread.

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

3. In AG mode, an additional transformation matrix multiplication is required.

   ```python
   if EP_AG and self.dispatcher.update_flag:
       expert_trans_tensor = self.dispatcher.get_expert_trans_tensor()
       trans_scores = torch.matmul(scores, expert_trans_tensor)
   ```

4. Insert load collection and weight replacement after `npu_moe_init_routing` and before `npu_grouped_matmul_finalize_routing` in the MoE forward pass.

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

### Class Descriptions

#### ExpertLoadCollector

```python
from mindiesd.eplb.collector import ExpertLoadCollector
```

| Parameter | Type | Required | Default | Description |
|------|------|------|--------|------|
| `expert_num` | `int` | Yes | - | Number of global experts |
| `lb_interval` | `int` | No | `1` | EPLB interval steps |

#### DynamicDispatcher

```python
from mindiesd.eplb.dispatcher import DynamicDispatcher
```

| Parameter | Type | Required | Default | Description |
|------|------|------|--------|------|
| `expert_num` | `int` | Yes | - | Number of global experts |
| `weight1` | `Tensor` | Yes | - | UP weight |
| `weight2` | `Tensor` | Yes | - | DOWN weight |
| `rank_in_group` | `int` | Yes | - | Rank number within the EP communication group |
| `ep_size` | `int` | Yes | - | Number of EPs |

#### construct_expert_info_transfer_pool

```python
from mindiesd.eplb.task_manager import construct_expert_info_transfer_pool
```

| Parameter | Type | Required | Default | Description |
|------|------|------|--------|------|
| `module` | `Module` | Yes | - | Initialized model |
| `rank_in_group` | `int` | Yes | - | Rank number within the EP communication group |
| `device` | `int` | Yes | - | Device number corresponding to the rank |
| `ip` | `str` | Yes | - | Same as the server IP |
| `port` | `int` | Yes | - | Same as the server port |
| `auth_key` | `str` | No | `secret_key` | Reads the `EPLB_AUTH_KEY` environment variable by default |
