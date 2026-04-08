# Dynamic EPLB Acceleration

## DyEPLB

- **Background**

  As visual generation models evolve toward DiT architectures, introducing MoE mechanisms to extend the scaling law has become a common industry direction. However, the large parameter scale of DiT-MoE models forces the use of expert parallelism (EP). Unlike LLM workloads, visual data has strong spatial locality, which can easily overload specific experts and cause severe load imbalance. In addition, expert activation distributions vary dynamically across denoising timesteps, which means traditional static load-balancing strategies fail under combined spatial and temporal heterogeneity. DyEPLB addresses this DiT-MoE scenario with dynamic expert load balancing to improve cluster utilization and inference performance.

  ![](../../zh/figures/DyEPLB-image-1.png)

- **Principle**

  Expert weights are adjusted dynamically across ranks according to load information so that expert load is balanced and model inference is accelerated.

- **Notes**

  - DyEPLB is designed to be minimally intrusive, so global synchronization checks and weight-update locations can be selected according to the model implementation and the FPA algorithm scenario.
  - In all-gather full-EP mode, the global synchronization check can be performed earlier to adapt custom operators such as `torch_npu.npu_moe_init_routing_v2`, which helps preserve token continuity when weights are replaced.
  - It is recommended to place the weight-update module between two Matmul operations to maximize the gain from the FPA algorithm. Because the workflow includes host-to-device data transfer, there may be bandwidth contention if DyEPLB is used together with offload. Adjust the scheduling of the two mechanisms to avoid blocking each other. Weight updates also introduce extra expert-weight memory consumption during the update process, which can raise peak memory usage. The FPA algorithm provides an EX mode to reduce expert-layout changes and mitigate this issue.

- **Integration flow**

  > [!NOTE]
  > To minimize the impact on the main inference path, the algorithm logic and expert-weight stitching are handled in additional threads and processes.

  For the input and output details of the involved interfaces, see [Class initialization and interface reference](#class-initialization-and-interface-reference).

  1. Start the EPLB scheduler process:

     ```console
     root@node134:/home# python -m mindiesd.eplb.eplb_scheduler --world_size 2 --host localhost -- port 50001 --mode A2A
     ```

     Common launch parameters include:

     - `world_size`: number of EP ranks
     - `expert_num`: total number of global experts
     - `block_num`: number of MoE layers
     - `max_move`: maximum number of moved experts in EX mode
     - `redundant`: number of redundant experts
     - `mode`: `A2A` for all-to-all EP, `AG` for all-gather EP, `EX` for controlled mode
     - `auth_key`: reads the `EPLB_AUTH_KEY` environment variable by default; falls back to `secret_key`

  2. Import the load collector and dispatcher:

     ```python
     from mindiesd.eplb.dispatcher import DynamicDispatcher
     from mindiesd.eplb.collector import ExpertLoadCollector
     ```

  3. Before inference, start the worker thread that handles data through a task queue. After model initialization, initialize the DyEPLB load collector and dispatcher at the MoE-layer granularity:

     ```python
     # model initialization
     model.init()

     # load collector
     model.moe_module.block.expert_load_collector = ExpertLoadCollector(expert_num, lb_interval)
     # dispatcher, holding the complete expert weights on the host side
     model.moe_module.block.dispatcher = DynamicDispatcher(expert_num, weight1, weight2, rank_in_group, ep_size)
     # start worker thread
     if eplb_enabled:
        from mindiesd.eplb.task_manager import construct_expert_info_transfer_pool
        # multiprocessing communication, auth_key must match the EPLB scheduler process
        construct_expert_info_transfer_pool(module=model, rank_in_group=rank_in_group, device=device, ip=host, port=port, auth_key=auth_key)

     # inference flow
     model.forward()
     ```

  4. In all-gather full-EP mode, add an extra matmul between the transform matrix and the expert scores to avoid manually reordering tokens, indices, and related variables later:

     ```python
     if EP_AG and self.dispatcher.update_flag:
         # transformation matrix generated from expert ordering, shape(global_expert_num * global_expert_num)
         expert_trans_tensor = self.dispatcher.get_expert_trans_tensor()
         trans_scores = torch.matmul(scores, expert_trans_tensor)
     ```

  5. Recommended enablement order inside MoE: `init_routing > collect_load > global_sync_check > weight_replace > GMM`.

     ```python
     expanded_tokens, expanded_row_idx, expanded_indices = torch_npu.npu_moe_init_routing(tokens, row_idx, indices, tokens.shape[0])

     # collect expert load
     self.expert_load_collector.collect_expert_load(expanded_indices)
     # global synchronization check
     self.dispatcher.check_consistency()
     # validate synchronization status
     if self.dispatcher.update_flag:
        weight1, weight2, local_expert_num, device_indices_map, local_expert_indices_map, local_expert_list = self.dispatcher.update_module_weight_and_map()
        self.weight1 = weight1
        self.weight2 = weight2
        self.local_expert_num = local_expert_num

     tokens = torch_npu.npu_grouped_matmul_finalize_routing()
     ```

## Class initialization and interface reference

- `ExpertLoadCollector`
  Parameters:
  - `expert_num`: total number of global experts
  - `lb_interval`: EPLB interval in steps; the default value `1` means every step participates in EPLB

  Return value: none

- `DynamicDispatcher`
  Parameters:
  - `expert_num`: total number of global experts
  - `weight1`: UP weights
  - `weight2`: DOWN weights
  - `rank_in_group`: rank index within the EP communication group
  - `ep_size`: EP size

  Return value: none

- `construct_expert_info_transfer_pool`
  Parameters:
  - `module`: initialized model
  - `rank_in_group`: rank index within the EP communication group
  - `device`: device index bound to the rank
  - `ip`: must match the configured server IP
  - `port`: must match the configured server port
  - `auth_key`: multiprocessing secret; reads `EPLB_AUTH_KEY` by default and falls back to `secret_key`

  Return value: none

- `get_expert_trans_tensor`
  Used in all-gather EP scenarios to obtain the transform matrix.

- `collect_expert_load`
  Parameters:
  - `expanded_indices`: token-cumsum values for each expert; the output of `npu_moe_init_routing` can be passed directly

  Return value: none

- `check_consistency`
  Performs an extra all-gather communication internally to verify synchronization status across ranks.

- `update_module_weight_and_map`
  Parameters: none

  Return values:
  - `weight1`: UP weights
  - `weight2`: DOWN weights
  - `local_expert_num`: number of local experts, including redundant experts
  - `device_indices_map`: for example `[0, 1, 1, 0]`, meaning which rank each expert index belongs to
  - `local_expert_indices_map`: for example on rank 0 `[0, -1, -1, 1]` and on rank 1 `[-1, 0, 1, -1]`, meaning the local position of each expert index in the local expert-weight tensor
  - `local_expert_list`: for example rank 0 `[0, 3]` and rank 1 `[1, 2]`, meaning the local expert layout
