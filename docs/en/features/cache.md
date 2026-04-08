# Cache-Based Acceleration Features

## DiTCache

- **Background**

  During inference, DiT models iterate through `T` steps, and every step computes the full set of blocks. Each block contains a large amount of compute, as shown below. However, latents from adjacent steps are highly similar, so nearly identical intermediate results are recomputed repeatedly, which introduces redundant compute and slows down inference.

  ![](../../zh/figures/ditcache-image-1.png)

- **Principle**

  By reusing local model features based on activation similarity between adjacent sampling steps or adjacent blocks, DiTCache skips selected DiT blocks, reduces redundant computation, and accelerates inference.

- **Optimization method**

  A search script first determines the minimum number of blocks that need to be skipped to achieve the target acceleration ratio. It then scans candidate start and end blocks and picks the configuration with the lowest MSE among all valid combinations. When the cache hits, the cached result from a selected block range at step `N` is reused directly at step `M`, so a full DiTBlock forward pass becomes a lightweight tensor read.

  ![](../../zh/figures/ditcache-image-2.png)

  1. Compute the minimum number of blocks that need to be cached for the target speedup ratio.
  2. Because `block0` must always be computed, start scanning from `block1` and search for the three best `(block_start, block_end)` candidates.
  3. Evaluate all candidate combinations by measuring the MSE before and after caching, then choose the one with the smallest MSE loss.
  4. Apply the resulting configuration in the model and enable cache during inference.

- **Workflow**

  1. Import `CacheConfig` and `CacheAgent`.

     ```python
     from mindiesd import CacheConfig, CacheAgent
     ```

  2. Initialize `CacheConfig` when the model is created.

     ```python
     config = CacheConfig(
             method="dit_block_cache",
             blocks_count=len(transformer.single_blocks), # number of cache-enabled blocks
             steps_count=args.infer_steps,                # total inference steps
             step_start=args.cache_start_steps,           # first step index that uses cache
             step_interval=args.cache_interval,           # forced recompute interval
             step_end=args.infer_steps-1,                 # last step index that uses cache
             block_start=args.single_block_start,         # first cache-enabled block in each step
             block_end=args.single_block_end              # last cache-enabled block in each step
         )
     ```

  3. Initialize the cache variable in the transformer `init` method.

     ```python
     self.cache = None
     ```

  4. Create `CacheAgent` and assign it to the block.

     ```python
     cache_agent = CacheAgent(config)
     # enable DiTCache
     pipeline.transformer.cache = CacheAgent(cache_config)
     ```

  5. Call `apply` in the transformer `forward` path. The first argument is the block itself, and the remaining arguments match the original implementation.

     ```python
     for index_block, block in enumerate(self.transformer_blocks):
       # enable DiTCache
       hidden_states, encoder_hidden_states = self.cache.apply(
           block,
           hidden_states=hidden_states,
           encoder_hidden_states=encoder_hidden_states,
           encoder_hidden_states_mask=encoder_hidden_states_mask,
           temb=temb,
           image_rotary_emb=image_rotary_emb,
           joint_attention_kwargs=attention_kwargs,
           txt_pad_len=txt_pad_len
       )
     ```

- **Example**

  See the [cache example](../../../examples/cache/cache.py) for a concrete implementation.

---

## AttentionCache

- **Background**

  Inference iterates over `T` steps, and each step contains multiple blocks with expensive compute, including STA, as shown below. Attention layers in blocks across adjacent steps are often highly similar, so nearly identical intermediate results are recomputed repeatedly and inference becomes slower.

  ![](../../zh/figures/attentioncache-image-1.png)

- **Principle**

  Unlike DiTCache, AttentionCache reuses the computed Attention result inside a block and skips selected Attention layers based on similarity across adjacent timesteps, reducing redundant computation and improving inference speed.

- **Optimization method**

  A search script first computes the minimum number of Attention executions that must be skipped for the target acceleration ratio. It then scans candidate start and end steps and chooses the configuration with the lowest MSE among all valid combinations. The key idea is to trade space for time by directly reusing cached Attention results from step `N` at step `M`.

  ![](../../zh/figures/attentioncache-image-2.png)

  1. Compute the minimum number of Attention executions that must be skipped from the requested speedup ratio.
  2. Based on the starting step and `min_skip_attention`, derive `min_interval` and `step_end`, traverse all valid candidates, and pick the one with minimum MSE loss.
  3. Apply the resulting configuration and enable AttentionCache during inference.

- **Workflow**

  1. Import `CacheConfig` and `CacheAgent`.

     ```python
     from mindiesd import CacheConfig, CacheAgent
     ```

  2. Initialize `CacheConfig`. For `attention_cache`, `block_start`, and `block_end`, the default values are usually sufficient.

     ```python
     config = CacheConfig(
                 method="attention_cache",
                 blocks_count=len(transformer.transformer_blocks), # number of blocks in the transformer
                 steps_count=args.infer_steps,                # total inference steps
                 step_start=args.start_step,                  # first step index that uses cache
                 step_interval=args.attentioncache_interval,  # forced recompute interval
                 step_end=args.end_step                       # last step index that uses cache
             )
     ```

  3. Initialize the cache variable in each transformer block.

     ```python
     self.cache = None
     ```

  4. Create `CacheAgent` and attach it to each block.

     ```python
     cache_agent = CacheAgent(config)
     # cache only the attention part inside each block
     for block in transformer.transformer_blocks:
         block.cache = cache_agent
     ```

  5. Use `apply` inside the block `forward` method. The first argument is the original attention function and the remaining arguments match the original implementation.

     ```python
     # enable attention cache
     attn_output = self.cache.apply(
         self.attn,
         hidden_states=img_modulated,
         encoder_hidden_states=txt_modulated,
         encoder_hidden_states_mask=encoder_hidden_states_mask,
         image_rotary_emb=image_rotary_emb,
         **joint_attention_kwargs,
     )
     ```

- **FAQ**

  Q: Why does Qwen-Image-Edit-2509 report `RuntimeError: NPU out of memory` after AttentionCache is enabled?

  A: AttentionCache increases graphics memory usage. On a single card, memory can become insufficient. Eight-card inference is recommended for this workload.

---

## Timestep optimization

- **Principle**

  By reducing, adjusting, or skipping selected denoising steps in diffusion models, timestep optimization lowers the number of executed DiT modules and avoids redundant compute while trying to preserve output quality.

- **Optimization method**

  - Modify the timestep count directly, for example from 50 steps down to 20 steps, to improve inference speed.
  - Use Adastep sampling, an adaptive and dynamic timestep-skipping algorithm. Its core idea is to evaluate the current latent state during inference and skip groups of steps whose changes are small enough to allow faster convergence. This method is currently used only in CogVideoX; on other models it has been replaced by DiTCache and AttentionCache.
