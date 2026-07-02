# DiTCache

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T09:59:59.641Z pushedAt=2026-06-08T02:24:29.092Z -->

During inference, diffusion models iterate over multiple time steps, with all blocks participating in computation at each step. However, the similarity of latents between consecutive steps leads to substantial redundant computation. To address this, MindIE SD provides the following caching mechanisms to reduce repetitive computations:

- **DiTCache**: Caches and reuses intermediate results at the block granularity, suitable for models with a large number of blocks.

- **AttentionCache**: Caches and reuses attention computation results at the Attention layer granularity, suitable for models where attention computation accounts for a high proportion of the workload.

- **Time step optimization**: Reduces or skips  certain time steps during diffusion, suitable for scenarios that require fine-grained control over the number of inference steps.

Each caching strategy can be used independently, or combined as best fits the model characteristics.

- **DiTCache (preferred)**: block-level caching with the best generality; recommended as the first option.

- **AttentionCache (alternative)**: offers finer granularity than DiTCache, suitable for models where attention dominates computation.

- **Time Step Optimization (auxiliary)**: complements other caching strategies to further reduce steps on top of DiTCache or AttentionCache.

## API Description

### `CacheConfig`

Cache configuration class, defining parameters such as the cache method, number of blocks, and number of steps.

```python
from mindiesd import CacheConfig
```

| Parameter       | Type  | Required | Default | Description                                              |
| --------------- | ----- | -------- | ------- | -------------------------------------------------------- |
| `method`        | `str` | Yes      | -       | Cache method, `"attention_cache"` or `"dit_block_cache"` |
| `blocks_count`  | `int` | Yes      | -       | Number of blocks per step                                |
| `steps_count`   | `int` | Yes      | -       | Total number of iteration steps                          |
| `step_start`    | `int` | No       | `0`     | Start step for caching                                   |
| `step_interval` | `int` | No       | `1`     | Caching interval in steps                                |
| `step_end`      | `int` | No       | `10000` | End step for caching                                     |
| `block_start`   | `int` | No       | `0`     | Start block index for caching                            |
| `block_end`     | `int` | No       | `10000` | End block index for caching                              |

### `CacheAgent`

Cache agent class, managing the application of cache based on the configuration.

```python
from mindiesd import CacheAgent
```

**Constructor Parameters**:

| Parameter | Type          | Required | Default | Description                |
| --------- | ------------- | -------- | ------- | -------------------------- |
| `config`  | `CacheConfig` | Yes      | -       | Cache configuration object |

**`apply`**:

```python
apply(function: callable, *args, **kwargs) -> Any
```

| Parameter  | Type       | Required | Description                                    |
| ---------- | ---------- | -------- | ---------------------------------------------- |
| `function` | `callable` | Yes      | The function to execute (block or attn module) |
| `*args`    | -          | No       | Positional arguments for the function          |
| `**kwargs` | -          | No       | Keyword arguments for the function             |

## DiTCache

- **Background**

During inference, the DiT model iterates over T steps. At each step t, all blocks are recomputed—each involving heavy computations (see figure below). However, the latents between adjacent steps are highly similar, leading to repeated computation of nearly identical intermediate results. This redundancy significantly slows down inference.

  ![](../../figures/dit_cache_image_1.png)

<br>

- **Principle**

Leverage activation similarity between adjacent sampling steps or blocks to reuse local features, skip specified DiT blocks, reduce redundant computation, and accelerate model inference.

<br>

- **Optimization Method**

A search script first determines the minimum number of blocks to skip based on the speedup ratio. It then iterates from the start and end blocks to find the configuration that minimizes MSE loss. The core optimization is that, on a cache hit, the computed results of a specific block range from step \(N\) are directly reused in step \(M\), turning the entire DiTBlock forward pass into a simple tensor read operation.

  ![](../../figures/dit_cache_image_2.png)

1. Calculate the minimum number of blocks that need to be cached based on the speedup ratio.

2. In each step, since block0 needs to be computed, start traversing from block1 and find the three smallest sets of block start and block end through computation.

3. Traverse all possible combinations obtained in the previous step, calculate the MSE before and after model caching, and find the configuration with the minimum MSE loss as the optimal solution.

4. Apply the resulting configuration to the model and enable caching during inference to achieve acceleration.

<br>

- **Optimization Process**

  1. Call the `CacheConfig` and `CacheAgent` interfaces.

      ```python
      from mindiesd import CacheConfig, CacheAgent
      ```

  2. Initialize `CacheConfig` in the model initialization method.

      ```python
      config = CacheConfig(
              method="dit_block_cache",
              blocks_count=len(transformer.single_blocks), # Number of blocks for which cache is enabled
              steps_count=args.infer_steps,                # Total number of iteration steps for model inference
              step_start=args.cache_start_steps,           # Step index at which caching starts
              step_interval=args.cache_interval,           # Interval of steps for forced recomputation
              step_end=args.infer_steps-1,                 # Step index at which caching stops
              block_start=args.single_block_start,         # Block index at which caching starts in each step
              block_end=args.single_block_end              # Block index at which caching stops in each step
          )
      ```

  3. Initialize the cache variable in the Transformer's `init` method.

      ```python
      self.cache = None
      ```

  4. Initialize `CacheAgent` and assign it to the block.

      ```python
      cache_agent = CacheAgent(config)
      # Enable ditcache
      pipeline.transformer.cache = CacheAgent(config)
      ```

  5. Use the `apply` method in the Transformer's `forward` method to enable the cache for inference. The first argument of the `apply` method is the block, and the remaining parameters remain consistent with the original code.

      ```python
      for index_block, block in enumerate(self.transformer_blocks):
        # Enable ditcache
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

<br>

- **Example**

  For details, see the [cache directory](../../../examples/cache/cache.py) under examples.

## AttentionCache

- **Background**

During inference, the model iterates through T steps. Each step consists of multiple blocks, and each block involves extensive computations, such as STA (see figure below). However, attention layers in blocks across adjacent steps are highly similar, causing the model to repeatedly compute nearly identical intermediate results. This redundancy slows down inference.

![](../../figures/attention_cache_image_1.png)

<br>

- **Principle**

Based on the feature similarity between adjacent time steps, unlike DitCache, AttentionCache reuses attention outputs from previous blocks to skip certain attention layers, reducing redundant computation and accelerating model inference.

<br>

- **Optimization Method**

A search script computes the minimum number of Attention skips required based on the speedup ratio. It then iterates over possible start and end steps to find the configuration with the lowest MSE loss as the optimal solution. The core optimization  leverages the space-for-time principle by reusing cached Attention layer outputs from step N directly in step M, significantly reducing Attention computation overhead.

  ![](../../figures/attention_cache_image_2.png)

1. Based on the acceleration ratio, compute the minimum number of attention skips.

2. Using the start step and `min_skip_attention`, derive `min_interval` and `step_end`, then iterate over all possible configurations. For each, compute the MSE of model outputs with and without cache, and select the configuration with minimal MSE loss as the optimal solution.

3. Apply the configuration to the model and enable caching during inference to achieve acceleration.

<br>

- **Optimization Process**

  1. Call the `CacheConfig` and `CacheAgent` interfaces.

      ```python
      from mindiesd import CacheConfig, CacheAgent
      ```

  2. Initialize `CacheConfig`. For `attention_cache`, `block_start`, and `block_end`, default values can be used.

      ```python
      config = CacheConfig(
                  method="attention_cache",
                  blocks_count=len(transformer.transformer_blocks), # Number of blocks contained in the transformer model
                  steps_count=args.infer_steps,                # Total number of iteration steps for model inference
                  step_start=args.start_step,                  # Step index at which caching starts
                  step_interval=args.attentioncache_interval,  # Interval of steps for forced recomputation
                  step_end=args.end_step                       # Step index at which caching stops
              )
      ```

  3. Initialize the `cache` variable in the `init` method of the Transformer's block module.

      ```python
      self.cache = None
      ```

  4. Initialize `CacheAgent` and assign it to the block.

      ```python
      cache_agent = CacheAgent(config)
      # Cache the attention part within the block
      for block in transformer.transformer_blocks:
          block.cache = cache_agent
      ```

  5. In the `forward` method of the Transformer's block module, use the `apply` method to enable the cache for inference. The first argument of this method is the original method for performing inference, and the remaining arguments remain consistent with the original code.

      ```python
      # Enable attention cache
      attn_output = self.cache.apply(
          self.attn,
          hidden_states=img_modulated,
          encoder_hidden_states=txt_modulated,
          encoder_hidden_states_mask=encoder_hidden_states_mask,
          image_rotary_emb=image_rotary_emb,
          **joint_attention_kwargs,
      )
      ```

- **FAQs**

**Q:** Enabling AttentionCache for Qwen-Image-Edit-2509 causes a `RuntimeError: NPU out of memory`.

  A: AttentionCache increases memory consumption, which can easily lead to out-of-memory errors on a single card. You are advised to use eight cards for inference.

## Time Step Optimization

- **Principle**

Reduce, restructure, or skip  certain denoising steps in the diffusion process to minimize the number of DiT modules and eliminate redundant computations—without significant loss of accuracy—thereby accelerating model inference.

<br>

- **Optimization Method**

  <ul>
      <li>Adjust timestep values: For example, reducing from 50 to 20 improves inference speed.</li>
      <li>Adastep sampling: An adaptive, dynamic timestep-skipping algorithm that evaluates the current latent state in real time during inference and dynamically skips steps with minimal inter-step differences to achieve faster convergence. Currently, this method is only used in CogVideoX and has been replaced by DiTCache and AttnCache in other models. </li>
  </ul>
