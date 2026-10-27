# FLUX.1-dev Model Inference Optimization Guide

Enable MindIE SD compilation optimization and the Cache-DiT DBCache feature to accelerate FLUX.1-dev model inference.

> This document targets end users who deploy FLUX.1-dev inference services and is a runnable end-to-end example.
> For the implementation principles and interface descriptions of the features (CompilationConfig, CacheConfig, and so on), see the developer documentation [Compilation Features](../../docs/en/features/compilation.md) and [Compute by Storage](../../docs/en/features/cache.md).

---

## Environment Information

The experimental environment of this guide is as follows:

| Component | Version |
|------|------|
| Server | Atlas 910B4 (HBM 64GB) |
| Python | 3.11 |
| diffusers | 0.36.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0 |
| mindiesd | 3.0.0 |
| cache_dit | 1.2.3 |

---

## 1. Prerequisites

### 1.1 Basic Environment Installation (Ascend NPU)

Complete the basic environment setup as described in the [MindIE SD Installation Guide](../../docs/en/installation.md), including:

- Driver and firmware installation
- CANN installation
- PyTorch and Torch NPU installation
- MindIE SD installation

### 1.2 Install Diffusers and Cache-DiT

After the basic environment is installed, install the following Python packages:

```bash
# Install Diffusers (a version that supports FLUX.1-dev)
pip install diffusers

# Install Cache-DiT
pip install cache-dit
```

### 1.3 Download the Model Weights

- The original weights are available from [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) (gated model, authentication required)
- For users in China, weights are available from [ModelScope](https://modelscope.cn/models/AI-ModelScope/FLUX.1-dev)

---

## 2. Model Inference Example

### 2.1 Quickest Verification (Based on the cache-dit CLI)

No inference code is required: use the `generate` command shipped with cache-dit to quickly verify the effect of MindIE SD compilation optimization:

```bash
# Set the FLUX.1-dev weights path
export FLUX_PATH=/path/to/FLUX.1-dev

# Without compilation (baseline)
python3 -m cache_dit.generate flux --model-path $FLUX_PATH

# Enable MindIE SD compilation optimization
python3 -m cache_dit.generate flux --model-path $FLUX_PATH --compile
```

Compare the time cost of the two runs to verify the acceleration effect of compilation optimization. On the NPU, `--compile` automatically compiles with `MindieSDBackend`.

### 2.2 Standard Inference Based on diffusers

The following is standard inference code based on the diffusers library, with no optimization enabled

```python
#!/usr/bin/env python3
import torch
import torch_npu
from diffusers import FluxPipeline

# Initialize the NPU environment
DEVICE_ID = 0
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"

# Load the model weights
model_path = ""  # Path where the model weights are stored
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# Prepare the input
prompt = "A cat holding a sign that says hello world"

# Run model inference
with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    output.save("flux_out.png")
```

### 2.3 Enabling Compilation Optimization (cache-dit Natively Supports MindIE SD)

When cache-dit detects that mindiesd is installed in an Ascend NPU environment, it automatically uses MindIE SD's `MindieSDBackend()` to apply compilation optimization to the transformer (equivalent to `torch.compile(pipe.transformer, backend=MindieSDBackend())`), and operators such as RMSNorm, RoPE, AdaLayerNorm, and fastGELU are automatically replaced with fused operators, **without modifying the inference code**. The first inference takes longer because it includes compilation warm-up; only runs from the second one onward reflect the real inference speed.

```bash
python3 -m cache_dit.generate flux --model-path $FLUX_PATH --compile
```

> [!NOTE]NOTE
> Starting from torch_npu 2.9 (MR 30358), the NPU over-adaptation of `Tensor.to` has been removed, and the dtype cast in the compiled graph changes from `torch.ops.npu._npu_dtype_cast` to `torch.ops.aten._to_copy`. MindIE SD 3.0.0 already adapts automatically to the torch version, so no extra configuration is required. If RMSNorm / RoPE fusion does not take effect in a torch 2.9+ environment, confirm that your MindIE SD version includes the adaptation for that version.

### 2.4 Enabling the Cache-DiT DBCache Feature

**Note**: Before enabling DBCache, set the environment variable `export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`; otherwise, out-of-memory errors may occur.

The FLUX.1-dev transformer contains dual-stream (`transformer_blocks`) and single-stream (`single_transformer_blocks`) blocks, which require separate cache configuration. For the recommended parameter combination verified in this environment, see the complete script in 2.5: the core setting is `residual_diff_threshold=0.4`, and because the error of single-stream blocks accumulates from dual-stream blocks, the threshold is set to three times that value (1.2). Call `cache_dit.refresh_context()` before each inference to refresh the cache context. For the meaning of parameters such as `DBCacheConfig`, `BlockAdapter`, and `ParamsModifier` and for more configuration options, see the [Cache-DiT CACHE_API documentation](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/).

### 2.5 Complete Single-Card Model Inference Example

The complete inference script `flux_optimized_infer.py`, which combines compilation optimization (enabled automatically by the CLI, see 2.3) with fine-grained Cache-DiT DBCache configuration, is as follows:

```python
#!/usr/bin/env python3
import torch
import torch_npu
import cache_dit
from diffusers import FluxPipeline
from cache_dit import (
    BlockAdapter,
    ForwardPattern,
    ParamsModifier,
    DBCacheConfig,
)

# Initialize the NPU environment
DEVICE_ID = 0
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"

# Load the model weights
model_path = ""  # Path where the model weights are stored
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# Enable the Cache-DiT DBCache feature
# (MindIE SD compilation optimization is enabled automatically by cache-dit in CLI --compile mode and does not need to be configured here)
cache_dit.enable_cache(
    BlockAdapter(
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_1,
        ],
    ),
    cache_config=DBCacheConfig(
        Fn_compute_blocks=2,
        Bn_compute_blocks=1,
        max_warmup_steps=4,
        max_cached_steps=8,
        max_continuous_cached_steps=4,
        residual_diff_threshold=0.4,
    ),
    params_modifiers=[
        ParamsModifier(
            cache_config=DBCacheConfig().reset(residual_diff_threshold=0.4),
        ),
        ParamsModifier(
            cache_config=DBCacheConfig().reset(residual_diff_threshold=0.4 * 3),
        ),
    ],
)

# Prepare the input
prompt = "A cat holding a sign that says hello world"

# Run model inference
with torch.inference_mode():
    for i in range(2):
        # Refresh the cache context before each inference
        cache_dit.refresh_context(
            pipe.transformer,
            num_inference_steps=28,
        )
        output = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        output.save(f"flux_out_{i}.png")
```

---

## References

- [MindIE SD Compilation Features](../../docs/en/features/compilation.md)
- [Cache-DiT Documentation](https://gitcode.com/vipshop/cache-dit)
- [FLUX.1-dev Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
