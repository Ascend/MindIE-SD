# Quantization

MindIE SD provides two types of quantization capabilities, applied to different parts of the model:

- **Linear Quantization**: Applies low-bit processing (INT8/FP8/W8A16, etc.) to linear layer weights and activations, reducing model storage and computation overhead.
- **FA Quantization**: Applies FP8 block quantization to Q/K/V activations in attention computation, reducing memory bandwidth requirements.

The following two sections describe the principles and usage of each quantization type.

## Linear Quantization

### General Principles

Quantization is the process of mapping model weights and activations from high precision (e.g., FP32) to low precision (e.g., INT8, FP8). Low-precision computation reduces memory usage and bandwidth requirements, improving inference throughput.

Based on whether retraining is needed, quantization is divided into Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). This section focuses on PTQ quantization, which falls into three main types:

- **Dynamic Quantization**: Only weights are quantized offline; activation quantization factors are dynamically computed during inference.
- **Static Quantization**: Both weights and activations are quantized offline.
- **Time-Aware Quantization**: Quantization strategy is dynamically adjusted based on the time dimension.

The following diagram shows an INT8 quantization example, mapping FP32 to INT8. `[-max(xf), max(xf)]` is the pre-quantization float range, and `[-128, 127]` is the post-quantization range.

![](../../figures/int8_image.png)

### Technical Features

This repository handles Linear quantization uniformly through the `quantize` API, supporting the following algorithms.

**Weight Quantization** (only weights are quantized; activations remain at original precision):

| Algorithm | Weight Precision | Description |
| ------ | ---------- | ------ |
| W8A16 | INT8 | Basic weight quantization |
| W4A16 | INT4 | Higher compression ratio |
| W4A16_AWQ | INT4 + AWQ | Activation-aware weight quantization |
| W8A16_GPTQ | INT8 + GPTQ | GPTQ post-training weight quantization |
| W4A16_GPTQ | INT4 + GPTQ | Same as above, INT4 version |

**Weight-Activation Quantization** (both weights and activations are quantized; computation done in low precision):

| Algorithm | Quantization Granularity | Description |
| ------ | ---------- | ------ |
| W8A8 | Per-layer | Basic INT8 weight-activation quantization |
| W8A8_TIMESTEP | Per-layer + timestep | Dynamically switch quantization strategy during inference |
| W8A8_DYNAMIC | Per-layer | Dynamic activation quantization |
| W8A8_PER_CHANNEL | Per-channel | Channel-granularity quantization |
| W8A8_PER_TENSOR | Per-tensor | Tensor-granularity quantization |
| W8A8_MXFP8 | Per-layer | MXFP8 format quantization |
| W4A4_DYNAMIC | Per-token + per-channel | INT4 weight-activation quantization |
| W4A4_MXFP4_SVD | Per-layer | MXFP4 format quantization |
| W4A4_MXFP4_DUALSCALE | Per-layer | MXFP4 dual-scale quantization |
| W4A4_MXFP4_DYNAMIC | Per-token + per-channel | MXFP4 dynamic quantization |

### API and Usage

All Linear quantization algorithms are triggered uniformly through the `quantize` API.

```python
from mindiesd import quantize
```

#### Parameters

| Parameter | Type | Required | Default | Description |
| ------ | ------ | ------ | -------- | ------ |
| `model` | `nn.Module` | Yes | - | Initialized floating-point model |
| `quant_des_path` | `str` | No | `None` | Path to the quantization descriptor JSON file. If not provided as a positional argument, configure `QuantConfig.quant_des_path`. |
| `quant_config` | `QuantConfig` | No | `None` | Unified quantization configuration. Supports `quant_des_path`, `dtype`, `use_nz`, time-step strategy, and `mxfp4_scale_alg`. |

`quantize` first parses the quantization descriptor JSON into a `QuantConfig`, then merges it with the user-provided `quant_config`. If the same field exists in both, the user-provided value takes precedence. Legacy parameters—`timestep_config`, `timestep_policy`, `dtype`, and `use_nz`—are still supported and are internally mapped to `QuantConfig`. The quantization descriptor path can be passed either as the second argument to `quantize` or via `QuantConfig(quant_des_path=...)`.

#### Usage Examples

Basic quantization:

```python
model = from_pretrain()
model = quantize(model, "quant_model_description_w8a16_0.json")
model.to("npu")
```

Equivalent configuration syntax:

```python
from mindiesd import QuantConfig, quantize

quant_config = QuantConfig(quant_des_path="quant_model_description_w8a16_0.json")
model = quantize(model, quant_config=quant_config)
model.to("npu")
```

Timestep quantization:

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 10), "static", target="w8a8_static_linear")

quant_config = QuantConfig(timestep_config=timestep_policy)
model = quantize(model, "quant_model_description_w8a8_timestep_0.json", quant_config=quant_config)

for i, t in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    ...
```

MXFP4 time-step rollback.

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig, quantize

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 4), "W4A8", target="w4a4_linear")
timestep_policy.register(range(4, 50), "W4A4", target="w4a4_linear")

quant_config = QuantConfig(
    timestep_config=timestep_policy,
    mxfp4_scale_alg=2,
)

model = quantize(model, "quant_model_description_w4a4_mxfp4_0.json", quant_config=quant_config)

for i, timestep in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    noise_pred = model(latents, timestep, encoder_hidden_states)
```

On the model side, `TimestepManager.set_timestep_idx(i)` must be set before each denoising step. This is consistent with the step‑by‑step iteration over `timesteps` used in Wan2.2 `wan/text2video.py`, but the policy semantics differ in this repository: here, the Linear policy toggles between `W4A4` and `W4A8`, rather than switching between dynamic and static quantization. The Linear/MM fallback only changes the activation quantization precision; weights remain in MXFP4.

#### Quantized Weight File Naming

Quantized weights and descriptor files are exported by the msmodelslim tool with the following naming convention:

- Weight file: `quant_model_weight_{quant_algo.lower()}_{rank}.safetensors`
- Descriptor file: `quant_model_description_{quant_algo.lower()}_{rank}.json`

For single-card quantization, `rank` is 0. For multi-card parallelism, each rank corresponds to its respective number.

## FA Quantization

### General Principles

FA (Flash Attention) quantization applies low-bit processing to Q/K/V activations in attention computation. Quantizing Q/K/V to FP8 before feeding into attention computation kernels significantly reduces memory bandwidth requirements and improves inference throughput. Unlike weight quantization, FA quantization processes dynamically generated activations during inference and requires block-level dynamic quantization strategies to balance accuracy and acceleration.

### Technical Features

This repository provides FA quantization via the `FP8_DYNAMIC` algorithm, with a three-step processing flow:

**Rotate**

Apply pre-trained rotation matrices (`q_rot`, `k_rot`) to Q and K, dispersing outliers across dimensions to mitigate FP8 quantization sensitivity to outliers.

**Block Quantization**

Dynamically quantize rotated Q/K/V into FP8 (`float8_e4m3fn`) block by block. Q uses a block size of 128, K/V use a block size of 256, performed via the `npu_dynamic_block_quant` operator.

**FP8 Attention**

Invoke MindIE-SD's own `torch.ops.mindiesd.fused_infer_attention_score_v2` operator, which routes to the migrated `FusedInferAttentionScore` implementation
in this repository, to perform attention computation in the FP8 domain with
outputs dequantized back to original precision.

### API Reference

FA quantization is triggered uniformly through the `quantize` API; no separate FA quantization API call is needed.

```python
from mindiesd import quantize
```

#### Usage Example

```python
from mindiesd import quantize

# Load original floating-point model
model = from_pretrain()

# Execute quantization conversion (automatically identifies Attention layers and injects FA quantization)
model = quantize(model, "path/to/exported/quantization/config")

# Move model to NPU and run inference
model.to("npu")
```

`quantize` internally traverses model layers, automatically calling `add_fa_quant` on matching Attention layers, injecting `FP8RotateQuantFA` modules, and replacing the forward computation with the rotate -> block quantize -> FP8 Attention flow.

FA quantization layers are implemented through the `FP8RotateQuantFA` module. See the rotate -> block quantize -> FP8 Attention flow description in this section.

MXFP4 FA usage example:

```python
from mindiesd import QuantConfig, TimestepManager, TimestepPolicyConfig, quantize

timestep_policy = TimestepPolicyConfig()
timestep_policy.register(range(0, 2), "FLOAT", target="fa")
timestep_policy.register(range(2, 8), "FP8", target="fa")
timestep_policy.register(range(8, 50), "MXFP4", target="fa")

quant_config = QuantConfig(
    timestep_config=timestep_policy,
    mxfp4_scale_alg=2,
)

model = quantize(model, "quant_model_description_mxfp4_dynamic_0.json", quant_config=quant_config)

for i, timestep in enumerate(timesteps):
    TimestepManager.set_timestep_idx(i)
    noise_pred = model(latents, timestep, encoder_hidden_states)
```

FA has no offline weight constraints, so any algorithm can be selected across different time steps. In contrast, Linear/MM can only switch activation precision under the same MXFP4 weight.

The `mxfp4_scale_alg` field in `QuantConfig` is passed through to the dynamic MX quantization path to align with the C7 inference parameters of the CANN `aclnnDynamicQuantV2` operator. If not set, the legacy interface default behavior is preserved. For details, refer to the [CANN `aclnnDynamicQuantV2` documentation](https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md). For an example of setting timesteps on the model side, see the [Wan2.2 text-to-video inference loop](https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/text2video.py).

#### Notes

- Hardware requirement: Only Atlas 800I A2 inference servers support this feature.
- Q/K/V input layout supports both `BNSD` and `BSND`.
- FA quantization weights (`q_rot`, `k_rot`) must be pre-exported using the msmodelslim model compression tool. See the msmodelslim tool documentation for details.
