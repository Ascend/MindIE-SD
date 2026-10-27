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

![INT8 quantization diagram](../../figures/int8_image.png)

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
| `quant_des_path` | `str` | No | `None` | Path to the quantization descriptor JSON. When it is not passed as a positional argument, `QuantConfig.quant_des_path` must be set. |
| `quant_config` | `QuantConfig` | No | `None` | Unified quantization config that carries `quant_des_path`, `dtype`, `use_nz`, the timestep policy, and `mxfp4_scale_alg`. |

`quantize` first parses the quantization descriptor JSON into a `QuantConfig` and then merges it with the user-provided `quant_config`. When the same field appears in both, the user-provided config takes precedence. The legacy `timestep_config`, `timestep_policy`, `dtype`, and `use_nz` arguments are still accepted and are folded into `QuantConfig` internally. The quantization descriptor path can be passed as the second positional argument of `quantize` or set through `QuantConfig(quant_des_path=...)`.

#### Usage Examples

Basic quantization:

```python
model = from_pretrain()
model = quantize(model, "quant_model_description_w8a16_0.json")
model.to("npu")
```

Equivalent config-based form:

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

MXFP4 timestep fallback:

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

On the model side, call `TimestepManager.set_timestep_idx(i)` before each denoise step. This matches how Wan2.2 `wan/text2video.py` iterates over `timesteps` step by step, but the policy semantics in this repository differ: here Linear switches between `W4A4` and `W4A8` rather than between dynamic and static quantization. The Linear/MM fallback switches only the activation quantization precision, while the weights stay in MXFP4.

#### Quantized Weight File Naming

Quantized weights and descriptor files are exported by the msmodelslim tool with the following naming convention:

- Weight file: `quant_model_weight_{quant_algo.lower()}_{rank}.safetensors`
- Descriptor file: `quant_model_description_{quant_algo.lower()}_{rank}.json`

For single-card quantization, `rank` is 0. For multi-card parallelism, each rank corresponds to its respective number.

#### Quantization Descriptor Validation

`quantize` validates the complete descriptor before constructing quantization configs or opening the weight file. The supported MindIE SD format is a flat JSON object:

```json
{
  "model_quant_type": "W8A8",
  "blocks.0.proj.weight": "W8A8",
  "blocks.0.proj.bias": "FLOAT"
}
```

- The file must be a UTF-8 standard JSON object without duplicate fields, unescaped control characters, or nonstandard constants such as `NaN` and `Infinity`.
- `model_quant_type` is required. Its value must be a string that exactly matches a `QuantAlgorithm` enum value, such as `W8A8` or `W4A4_MXFP4`. It is file-level metadata, not a layer config.
- Known AscendV1 exporter metadata is optional: `version`, `kv_cache_type`, `kv_quant_type`, `fa_quant_type`, and `reduce_quant_type` must be strings; `group_size` must be a non-negative integer, excluding booleans; `metadata` and `optional` must be JSON objects. The string values of fields that MindIE SD does not consume are intentionally not restricted to today's exporter enum values, so newer exporters remain forward compatible. These fields are validated and excluded from layer configs; they do not enable grouped, KV-cache, communication, or rotation features. Objects inside `metadata` and `optional` are exporter extensions and are not interpreted or used to open additional files.
- All remaining fields are dynamic layer or tensor names. Names must not be entirely whitespace or contain ASCII control characters. Values must be strings containing a `QuantAlgorithm` enum value (case-insensitive) or exactly `FLOAT`.
- Fields that violate this format are rejected instead of silently ignored. Dynamic names are not a fixed whitelist: an unfamiliar name with a valid algorithm value is a layer entry. All JSON numbers, including those in metadata objects, must be finite.
- Validation failures raise `ParametersInvalid` with the field name or JSON line and column. The weight file is not opened after a validation failure.

This checks the descriptor structure, types, and values. It does not verify that dynamic names correspond to actual model layers, or validate weight tensor shapes, values, or quantization accuracy. Existing path, symbolic link, file size, and permission checks still apply.

## FA Quantization

### FA General Principles

FA (Flash Attention) quantization applies low-bit processing to Q/K/V activations in attention computation. Quantizing Q/K/V to FP8 before feeding into attention computation kernels significantly reduces memory bandwidth requirements and improves inference throughput. Unlike weight quantization, FA quantization processes dynamically generated activations during inference and requires block-level dynamic quantization strategies to balance accuracy and acceleration.

### FA Technical Features

This repository provides FA quantization via the `FP8_DYNAMIC` and `MXFP4_DYNAMIC` algorithms, with a three-step processing flow:

#### Rotate

If the quantized weights contain pre-trained rotation matrices (`q_rot`, `k_rot`), apply them to Q and K so outliers are spread across dimensions and FP8 quantization is less sensitive to those outliers. If the two tensors are absent, rotation is skipped and module initialization still succeeds.

#### Block Quantization

Dynamically quantize Q/K/V into FP8 (`float8_e4m3fn`) block by block via `npu_dynamic_block_quant`. Q uses a 128-token block and K uses a 256-token block; the V token block is selected by `FP8FAMode` below.

#### FP8 Attention

Invoke MindIE SD's own `torch.ops.mindiesd.fused_infer_attention_score_v2` operator, which routes to the migrated `FusedInferAttentionScore` implementation
in this repository, to perform attention computation in the FP8 domain with
outputs dequantized back to original precision. `FP8RotateQuantFA` selects the quantization blocks, `value_quant_mode`, and `inner_precise` together through the `FP8FAMode` enum, so those switches stay aligned.

| Mode | Q/K/V quant mode | `inner_precise` | V quant block |
| ------ | ------ | ------ | ------ |
| `HIGH_PRECISION` (default) | `7/7/7` | omitted (original high-precision path) | 256×128 |
| `C8V16_TILING512` | `7/7/12` | `4` (C8V16) | 512×64 |

Later features should be added as new enum members instead of unbound quant switches.

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

`quantize` internally traverses model layers, automatically calling `add_fa_quant` on matching Attention layers, injecting `FP8RotateQuantFA` modules, and replacing the forward computation with the rotate -> block quantize -> FP8 Attention flow. The default mode is `FP8FAMode.HIGH_PRECISION`. To use C8V16 with a 512-token V tile:

```python
from mindiesd import FP8FAMode, QuantConfig, quantize

model = quantize(
    model,
    "path/to/exported/quantization/config",
    quant_config=QuantConfig(fp8_fa_mode=FP8FAMode.C8V16_TILING512),
)
```

FA quantization layers are implemented through the `FP8RotateQuantFA` module. See the rotate -> block quantize -> FP8 Attention flow description in this section. `C8V16_TILING512` requires the FIA C8V16 + V512 path (`inner_precise=4`, `value_quant_mode=12`).

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

FA has no offline weight constraint, so any algorithm can be selected at different timesteps; Linear/MM can only switch activation precision under the same MXFP4 weights.

`mxfp4_scale_alg` in `QuantConfig` is passed through to the dynamic MX quantization path to align with the C7 inference parameters of CANN `aclnnDynamicQuantV2`; when it is not set, the legacy interface default is preserved. For the related API, see the CANN `aclnnDynamicQuantV2` documentation: <https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md>. For how the model side sets the timestep, see the Wan2.2 text-to-video inference loop: <https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/text2video.py>.

#### Notes

- Hardware requirement: Only Atlas 800I A2 inference server support this feature.
- Q/K/V input layout supports both `BNSD` and `BSND`.
- K/V may have fewer heads than Q (GQA, Grouped Query Attention). `FP8RotateQuantFA` passes `num_key_value_heads` from the K head count.
- Rotation matrices (`q_rot`, `k_rot`) are optional. When used, they must be pre-exported with the msmodelslim model compression tool; see the msmodelslim documentation. If the two tensors are absent, `FP8RotateQuantFA` skips rotation.
