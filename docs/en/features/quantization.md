# Quantization

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:14:59.801Z pushedAt=2026-06-08T08:22:41.196Z -->

MindIE SD provides two types of quantization capabilities, which act on different parts of the model:

- **Linear quantization**: Applies low-bit (INT8/FP8/W8A16, etc.) quantization to the weights and activations of linear layers, reducing model storage and computational overhead.

- **FA quantization**: Applies FP8 block quantization on Q/K/V activations in attention mechanisms to lower memory bandwidth demand for attention computation.

The following two sections introduce the principles and usage methods of the two quantization types respectively.

## Linear Quantization

### General Principles

Quantization maps model weights and activations from high precision (e.g., FP32) to lower precision (e.g., INT8, FP8). Low-precision computation reduces memory usage and bandwidth demand, improving inference throughput.

Quantization is categorized into Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT), depending on whether retraining is required. This section focuses on PTQ and covers the following three types:

- **Dynamic quantization**: Only weights are quantized offline; activation scaling factors are computed dynamically during inference.

- **Static quantization**: Both weights and activations are quantized offline.

- **Time-aware quantization**: The quantization strategy is dynamically adjusted along the time dimension.

The following figure shows an INT8 quantization example, mapping FP32 to INT8. Here, `[-max(xf), max(xf)]` is the floating-point range before quantization, and `[-128, 127]` is the range after quantization.

![](../../figures/int8_image.png)

### Technical Features

This repository provides a unified `quantize` interface for linear quantization, supporting the following algorithms.

**Weight quantization** (only weights are quantized, activations remain at original precision):

| Algorithm | Weight Precision | Description |
|------|----------|------|
| W8A16 | INT8 | Basic weight quantization |
| W4A16 | INT4 | Higher compression ratio |
| W4A16_AWQ | INT4 + AWQ | Activation-aware weight quantization |
| W8A16_GPTQ | INT8 + GPTQ | Weight quantization based on GPTQ post-training |
| W4A16_GPTQ | INT4 + GPTQ | Same as above, INT4 version |

**Weight activation quantization** (both weights and activations are quantized, and computation is performed at low precision):

| Algorithm | Quantization Granularity | Description |
|------|----------|------|
| W8A8 | Per-layer | Basic INT8 weight activation quantization |
| W8A8_TIMESTEP | Per-layer + Timestep | Dynamical switch of  quantization strategy during inference |
| W8A8_DYNAMIC | Per-layer | Dynamic activation quantization |
| W8A8_PER_CHANNEL | Per-channel | Quantization by channel granularity |
| W8A8_PER_TENSOR | Per-tensor | Quantization by tensor granularity |
| W8A8_MXFP8 | Per-layer | MXFP8 format quantization |
| W4A4_DYNAMIC | Per-token + Per-channel | INT4 weight activation quantization |
| W4A4_MXFP4_SVD | Per-layer | MXFP4 format quantization |
| W4A4_MXFP4_DUALSCALE | Per-layer | MXFP4 dual-scale quantization |
| W4A4_MXFP4_DYNAMIC | Per-token + Per-channel | MXFP4 dynamic quantization |

### API and Usage

All linear quantization algorithms are uniformly triggered through the `quantize` API.

```python
from mindiesd import quantize
```

#### Parameters

| Parameter| Type| Required/Optional| Default Value| Description|
|------|------|------|--------|------|
| `model` | `nn.Module` | Required| - | Initialized floating-point model|
| `quant_des_path` | `str` | Optional| `None` | Quantization descriptor JSON path. If this parameter is not passed as a positional parameter, `QuantConfig.quant_des_path` must be configured.|
| `quant_config` | `QuantConfig` | Optional| `None` | Unified quantization configuration, which can contain `quant_des_path`, `dtype`, `use_nz`, time step policy, and `mxfp4_scale_alg`.|

`quantize` parses the quantization descriptor JSON into `QuantConfig`, and then combines `QuantConfig` with `quant_config` passed by the user. If the same field exists at the same time, the configuration passed by the user is used. Legacy parameters (`timestep_config`, `timestep_policy`, `dtype, use_nz`) are still supported and are automatically mapped to `QuantConfig` internally. The quantization descriptor path can be passed as the second parameter of `quantize` or written to `QuantConfig(quant_des_path=...)`.

#### Example

Basic quantization:

```python
model = from_pretrain()
model = quantize(model, "quant_model_description_w8a16_0.json")
model.to("npu")
```

Equivalent configuration:

```python
from mindiesd import QuantConfig, quantize

quant_config = QuantConfig(quant_des_path="quant_model_description_w8a16_0.json")
model = quantize(model, quant_config=quant_config)
model.to("npu")
```

Time step quantization:

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

MXFP4 time step rollback:

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

On the model side, `TimestepManager.set_timestep_idx(i)` needs to be set before each denoise step. This is the same as the method of traversing `timesteps` by step in Wan2.2 `wan/text2video.py`. However, the policy semantics of this repository are different. Linear indicates the switching between `W4A4` and `W4A8`, not the original dynamic and static quantization switching. Linear/MM rollback only switches the quantization precision of activations, and the weight remains MXFP4.

#### Quantized Weight File Naming

Quantized weights and descriptor files are exported by the msmodelslim tool, with the following naming conventions:

- Weight file: `quant_model_weight_{quant_algo.lower()}_{rank}.safetensors`

- Descriptor file: `quant_model_description_{quant_algo.lower()}_{rank}.json`

For single-card quantization, `rank` is `0`. For multi-card parallelism, each rank corresponds to its own number.

## FA Quantization

### General Principles

Flash Attention (FA) quantization applies low-bit processing to Q/K/V activations in attention computation. By quantizing Q/K/V to FP8 before feeding them into the attention kernel, it significantly reduces memory bandwidth demand and improves inference throughput. Unlike weight quantization, FA quantization handles dynamically generated activations during inference, requiring block-level dynamic quantization to balance accuracy and speed.

### Technical Features

This repository provides FA quantization capability through the `FP8_DYNAMIC` algorithm, whose processing flow is divided into three steps:

**Rotate**

Apply pre-trained rotation matrices (`q_rot`, `k_rot`) to Q and K to distribute outliers across dimensions, mitigating the sensitivity of FP8 quantization to outliers.

**Block Quant**

Dynamically quantize the rotated Q/K/V into FP8 (`float8_e4m3fn`) per block, using the `npu_dynamic_block_quant` operator. The block size is 128 for Q and 256 for K/V.

**FP8 Attention**

Calls the Ascend `npu_fused_infer_attention_score_v2` kernel to perform attention computation in the FP8 domain. The output is then dequantized back to the original precision.

### API Description

FA quantization is triggered uniformly through the `quantize` API, without the need to call a separate FA quantization API.

```python
from mindiesd import quantize
```

#### Example

```python
from mindiesd import quantize

# Load the original floating-point model
model = from_pretrain()

# Perform quantization conversion (automatically identify Attention layers and inject FA quantization)
model = quantize(model, "Path to the exported quantization configuration file")

# Move the model to the NPU and perform inference
model.to("npu")
```

`quantize` internally traverses each layer of the model, automatically calls `add_fa_quant` on matched Attention layers, injects the `FP8RotateQuantFA` module, and replaces the forward computation with the process of rotation → block-wise quantization → FP8 Attention.

The FA quantization layer is implemented via the `FP8RotateQuantFA` module — see the Rotate → Block Quantization → FP8 Attention workflow in this section.

The following is an example of using MXFP4 FA:

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

FA has no offline weight constraint. Therefore, any algorithm can be selected at different time steps. Linear/MM can switch the activation precision only under the same MXFP4 weight.

`mxfp4_scale_alg` in `QuantConfig` is transparently transmitted to the dynamic MX quantization path to align the C7 inference parameters of CANN `aclnnDynamicQuantV2`. If `mxfp4_scale_alg` is not set, the default behavior of the old API is retained. For details about related APIs, see the CANN `aclnnDynamicQuantV2` document: <https://gitcode.com/cann/ops-nn/blob/master/quant/dynamic_quant_v2/docs/aclnnDynamicQuantV2.md>. For details about how to set the timestep on the model side, see the Wan2.2 text-to-video inference loop: <https://modelers.cn/models/MindIE/Wan2.2/blob/main/wan/text2video.py>.

#### **Precautions**

- Hardware requirement: Only the Atlas 800I A2 inference server supports this feature.

- Q/K/V input layout supports `BNSD` and `BSND`.

 FA quantized weights (`q_rot`, `k_rot`) must be pre-exported using msModelSlim. For details, see the msModelSlim tool documentation.
