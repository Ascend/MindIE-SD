# Empty Weight Run Verification

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:15:44.371Z pushedAt=2026-06-09T01:22:37.207Z -->

Construct generative models on NPU devices **without real weights** (only downloading config files, tens of KB), and perform forward inference validation.

## Directory Structure

```text
examples/dummy_run/
├── model/
│   ├── __init__.py                       # check_npu(), resolve_config_path(), _PhaseTimer
│   ├── wan_model.py                      # build_wan_pipeline()
│   ├── qwen_image_model.py               # build_qwen_image_pipeline()
│   └── flux_model.py                     # build_flux_pipeline()
├── wan_infer.py                          # Wan2.2 entry script
├── qwen_image_infer.py                   # Qwen-Image entry script
├── flux_infer.py                         # FLUX.1-dev entry script
├── requirements.txt                      # Dependency declaration
└── README.md
```

## Prerequisites

```shell
pip install -r examples/dummy_run/requirements.txt
```

| Dependency | Minimum Version |
|---|---|
| Python | 3.10 |
| torch / torch_npu | Matches CANN version |
| diffusers | >= 0.34.0 |
| transformers | >= 4.44.0 |
| huggingface_hub | >= 0.23.0 |

Ensure NPU is available: `npu-smi info -l`

## CLI Parameters (Unified for Three Models)

```shell
python <model>_infer.py --device_id <N> --num_layers <N>
```

| Parameter | Default Value | Description |
|---|---|---|
| `--device_id` | 0 | NPU device index |
| `--config_cache` | None | Local configuration file directory for offline usage|
| `--num_layers` | 2 | Number of Transformer layers |
| `--compile` | False | Whether to enable MindieSDBackend compilation |
| `--profile` | False | Whether to enable NPU profiling (level=l1) |
| `--skip-vae` / `--no-skip-vae` | True | Skips VAE decode by default. Use `--no-skip-vae` to enable VAE decode. |

## Configuration Cache

Configuration files (JSON, tokenizer, KB-level) are automatically downloaded to the Hugging Face Hub cache on first run, with subsequent runs requiring no network connection.
An offline cache directory can be specified via `--config_cache /path/to/config`.

For gated models (e.g., FLUX.1-dev), set `HF_TOKEN` or download via ModelScope and load with `--config_cache`.

## Verification Records (910B, 64GB HBM, NPU Direct Connection, 2 Layers)

| Model | Build (ms) | Timed (ms) | Peak Mem | Status |
|---|---|---|---|---|
| Wan2.2 | 1,200 | 7,000 | 10.18 GB | PASSED |
| Qwen-Image | 7,000 | 100 | 6.26 GB | PASSED |
| FLUX.1-dev | 20,500 | 900 | 24.20 GB | PASSED |

## Wan2.2

### Model Components

| Component | Class | Layers |
|---|---|---|
| Transformer | `WanTransformer3DModel` | 2 (original: 40) |
| Transformer_2 | `WanTransformer3DModel` | 2 |
| Text Encoder | `UMT5EncoderModel` | 2 (original: 28) |
| VAE | `AutoencoderKLWan` | — |
| Scheduler | `UniPCMultistepScheduler` | — |

### Usage

```shell
python wan_infer.py --device_id 0
python wan_infer.py --device_id 0 --num_layers 4
python wan_infer.py --device_id 0 --no-skip-vae      # Output video frames
python wan_infer.py --device_id 0 --config_cache /path/to/config
python wan_infer.py --device_id 0 --compile
python wan_infer.py --device_id 0 --profile
```

### Embedded Default Values

- height: 720, width: 1280, num_frames: 81

- num_inference_steps: 1 (warmup 1, timed 1)

- guidance_scale: 1.0, prompt: "test"

## Qwen-Image

### Model Components

| Component | Class | Layers |
|---|---|---|
| Transformer | `QwenImageTransformer2DModel` | 2 (original: 60) |
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | 2 (original: 28) |
| VAE | `AutoencoderKLQwenImage` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |
| Tokenizer | `Qwen2Tokenizer` | — |

### Usage

```shell
python qwen_image_infer.py --device_id 0
python qwen_image_infer.py --device_id 0 --num_layers 4
python qwen_image_infer.py --device_id 0 --no-skip-vae    # Output image
python qwen_image_infer.py --device_id 0 --config_cache /path/to/config
python qwen_image_infer.py --device_id 0 --compile
python qwen_image_infer.py --device_id 0 --profile
```

### Embedded Default Value

- height: 1024, width: 1024

- num_inference_steps: 1 (warmup 1, timed 1)

- true_cfg_scale: 1.0, prompt: "test"

## FLUX.1-dev

### Model Components

| Component | Class | Number of Layers |
|---|---|---|
| Transformer | `FluxTransformer2DModel` | 2 |
| Text Encoder (CLIP) | `CLIPTextModel` | 1 (original: 12) |
| Text Encoder (T5) | `T5EncoderModel` | 2 (original: 24) |
| VAE | `AutoencoderKL` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |

### Gated Model Configuration

FLUX.1-dev requires authentication. Choose one of the following:

```shell
# Method A: Set HF_TOKEN
export HF_TOKEN=hf_xxx
python flux_infer.py --device_id 0

# Method B: Download offline from ModelScope and specify the cache
python flux_infer.py --device_id 0 --config_cache /home/lb/workspace/flux_configs
```

### Usage

```shell
python flux_infer.py --device_id 0
python flux_infer.py --device_id 0 --num_layers 4
python flux_infer.py --device_id 0 --no-skip-vae     # Output image
python flux_infer.py --device_id 0 --config_cache /path/to/config
python flux_infer.py --device_id 0 --compile
python flux_infer.py --device_id 0 --profile
```

### Embedded Default Value

- height: 1024, width: 1024

- num_inference_steps: 1 (warmup 1, timed 1)

- guidance_scale: 1.0, max_sequence_length: 512, prompt: "test"

## Known Limitations

| Issue | Description |
|---|---|
| Tokenizer compatibility | `Pipeline.from_config()` in diffusers has a bug affecting the tokenizer. As a workaround, construct the pipeline manually component by component. |
| `expandable_segments:True` | In certain NPU environments, the lock pool may lead to an out-of-memory (OOM) error. While Wan2.2 remains unaffected with this configuration, removing it restores normal memory allocation for Qwen/FLUX. |
| `torch.compile` + CPU offload | Incompatible (`InternalTorchDynamoError`), only available in NPU direct-connect mode. |
| ModelScope offline configuration | When uploading `spiece.model` (a protobuf file for FLUX.1-dev), CRLF→LF conversion must be disabled. |
