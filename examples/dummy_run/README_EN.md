# Empty Weight Runtime Verification

On the NPU, a class model is constructed and verified by performing forward inference **without using real weights** (only the configuration file is downloaded, which is only dozens of KB).

## Directory Structure

```text
examples/dummy_run/
├── model/
│   ├── __init__.py                       # check_npu(), resolve_config_path(), _PhaseTimer
│   ├── wan_model.py                      # build_wan_pipeline()
│   ├── qwen_image_model.py               # build_qwen_image_pipeline()
│   └── flux_model.py                     # build_flux_pipeline()
├── wan_infer.py                          # Entry script of Wan2.2
├── qwen_image_infer.py                   # Entry script of Qwen-Image
├── flux_infer.py                         # Entry script of FLUX.1-dev
├── requirements.txt                      # Dependency declaration
└── README.md
```

## Prerequisites

```shell
pip install -r examples/dummy_run/requirements.txt
```

| Dependency| Earliest Version|
|---|---|
| Python | 3.10 |
| torch / TorchNPU | Matching the CANN version|
| diffusers | >= 0.34.0 |
| transformers | >= 4.44.0 |
| huggingface_hub | >= 0.23.0 |

Ensure that the NPU is available: `npu-smi info -l`

## CLI Parameters (Unified for Three Models)

```shell
python <model>_infer.py --device_id <N> --num_layers <N>
```

| Parameter| Default Value| Description|
|---|---|---|
| `--device_id` | 0 | NPU device index.|
| `--config_cache` | None| Local configuration file directory in offline mode.|
| `--num_layers` | 2 | Number of Transformer layers.|
| `--compile` | False | Enables MindieSDBackend compilation.|
| `--profile` | False | Enables NPU profiling (level=l1).|
| `--skip-vae` / `--no-skip-vae` | True | Skips VAE decoding (default). `--no-skip-vae` is enabled.|

## Configuring the Cache

The configuration file (JSON, tokenizer, KB-level) is automatically downloaded to the HuggingFace Hub cache upon the first run, and no Internet access is required for subsequent runs.
You can specify the offline cache directory using `--config_cache /path/to/config`.

For **gated model** (such as FLUX.1-dev), set `HF_TOKEN` or download it from ModelScope.
It is loaded through `--config_cache`.

## Verification Record (910B, 64 GB HBM, NPU Direct Connection, 2 layers)

| Model | Build (ms) | Timed (ms) | Peak Mem | Status |
|---|---|---|---|---|
| Wan2.2 | 1,200 | 7,000 | 10.18 GB | PASSED |
| Qwen-Image | 7,000 | 100 | 6.26 GB | PASSED |
| FLUX.1-dev | 20,500 | 900 | 24.20 GB | PASSED |
| MiniMax-H3 (bf16) | 800 | 51 | 13.50 GB | PASSED |

> The default bf16 (`--quant bf16`) of MiniMax-H3 is 338 ms/21.90 GB in eager fp32 mode.

---

## Wan2.2

### Model Components

| Component| Class| Layer Quantity|
|---|---|---|
| Transformer | `WanTransformer3DModel` | 2 (original: 40)|
| Transformer_2 | `WanTransformer3DModel` | 2 |
| Text Encoder | `UMT5EncoderModel` | 2 (original: 28)|
| VAE | `AutoencoderKLWan` | — |
| Scheduler | `UniPCMultistepScheduler` | — |

### Usage

```shell
python wan_infer.py --device_id 0
python wan_infer.py --device_id 0 --num_layers 4
python wan_infer.py --device_id 0 --no-skip-vae      # Output video frames.
python wan_infer.py --device_id 0 --config_cache /path/to/config
python wan_infer.py --device_id 0 --compile
python wan_infer.py --device_id 0 --profile
```

### Embedded Default Value

- height: 720, width: 1280, num_frames: 81
- num_inference_steps: 1 (warmup 1, timed 1)
- guidance_scale: 1.0, prompt: "test"

---

## Qwen-Image

### Model Components

| Component| Class| Layer Quantity|
|---|---|---|
| Transformer | `QwenImageTransformer2DModel` | 2 (original: 60)|
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | 2 (original: 28)|
| VAE | `AutoencoderKLQwenImage` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |
| Tokenizer | `Qwen2Tokenizer` | — |

### Usage

```shell
python qwen_image_infer.py --device_id 0
python qwen_image_infer.py --device_id 0 --num_layers 4
python qwen_image_infer.py --device_id 0 --no-skip-vae # Output image
python qwen_image_infer.py --device_id 0 --config_cache /path/to/config
python qwen_image_infer.py --device_id 0 --compile
python qwen_image_infer.py --device_id 0 --profile
```

### Embedded Default Value

- height: 1024, width: 1024
- num_inference_steps: 1 (warmup 1, timed 1)
- true_cfg_scale: 1.0, prompt: "test"

---

## FLUX.1-dev

### Model Components

| Component| Class| Layer Quantity|
|---|---|---|
| Transformer | `FluxTransformer2DModel` | 2 |
| Text Encoder (CLIP) | `CLIPTextModel` | 1 (original: 12)|
| Text Encoder (T5) | `T5EncoderModel` | 2 (original: 24)|
| VAE | `AutoencoderKL` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |

### Gated model configuration

FLUX.1-dev requires authentication. Choose either item.

```shell
# Method A: Set HF_TOKEN.
export HF_TOKEN=hf_xxx
python flux_infer.py --device_id 0

# Method B: Specify the cache after offline download from ModelScope.
python flux_infer.py --device_id 0 --config_cache /home/lb/workspace/flux_configs
```

### Usage

```shell
python flux_infer.py --device_id 0
python flux_infer.py --device_id 0 --num_layers 4
python flux_infer.py --device_id 0 --no-skip-vae # Output image
python flux_infer.py --device_id 0 --config_cache /path/to/config
python flux_infer.py --device_id 0 --compile
python flux_infer.py --device_id 0 --profile
```

### Embedded Default Value

- height: 1024, width: 1024
- num_inference_steps: 1 (warmup 1, timed 1)
- guidance_scale: 1.0, max_sequence_length: 512, prompt: "test"

---

## Known Limitations

| Question| Description|
|---|---|
| Tokenizer compatibility| The diffusers `Pipeline.from_config()` has a bug for the tokenizer. Therefore, the tokenizer is manually constructed component by component.|
| `expandable_segments:True` | In some NPU environments, the lock pool may cause OOM. Wan2.2 uses this configuration, which does not affect the system. After Qwen/FLUX is removed, the allocation is normal.|
| `torch.compile` + CPU offload | Incompatible (`InternalTorchDynamoError`), and available only in NPU direct connection mode.|
| ModelScope offline configuration| The spiece.model of FLUX.1-dev is a protobuf file. During the upload, the CRLF-to-LF conversion is not allowed.|
