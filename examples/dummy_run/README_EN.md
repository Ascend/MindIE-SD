# Dummy-Weight Run Verification

Construct generative models on NPU devices **without relying on real weights** (only configuration files are downloaded, which are tens of KB) and complete forward inference verification.

## Directory Structure

```text
examples/dummy_run/
├── model/
│   ├── __init__.py                       # check_npu(), resolve_config_path(), _PhaseTimer
│   ├── wan_model.py                      # build_wan_pipeline()
│   ├── qwen_image_model.py               # build_qwen_image_pipeline()
│   ├── flux_model.py                     # build_flux_pipeline()
│   └── minimax_h3_model.py               # build_minimax_h3_pipeline()
├── wan_infer.py                          # Wan2.2 entry script
├── qwen_image_infer.py                   # Qwen-Image entry script
├── flux_infer.py                         # FLUX.1-dev entry script
├── minimax_h3_infer.py                   # MiniMax-H3 entry script
├── requirements.txt                      # Dependency declarations
└── README.md
```

## Prerequisites

```shell
pip install -r examples/dummy_run/requirements.txt
```

| Dependency | Minimum version |
|---|---|
| Python | 3.10 |
| torch / TorchNPU | Matches the CANN version |
| diffusers | >= 0.40.0 (MiniMax-H3 requires 0.40.0+; 0.34.0 is sufficient for the other models) |
| transformers | >= 4.56.0 (Qwen3-VL) |
| huggingface_hub | >= 0.23.0 |

Ensure that the NPU is available: `npu-smi info -l`

## CLI Parameters (Unified for All Four Models)

```shell
python <model>_infer.py --device_id <N> --num_layers <N>
```

| Parameter | Default value | Description |
|---|---|---|
| `--device_id` | 0 | NPU device index |
| `--config_cache` | None | Offline mode; specifies the local configuration directory |
| `--num_layers` | 2 | Number of Transformer layers |
| `--compile` | False | Enables MindieSDBackend compilation |
| `--profile` | False | Enables NPU profiling (level=l1) |
| `--skip-vae` / `--no-skip-vae` | True | Skips VAE decode (default). `--no-skip-vae` turns it on (Wan2.2 / Qwen-Image / FLUX.1-dev) |
| `--quant` | bf16 | Quantization/compute mode: `bf16` / `w8a8` / `fp32` (Wan2.2 / MiniMax-H3 / Qwen-Image / FLUX.1-dev). bf16 = model-layer weights are cast and activations use native bf16, with zero implicit precision conversion on the compilation side (GEMM ~15x faster); w8a8 = W8A8 online quantization (Matmul only, FA not enabled; the format depends on the device: Ascend 950PR&950DT products → MXFP8, Atlas 800I A2 / Atlas 800I A3 → INT8); fp32 = the original fp32 computation |

Additional MiniMax-H3 parameters (see the [MiniMax-H3](#minimax-h3) section): `--height`, `--width`, `--num_frames`, and `--num_inference_steps`; there is no `--skip-vae` (the output is a fixed latent without decoding).

## Configuration Cache

Configuration files (JSON and tokenizer, at the KB level) are downloaded automatically to the HuggingFace Hub cache on the first run, and later runs do not require network access.
Use `--config_cache /path/to/config` to specify an offline cache directory.

For **gated models** (such as FLUX.1-dev), set `HF_TOKEN` or download the files from modelscope and then
load them through `--config_cache`.

## Verification Record (910B, 64GB HBM, Direct NPU Connection, 2 Layers)

| Model | Build (ms) | Timed (ms) | Peak Mem | Status |
|---|---|---|---|---|
| Wan2.2 | 1,200 | 7,000 | 10.18 GB | PASSED |
| Qwen-Image | 7,000 | 100 | 6.26 GB | PASSED |
| FLUX.1-dev | 20,500 | 900 | 24.20 GB | PASSED |
| MiniMax-H3 (bf16) | 800 | 51 | 13.50 GB | PASSED |

> MiniMax-H3 defaults to bf16 (`--quant bf16`): under eager fp32, Timed is 338 ms / 21.90 GB.

---

## Wan2.2

### Model Components

| Component | Class | Layers |
|---|---|---|
| Transformer | `WanTransformer3DModel` | 2 (originally 40) |
| Transformer_2 | `WanTransformer3DModel` | 2 |
| Text Encoder | `UMT5EncoderModel` | 2 (originally 28) |
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

### Built-in Defaults

- height: 720, width: 1280, num_frames: 81
- num_inference_steps: 1 (warmup 1, timed 1)
- guidance_scale: 1.0, prompt: "test"

---

## Qwen-Image

### Model Components

| Component | Class | Layers |
|---|---|---|
| Transformer | `QwenImageTransformer2DModel` | 2 (originally 60) |
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | 2 (originally 28) |
| VAE | `AutoencoderKLQwenImage` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |
| Tokenizer | `Qwen2Tokenizer` | — |

### Usage

```shell
python qwen_image_infer.py --device_id 0
python qwen_image_infer.py --device_id 0 --num_layers 4
python qwen_image_infer.py --device_id 0 --no-skip-vae    # Output images
python qwen_image_infer.py --device_id 0 --config_cache /path/to/config
python qwen_image_infer.py --device_id 0 --compile
python qwen_image_infer.py --device_id 0 --profile
```

### Built-in Defaults

- height: 1024, width: 1024
- num_inference_steps: 1 (warmup 1, timed 1)
- true_cfg_scale: 1.0, prompt: "test"

---

## FLUX.1-dev

### Model Components

| Component | Class | Layers |
|---|---|---|
| Transformer | `FluxTransformer2DModel` | 2 |
| Text Encoder (CLIP) | `CLIPTextModel` | 1 (originally 12) |
| Text Encoder (T5) | `T5EncoderModel` | 2 (originally 24) |
| VAE | `AutoencoderKL` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |

### Gated Model Configuration

FLUX.1-dev requires authentication. Choose one of the following:

```shell
# Option A: set HF_TOKEN
export HF_TOKEN=hf_xxx
python flux_infer.py --device_id 0

# Option B: download offline from modelscope and specify the cache
python flux_infer.py --device_id 0 --config_cache /home/lb/workspace/flux_configs
```

### Usage

```shell
python flux_infer.py --device_id 0
python flux_infer.py --device_id 0 --num_layers 4
python flux_infer.py --device_id 0 --no-skip-vae     # Output images
python flux_infer.py --device_id 0 --config_cache /path/to/config
python flux_infer.py --device_id 0 --compile
python flux_infer.py --device_id 0 --profile
```

### Built-in Defaults

- height: 1024, width: 1024
- num_inference_steps: 1 (warmup 1, timed 1)
- guidance_scale: 1.0, max_sequence_length: 512, prompt: "test"

---

## MiniMax-H3

### Model Components

MiniMax-H3 (a 33B omni-modal generation model covering the T2VA/FL2VA/Ref2VA workflows) is integrated through the diffusers `MiniMaxH3ModularPipeline`. The
dummy run covers the `t2va` (text-to-audio-video) workflow: text encoding plus denoising of the packed sequence (video + audio). The reference weights are
[Hugging Face MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (the root directory is in diffusers format, and
`FL2VA/` and `Ref2VA/` are in vLLM-Omni format).

| Component | Class | Layers |
|---|---|---|
| Transformer | `MiniMaxH3Transformer3DModel` | 2 (originally 50) |
| Text Encoder | `Qwen3VLForConditionalGeneration` | 2 (originally 64) |
| Video VAE | `AutoencoderKLMiniMaxH3` | — (parameter count only, not involved in the forward pass) |
| Audio VAE | `AutoencoderKLMiniMaxH3Audio` | — (parameter count only, not involved in the forward pass) |
| Scheduler | `MiniMaxH3Scheduler` (shift=12.0) | — |
| Audio Scheduler | `MiniMaxH3Scheduler` (shift=3.0) | — |
| Tokenizer / Processor | `Qwen2TokenizerFast` / `Qwen3VLProcessor` | — |

### Obtaining Configuration Files (modelscope)

MiniMax-H3 is a **gated model** on HF and requires approval; the modelscope mirror `MiniMax/MiniMax-H3` requires no authentication.
The script prefers to download configuration files through modelscope (KB level; only json/py/txt/tokenizer files are pulled, not safetensors):

```shell
# Option A: download the configuration from modelscope automatically (no token required)
python minimax_h3_infer.py --device_id 0

# Option B: download manually offline and specify the cache directory
pip install modelscope
python -c "from modelscope import snapshot_download; \
    print(snapshot_download('MiniMax/MiniMax-H3', allow_patterns=['*.json','*.txt','*.py','tokenizer*'], \
    ignore_patterns=['*.safetensors','*.bin']))"
python minimax_h3_infer.py --device_id 0 --config_cache /path/to/MiniMax-H3-configs
```

> Note: `FL2VA/` in the model weights directory (`hf download MiniMaxAI/MiniMax-H3` or the vLLM-Omni deployment directory) is in
> vLLM-Omni format and cannot be used as `--config_cache`; use the diffusers-format configuration in the repository **root directory** instead.

### Usage

```shell
python minimax_h3_infer.py --device_id 0                    # Default bf16
python minimax_h3_infer.py --device_id 0 --num_layers 4
python minimax_h3_infer.py --device_id 0 --height 512 --width 768
python minimax_h3_infer.py --device_id 0 --quant fp32   # fp32 baseline (about 10x slower)
python minimax_h3_infer.py --device_id 0 --quant w8a8   # W8A8 online quantization (Ascend 950PR&950DT products → MXFP8, Atlas 800I A2 / Atlas 800I A3 → INT8)
python minimax_h3_infer.py --device_id 0 --config_cache /path/to/config
python minimax_h3_infer.py --device_id 0 --compile
python minimax_h3_infer.py --device_id 0 --profile
```

### Built-in Defaults

- compute precision: **bf16 (default)**. The MiniMax-H3 DiT has no forced fp32 island (the dtype is fully determined by the parameters), so
  after the weights are cast to bf16 the entire DiT block stack computes in native bf16, with zero implicit precision conversion on the compilation side
- height: 256, width: 384 (small canvas: MiniMax-H3 applies full self-attention O(seq^2) to the packed sequence, and 768x1344 is infeasible on a single card)
- num_frames: 124 (17x7+5, a minimum of 5s at 24fps, range 5-15s)
- num_inference_steps: 2 (`MiniMaxH3Scheduler` requires >= 2; 2 steps = 1 transformer forward)
- prompt: "test"

### Differences from the Other Dummy Runs

- **Fixed output latent**: to avoid the Video/Audio VAE decode of the custom remote code (whose NPU compatibility is not verified),
  the dummy run removes the decode block of the pipeline, and `t2va` returns the video/audio latents directly after denoising, with no `--skip-vae` parameter.
- **Text encoding timing**: the MiniMax-H3 text encoding block directly drives the `text_encoder.model` submodule, and `_PhaseTimer` attaches a separate
  timing hook to that submodule.
- **text_encoder_layer**: the full model conditions on the hidden state of layer 50 of Qwen3-VL; after truncation to 2 layers it uses layer 1 instead.
- **bf16 mechanism**: unlike Wan, the MiniMax-H3 DiT has no `.float()` island and requires no source-level rewrite; `--quant bf16`
  only casts the weights (see minimax-h3-notes.md in the dummy-run-dev skill for details).
- **--compile**: applies MindieSDBackend to the transformer (the `_CompiledDiT` wrapper keeps the original forward signature,
  because the denoise block filters row index parameters by `inspect.signature`); the compiled graph is verified to contain no fp32 compute nodes.

---

## model/common Shared Capabilities

Shared cross-model capabilities are grouped by responsibility, and each `*_infer.py` uniformly uses `from model.common import ...`:

| Module | Responsibility | Key APIs |
|---|---|---|
| `model/common/precision.py` | Model-level bf16/fp32 precision mechanism (`--quant bf16` / `fp32`) | `apply_compute_precision` / `verify_compute_precision_graph` |
| `model/common/compile_patches.py` | Performance patches for the compile graph (wasted dropout kernel, qwen pos_embed hotspot) | `replace_zero_dropout` / `replace_pos_embed_with_buffers` |
| `model/common/quantization.py` | W8A8 online quantization with device awareness (`--quant w8a8`) | `apply_w8a8_quant` / `report_quant_layers` |

`--quant w8a8` selects the format automatically by device:

<!-- npu="950" id1 -->
- **Ascend 950PR&950DT products → MXFP8**.
<!-- end id1 -->
<!-- npu="910b" id2 -->
- **Atlas 800I A2 (910B) → INT8**;
<!-- end id2 -->
<!-- npu="A3" id3 -->
- **Atlas 800I A3 (910C) → INT8**;
<!-- end id3 -->

Only Matmul (`nn.Linear`) is quantized; other vector operations stay in bf16 and FA is not quantized (verified with the kernel: the graph contains no fp32 compute nodes).

> For the complete module responsibilities / APIs / integration approach, see `references/model-common.md` in the dummy-run-dev skill.

---

## Known Limitations

| Issue | Description |
|---|---|
| tokenizer compatibility | diffusers `Pipeline.from_config()` has a bug for tokenizers, so components are constructed manually one by one |
| `expandable_segments:True` | In some NPU environments this may lock the memory pool and cause OOM. Wan2.2 uses this setting without impact; Qwen/FLUX allocate normally after removing it |
| `torch.compile` + CPU offload | Incompatible (`InternalTorchDynamoError`); supported only in direct NPU connection mode |
| modelscope offline configuration | The spiece.model of FLUX.1-dev is a protobuf file, and CRLF to LF conversion must be disabled when uploading it |
