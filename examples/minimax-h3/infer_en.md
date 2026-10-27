# MiniMax-H3

## Overview

On August 3, 2026, MiniMax officially open-sourced MiniMax-H3, a general omni-modal generation model with 33B parameters that covers workflows such as text-to-audio-video (T2VA), first/last frame to audio-video (FL2VA), and multimodal reference to audio-video (Ref2VA). The model uniformly understands multimodal context such as text, images, video, and audio, and generates native dual-channel audio-video with up to 15 seconds of 2K output, showing strong controllable generation capabilities in scenarios such as instruction following, brand and text rendering, and video motion transfer.

## Environment Preparation

### Model Weights

MiniMax-H3: [Download the model weights](https://huggingface.co/MiniMaxAI/MiniMax-H3)

```bash
export MODEL=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL}"
```

After downloading, the weights directory structure should be as follows:

```text
MiniMax-H3/
├── FL2VA/          # Weights shared by the T2VA/FL2VA tasks
├── Ref2VA/         # Weights for the Ref2VA task
└── model_index.json
```

4-step FlashGen inference loads an online LoRA and does not require merging weights; for the LoRA weights, see [FlashGen 4-Step Online LoRA (T2VA)](#flashgen-4-step-online-lora-t2va).

### Deployment Environment

#### 1) Official Docker Image

You can deploy by downloading the officially released vllm-omni Docker image from the [vllm-omni image repository](https://quay.io/repository/ascend/vllm-omni?tab=tags). The official image already has the matching vllm/vllm-ascend/vllm-omni installed. The procedure is as follows, using the Atlas 800I A3 as an example:

```bash
# Pull the Atlas 800I A3 image; the latest version is recommended, and the newest release so far is 0.28.0
docker pull quay.io/ascend/vllm-omni:v0.28.0-a3

# Create the container
export IMAGE=quay.io/ascend/vllm-omni:v0.28.0-a3
export CONTAINER_NAME=h3

docker run -it -u root --name ${CONTAINER_NAME} \
  --privileged=true \
  --shm-size=2000g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /home:/home \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  ${IMAGE} \
  /bin/bash
```

In the Ascend 950 scenario, add `-v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo` when creating the container to mount the host driver topology directory to the same path in the container.

#### 2) (Optional) Installing MindIE SD

If mindiesd is not installed in the official image, build and install it from source as follows:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

#### 3) (Optional) Installing the Latest vLLM-Omni

To try the latest vllm-omni, follow the guidance below:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

#### 4) (Optional) Installing Other Dependencies

Only the Ref2VA workflow requires the following dependencies; preprocessing of the reference video/audio depends on ffmpeg and decord.

```bash
apt update

# Install ffmpeg
apt install -y ffmpeg

# Install decord
apt install -y pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev libavdevice-dev
git clone --depth 1 --recursive https://github.com/dmlc/decord.git
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DDECODE_FFMPEG=1 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
pip3 install . --no-build-isolation
```

## Start the Service

T2VA/FL2VA tasks share the same model weights, while Ref2VA uses different weights. When starting the service, pass the **--task-type** parameter to specify the task type and load the corresponding model weights. All other environment variables and parameter configurations are shared by T2VA/FL2VA/Ref2VA. If it is not specified, only the T2VA/FL2VA weights are loaded by default.

Start the T2VA/FL2VA service:

```bash
--task-type fl2va
```

Start the Ref2VA service:

```bash
--task-type ref2va
```

<!-- npu="A3,910b" id1 -->
### Atlas 800I A2 / Atlas 800I A3

The following 8-card startup commands are adapted for the Atlas 800I A2 / Atlas 800I A3; all of them use 8-card USP, 8-card text encoder TP, layerwise offload, and VAE `tile` parallelism.
The recommended lossless configuration section enables only lossless optimizations (fused operators and the high-performance FA backend). The recommended lossy configuration section adds DiT sparse FA and online INT8 quantization on top of the recommended lossless configuration. Other optional lossy configurations are also provided.

#### Recommended Lossless Configuration

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-distributed-layerwise-offload \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-backend FLASH_ATTN
```

#### Recommended Lossy Configuration

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-distributed-layerwise-offload \
  --enable-diffusion-pipeline-profiler \
  --diffusion-quantization-config '{"transformer":{"method":"int8"}} \
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

#### Other Optional Performance Optimization Configurations

The following subsections list only the **incremental or replacement parameters** relative to the [Recommended Lossless Configuration (Atlas 800I A2 / Atlas 800I A3)](#recommended-lossless-configuration); the environment variables and other startup parameters remain unchanged.

##### Enabling cache-dit

Append the following parameters to the end of the startup command to enable DiT block caching and TaylorSeer extrapolation:

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```
<!-- end id1 -->

<!-- npu="950" id2 -->
### Ascend 950PR&950DT Products

The following 4-card startup commands are adapted for Ascend 950PR&950DT products; all of them use 4-card USP, 4-card text encoder TP, and VAE `tile` parallelism.
The recommended lossless configuration section enables only lossless optimizations (fused operators and the high-performance FA backend).
The recommended lossy configuration section adds DiT sparse FA and online MXFP8 quantization on top of the recommended lossless configuration. Other optional lossy configurations are also provided.
The recommended lossless configuration uses the DLO layerwise weight offload strategy to avoid activation OOM at longer durations; the recommended lossy configuration has no OOM issue at the maximum specification supported by minimax-h3, so DLO is not enabled for performance reasons.

#### Recommended Lossless Configuration (Ascend 950PR&950DT Products)

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 4 \
  --enable-diffusion-pipeline-profiler \
  --enable-distributed-layerwise-offload \
  --diffusion-attention-backend FLASH_ATTN
```

#### Recommended Lossy Configuration (Ascend 950PR&950DT products)

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 4 \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":8,"end_step":12}}}' \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

#### Other Optional Performance Optimization Configurations (Ascend 950PR&950DT products)

The following subsections list only the **incremental or replacement parameters** relative to the [Recommended Lossless Configuration (Ascend 950PR&950DT products)](#recommended-lossless-configuration-ascend-950pr950dt-products); the environment variables and other startup parameters remain unchanged.

##### Enabling Sparse Attention Optimization (eagleQBSA)

EQBSA uses mixed Q/K INT8 and V FP8 quantization on top of block sparse attention. Replace `--diffusion-attention-backend FLASH_ATTN` in the recommended configuration with:

```bash
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":8,"end_step":13,"precision":"mix"}}}'
```

The default value of `precision` is `bf16`, and enabling EQBSA requires setting it explicitly to `mix`. `start_step` and `end_step` specify how many steps at the beginning and at the end fall back to dense attention, not step indices. For 1344x768 T2VA with 50 steps, choose according to the scenario:

| Configuration | `start_step` | `end_step` | Assessment |
| --- | ---: | ---: | --- |
| Quality first (recommended) | 8 | 13 | General and complex motion scenarios; image quality first |
| Balanced performance | 0 | 13 | Balanced |
| Speed first | 0 | 0 | Simple scenes with little motion; speed first |

If the image becomes discontinuous or details become unstable, increase `end_step` first and then `start_step`; do not replace the dense fallback merely by lowering `sparsity`. The combinations above were verified only at 1344x768 with 50 steps; other resolutions or sampling steps must be re-evaluated.

The recommended lossy configuration uses a **start=8 / end=12** fallback policy; for higher performance, you can also try fewer fallback steps.

##### Enabling cache-dit (Ascend 950PR&950DT Products)

Append the following parameters to the end of the startup command to enable DiT block caching and TaylorSeer extrapolation:

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

##### Enabling Online mxfp4 Quantization

Append the following parameters to the end of the startup command to enable online MXFP4 quantization of the DiT:

```bash
  --diffusion-quantization-config '{"transformer":{"method":"mxfp4"}}'
```

And delete the following parameter:

```bash
--enable-distributed-layerwise-offload
```
<!-- end id2 -->

## Send a Request

### T2VA Text-to-Video

After the T2VA/FL2VA service starts, send the following request:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT=""

HDR_FILE=$(mktemp)
curl -sS -D "$HDR_FILE" -X POST "${API_URL}" \
  -F "prompt=${PROMPT}" \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F "flow_shift=12.0" \
  -F "audio_flow_shift=3.0" \
  -F "seed=1101" \
  -F 'extra_params={"task":"t2va","duration":5.0}' \
  -o "t2va.mp4"
```

### FL2VA Reference First/Last Frame to Video

After the T2VA/FL2VA service starts, send the following request:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT=""
export FIRST_FRAME=""

HDR_FILE=$(mktemp)
curl -sS -D "$HDR_FILE" -X POST "${API_URL}" \
  -F "prompt=${PROMPT}" \
  -F 'short_edge=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F "flow_shift=12.0" \
  -F "audio_flow_shift=3.0" \
  -F "seed=1101" \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -F 'extra_params={"task":"fl2va","duration":5.0}' \
  -o "fl2va.mp4"
```

FL2VA uses `short_edge` to control the short edge length; the aspect ratio is determined by `aspect_ratio` or by the first-frame image.

### Ref2VA Reference Video/Audio to Video

After the Ref2VA service starts, send the following request:

```bash
# Text + reference video to audio-video
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT=""
export VIDEO_REF=""

HDR_FILE=$(mktemp)
curl -sS -D "$HDR_FILE" -X POST "${API_URL}" \
  -F "prompt=${PROMPT}" \
  -F 'short_edge=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F "flow_shift=12.0" \
  -F "audio_flow_shift=3.0" \
  -F "seed=2101" \
  -F 'extra_params={"task":"ref2va","duration":5.0}' \
  -F "input_references=@${VIDEO_REF};type=video/mp4" \
  -o "ref2va.mp4"
```

### Request Parameters

The request parameters shared by the three tasks are as follows:

| Parameter | Type | Required | Description |
| ------ | ------ | ------ | ------ |
| `prompt` | `str` | Yes | Text prompt |
| `width` / `height` | `int` | Either this or `short_edge` | Output resolution in pixels; used by the T2VA example |
| `short_edge` | `int` | Either this or `width`/`height` | Output short edge length in pixels; used by the FL2VA / Ref2VA examples |
| `aspect_ratio` | `str` | No | Output aspect ratio (for example, `16:9`); used together with `short_edge` |
| `fps` | `int` | No | Output frame rate |
| `num_inference_steps` | `int` | No | Number of denoising inference steps; the base model defaults to 50; the FlashGen online LoRA requires `4`, see [FlashGen 4-Step Online LoRA (T2VA)](#flashgen-4-step-online-lora-t2va) |
| `flow_shift` | `float` | No | Video flow shift parameter |
| `audio_flow_shift` | `float` | No | Audio flow shift parameter |
| `seed` | `int` | No | Random seed |
| `extra_params.task` | `str` | Yes | Task type: `t2va` / `fl2va` / `ref2va` |
| `extra_params.duration` | `float` | No | Duration of the generated audio-video in seconds, up to 15 seconds |
| `input_references` | `file` | No | Reference input file: FL2VA passes the first-frame image (`image/png`, etc.), and Ref2VA passes the reference video/audio (`video/mp4`, etc.) |
| `lora` | `str` (JSON) | No | Online LoRA configuration; FlashGen 4-step requires `name` / `path` / `scale`, see [FlashGen 4-Step Online LoRA (T2VA)](#flashgen-4-step-online-lora-t2va) |

## FlashGen 4-Step Online LoRA (T2VA)

The FlashGen 4-step weights are a single native-layout LoRA file (`key_format=minimax-h3-native`), **loaded at runtime** by vLLM-Omni and not merged into the base model.

The weights are published on ModelScope: [FlashGen/Minimax-H3-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen)

### Download the LoRA

```bash
pip install modelscope
export FLASHGEN_DIR=/path/to/minimax-h3-flashgen-lora
export FLASHGEN_FILE=minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

### Start the Service (Added on Top of the Recommended Configuration)

Relative to the [Recommended Lossy Configuration (Atlas 800I A2 / Atlas 800I A3)](#recommended-lossy-configuration), add the LoRA parameters after `--trust-remote-code`, and replace the following in that configuration

```bash
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

with

```bash
  --diffusion-attention-backend FLASH_ATTN
```

Append the LoRA parameters:

```bash
  --lora-backend peft \
  --lora-path "${FLASHGEN_LORA}" \
```

The remaining parameters (USP8, DLO, VAE tile, and so on) are the same as in the recommended configuration.

Complete example for the Atlas 800I A2 / Atlas 800I A3:

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export MODEL="${MODEL_ROOT}/FL2VA"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --lora-backend peft \
  --lora-path "${FLASHGEN_LORA}" \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-distributed-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-backend FLASH_ATTN
```

Notes:

- `--lora-backend peft` enables the runtime LoRA manager; the native layout is detected automatically from the LoRA file metadata, so no separate backend is required.
- Only **T2VA** is supported; FL2VA / Ref2VA are not supported.
- `--enable-cpu-offload` and `--enable-layerwise-offload` (ordinary layerwise offload) are not supported.

### Sending a 4-Step T2VA Request

Based on [T2VA Text-to-Video](#t2va-text-to-video), change `num_inference_steps` to `4` and pass the `lora` field:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT=""

HDR_FILE=$(mktemp)
curl -sS -D "$HDR_FILE" -X POST "${API_URL}" \
  -F "prompt=${PROMPT}" \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F "seed=1101" \
  -F 'extra_params={"task":"t2va","duration":5.2}' \
  -F "lora={\"name\":\"h3-flashgen-v1.0\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}" \
  -o "t2va_4step.mp4"
```

Parameter highlights:

| Parameter | Value | Description |
| ------ | ------ | ------ |
| `extra_params.task` | `t2va` | native LoRA supports T2VA only |
| `num_inference_steps` | `4` | Number of intervals (4 denoise runs), not 5 |
| `lora.scale` | `1.0` | Activates LoRA; `0.0` compares against the base model output |
| `flow_shift` / `audio_flow_shift` | Optional | The schedule is provided by `base_schedule` in the LoRA metadata |

## Optimizations Adapted So Far

1. Ascend-affinity attention operators: the MindIE SD ascend_laser_attention operator; for details, see [core_layers.md](../../docs/en/features/core_layers.md)
2. Ascend-affinity fused operators: RMSNorm, AddRMSNorm, GQA, SwiGLU, and rotary_position_embedding; for details, see [core_layers.md](../../docs/en/features/core_layers.md)
3. Sparse FA operators; for details, see [sparse.md](../../docs/en/features/sparse.md)
4. Online quantization: int8, mxfp8, and mxfp4
5. USP\TP\Tile parallelism
6. Post-training: few-step distillation

## Benchmark

<!-- npu="910b" id3 -->
### Atlas 800I A2

| Task Type | Output Resolution | Output Frames | Output Duration (s) | Optimization Config | E2E Latency (s) | DiT Total Latency (s) | DiT Per-Step Latency (s) | DiT Actual Steps | Text Encoding Latency (s) | Reference Video Encoding Latency (s) | Reference Audio Encoding Latency (s) | Decoding Latency (s) | Reference Video Preprocessing (s) | Post-processing (s) | CPU MP4 (s) | Resident Weight Memory (GB) | Peak Inference Memory (GB) |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | Lossless | 150.06 | 139.9 | 2.85 | 49 | 0.83 | NA | NA | 6.76 | NA | 0.462 | 1.18 | 2.97 | 24.7 |
| t2va | 1344×768 | 362 | 15 | Lossless | 786.239 | 761.357 | 15.53 | 49 | 1.196 | NA | NA | 15.87 | NA | 1.977 | 2.95 | 2.97 | 36.21 |
| t2va | 1344×768 | 124 | 5 | Lossy | 108.1 | 95.85 | 1.95 | 49 | 0.9 | NA | NA | 6.73 | NA | 0.745 | 1.04 | 2.55 | 22.47 |
| t2va | 1344×768 | 362 | 15 | Lossy | 552.51 | 526.041 | 10.73 | 49 | 1.28 | NA | NA | 16.07 | NA | 1.903 | 3.76 | 2.55 | 35.11 |
| ref2va | 1344×768 | 124 | 5 | Lossless | 474.89 | 421.678 | 8.6 | 49 | 2.42 | 13.31 | 0.38 | 6.469 | 3.94 | 0.489 | 1.157 | 3.4 | 23.45 |
| ref2va | 1344×768 | 362 | 15 | Lossless | 3414.184 | 3336.981 | 68.08 | 49 | 6.48 | 30.85 | 0.59 | 15.8 | 12.59 | 1.988 | 3.08 | 3.4 | 36.28 |
| ref2va | 1344×768 | 124 | 5 | Lossy | 380.915 | 326.495 | 6.66 | 49 | 2.47 | 13.48 | 0.38 | 6.36 | 4.68 | 0.591 | 1.217 | 2.55 | 22.57 |
| ref2va | 1344×768 | 362 | 15 | Lossy | 2332.041 | 2249.63 | 45.89 | 49 | 6.33 | 35.1 | 0.59 | 15.72 | 11.07 | 1.345 | 2.3 | 2.55 | 35.58 |
<!-- end id3 -->

<!-- npu="A3" id4 -->
### Atlas 800I A3

| Task Type | Output Resolution | Output Frames | Output Duration (s) | Optimization Config | E2E Latency (s) | DiT Total Latency (s) | DiT Per-Step Latency (s) | DiT Actual Steps | Text Encoding Latency (s) | Reference Video Encoding Latency (s) | Reference Audio Encoding Latency (s) | Decoding Latency (s) | Reference Video Preprocessing (s) | Post-processing (s) | CPU MP4 (s) | Resident Weight Memory (GB) | Peak Inference Memory (GB) |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | Lossless | 114.187 | 105.871 | 2.16 | 49 | 0.483 | NA | NA | 6.56 | NA | 0.001 | 0.461 | 3.49 | 24.7 |
| t2va | 1344×768 | 362 | 15 | Lossless | 599.29 | 580.383 | 11.84 | 49 | 1.826 | NA | NA | 14.95 | NA | 0.001 | 1.021 | 3.49 | 36.21 |
| t2va | 1344×768 | 124 | 5 | Lossy | 84.824 | 77.473 | 1.58 | 49 | 0.566 | NA | NA | 6.23 | NA | 0.001 | 0.608 | 2.63 | 22.47 |
| t2va | 1344×768 | 362 | 15 | Lossy | 401.434 | 383.559 | 7.82 | 49 | 1.716 | NA | NA | 14.83 | NA | 0.001 | 1.14 | 2.63 | 35.11 |
| ref2va | 1344×768 | 124 | 5 | Lossless | 355.534 | 311.859 | 6.36 | 49 | 1.888 | 12.367 | 0.244 | 5.55 | 2.493 | 0.001 | 0.433 | 3.49 | 23.45 |
| ref2va | 1344×768 | 362 | 15 | Lossless | 2339.556 | 2277.606 | 46.47 | 49 | 5.613 | 34.46 | 0.3635 | 13.607 | 6.833 | 0.001 | 1.114 | 3.49 | 36.28 |
| ref2va | 1344×768 | 124 | 5 | Lossy | 306.816 | 247.716 | 5.05 | 49 | 1.877 | 12.712 | 0.24 | 5.237 | 2.447 | 0.001 | 0.535 | 2.63 | 22.57 |
| ref2va | 1344×768 | 362 | 15 | Lossy | 1695.256 | 1634.584 | 33.35 | 49 | 5.568 | 33.269 | 0.3564 | 13.51 | 6.96 | 0.001 | 1.221 | 2.63 | 35.58 |
<!-- end id4 -->

<!-- npu="950" id5 -->
### Ascend 950PR&950DT Products

| Task Type | Output Resolution | Output Frames | Output Duration (s) | Optimization Config | E2E Latency (s) | DiT Total Latency (s) | DiT Per-Step Latency (s) | DiT Actual Steps | Text Encoding Latency (s) | Reference Video Encoding Latency (s) | Reference Audio Encoding Latency (s) | Decoding Latency (s) | Reference Video Preprocessing (s) | Post-processing (s) | CPU MP4 (s) | Resident Weight Memory (GB) | Peak Inference Memory (GB) |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | Lossless | 203.949 | 198.096 | 4.04 | 49 | 0.574 | NA | NA | 5.05 | NA | 0.001 | 0.225 | 3.97 | 22.2 |
| t2va | 1344×768 | 362 | 15 | Lossless | 1489.165 | 1474.353 | 30.08 | 49 | 0.69 | NA | NA | 13.449 | NA | 0.001 | 0.815 | 3.97 | 32.14 |
| t2va | 1344×768 | 124 | 5 | Lossy | 113.857 | 109.809 | 2.24 | 49 | 0.028 | NA | NA | 3.82 | NA | 0.001 | 0.214 | 56.99 | 64.38 |
| t2va | 1344×768 | 362 | 15 | Lossy | 762.169 | 749.397 | 15.29 | 49 | 0.028 | NA | NA | 12.19 | NA | 0.001 | 0.606 | 56.99 | 74.26 |
| ref2va | 1344×768 | 124 | 5 | Lossless | 745.427 | 730.219 | 14.9 | 49 | 1.159 | 8.148 | 0.134 | 5.119 | 0.408 | 0.001 | 0.278 | 3.97 | 23.82 |
| ref2va | 1344×768 | 362 | 15 | Lossless | 5887.24 | 5847.567 | 119.33 | 49 | 2.54 | 22.066 | 0.2262 | 13.34 | 0.808 | 0.001 | 0.702 | 3.97 | 32.53 |
| ref2va | 1344×768 | 124 | 5 | Lossy | 400.404 | 387.815 | 7.91 | 49 | 0.951 | 7.048 | 0.085 | 3.91 | 0.405 | 0.001 | 0.21 | 56.6 | 66.03 |
| ref2va | 1344×768 | 362 | 15 | Lossy | 3037.842 | 3001.314 | 61.24 | 49 | 2.278 | 20.64 | 0.145 | 12.1 | 0.806 | 0.001 | 0.695 | 56.6 | 74.76 |
<!-- end id5 -->

The test data in the tables above was obtained with vllm-omni 0.28.0: the lossless optimizations use the recommended lossless configuration of the corresponding device model, and the lossy optimizations use the recommended lossy configuration.

## Follow-Up Plans

Performance optimizations for MiniMax-H3 are being updated continuously; stay tuned.
