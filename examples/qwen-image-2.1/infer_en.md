# Qwen-Image-2.1

## Overview

Qwen-Image-2.1 is an image generation model that covers workflows such as text-to-image and reference image editing. The model uses Qwen3-VL to encode text and reference images, and generates images through a single-stream DiT.

> [!NOTE]NOTE
> The model support code comes from [vLLM-Omni PR #7759](https://github.com/vllm-project/vllm-omni/pull/7759). Install the specified source commit as described below. Downloading the weights requires access to the model repository.

## Supported Devices

- Atlas 350.

**Other device models will be updated later.**

## Environment Preparation

### Model Weights

Qwen-Image-2.1: [Download the model weights](https://huggingface.co/Qwen/Qwen-Image-2.1)

```bash
export MODEL=/path/to/Qwen-Image-2.1
hf download Qwen/Qwen-Image-2.1 --local-dir "${MODEL}"
```

### Deployment Environment

#### 1) Official Docker Image

Select the official image for Atlas 350 from the [vllm-omni image repository](https://quay.io/repository/ascend/vllm-omni?tab=tags) as the base environment. MindIE SD is already included in the image, so no separate installation is required. Replace the image tag placeholder below, and install the model support code from the PR as described in step 2:

```bash
# Pull the official image for Atlas 350 (replace the tag first)
export IMAGE='quay.io/ascend/vllm-omni:<IMAGE_TAG>'
docker pull "${IMAGE}"

# Create the container (single card)
export CONTAINER_NAME=qwen21
docker run -it -u root --name ${CONTAINER_NAME} \
  -e MODEL="${MODEL}" \
  --privileged=true \
  --shm-size=200g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home:/home \
  -v "${MODEL}":"${MODEL}" \
  ${IMAGE} \
  /bin/bash
```

In the 8-card scenario, add the following device parameters when creating the container, and use `ASCEND_RT_VISIBLE_DEVICES` to specify the visible cards:

```bash
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7
```

#### 2) Installing vLLM-Omni

Qwen-Image-2.1 requires the model support code from PR #7759. Install it in the container as follows:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git fetch origin refs/pull/7759/head
git checkout --detach 3ea9e605011fc2a0bfa9f3e8531286fb683068dd
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

## Start the Service

### Atlas 350

#### Recommended Single-Card Configuration

The following command starts the service on a single card with BF16 precision and MindIE SD attention operators:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN
```

#### Recommended Multi-Card Configuration (CFG + Ulysses)

Multi-card inference prefers hybrid CFG and Ulysses parallelism. The following 8-card configuration uses 2-way CFG parallelism and 4-way Ulysses sequence parallelism. Mount eight cards as described above first, and provide a negative prompt and a `true_cfg_scale` greater than 1 in the request:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 8 \
  --cfg-parallel-size 2 \
  --usp 4 \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN
```

## Send a Request

### Text-to-Image

After the service starts, generate images with the following request. Single-card and multi-card services use the same request format:

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "negative_prompt": "blurry, low quality, text, watermark",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }'
```

### Reference Image Editing

Upload reference images through `/v1/images/edits`. The following example combines two images; replace the paths with actual file paths on the requesting client:

```bash
curl http://localhost:8091/v1/images/edits \
  -F "model=Qwen/Qwen-Image-2.1" \
  -F "image=@/path/to/input1.png" \
  -F "image=@/path/to/input2.png" \
  -F "prompt=Combine these images into a single scene" \
  -F "negative_prompt=blurry, low quality, text, watermark" \
  -F "size=1024x1024" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=4.0" \
  -F "seed=42"
```

### Request Parameters

| Parameter | Example | Description |
| ------ | ------ | ------ |
| `prompt` | — | Text prompt |
| `negative_prompt` | — | Negative prompt; enables true CFG together with `true_cfg_scale > 1` |
| `true_cfg_scale` | 4.0 | Strength of true CFG guidance |
| `num_inference_steps` | 50 | Number of denoising steps |
| `size` | 1024x1024 | Output image size; width and height must be multiples of 32. If omitted in an edit request, the size is derived from the aspect ratio of the last reference image |
| `seed` | 42 | Random seed |
| `image` | — | Reference image file for edit requests; the source version used in this guide accepts at most 4 images per request |

The maximum number of reference images is defined by `QWEN_IMAGE_21_MAX_INPUT_IMAGES` in `vllm_omni/diffusion/model_metadata.py`, and the pipeline checks the input count. Both the pinned version in this guide and the `3ea9e605` version of PR #7759 use 4; this value is not the capability limit of the model itself.

## Optimizations Adapted So Far

1. Ascend-affinity attention operators: call MindIE SD attention operators through the `FLASH_ATTN` backend. For details, see [Core Acceleration API](../../docs/en/features/core_layers.md).
2. Hybrid CFG and Ulysses parallelism: set the CFG parallel degree with `--cfg-parallel-size` and the sequence parallel degree with `--usp`.

## Known Limitations

- Sequence parallelism supports Ulysses only.
