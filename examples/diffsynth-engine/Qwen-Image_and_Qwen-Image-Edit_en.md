# DiffSynth-Engine Qwen-Image Inference Optimization Guide

This document describes how to use the attention, compilation, and fused operator capabilities of MindIE SD through DiffSynth-Engine on an Ascend NPU to run Qwen-Image text-to-image and Qwen-Image-Edit-2509 image editing inference.

> This document targets developers who deploy Qwen-Image series models with DiffSynth-Engine and is a runnable end-to-end example. For the implementation principles and interface descriptions of MindIE SD features, see developer documents such as [Compilation Features](../../docs/en/features/compilation.md).

---

## Environment Information

This example has been verified in the following environment:

| Component | Version |
|------|------|
| NPU | Ascend 950PR |
| Python | 3.11.6 |
| CANN | 9.2 |
| torch | 2.9.0+cpu |
| torchvision | 0.24.0+cpu |
| torch_npu | 2.9.0.post7.dev20260824 |
| mindiesd | 3.2.0.dev202609040106 |
| diffusers | 0.36.0 |
| transformers | 4.57.6 |
| DiffSynth-Engine | PR 272, commit `a837164ab4546ff6134790cd2e6137a0479e9897` |

The versions in the table describe the verified environment of this example and do not change the official dependency floor of DiffSynth-Engine. If you use other versions, make sure that torch, torch_npu, CANN, and MindIE SD match each other.

---

## 1. Prerequisites

### 1.1 Installing the Ascend Basic Environment

You can install into an existing Ascend environment or use the official MindIE container image directly. First, choose a working directory on the host for the source code and inference results, and set the model root directory:

```bash
mkdir -p mindiesd-diffsynth
cd mindiesd-diffsynth

export MODEL_PATH="/path/to/models"
mkdir -p "${MODEL_PATH}"
```

`MODEL_PATH` must be replaced with a real absolute path on the host. After completing either base environment route, run the same source and dependency installation steps from section 1.2 in the current working directory.

#### 1.1.1 Using an Existing Ascend Environment

Complete the basic environment setup as described in the [MindIE SD Installation Guide](../../docs/en/installation.md), including:

- Ascend driver and firmware;
- CANN Toolkit and operator packages;
- PyTorch and torch_npu;
- MindIE SD.

After entering the inference environment, load the CANN environment variables:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 1.1.2 Using the Official MindIE Image

Pull the Ascend 950 image that contains CANN 9.1, torch_npu 2.9, and MindIE 3.1:

```bash
docker pull quay.io/ascend/mindie:3.1.0-cann9.1.0-torch_npu2.9.0.post6-950-openeuler24.03_lts_sp4-py3.11
```

The software stack versions of this image differ from the verified environment in the table above. When choosing the image route, take the CANN, PyTorch, torch_npu, and MindIE SD versions bundled in the image as the baseline, and do not reinstall or replace these core components in later steps; only add the remaining Python dependencies required by DiffSynth-Engine.

The following commands run in the working directory just created. The current directory is mounted to `/workspace` in the container, and the model directory is mounted at the same absolute path. The example maps host NPUs 0 to 3, so both the single-card and four-card examples below can run:

```bash
export CONTAINER_NAME="mindiesd-diffsynth"

test -s /etc/hccl_rootinfo.json

docker run -it \
  --name "${CONTAINER_NAME}" \
  --runtime=ascend \
  --privileged \
  --network host \
  --shm-size=256g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
  -v /etc/hccn.conf:/etc/hccn.conf:ro \
  -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "$(pwd)":/workspace \
  -v "${MODEL_PATH}":"${MODEL_PATH}" \
  -w /workspace \
  -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e OMP_NUM_THREADS=1 \
  quay.io/ascend/mindie:3.1.0-cann9.1.0-torch_npu2.9.0.post6-950-openeuler24.03_lts_sp4-py3.11 \
  bash
```

After exiting, you can re-enter the same container from the host:

```bash
docker start -ai "${CONTAINER_NAME}"
```

The container retains the working directory and `MODEL_PATH` set at creation time. A multi-card container must mount a valid HCCL/rootInfo configuration generated on the current host; do not bake topology files from another server into the image. If device files or driver installation paths differ on the host, adjust the corresponding mounts for the actual Ascend container runtime environment.

### 1.2 Installing DiffSynth-Engine

The current MindIE SD adaptation code is in [DiffSynth-Engine PR 272](https://github.com/modelscope/DiffSynth-Engine/pull/272). Before that PR is merged into the official branch, use the verified commit:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/modelscope/DiffSynth-Engine.git \
  DiffSynth-Engine

git -C DiffSynth-Engine fetch origin pull/272/head:pr-272
git -C DiffSynth-Engine checkout --detach \
  a837164ab4546ff6134790cd2e6137a0479e9897
```

`GIT_LFS_SKIP_SMUDGE=1` skips the repository test data that this example does not need, preventing `git clone` from staying at `Filtering content` for a long time. If the `DiffSynth-Engine` directory already exists, do not overwrite it directly; check its contents first, or switch to a new working directory.

The following dependencies are installed directly into the default site-packages of the current Python environment. The installation commands must keep `--no-deps`; PR 272 declares `torch>=2.10`, so ordinary dependency resolution may replace the already matched torch and torch_npu in the base environment, or even pull in a CUDA/NVIDIA build of torch.

First install the torchvision that matches torch 2.9.0 from the PyTorch CPU wheel index:

```bash
python -m pip install \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  "torchvision==0.24.0+cpu"
```

Then install the remaining pinned dependencies used in the verified environment:

```bash
python -m pip install \
  --no-deps \
  transformers==4.57.6 \
  diffusers==0.36.0 \
  modelscope==1.40.0 \
  modelscope-hub==0.4.2 \
  flufl.lock==9.1.0 \
  atpublic==7.0.0 \
  imageio==2.37.4 \
  imageio-ffmpeg==0.6.0 \
  librosa==0.11.0 \
  moviepy==2.2.1 \
  audioread==3.1.0 \
  cffi==2.1.1 \
  cryptography==50.0.1 \
  huggingface-hub==0.36.2 \
  lazy_loader==0.5 \
  llvmlite==0.49.0 \
  msgpack==1.2.2 \
  numba==0.67.0 \
  pooch==1.9.0 \
  proglog==0.1.12 \
  pycparser==3.0 \
  python-dotenv==1.2.3 \
  soundfile==0.14.0 \
  soxr==1.1.0
```

`--no-deps` installs only the packages explicitly listed in the command and does not resolve dependencies recursively. The list above assumes that the base environment already provides common dependencies such as NumPy, Pillow, safetensors, requests, tokenizers, accelerate, and einops. If you use a different base environment, check that these packages exist first, and do not replace torch, torch_npu, MindIE SD, or CANN directly.

Finally, install DiffSynth-Engine itself in editable mode:

```bash
python -m pip install --no-deps -e DiffSynth-Engine
```

`--no-deps` likewise prevents re-resolving the dependencies of DiffSynth-Engine during installation. After installation, Python imports the new dependencies from the default site-packages and, through the editable install, points to the `DiffSynth-Engine` source in the current working directory, so `PYTHONPATH` is no longer needed.

After PR 272 is merged, you can check out the official branch that contains the adaptation directly, without fetching the PR ref or pinning the commit above.

### 1.3 Downloading the Model Weights

You can download Qwen-Image and Qwen-Image-Edit-2509 with ModelScope:

```bash
modelscope download --model Qwen/Qwen-Image \
  --local_dir "${MODEL_PATH}/Qwen-Image"

modelscope download --model Qwen/Qwen-Image-Edit-2509 \
  --local_dir "${MODEL_PATH}/Qwen-Image-Edit-2509"
```

Complete the weight download before multi-card inference to avoid multiple workers accessing the remote model repository at the same time.

---

## 2. Model Inference Example

### 2.1 Qwen-Image Text-to-Image

Save the following code as `qwen_image_mindiesd.py`:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401, registers torch.npu

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig


def parse_args():
    parser = argparse.ArgumentParser()
    # Model directory and the save location for generated images.
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="qwen_image.png")
    # Compilation and parallelism parameters can be adjusted from the command line.
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cfg-parallel", action="store_true")
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def main():
    args = parse_args()
    # Select the attention, compilation, and parallelism strategy here.
    config = QwenImagePipelineConfig(
        model_path=args.model_path,
        device="npu",
        model_dtype=torch.bfloat16,
        attn_type="mindie",
        use_torch_compile=args.compile,
        parallelism=args.parallelism,
        use_cfg_parallel=args.cfg_parallel,
        sp_ulysses_degree=args.ulysses,
        sp_ring_degree=1,
        tp_degree=1,
    )

    engine = None
    try:
        engine = DiffSynthEngine.from_pretrained(
            config,
            master_port=args.master_port,
        )
        result = engine.generate(
            # Modify the prompt, output size, and number of inference steps here.
            prompt="A painting of a cat in a zen garden",
            negative_prompt="ugly, blurry, low quality",
            true_cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            # Changing the random seed produces different results.
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        # The output path is specified by --output.
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
```

Run on a single card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --output outputs/qwen_image.png
```

The `attn_type="mindie"` setting in the configuration selects the MindIE attention backend, and the attention computation is performed by `mindiesd.layers.flash_attn.attention_forward`.

### 2.2 Qwen-Image-Edit-2509 Image Editing

Save the following code as `qwen_image_edit_mindiesd.py`. `--input` accepts one or more reference images:

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401, registers torch.npu
from PIL import Image

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    # --input accepts one or more reference images; --prompt specifies the editing instruction.
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="qwen_image_edit.png")
    # The target size can be adjusted for the input case.
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # Compilation and parallelism parameters can be adjusted from the command line.
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cfg-parallel", action="store_true")
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29501)
    return parser.parse_args()


def main():
    args = parse_args()
    # Input images are passed to the model in command-line order.
    input_images = [Image.open(path).convert("RGB") for path in args.input]
    # Select the attention, compilation, and parallelism strategy here.
    config = QwenImagePipelineConfig(
        model_path=args.model_path,
        device="npu",
        model_dtype=torch.bfloat16,
        attn_type="mindie",
        use_torch_compile=args.compile,
        parallelism=args.parallelism,
        use_cfg_parallel=args.cfg_parallel,
        sp_ulysses_degree=args.ulysses,
        sp_ring_degree=1,
        tp_degree=1,
    )

    engine = None
    try:
        engine = DiffSynthEngine.from_pretrained(
            config,
            master_port=args.master_port,
        )
        result = engine.generate(
            image=input_images,
            # The prompt, CFG scale, size, and number of inference steps can all be adjusted per case.
            prompt=args.prompt,
            negative_prompt=" ",
            true_cfg_scale=4.0,
            width=args.width,
            height=args.height,
            num_inference_steps=40,
            # Changing the random seed produces different results.
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        # The output path is specified by --output.
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
```

Run on a single card:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python qwen_image_edit_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image-Edit-2509" \
  --input /path/to/reference_1.png /path/to/reference_2.png \
  --prompt "Replace the object in image 1 with the object in image 2" \
  --output outputs/qwen_image_edit.png
```

### 2.3 Enabling Compilation and Fused Operators

After `--compile` is passed to the script, DiffSynth-Engine calls `torch.compile` on the repeated Transformer Blocks. When the NPU platform capability check passes, the compilation backend uses MindIE SD's `MindieSDBackend()`:

```bash
python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --compile \
  --output outputs/qwen_image_compile.png
```

The first forward completes TorchDynamo graph capture and compilation warm-up, so the first inference usually takes longer than subsequent ones.

Set `USE_MINDIESD_FUSE=true` before starting Python to enable the fused operator path in the DiffSynth-Engine NPU adaptation layer:

```bash
USE_MINDIESD_FUSE=true \
python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --compile \
  --output outputs/qwen_image_optimized.png
```

The Qwen-Image path includes the following optimizations:

- MindIE SD RoPE fused operator;
- MindIE SD LayerNorm + Scale + Shift fused operator;
- torch_npu RMSNorm fused operator.

`USE_MINDIESD_FUSE` is read by the DiffSynth-Engine NPU adaptation layer and must be set before DiffSynth-Engine is imported.

### 2.4 Four-Card Parallel Inference

DiffSynthEngine creates workers internally according to `parallelism`, so the end-to-end examples below use `python` directly and do not need an extra `torchrun` wrapper.

Four-card Ulysses sequence parallelism:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --parallelism 4 \
  --ulysses 4 \
  --output outputs/qwen_image_ulysses4.png
```

Four-card CFG + Ulysses2:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --parallelism 4 \
  --cfg-parallel \
  --ulysses 2 \
  --output outputs/qwen_image_cfg_ulysses2.png
```

---

## References

- [MindIE SD Installation Guide](../../docs/en/installation.md)
- [MindIE SD Compilation Features](../../docs/en/features/compilation.md)
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)
- [DiffSynth-Engine PR 272](https://github.com/modelscope/DiffSynth-Engine/pull/272)
- [Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image)
- [Qwen-Image-Edit-2509](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)
