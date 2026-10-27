# DiffSynth-Engine Qwen-Image 推理优化指南

本文介绍如何在昇腾 NPU 上通过 DiffSynth-Engine 使用 MindIE-SD 的 Attention、编译和融合算子能力，完成 Qwen-Image 文生图与 Qwen-Image-Edit-2509 图像编辑推理。

> 本文面向使用 DiffSynth-Engine 部署 Qwen-Image 系列模型的开发者，是可直接运行的端到端示例。MindIE-SD 的特性原理和接口说明请参见[编译特性](../../docs/zh/features/compilation.md)等开发文档。

---

## 环境信息

本示例已在以下环境完成验证：

| 组件 | 版本 |
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
| DiffSynth-Engine | PR 272，提交 `a837164ab4546ff6134790cd2e6137a0479e9897` |

表中版本用于说明本示例的实测环境，不代表修改了 DiffSynth-Engine 的正式依赖下限。使用其他版本时，请确保 torch、torch_npu、CANN 和 MindIE-SD 相互匹配。

---

## 1. 前置准备

### 1.1 安装昇腾基础环境

可以在已有昇腾环境中安装，也可以直接使用 MindIE 官方容器镜像。首先在宿主机选择一个用于保存源码和推理结果的工作目录，并设置模型根目录：

```bash
mkdir -p mindiesd-diffsynth
cd mindiesd-diffsynth

export MODEL_PATH="/path/to/models"
mkdir -p "${MODEL_PATH}"
```

`MODEL_PATH` 必须替换为宿主机上的真实绝对路径。任选一种基础环境路线完成后，从 1.2 节开始在当前工作目录执行相同的源码和依赖安装步骤。

#### 1.1.1 使用已有昇腾环境

请参考 [MindIE-SD 安装指导](../../docs/zh/installation.md)完成基础环境搭建，包括：

- 昇腾驱动和固件；
- CANN Toolkit 与算子包；
- PyTorch 与 torch_npu；
- MindIE-SD。

进入推理环境后加载 CANN 环境变量：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 1.1.2 使用 MindIE 官方镜像

拉取包含 CANN 9.1、torch_npu 2.9 和 MindIE 3.1 的 Ascend 950 镜像：

```bash
docker pull quay.io/ascend/mindie:3.1.0-cann9.1.0-torch_npu2.9.0.post6-950-openeuler24.03_lts_sp4-py3.11
```

该镜像的软件栈版本与上表中的实测环境不同。选择镜像路线时，应以镜像内置的 CANN、PyTorch、torch_npu 和 MindIE-SD 版本为准，不要在后续步骤中重新安装或替换这些核心组件；后续只补齐 DiffSynth-Engine 所需的其余 Python 依赖。

以下命令在刚才创建的工作目录中执行。当前目录会挂载到容器的 `/workspace`，模型目录会以相同的绝对路径挂载到容器中。示例映射宿主机 0 至 3 号 NPU，可同时运行后文的单卡和四卡示例：

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

退出后可从宿主机重新进入同一个容器：

```bash
docker start -ai "${CONTAINER_NAME}"
```

容器会保留创建时设置的工作目录和 `MODEL_PATH`。多卡容器必须挂载当前宿主机生成的有效 HCCL/rootInfo 配置，不要将其他服务器的拓扑文件打入镜像复用。若宿主机设备文件或驱动安装路径不同，请按实际 Ascend 容器运行环境调整对应挂载。

### 1.2 安装 DiffSynth-Engine

当前 MindIE-SD 适配代码位于 [DiffSynth-Engine PR 272](https://github.com/modelscope/DiffSynth-Engine/pull/272)。在该 PR 合入正式分支前，可使用已验证提交：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://github.com/modelscope/DiffSynth-Engine.git \
  DiffSynth-Engine

git -C DiffSynth-Engine fetch origin pull/272/head:pr-272
git -C DiffSynth-Engine checkout --detach \
  a837164ab4546ff6134790cd2e6137a0479e9897
```

`GIT_LFS_SKIP_SMUDGE=1` 会跳过当前示例不需要的仓库测试数据，避免 `git clone` 长时间停留在 `Filtering content`。如果 `DiffSynth-Engine` 目录已经存在，不要直接覆盖；应先确认目录内容，或换到新的工作目录。

以下依赖直接安装到当前 Python 环境的默认 site-packages。安装命令必须保留 `--no-deps`；PR 272 声明了 `torch>=2.10`，普通依赖解析可能替换基础环境中已经匹配的 torch 和 torch_npu，甚至引入 CUDA/NVIDIA 版本的 torch。

先从 PyTorch CPU wheel 源安装与 torch 2.9.0 配套的 torchvision：

```bash
python -m pip install \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  "torchvision==0.24.0+cpu"
```

再安装实测环境中使用的其余锁定依赖：

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

`--no-deps` 表示只安装命令中明确列出的软件包，不递归解析依赖。上述列表依赖基础环境已经提供 NumPy、Pillow、safetensors、requests、tokenizers、accelerate 和 einops 等公共依赖；如果使用的基础环境不同，应先核对这些包是否存在，不要直接改装 torch、torch_npu、MindIE-SD 或 CANN。

最后以 editable 模式安装 DiffSynth-Engine 本身：

```bash
python -m pip install --no-deps -e DiffSynth-Engine
```

`--no-deps` 同样会阻止安装 DiffSynth-Engine 时重新解析其依赖。安装完成后，Python 会从默认 site-packages 导入新增依赖，并通过 editable 安装指向当前工作目录下的 `DiffSynth-Engine` 源码，不再需要设置 `PYTHONPATH`。

PR 272 合入后，可以直接检出包含该适配的正式分支，不再需要获取 PR ref 或固定上述提交。

### 1.3 下载模型权重

可以使用 ModelScope 下载 Qwen-Image 和 Qwen-Image-Edit-2509：

```bash
modelscope download --model Qwen/Qwen-Image \
  --local_dir "${MODEL_PATH}/Qwen-Image"

modelscope download --model Qwen/Qwen-Image-Edit-2509 \
  --local_dir "${MODEL_PATH}/Qwen-Image-Edit-2509"
```

多卡推理前应完成权重下载，避免多个 worker 同时访问远端模型仓库。

---

## 2. 模型推理示例

### 2.1 Qwen-Image 文生图

将以下代码保存为 `qwen_image_mindiesd.py`：

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401，注册 torch.npu

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig


def parse_args():
    parser = argparse.ArgumentParser()
    # 模型目录和生成图片的保存位置。
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="qwen_image.png")
    # 编译与并行参数可直接从命令行调整。
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cfg-parallel", action="store_true")
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29500)
    return parser.parse_args()


def main():
    args = parse_args()
    # 在这里选择 Attention、编译和并行策略。
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
            # 可在这里修改提示词、输出尺寸和推理步数。
            prompt="A painting of a cat in a zen garden",
            negative_prompt="ugly, blurry, low quality",
            true_cfg_scale=4.0,
            width=1328,
            height=1328,
            num_inference_steps=28,
            # 修改随机种子可以生成不同结果。
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        # 输出路径由 --output 指定。
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
```

单卡运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --output outputs/qwen_image.png
```

配置中的 `attn_type="mindie"` 会选择 MindIE Attention 后端，Attention 计算由 `mindiesd.layers.flash_attn.attention_forward` 完成。

### 2.2 Qwen-Image-Edit-2509 图像编辑

将以下代码保存为 `qwen_image_edit_mindiesd.py`。`--input` 可以接收一张或多张参考图：

```python
#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch_npu  # noqa: F401，注册 torch.npu
from PIL import Image

from diffsynth_engine import DiffSynthEngine
from diffsynth_engine.configs import QwenImagePipelineConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    # --input 支持传入一张或多张参考图；--prompt 指定编辑要求。
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="qwen_image_edit.png")
    # 可根据输入案例修改目标尺寸。
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    # 编译与并行参数可直接从命令行调整。
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--cfg-parallel", action="store_true")
    parser.add_argument("--ulysses", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29501)
    return parser.parse_args()


def main():
    args = parse_args()
    # 输入图片会按命令行顺序传给模型。
    input_images = [Image.open(path).convert("RGB") for path in args.input]
    # 在这里选择 Attention、编译和并行策略。
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
            # prompt、CFG scale、尺寸和推理步数均可按案例调整。
            prompt=args.prompt,
            negative_prompt=" ",
            true_cfg_scale=4.0,
            width=args.width,
            height=args.height,
            num_inference_steps=40,
            # 修改随机种子可以生成不同结果。
            generator=torch.Generator(device="cpu").manual_seed(42),
        )

        # 输出路径由 --output 指定。
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(output)
    finally:
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
```

单卡运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

python qwen_image_edit_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image-Edit-2509" \
  --input /path/to/reference_1.png /path/to/reference_2.png \
  --prompt "Replace the object in image 1 with the object in image 2" \
  --output outputs/qwen_image_edit.png
```

### 2.3 使能编译和融合算子

向脚本传入 `--compile` 后，DiffSynth-Engine 会对重复的 Transformer Block 调用 `torch.compile`。在 NPU 平台能力检查通过时，编译后端使用 MindIE-SD 的 `MindieSDBackend()`：

```bash
python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --compile \
  --output outputs/qwen_image_compile.png
```

首次 forward 会完成 TorchDynamo 图捕获和编译预热，因此首次推理耗时通常高于后续推理。

在启动 Python 前设置 `USE_MINDIESD_FUSE=true`，可使能 DiffSynth-Engine NPU 适配层中的融合算子路径：

```bash
USE_MINDIESD_FUSE=true \
python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --compile \
  --output outputs/qwen_image_optimized.png
```

Qwen-Image 路径中包含以下优化：

- MindIE-SD RoPE 融合算子；
- MindIE-SD LayerNorm + Scale + Shift 融合算子；
- torch_npu RMSNorm 融合算子。

`USE_MINDIESD_FUSE` 由 DiffSynth-Engine NPU 适配层读取，需要在导入 DiffSynth-Engine 前设置。

### 2.4 四卡并行推理

DiffSynthEngine 会根据 `parallelism` 在内部创建 worker，因此以下端到端示例直接使用 `python`，不需要再套一层 `torchrun`。

四卡 Ulysses 序列并行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

python qwen_image_mindiesd.py \
  --model-path "${MODEL_PATH}/Qwen-Image" \
  --parallelism 4 \
  --ulysses 4 \
  --output outputs/qwen_image_ulysses4.png
```

四卡 CFG + Ulysses2：

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

## 参考链接

- [MindIE-SD 安装指导](../../docs/zh/installation.md)
- [MindIE-SD 编译特性](../../docs/zh/features/compilation.md)
- [DiffSynth-Engine](https://github.com/modelscope/DiffSynth-Engine)
- [DiffSynth-Engine PR 272](https://github.com/modelscope/DiffSynth-Engine/pull/272)
- [Qwen-Image](https://modelscope.cn/models/Qwen/Qwen-Image)
- [Qwen-Image-Edit-2509](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2509)
