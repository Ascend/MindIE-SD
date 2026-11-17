# MiniMax-H3-Image

## 简介

MiniMax-H3-Image 是 MiniMax-H3 家族的图像生成/编辑模型，于 2026 年 11 月开源。模型将文生图（t2i）与参考图编辑（edit）合并在同一份 checkpoint 中：DiT 沿用 H3 视频模型的 33.1B 单流结构，去掉音频分支；文本编码器直接复用 MiniMax-H3 未改动的 Qwen3-VL。
本文给出 MiniMax-H3-Image 在昇腾上的推理部署配置。

## 已支持的机型

- Atlas 800I A2（1/2/4/8 卡）
- Atlas 800T A3（1/2/4/8 die）
- Atlas 350（A5）（1/2/4/8 卡）

## 环境准备

### 模型权重

MiniMax-H3-Image: [下载模型权重](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)

模型仓库**不包含文本编码器**。`model_index.json` 指向 MiniMax-H3 的 `text_encoder` 子目录：

```bash
export MODEL=/path/to/MiniMax-H3-Image

# ModelScope 下载
modelscope download --model MiniMax/MiniMax-H3-Image \
  --local_dir "${MODEL}" --max-workers 8

# 复用 MiniMax-H3 的文本编码器
ln -sfn /path/to/MiniMax-H3/FL2VA/text_encoder "${MODEL}/text_encoder"
```

下载完成后，权重目录结构应如下：

```text
MiniMax-H3-Image/
├── model_index.json
├── pdd_config.json                 # 6 步蒸馏 sigma schedule
├── pdd_fused_heads.safetensors     # 每步输出头
├── processor/  tokenizer/          # Qwen3-VL tokenizer/processor
├── transformer/                    # DiT，12 分片，BF16，约 61.7 GiB
├── vae/                            # 图像 VAE，FP32，约 2.2 GiB
└── text_encoder -> ...             # 复用的 Qwen3-VL（可选软链）
```

### 部署环境

#### 官方 Docker 镜像

MiniMax-H3-Image 从 vLLM-Omni 0.33（当前最新版本）开始适配，后续 vLLM-Omni 版本会继续更新并保持支持。请通过 [vLLM-Omni 镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)选择与所用机型匹配、包含 vLLM-Omni 0.33 或更高版本的官方镜像：

| 适用产品 | 镜像 Tag |
| --- | --- |
| Atlas 800I A2 | `quay.io/ascend/vllm-omni:v0.33.0rc1` |
| Atlas 800T A3 | `quay.io/ascend/vllm-omni:v0.33.0rc1-a3` |
| Atlas 350（A5） | `quay.io/ascend/vllm-omni:v0.33.0rc1-a5` |

```bash
export IMAGE=quay.io/ascend/vllm-omni:v0.33.0rc1-a3
export CONTAINER_NAME=h3-image

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

> [!NOTE]说明
> Atlas 350（A5）不存在 `/dev/devmm_svm` 设备，创建容器时请删除该映射；`/dev/davinciN` 按实际可见卡数保留。
> Atlas 350（A5）场景还需增加 `-v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo`。

#### 安装最新 vLLM-Omni【可选】

MiniMax-H3-Image 的模型支持由 vLLM-Omni 上游引入（[vllm-project/vllm-omni#10120](https://github.com/vllm-project/vllm-omni/pull/10120)）。若镜像内置版本较旧，可在容器内安装最新源码：

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

#### 安装 MindIE SD【可选】

官方镜像中如未安装 mindiesd，可源码编译安装，参考如下指导：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

## 启动服务

本文中只列出 1 卡与 8 卡两种参考配置，若需 2、4 卡部署，按实际卡数调整 `--num-gpus`、`--usp` 与 `--text-encoder-tp-size` 即可。
三种机型的请求方式完全一致，见[发送请求](#发送请求)。

### Atlas 800T A3 / Atlas 800I A2

#### Atlas 800T A3 / Atlas 800I A2 1 卡

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit","text_encoder"]}'
```

#### Atlas 800T A3 8 die / Atlas 800I A2 8 卡

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 8 --tensor-parallel-size 8 --usp 1 --text-encoder-tp-size 8
```

### Atlas 350（A5）

#### Atlas 350（A5）1 卡

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 1 \
  --diffusion-offload-config '{"mode":"layer","components":["dit"]}'
```

#### Atlas 350（A5）8 卡

```bash
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" --omni \
  --host 127.0.0.1 --port 18089 \
  --served-model-name MiniMaxAI/MiniMax-H3-Image \
  --enforce-eager --num-gpus 8 --usp 8 --text-encoder-tp-size 8
```

## 发送请求

### 文生图

```bash
curl -fsS http://127.0.0.1:18089/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMaxAI/MiniMax-H3-Image",
    "prompt":"Documentary photograph of a lighthouse keeper on a cliff at dawn, Kodak Portra 400",
    "size":"2048x2048",
    "seed":42,
    "response_format":"b64_json"
  }' | jq -er '.data[0].b64_json' | base64 -d > t2i-2048.png
```

### 参考图像编辑

```bash
ASSETS=/path/to/MiniMax-H3-Image/examples

curl -fsS http://127.0.0.1:18089/v1/images/edits \
  -F 'model=MiniMaxAI/MiniMax-H3-Image' \
  -F "image=@$ASSETS/reference1.png;type=image/png" \
  -F "prompt=<$ASSETS/prompt.edit.txt" \
  -F 'size=4096x4096' \
  -F 'seed=1234' -F 'generator_device=cpu' \
  -F 'response_format=b64_json' -F 'output_format=png' \
  | jq -er '.data[0].b64_json' | base64 -d > edit-4096.png
```

`prompt=<file` 表示从文件读取提示词。参考图按上传顺序对应 `Picture 1`、`Picture 2`…，单次最多 9 张。

### 请求参数说明

| 参数 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| `prompt` | `str` | 是 | 整幅图像的详细英文描述，超过 4096 tokens 会被截断 |
| `size` | `str` | 是 | `宽x高`，宽高均须为 32 的倍数；返回图片尺寸与请求完全一致 |
| `image` | `file` | 仅编辑 | 参考图像，每个文件一个 `image` 字段，1～9 张；上传顺序对应 `Picture 1`、`Picture 2`… |
| `seed` | `int` | 建议 | 随机种子 |
| `generator_device` | `str` | 建议 | 设为 `cpu` 时，初始噪声在各机器上一致 |
| `response_format` | `str` | 建议 | `b64_json`；图片位于 `data[0].b64_json` |
| `output_format` | `str` | 可选 | `png`（默认）或 `jpeg` |
| `extra_args.min_generation_area` | `int` | 可选 | 最小生成画布面积（像素），32²～4096²，默认 4194304（2048×2048）；更小的 `size` 会按相同宽高比在更大画布上生成后缩放回目标尺寸 |
| `num_inference_steps` | `int` | 可选 | 省略或设为 `6`；其他值会被拒绝 |
| `guidance_scale` | `float` | 可选 | 省略或设为 `1`；其他值会被拒绝（模型已做 CFG 蒸馏） |
| `negative_prompt` | `str` | 不支持 | 请求会被拒绝 |

### 输入与输出尺寸

模型的有效工作范围约为 4MP～20MP，16MP 附近效果最佳。官方 16MP 尺寸如下：

| 宽高比 | 尺寸（宽 x 高） |
| --- | --- |
| 1:1 | 4096x4096 |
| 4:3 / 3:4 | 4736x3552 / 3552x4736 |
| 3:2 / 2:3 | 5024x3360 / 3360x5024 |
| 16:9 / 9:16 | 5472x3072 / 3072x5472 |
| 21:9 | 6272x2688 |

vLLM-Omni 接受任意宽高均为 32 倍数的 `WIDTHxHEIGHT`，并原样返回该尺寸，因此上述官方尺寸均可直接使用。`t2i` 与 `edit` 使用相同的尺寸规则。

| 宽高比 | 尺寸 | 32 像素对齐 | vLLM-Omni |
| --- | --- | --- | --- |
| 1:1 | 4096x4096 | 是 | 支持 |
| 4:3 | 4736x3552 | 是 | 支持 |
| 3:4 | 3552x4736 | 是 | 支持 |
| 3:2 | 5024x3360 | 是 | 支持 |
| 2:3 | 3360x5024 | 是 | 支持 |
| 16:9 | 5472x3072 | 是 | 支持 |
| 9:16 | 3072x5472 | 是 | 支持 |
| 21:9 | 6272x2688 | 是 | 支持 |

小于 2048×2048 的请求会先按相同宽高比在至少 2048×2048 的画布上生成（边长按 32 像素对齐），再在设备端用 LANCZOS 缩放到目标尺寸。例如请求 `1024x1024` 时，输出精确为 1024×1024。可通过 `extra_args.min_generation_area` 调整画布下限。编辑请求建议直接请求画布尺寸，以跳过最终缩放，保持与参考图的最佳对齐。

## 验证

确认服务已加载模型：

```bash
curl -s http://127.0.0.1:18089/v1/models | jq -r '.data[].id'
# MiniMaxAI/MiniMax-H3-Image
```

生成响应中的图片位于 `data[0].b64_json`；base64 解码后的 PNG 宽高应与请求的 `WIDTHxHEIGHT` 完全一致。

## 性能

### Atlas 350（A5）性能测试

客户端端到端时延，统计到收到完整 HTTP 响应；前三列为一个用例预热一次后三次请求的中位数，4096² 编辑为单次请求。T2I 使用 lighthouse prompt、seed 42 与默认最小生成面积（1024² 在 2048² 画布生成后缩放）；编辑使用发布版 1024² 参考图与结构化 prompt、CPU generator、seed 1234。

| Profile | T2I 1024² | T2I 2048² | Edit 2048² | Edit 4096² | 峰值 HBM/NPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 卡，Ulysses SP4，文本编码器 TP4，全常驻 | 1.52 s | 1.49 s | 2.47 s | 8.56 s | 85.61 GiB |
| 2 卡，Ulysses SP2，文本编码器 TP2，DiT DLO AllGather | 6.45 s | 6.45 s | 7.27 s | 18.89 s | 39.65 GiB |
| 1 卡，DiT 逐层卸载 | 7.79 s | 7.76 s | 8.36 s | 31.10 s | 64.20 GiB |

### Atlas 800I A2 性能测试

测试用例同 Atlas 350（A5）。

| Profile | T2I 1024² | T2I 2048² | Edit 2048² | Edit 4096² | 峰值 HBM/NPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 卡，DiT TP4，权重常驻 | 5.28 s | 4.98 s | 7.75 s | 24.66 s | 39.10 GiB |
| 2 卡，Ulysses SP2，DiT DLO AllGather | 16.14 s | 16.09 s | 18.72 s | 55.83 s | 38.44 GiB |
| 1 卡，DiT + 文本编码器逐层卸载 | 24.54 s | 24.86 s | 26.04 s | 98.19 s | 12.69 GiB |

## 支持的优化项

| 优化项 | 开启配置 |
| --- | --- |
| DiT USP 并行 | `--num-gpus N --usp N` |
| DiT TP 并行 | `--num-gpus N --tensor-parallel-size N --usp 1` |
| 文本编码器 TP 并行 | `--text-encoder-tp-size N` |
| DiT DLO | `--diffusion-offload-config '{"mode":"layer","components":["dit"]}'`（多卡叠加 `"layer_options":{"dit":{"weight_transfer":"allgather"}}`） |
| 文本编码器 DLO | `--diffusion-offload-config '{"mode":"layer","components":["text_encoder"]}'`（多卡叠加 `"layer_options":{"text_encoder":{"weight_transfer":"rank-local"}}`） |
| VAE tile 并行 | 多卡默认开启（`additional_config.vae_tile_parallel`） |

## 后续计划

MiniMax-H3-Image 的融合算子、稀疏注意力、量化与缓存优化持续适配中，敬请关注。
