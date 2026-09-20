# Qwen-Image-2.1

## 简介

Qwen-Image-2.1 是图像生成模型，覆盖文生图及参考图像编辑等工作流。该模型使用 Qwen3-VL 编码文本与参考图像，通过单流 DiT 生成图像。

> [!NOTE]说明
> 模型支持代码来自 [vLLM-Omni PR #7759](https://github.com/vllm-project/vllm-omni/pull/7759)，请按下文安装指定源码提交。下载权重需具备模型仓库访问权限。

## 已支持的机型

- Atlas 350。

**其他机型待更新。**

## 环境准备

### 模型权重

Qwen-Image-2.1: [下载模型权重](https://huggingface.co/Qwen/Qwen-Image-2.1)

```bash
export MODEL=/path/to/Qwen-Image-2.1
hf download Qwen/Qwen-Image-2.1 --local-dir "${MODEL}"
```

### 部署环境

#### 1）官方 Docker 镜像

通过[vllm-omni镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)选择适用于 Atlas 350 的官方镜像作为基础环境。镜像中已携带MindIE-SD，无需单独安装。请替换下方镜像标签占位，并按第 2 步安装 PR 中的模型支持代码：

```bash
# 拉取适用于 Atlas 350 的官方镜像（先替换标签）
export IMAGE='quay.io/ascend/vllm-omni:<IMAGE_TAG>'
docker pull "${IMAGE}"

# 创建容器（单卡）
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

8 卡场景在创建容器时追加以下设备参数，后续通过 `ASCEND_RT_VISIBLE_DEVICES` 指定可见卡：

```bash
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7
```

#### 2）安装vllm-Omni

Qwen-Image-2.1 需要安装 PR #7759 中的模型支持代码，请在容器内参考如下指导进行安装：

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git fetch origin refs/pull/7759/head
git checkout --detach 3ea9e605011fc2a0bfa9f3e8531286fb683068dd
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

## 启动服务

### Atlas 350

#### 推荐单卡配置

以下是单卡服务启动命令，使用 BF16 精度及 MindIE-SD 注意力算子：

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

#### 推荐多卡配置（CFG + Ulysses）

多卡推理优先使用 CFG 与 Ulysses 混合并行。以下为 8 卡配置，使用 2 路 CFG 并行及 4 路 Ulysses 序列并行。请先按前述说明挂载 8 张卡，并在请求中提供负向提示词及大于 1 的 `true_cfg_scale`：

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

## 发送请求

### 文生图

服务启动后，可通过以下请求生成图像。单卡与多卡服务使用相同的请求格式：

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

### 参考图像编辑

通过 `/v1/images/edits` 上传参考图像。以下示例组合两张图像，请将路径替换为请求客户端上的实际文件路径：

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

### 请求参数说明

| 参数 | 示例值 | 说明 |
| ------ | ------ | ------ |
| `prompt` | — | 文本提示词 |
| `negative_prompt` | — | 负向提示词，配合 `true_cfg_scale > 1` 启用 true CFG |
| `true_cfg_scale` | 4.0 | true CFG 引导强度 |
| `num_inference_steps` | 50 | 去噪步数 |
| `size` | 1024x1024 | 输出图像尺寸，宽高必须是 32 的倍数；编辑请求不填时根据最后一张参考图像的宽高比推导 |
| `seed` | 42 | 随机种子 |
| `image` | — | 编辑请求的参考图像文件；本指导源码版本单次最多 4 张 |

参考图像数量上限由 `vllm_omni/diffusion/model_metadata.py` 中的 `QWEN_IMAGE_21_MAX_INPUT_IMAGES` 定义，Pipeline 会检查输入数量。本指导固定版本与 PR #7759 的 `3ea9e605` 版本均为 4；该值不是模型本身的能力上限。

## 当前已适配的优化点

1. 昇腾亲和的注意力算子：通过 `FLASH_ATTN` 后端调用 MindIE-SD 注意力算子，详情见[核心层接口](../../docs/zh/features/core_layers.md)。
2. CFG 与 Ulysses 混合并行：通过 `--cfg-parallel-size` 配置 CFG 并行度，通过 `--usp` 配置序列并行度。

## 已知限制

- 序列并行仅支持 Ulysses。
