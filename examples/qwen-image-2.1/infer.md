# Qwen-Image-2.1

## 简介

Qwen-Image-2.1 是图像生成模型，覆盖文生图及参考图像编辑等工作流。该模型使用 Qwen3-VL 编码文本与参考图像，通过单流 DiT 生成图像。

> [!NOTE]说明
> 下载权重需具备模型仓库访问权限。

## 已支持的机型

- Atlas 350。
- Atlas 800I A2 推理服务器。
- Atlas 800I A3 超节点服务器。

**其他机型待更新。**

## 环境准备

### 模型权重

Qwen-Image-2.1: [下载模型权重](https://huggingface.co/Qwen/Qwen-Image-2.1)

```bash
export MODEL=/path/to/Qwen-Image-2.1
hf download Qwen/Qwen-Image-2.1 --local-dir "${MODEL}"
```

### 部署环境

#### 官方 Docker 镜像

通过 [vLLM-Omni 镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)选择支持 Qwen-Image-2.1、与所用机型匹配的官方镜像。镜像中已安装配套的 vLLM、vLLM-Ascend、vLLM-Omni 和 MindIE-SD，可直接启动服务。请替换下方镜像标签占位：

```bash
# 拉取与所用机型匹配的官方镜像（先替换标签）
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

## 启动服务

### 单卡配置

以下示例使用 BF16 精度及 MindIE-SD 注意力算子，通过 `FLASH_ATTN` 后端调用。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN
```

### 8 卡配置（Ulysses）

以下为 8 卡 Ulysses 序列并行配置，请先按前述说明挂载 8 张卡。`--usp` 配置序列并行，不是张量并行（TP）。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 8 \
  --cfg-parallel-size 1 \
  --usp 8 \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN
```

### CFG 配置

`true_cfg_scale` 是请求中的引导强度，`--cfg-parallel-size` 是服务启动时的 CFG 并行度，两者含义不同：

- 不传 `negative_prompt`，或 `true_cfg_scale <= 1` 时，不启用 true CFG，应使用 `--cfg-parallel-size 1`。配置为 2 不会自动退回单路，会产生冗余计算。
- 需要 true CFG 时，由用户提供 `negative_prompt`，并设置 `true_cfg_scale > 1`。单路 CFG 可顺序执行两个分支；8 卡服务也可将上述参数改为 `--cfg-parallel-size 2 --usp 4`，并行执行两个分支。

Ulysses 不切分模型权重，不能将卡数增加理解为每卡模型权重显存按比例下降。

### 量化配置

支持在线 W8A8 量化，直接使用原始模型权重，无需额外下载量化权重。以下为单卡启动示例，请将 `MODEL` 替换为实际模型路径。

#### Atlas 800I A2 推理服务器、Atlas 800I A3 超节点服务器：INT8

```bash
export MODEL=/path/to/Qwen-Image-2.1
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --cfg-parallel-size 1 \
  --enforce-eager \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"int8"},"text_encoder":null}'
```

#### Atlas 350：MXFP8

```bash
export MODEL=/path/to/Qwen-Image-2.1
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --cfg-parallel-size 1 \
  --enforce-eager \
  --dtype bfloat16 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"},"text_encoder":null}'
```

上述配置仅量化 Transformer 中支持量化的线性层，文本编码器、VAE 和 Attention 保持 BF16。请求格式与下述文生图、参考图像编辑示例相同。量化可降低显存占用，生成效果请结合实际业务场景评估。

## 发送请求

### 文生图

服务启动后，可通过以下请求生成图像。单卡与多卡服务使用相同的请求格式：

```bash
curl http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "true_cfg_scale": 1.0,
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
  -F "size=1024x1024" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=1.0" \
  -F "seed=42"
```

### 请求参数说明

| 参数 | 示例值 | 说明 |
| ------ | ------ | ------ |
| `prompt` | — | 文本提示词 |
| `negative_prompt` | blurry, low quality | 可选，由用户提供；配合 `true_cfg_scale > 1` 启用 true CFG |
| `true_cfg_scale` | 1.0 | true CFG 引导强度；不传负向提示词时不启用 true CFG |
| `num_inference_steps` | 50 | 去噪步数 |
| `size` | 1024x1024 | 输出图像尺寸，宽高须为 32 的倍数；建议显式指定，编辑接口省略时以第一张参考图像的尺寸为依据进行对齐 |
| `seed` | 42 | 随机种子 |
| `image` | — | 编辑请求的参考图像文件，单次最多 4 张 |

## 已知限制

- 序列并行仅支持 Ulysses。
