# MiniMax-H3

## 简介

2026年8月3日，MiniMax正式开源通用全模态生成模型 MiniMax-H3，有33B参数量，覆盖文本生成音视频（T2VA）、首帧/末帧生成音视频（FL2VA）及多模态参考生成音视频（Ref2VA）等工作流。该模型统一理解文本、图像、视频和声音等多模态上下文，并生成原生双声道音视频，最高支持 15 秒 2K 分辨率输出，在指令遵循、品牌与文字呈现、视频动作迁移等场景展现出较强的可控生成能力。

## 环境准备

### 模型权重

MiniMax-H3: [下载模型权重](https://huggingface.co/MiniMaxAI/MiniMax-H3)

```bash
export MODEL=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL}"
```

下载完成后，权重目录结构应如下：

```text
MiniMax-H3/
├── FL2VA/          # T2VA/FL2VA 任务共用权重
├── Ref2VA/         # Ref2VA 任务权重
└── model_index.json
```

4 步 FlashGen 可使用在线 LoRA 加载，LoRA 权重见[FlashGen 4 步在线 LoRA](#flashgen-4-步在线-lora)。原生 C7 W4A4 路径使用离线合并的 transformer，见[SLA 与 QAT 推理](#sla-与-qat-推理)。

### 部署环境

#### 1）官方 Docker 镜像

您可以通过[vllm-omni镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)下载vllm-omni官方发布的docker镜像来进行部署。官方镜像中已安装好配套的vllm/vllm-ascend/vllm-omni。具体流程如下，以 Atlas 800I A3超节点服务器为例：

```bash
# 拉取Atlas 800I A3超节点服务器镜像，建议拉取最新版本的镜像，截至目前最新发布的版本是0.28.0
docker pull quay.io/ascend/vllm-omni:v0.28.0-a3

# 创建容器
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

Ascend 950 场景下，创建容器时需增加 `-v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo`，将宿主机的驱动拓扑目录挂载到容器内的相同路径。

#### 2）安装MindIE SD【可选】

官方镜像中如未安装mindiesd，可源码编译安装，参考如下指导：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

#### 3）安装最新的vllm-Omni【可选】

如果想体验最新的vllm-omni，参考如下指导：

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

#### 4）安装其他依赖【可选】

仅Ref2VA工作流需要安装如下依赖，对参考视频/音频的预处理依赖ffmpeg、decord。

```bash
apt update

# 安装ffmpeg
apt install -y ffmpeg

# 安装decord
apt install -y pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavfilter-dev libavdevice-dev
git clone --depth 1 --recursive https://github.com/dmlc/decord.git
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DDECODE_FFMPEG=1 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
pip3 install . --no-build-isolation
```

## 启动服务

T2VA/FL2VA任务共享同一份模型权重，Ref2VA的模型权重与之不同。在启动服务时通过传入**--task-type**参数指定任务类型来加载对应的模型权重。其他环境变量和传参配置均为T2VA/FL2VA/Ref2VA通用。如果不指定的话，默认情况下只会加载T2VA/FL2VA的模型权重。

启动 T2VA/FL2VA服务：

```bash
--task-type fl2va
```

启动 Ref2VA 服务：

```bash
--task-type ref2va
```

<!-- npu="A3,910b" id1 -->
### Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器

以下是Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器适配的 8 卡启动命令，均使用了 8 卡 USP、8 卡文本编码器 TP、逐层卸载、VAE `tile` 并行。
推荐无损优化配置章节只启动了无损优化（融合算子、高性能fa后端）。推荐有损优化配置章节在推荐无损优化配置的基础上叠加了Dit的稀疏fa和在线int8量化。另外还提供了其他可选的有损优化配置。

#### 推荐无损优化配置

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

#### 推荐有损优化配置

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

#### 其他可选性能优化配置

以下小节只列出相对 [推荐无损优化配置（Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器）](#推荐无损优化配置) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

##### 使能cache-dit

在启动命令末尾追加以下参数，启用 DiT block 缓存与 TaylorSeer 外推：

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```
<!-- end id1 -->

<!-- npu="950" id2 -->
### Ascend 950PR&950DT系列产品

以下是 Ascend 950PR&950DT系列产品适配的 4 卡启动命令，均使用了 4 卡 USP、4 卡文本编码器 TP、VAE `tile` 并行。
推荐无损优化配置章节只启动了无损优化（融合算子、高性能fa后端）。
推荐有损优化配置章节在推荐无损优化配置的基础上叠加了Dit的稀疏fa和在线mxfp8量化。另外还提供了其他可选的有损优化配置。
推荐无损优化配置使用了DLO权重逐层卸载策略来避免较大时长情况下出现的激活OOM问题，推荐有损优化配置在minimax-h3支持的最大规格下不存在OOM问题，因此为了性能考虑不开启DLO。

#### 推荐无损优化配置（Ascend 950PR&950DT系列产品）

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

#### 推荐有损优化配置（Ascend 950PR&950DT系列产品）

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

#### 其他可选性能优化配置（Ascend 950PR&950DT系列产品）

以下小节只列出相对 [推荐无损优化配置（Ascend 950PR&950DT系列产品）](#推荐无损优化配置ascend-950pr950dt系列产品) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

##### 使能稀疏注意力优化(eagleQBSA)

EQBSA 在块稀疏注意力基础上采用 Q/K INT8 与 V FP8 混合量化。将推荐配置中的 `--diffusion-attention-backend FLASH_ATTN` 替换为：

```bash
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":8,"end_step":13,"precision":"mix"}}}'
```

`precision` 默认值为 `bf16`，启用 EQBSA 必须显式设置为 `mix`。`start_step` 和 `end_step` 分别表示开头和结尾回退到 dense attention 的步数，并非步序号。对于 1344×768、50 步 T2VA，建议按照场景选择：

| 配置定位 | `start_step` | `end_step` | 评价 |
| --- | ---: | ---: | --- |
| 质量优先（推荐） | 8 | 13 | 通用及复杂运动场景，画质优先 |
| 性能均衡 | 0 | 13 | 均衡 |
| 速度优先 | 0 | 0 | 画面简单、低运动量场景，速度优先 |

出现画面不连续或细节不稳定时，应优先增加 `end_step`，再增加 `start_step`；不建议仅通过降低 `sparsity` 替代 dense 回退。以上配伍仅在 1344×768、50 步条件下完成验证，其它分辨率或采样步数需重新评估。

在推荐有损配置中采用了**start=8 / end=12**的回退策略，若想获取更高性能，还可尝试更少的回退step。

##### 使能cache-dit（Ascend 950PR&950DT系列产品）

在启动命令末尾追加以下参数，启用 DiT block 缓存与 TaylorSeer 外推：

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

##### 使能在线mxfp4量化

在启动命令末尾追加以下参数，使能Dit的在线mxfp4量化：

```bash
  --diffusion-quantization-config '{"transformer":{"method":"mxfp4"}}'
```

并且删除以下参数：

```bash
--enable-distributed-layerwise-offload
```
<!-- end id2 -->

## 发送请求

### T2VA 文生视频

T2VA/FL2VA服务启动后，可发起以下请求：

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

### FL2VA 参考首帧/末帧生视频

T2VA/FL2VA服务启动后，可发起以下请求：

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

FL2VA 通过 `short_edge` 控制短边长度，宽高比由 `aspect_ratio` 或首帧图像决定。

### Ref2VA 参考视频/音频生视频

Ref2VA服务启动后，可发起以下请求：

```bash
# 文本 + 参考视频生成音视频
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

### 请求参数说明

三个任务共用的请求参数如下：

| 参数 | 类型 | 必选 | 说明 |
| ------ | ------ | ------ | ------ |
| `prompt` | `str` | 是 | 文本提示词 |
| `width` / `height` | `int` | 与 `short_edge` 二选一 | 输出分辨率（像素），T2VA 示例采用此方式 |
| `short_edge` | `int` | 与 `width`/`height` 二选一 | 输出短边长度（像素），FL2VA / Ref2VA 示例采用此方式 |
| `aspect_ratio` | `str` | 否 | 输出宽高比（如 `16:9`），与 `short_edge` 搭配使用 |
| `fps` | `int` | 否 | 输出帧率 |
| `num_inference_steps` | `int` | 否 | 去噪推理步数；基座默认 50 步；FlashGen 在线 LoRA 须为 `4`，见[FlashGen 4 步在线 LoRA](#flashgen-4-步在线-lora) |
| `flow_shift` | `float` | 否 | 视频流偏移参数 |
| `audio_flow_shift` | `float` | 否 | 音频流偏移参数 |
| `seed` | `int` | 否 | 随机种子 |
| `extra_params.task` | `str` | 是 | 任务类型：`t2va` / `fl2va` / `ref2va` |
| `extra_params.duration` | `float` | 否 | 生成音视频时长（秒），最高 15 秒 |
| `extra_params.frame_indices` | `list[int]` | 否 | 仅用于 FL2VA，按上传顺序指定参考图像的位置。单图默认首帧 `[0]`，末帧用 `[-1]`，双图首尾帧用 `[0,-1]` |
| `input_references` | `file` | FL2VA、Ref2VA 示例必传 | FL2VA 传参考帧图像，Ref2VA 传参考图像、视频或音频。上传多个文件时，重复使用此字段 |
| `lora` | `str` (JSON) | 否 | 在线 LoRA 配置；FlashGen 4 步须传 `name` / `path` / `scale`，见[FlashGen 4 步在线 LoRA](#flashgen-4-步在线-lora) |

## FlashGen 4 步在线 LoRA

FlashGen 4 步权重为 native-layout LoRA 单文件（`key_format=minimax-h3-native`），通过 vLLM-Omni **运行时加载**，无需合并到基座。

FL2VA 和 Ref2VA（R2VA）需要包含 [vLLM-Omni #10596](https://github.com/vllm-project/vllm-omni/pull/10596) 中 FlashGen 适配代码的版本。T2VA 和 FL2VA 共用基座。每个场景需加载该场景专用的 LoRA 文件。

| 场景 | 基座目录（相对于 `MODEL_ROOT`） | 启动参数 `--task-type` | 请求中的 `extra_params.task` | 示例参考输入 |
| ------ | ------ | ------ | ------ | ------ |
| T2VA 文生音视频 | `FL2VA/` | `fl2va` | `t2va` | 无 |
| FL2VA 首帧生成音视频 | `FL2VA/` | `fl2va` | `fl2va` | 一张首帧图像 |
| Ref2VA（R2VA）参考生成音视频 | `Ref2VA/` | `ref2va` | `ref2va` | 一张参考图像 |

### 准备 LoRA

T2VA 权重可从 ModelScope 的 [FlashGen/Minimax-H3-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen) 下载。

```bash
pip install modelscope
export FLASHGEN_DIR=/path/to/minimax-h3-flashgen-lora
export FLASHGEN_FILE=minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

FL2VA 权重从独立的 [FlashGen/Minimax-H3-FL2VA-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-FL2VA-4step-lora-flashgen) 仓库下载：

```bash
export FLASHGEN_FILE=minimax_h3_fl2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-FL2VA-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

Ref2VA 权重从独立的 [FlashGen/Minimax-H3-Re2VA-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-Re2VA-4step-lora-flashgen) 仓库下载。注意仓库名使用 `Re2VA`，权重文件名使用 `ref2va`：

```bash
export FLASHGEN_FILE=minimax_h3_ref2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-Re2VA-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

按场景选择对应的下载命令，并将 `FLASHGEN_DIR` 设为服务容器内可访问的绝对路径。下方启动示例使用该目录中的场景专用 LoRA 文件。

在以下启动命令中，将 `MODEL_ROOT` 设为[模型权重](#模型权重)小节下载的 MiniMax-H3 根目录。

### 启动 T2VA 服务

相对 [推荐有损优化配置（Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器）](#推荐有损优化配置)，在 `--trust-remote-code` 之后追加 LoRA 参数，并将该配置中的

```bash
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

替换成

```bash
  --diffusion-attention-backend FLASH_ATTN
```

追加 LoRA 参数：

```bash
  --lora-backend peft \
  --lora-path "${FLASHGEN_LORA}" \
```

其余参数（USP8、DLO、VAE tile 等）与推荐配置相同。

Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器完整示例：

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
export MODEL_ROOT=/path/to/MiniMax-H3
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

说明：

- `--lora-backend peft` 启用运行时 LoRA manager；native 布局由 LoRA 文件 metadata 自动识别，无需单独 backend。
- 不支持 `--enable-cpu-offload` 或 `--enable-layerwise-offload`（普通逐层卸载）。

### 启动 FL2VA 或 Ref2VA 服务

以 [Ascend 950PR 四卡推荐无损优化配置](#推荐无损优化配置ascend-950pr950dt系列产品) 为基础。先按场景设置以下变量，再将原启动命令中的 `--task-type fl2va` 替换为 `--task-type "${TASK_TYPE}"`，并在命令末尾追加 `--lora-backend peft --lora-path "${FLASHGEN_LORA}"`。

下方 768p、5 秒的四卡预检使用 `MINDIE_SD_FA_TYPE=fused_attn_score`，并省略 `--enable-distributed-layerwise-offload` 和 `--enable-diffusion-pipeline-profiler`。生成更长的视频时，保留推荐配置中的 DLO，以降低激活内存不足的风险。

```bash
export MODEL_ROOT=/path/to/MiniMax-H3

# FL2VA
export MODEL="${MODEL_ROOT}/FL2VA"
export TASK_TYPE=fl2va
export FLASHGEN_LORA="${FLASHGEN_DIR}/minimax_h3_fl2va_flashgen_4step_v1.0_768p_bf16.safetensors"

# 启动 Ref2VA 时，使用下面三行替换 FL2VA 配置
# export MODEL="${MODEL_ROOT}/Ref2VA"
# export TASK_TYPE=ref2va
# export FLASHGEN_LORA="${FLASHGEN_DIR}/minimax_h3_ref2va_flashgen_4step_v1.0_768p_bf16.safetensors"
```

### 发送 4 步 T2VA 请求

在 [T2VA 文生视频](#t2va-文生视频) 基础上，`num_inference_steps` 改为 `4`，并传入 `lora` 字段：

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

### 发送 4 步 FL2VA 请求

上传首帧图像，使用 `extra_params.frame_indices=[0]` 指定首帧位置。`FLASHGEN_LORA` 使用 FL2VA 服务启动时的路径。

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT="镜头缓慢向前推进，保留首帧构图，背景有轻微风声。"
export FIRST_FRAME=/path/to/first_frame.png

curl --fail-with-body -sS "${API_URL}" \
  --form-string "prompt=${PROMPT}" \
  -F 'short_edge=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"fl2va","duration":5.0,"frame_indices":[0]}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  --form-string "lora={\"name\":\"h3-flashgen-fl2va\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}" \
  -o fl2va_4step.mp4
```

### 发送 4 步 Ref2VA（R2VA）请求

上传一张参考图像。`FLASHGEN_LORA` 使用 Ref2VA 服务启动时的路径，请求任务名为 `ref2va`。

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
export PROMPT="以参考图像中的主体为中心，生成环绕镜头，并加入轻微的环境声。"
export REF_IMAGE=/path/to/reference.png

curl --fail-with-body -sS "${API_URL}" \
  --form-string "prompt=${PROMPT}" \
  -F 'short_edge=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=4' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"ref2va","duration":5.0}' \
  -F "input_references=@${REF_IMAGE};type=image/png" \
  --form-string "lora={\"name\":\"h3-flashgen-ref2va\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}" \
  -o ref2va_4step.mp4
```

### 三个场景的请求参数

| 参数 | 取值 | 说明 |
| ------ | ------ | ------ |
| `extra_params.task` | `t2va` / `fl2va` / `ref2va` | 必须与请求加载的 LoRA 对应 |
| `num_inference_steps` | `4` | 调度包含 5 个边界点，对应 4 次去噪 |
| `lora.scale` | `1.0` | 按原始强度应用 LoRA |
| `flow_shift` / `audio_flow_shift` | 可省略 | 音视频调度由 LoRA 元数据提供 |

### T2VA RTF<1

在 8 卡 Ascend 950PR 上验证 FlashGen 4 步 T2VA 的 `RTF < 1`。使用原始浮点 `FL2VA/` 权重、提供 `mindiesd.quant_attention` 的 MindIE-SD。

#### 启动低时延 T2VA 服务

将 `MODEL_ROOT` 设为 MiniMax-H3 根目录，并按前文准备 T2VA 专用的 `FLASHGEN_LORA`。此配置让 DiT 和 VAE 权重常驻 NPU，RainFusion 从第 0 步启用。

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True HCCL_NPU_SOCKET_PORT_RANGE=auto
unset VLLM_OMNI_MINIMAX_H3_DISABLED_AUDIO_TCONV_CONV1D

vllm serve "${MODEL_ROOT}/FL2VA" --omni --host 0.0.0.0 --port "${PORT}" \
  --trust-remote-code --task-type fl2va \
  --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 8 \
  --lora-backend peft --lora-application startup-merge \
  --lora-path "${FLASHGEN_LORA}" --lora-scale 1.0 \
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":0}}}' \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"},"video_vae":{"method":"mxfp8"}}' \
  --additional-config '{"video_vae_fused_ops":true}'
```

#### 发送 4 步 T2VA 请求并测量 RTF

LoRA 已在服务启动时固定，请求省略 `lora` 字段。新增流式掩盖配置`preencode_mp4`。服务就绪后先发送一次预热，再用相同请求测量。端到端 RTF 为 HTTP 请求总耗时除以生成的 MP4 实际时长。

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
  -F 'extra_params={"task":"t2va","duration":15.0,"preencode_mp4":true}' \
  -o "t2va_4step_rtf1.mp4"
```

## SLA 与 QAT 推理

本节面向 Ascend 950（A5），介绍四步模型、原生 C7 W4A4 与 FP8 SLA 的组合推理。SLA 与 QAT 可分别启用：SLA 选择注意力后端，QAT 配置线性层的权重和激活量化。

### 环境要求

- vLLM-Omni 需包含 [SLA 接入 #11089](https://github.com/vllm-project/vllm-omni/pull/11089) 与 [QAT/C7 接入 #11137](https://github.com/vllm-project/vllm-omni/pull/11137)。仅设置量化参数或安装 MindIE SD 不会自动补齐框架侧接入。
- MindIE SD 需提供 `mindiesd.layers.SparseLinearAttention`；使用 FP8 SLA 时，还需支持 FP8 `BlockSparseAttention` 及其反量化 scale 参数，并安装配套 triton-ascend。
- 原生 C7 使用支持 FP4 `npu_quant_matmul` 的 torch-npu/CANN。每个量化线性层的本地 K 维必须能被 64 整除。
- 示例使用八卡、USP8、Ring1、文本编码器 TP8 和 VAE tile 并行8，不开启 DLO、CPU offload 或普通逐层 offload。

固定版本的 A5 基础镜像可从 Quay 拉取：

```bash
docker pull --platform linux/arm64 \
  quay.io/ascend/vllm-omni:nightly-20261125-574fb73-a5
```

该固定版本基础镜像尚不包含上述 SLA 与 QAT/C7 PR。使用前需引入 [vLLM-Omni #11089](https://github.com/vllm-project/vllm-omni/pull/11089) 和 [#11137](https://github.com/vllm-project/vllm-omni/pull/11137)，并安装配套 MindIE SD。

### 准备离线合并的 transformer

先使用权重交付方提供的合并工具准备完整的 BF16 transformer：

1. 选择任务对应的官方分区：T2VA/FL2VA 使用 `FL2VA`，Ref2VA 使用 `Ref2VA`。
2. 合并该任务的四步 LoRA；若基座已包含该 LoRA，则跳过，避免重复合并。
3. 若交付 QAT LoRA，将其按交付的 rank、alpha 和 merge scale 合并到匹配的四步基座。若交付完整 QAT transformer，则直接使用交付产物。
4. SLA 的 `proj_l` 参数随最终 transformer 一起打包，并更新 safetensors 索引。若 SLA 训练同时修改了主干，最终主干的选择或合并方式应由权重交付方明确，不能只复制 `proj_l` 后视为完整 SLA 模型。

LoRA 合并与量化是两步操作。BF16 合并产物没有预先打包的 FP4 张量或离线 scale；原生 C7 在加载时编码权重，在每次线性层调用时编码激活，再执行 FP4 矩阵乘。`mxfp4_fake` 则是浮点 QDQ 参考路径，不能用来替代原生 FP4 执行。

产物目录应为：

```text
h3-fl2va-merged/
├── model_index.json
└── transformer/
    ├── config.json
    ├── model.safetensors.index.json
    └── model-*.safetensors
```

`transformer/config.json` 的架构需与官方分区一致。父目录 `model_index.json` 的 `_minimax_h3` 需保留交付模型的 `partition`、`tasks`、`base_schedule` 和 `sigma_shift_scales`。例如，下面只适用于明确交付该调度的 FL2VA 四步模型：

```json
{
  "_minimax_h3": {
    "partition": "fl2va",
    "tasks": ["fl2va"],
    "base_schedule": [1.0, 0.7, 0.4, 0.15, 0.0],
    "sigma_shift_scales": {"video": 12.0, "audio": 3.0}
  }
}
```

请保留原 `model_index.json` 的其余字段，不要用上面的片段覆盖整个文件。T2VA 的任务声明应为 `t2va`；Ref2VA 的分区和任务均为 `ref2va`。仅把请求的 `num_inference_steps` 改成 `4`，不能替代四步权重及调度元数据。

通过 `transformer_checkpoint` 指定此目录后，文本编码器、tokenizer 和音视频 VAE 仍从官方模型加载。该路径不接受启动时或请求时 LoRA：应删除服务的 `--lora-*` 参数，并且不在请求中发送 `lora` 字段。

### 配置 SLA

在运行推理服务的容器内设置下列变量：

```bash
export ATTENTION_CONFIG='{
  "default": {"backend": "FLASH_ATTN"},
  "per_role": {
    "self": {
      "backend": "SLA_ATTN",
      "sla": {
        "topk": 0.05,
        "blkq": 64,
        "blkk": 64,
        "feature_map": "softmax",
        "precision": "fp8",
        "zero_init": false
      }
    },
    "minimax_h3.token_refiner": {"backend": "FLASH_ATTN"}
  }
}'
```

主干 self-attention 使用 SLA，token refiner 保留普通注意力。不要把全局默认后端直接设为 `SLA_ATTN`。`precision=fp8` 表示稀疏分支 Q/K/V 使用块缩放 E4M3；输入、输出及线性修正仍为 BF16。将其改为 `bf16` 可使用 BF16 稀疏分支，默认值也是 `bf16`。

每个主干 block 需要以下参数，参数名是 checkpoint 中的 native 名称：

```text
blocks.<层号>.attn.sla_module.proj_l.weight
blocks.<层号>.attn.sla_module.proj_l.bias
```

正式 SLA 权重使用 `zero_init=false`，缺少参数应报错。仅检查推理链路时，可以使用不含 `proj_l` 的官方基座及四步合并权重，并将 `zero_init` 改为 `true`；缺失的投影会置零，已有投影仍正常加载。

`topk=0.05` 表示稀疏分支名义上保留 5% 的 KV 块，不等于测得整个模型达到 95% 稀疏率；线性修正分支仍执行。特征映射和块大小应与权重交付的推理配置一致。本路径使用 `ring=1`，不支持 Ring/AllGather-KV、任意 attention mask 或 reference-KV 复用等组合。

### 配置原生 C7 W4A4

将路径替换为容器内可访问的模型与合并产物：

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="${MODEL_ROOT}/FL2VA"
export TASK_TYPE=fl2va
export TRANSFORMER_DIR=/path/to/h3-fl2va-merged/transformer
```

下面的示例仅量化主干 attention 和 MLP 线性层，保留输入输出、时间嵌入、AdaLN 与 token refiner 为 BF16。如果 checkpoint 定义了 `ignored_layers`，直接使用交付清单，运行时清单必须与其一致；没有清单时，使用以下显式示例策略，并由交付方确认是否适用于目标 QAT 权重：

```bash
export QUANTIZATION_CONFIG="$(python3 - <<'PY'
import json
import os
from pathlib import Path

config = json.loads((Path(os.environ["TRANSFORMER_DIR"]) / "config.json").read_text())
disk_quant = config.get("quantization_config") or {}
ignored = disk_quant.get("ignored_layers") if isinstance(disk_quant, dict) else None
if ignored is None:
    ignored = [
        "time_embedder.proj_in", "time_embedder.proj_out",
        "video_patch_proj", "audio_patch_proj", "condition_proj",
        "final_layer.video_out", "final_layer.audio_out",
        "final_layer.adaln_proj.linear",
    ]
    ignored += [f"blocks.{i}.adaln_proj.linear" for i in range(config.get("num_layers", 50))]
    refiner_layers = config.get("token_refiner_num_layers", config.get("num_refiner_layers", 2))
    ignored += [
        f"token_refiner.blocks.{i}.{name}"
        for i in range(refiner_layers)
        for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    ]
print(json.dumps({
    "transformer": {
        "method": "mxfp4",
        "mxfp4_recipe": "mxfp4_c7",
        "ignored_layers": ignored,
    },
    "default": None,
}))
PY
)"
```

`--diffusion-quantization-config` 的 CLI 配置同时接受 `method` 和 `quant_method`，两者是合法别名；本文统一使用 `method`，与前文示例保持一致。此约定仅针对 CLI 示例，不要求修改 checkpoint 中的 `quantization_config.quant_method` 字段。

`mxfp4_recipe=mxfp4_c7` 是显式选择训练匹配的 C7 规则，不能只写 `method=mxfp4` 就认为等价。该原生路径要求浮点合并权重，不接受预序列化低比特权重或 W4A8 fallback。SLA 的 `proj_l` 保持浮点，不参与线性层 W4A4 量化。

### 启动组合推理服务

在容器内、设置完上述变量后执行。此示例不修改镜像源码，不启用运行时 LoRA 或 DLO：

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_ALGO=mesh

python -m vllm_omni.entrypoints.cli.main serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type "${TASK_TYPE}" \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-config "${ATTENTION_CONFIG}" \
  --diffusion-quantization-config "${QUANTIZATION_CONFIG}" \
  --additional-config "{\"transformer_checkpoint\":\"${TRANSFORMER_DIR}\"}" \
  --init-timeout 1800 \
  --stage-init-timeout 1800
```

如需单独启用一个特性，在启动前替换对应变量：

| 模式 | `ATTENTION_CONFIG` | `QUANTIZATION_CONFIG` |
| --- | --- | --- |
| SLA，不量化线性层 | 保留 SLA 配置 | `{"default":null}`，并使用浮点 checkpoint |
| 原生 C7 W4A4，不使用 SLA | `{"default":{"backend":"FLASH_ATTN"}}`，checkpoint 不含 SLA 投影 | 保留 C7 配置 |
| 原生 W4A4 + FP8 SLA | 保留 `precision=fp8` 的 SLA 配置 | 保留 C7 配置 |

Ref2VA 启动时，将 `MODEL` 改为 `${MODEL_ROOT}/Ref2VA`、`TASK_TYPE` 改为 `ref2va`，并使用对应 Ref2VA 合并产物。T2VA 仍使用 `FL2VA` 分区和 `TASK_TYPE=fl2va`，但 transformer 及其元数据应对应 T2VA 四步权重。

### 发送四步请求与检查结果

服务前台保持运行，在另一个容器终端设置相同的 `PORT`，先检查服务状态：

```bash
export PORT=9098
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
curl --fail -sS "http://127.0.0.1:${PORT}/health"
```

FL2VA 示例上传首帧图像，不发送 `lora` 字段：

```bash
export FIRST_FRAME=/path/to/first_frame.png
curl --fail-with-body -sS "${API_URL}" \
  --form-string 'prompt=镜头缓慢向前推进，保留首帧构图，背景有轻微风声。' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' \
  -F 'fps=24' -F 'num_inference_steps=4' -F 'seed=42' \
  -F 'flow_shift=12.0' -F 'audio_flow_shift=3.0' \
  -F 'extra_params={"task":"fl2va","duration":4.0,"frame_indices":[0],"preencode_mp4":true}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -o fl2va_w4a4_fp8_sla.mp4
```

T2VA 使用该任务的合并产物启动服务后，删除 `input_references`，将 `extra_params` 改为：

```json
{"task":"t2va","duration":4.0,"preencode_mp4":true}
```

Ref2VA 使用其独立服务，上传参考图像，并将 `extra_params` 改为：

```json
{"task":"ref2va","duration":4.0,"preencode_mp4":true}
```

示例的音视频 shift 需与目标四步权重的元数据一致。确认请求 HTTP 200 后，使用已安装的 FFmpeg 完整解码输出：

```bash
ffprobe -v error -show_entries stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate,channels \
  -of json fl2va_w4a4_fp8_sla.mp4
ffmpeg -v error -xerror -i fl2va_w4a4_fp8_sla.mp4 \
  -map 0:v:0 -map 0:a:0 -f null -
```

服务日志应出现 `Executed MindIE-SD SparseLinearAttention` 和 `Executed native C7 MXFP4 W4A4 npu_quant_matmul`。这两条日志在首次成功执行对应路径后打印；仅有 `mxfp4` 配置字符串或服务就绪信息不能证明两个后端都已执行。测试结束后，在服务前台终端按 `Ctrl+C` 停止服务。

本节的 W4A4F8 指线性层 W4A4 与 SLA 稀疏分支 FP8 的组合，线性修正及其他保留模块仍使用 BF16。`proj=0`、普通四步 LoRA 或合成算子用例可以检查功能链路，但不能替代正式 QAT/SLA 权重的效果验证。

## 当前已适配的优化点

1. 昇腾亲和的注意力算子：MindIE SD的ascend_laser_attention算子，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
2. 昇腾亲和的融合算子：RMSNorm、AddRMSNorm、GQA、SwiGLU、rotary_position_embedding，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
3. 稀疏fa算子，详情见 [sparse.md](../../docs/zh/features/sparse.md)
4. 在线量化：int8、mxfp8、mxfp4
5. USP\TP\Tile 并行
6. 后训练：少步蒸馏

## Benchmark

<!-- npu="910b" id3 -->
### Atlas 800I A2推理服务器

| 任务类型 | 输出分辨率 | 输出帧数 | 输出时长（s） | 优化配置 | E2E总时延(s) | Dit总时延(s) | Dit单步时延(s) | Dit实际执行步数 | 文本编码时延(s) | 参考视频编码时延(s) | 参考音频编码时延(s) | 解码时延(s) | 参考视频预处理(s) | 后处理(s) | CPU MP4(s) | 常驻权重显存占用（GB） | 推理过程峰值显存占用（GB） |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | 无损 | 150.06 | 139.9 | 2.85 | 49 | 0.83 | NA | NA | 6.76 | NA | 0.462 | 1.18 | 2.97 | 24.7 |
| t2va | 1344×768 | 362 | 15 | 无损 | 786.239 | 761.357 | 15.53 | 49 | 1.196 | NA | NA | 15.87 | NA | 1.977 | 2.95 | 2.97 | 36.21 |
| t2va | 1344×768 | 124 | 5 | 有损 | 108.1 | 95.85 | 1.95 | 49 | 0.9 | NA | NA | 6.73 | NA | 0.745 | 1.04 | 2.55 | 22.47 |
| t2va | 1344×768 | 362 | 15 | 有损 | 552.51 | 526.041 | 10.73 | 49 | 1.28 | NA | NA | 16.07 | NA | 1.903 | 3.76 | 2.55 | 35.11 |
| ref2va | 1344×768 | 124 | 5 | 无损 | 474.89 | 421.678 | 8.6 | 49 | 2.42 | 13.31 | 0.38 | 6.469 | 3.94 | 0.489 | 1.157 | 3.4 | 23.45 |
| ref2va | 1344×768 | 362 | 15 | 无损 | 3414.184 | 3336.981 | 68.08 | 49 | 6.48 | 30.85 | 0.59 | 15.8 | 12.59 | 1.988 | 3.08 | 3.4 | 36.28 |
| ref2va | 1344×768 | 124 | 5 | 有损 | 380.915 | 326.495 | 6.66 | 49 | 2.47 | 13.48 | 0.38 | 6.36 | 4.68 | 0.591 | 1.217 | 2.55 | 22.57 |
| ref2va | 1344×768 | 362 | 15 | 有损 | 2332.041 | 2249.63 | 45.89 | 49 | 6.33 | 35.1 | 0.59 | 15.72 | 11.07 | 1.345 | 2.3 | 2.55 | 35.58 |
<!-- end id3 -->

<!-- npu="A3" id4 -->
### Atlas 800I A3超节点服务器

| 任务类型 | 输出分辨率 | 输出帧数 | 输出时长（s） | 优化配置 | E2E总时延(s) | Dit总时延(s) | Dit单步时延(s) | Dit实际执行步数 | 文本编码时延(s) | 参考视频编码时延(s) | 参考音频编码时延(s) | 解码时延(s) | 参考视频预处理(s) | 后处理(s) | CPU MP4(s) | 常驻权重显存占用（GB） | 推理过程峰值显存占用（GB） |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | 无损 | 114.187 | 105.871 | 2.16 | 49 | 0.483 | NA | NA | 6.56 | NA | 0.001 | 0.461 | 3.49 | 24.7 |
| t2va | 1344×768 | 362 | 15 | 无损 | 599.29 | 580.383 | 11.84 | 49 | 1.826 | NA | NA | 14.95 | NA | 0.001 | 1.021 | 3.49 | 36.21 |
| t2va | 1344×768 | 124 | 5 | 有损 | 84.824 | 77.473 | 1.58 | 49 | 0.566 | NA | NA | 6.23 | NA | 0.001 | 0.608 | 2.63 | 22.47 |
| t2va | 1344×768 | 362 | 15 | 有损 | 401.434 | 383.559 | 7.82 | 49 | 1.716 | NA | NA | 14.83 | NA | 0.001 | 1.14 | 2.63 | 35.11 |
| ref2va | 1344×768 | 124 | 5 | 无损 | 355.534 | 311.859 | 6.36 | 49 | 1.888 | 12.367 | 0.244 | 5.55 | 2.493 | 0.001 | 0.433 | 3.49 | 23.45 |
| ref2va | 1344×768 | 362 | 15 | 无损 | 2339.556 | 2277.606 | 46.47 | 49 | 5.613 | 34.46 | 0.3635 | 13.607 | 6.833 | 0.001 | 1.114 | 3.49 | 36.28 |
| ref2va | 1344×768 | 124 | 5 | 有损 | 306.816 | 247.716 | 5.05 | 49 | 1.877 | 12.712 | 0.24 | 5.237 | 2.447 | 0.001 | 0.535 | 2.63 | 22.57 |
| ref2va | 1344×768 | 362 | 15 | 有损 | 1695.256 | 1634.584 | 33.35 | 49 | 5.568 | 33.269 | 0.3564 | 13.51 | 6.96 | 0.001 | 1.221 | 2.63 | 35.58 |
<!-- end id4 -->

<!-- npu="950" id5 -->
### Ascend 950PR&950DT系列产品性能数据

| 任务类型 | 输出分辨率 | 输出帧数 | 输出时长（s） | 优化配置 | E2E总时延(s) | Dit总时延(s) | Dit单步时延(s) | Dit实际执行步数 | 文本编码时延(s) | 参考视频编码时延(s) | 参考音频编码时延(s) | 解码时延(s) | 参考视频预处理(s) | 后处理(s) | CPU MP4(s) | 常驻权重显存占用（GB） | 推理过程峰值显存占用（GB） |
| -------- | -------- | ---- | --------- | ------------ | ------------ | ------------ | -------------- | -------------- | ------------- | ------------- | ----------- | ------------------------ | ------------ | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| t2va | 1344×768 | 124 | 5 | 无损 | 203.949 | 198.096 | 4.04 | 49 | 0.574 | NA | NA | 5.05 | NA | 0.001 | 0.225 | 3.97 | 22.2 |
| t2va | 1344×768 | 362 | 15 | 无损 | 1489.165 | 1474.353 | 30.08 | 49 | 0.69 | NA | NA | 13.449 | NA | 0.001 | 0.815 | 3.97 | 32.14 |
| t2va | 1344×768 | 124 | 5 | 有损 | 113.857 | 109.809 | 2.24 | 49 | 0.028 | NA | NA | 3.82 | NA | 0.001 | 0.214 | 56.99 | 64.38 |
| t2va | 1344×768 | 362 | 15 | 有损 | 762.169 | 749.397 | 15.29 | 49 | 0.028 | NA | NA | 12.19 | NA | 0.001 | 0.606 | 56.99 | 74.26 |
| ref2va | 1344×768 | 124 | 5 | 无损 | 745.427 | 730.219 | 14.9 | 49 | 1.159 | 8.148 | 0.134 | 5.119 | 0.408 | 0.001 | 0.278 | 3.97 | 23.82 |
| ref2va | 1344×768 | 362 | 15 | 无损 | 5887.24 | 5847.567 | 119.33 | 49 | 2.54 | 22.066 | 0.2262 | 13.34 | 0.808 | 0.001 | 0.702 | 3.97 | 32.53 |
| ref2va | 1344×768 | 124 | 5 | 有损 | 400.404 | 387.815 | 7.91 | 49 | 0.951 | 7.048 | 0.085 | 3.91 | 0.405 | 0.001 | 0.21 | 56.6 | 66.03 |
| ref2va | 1344×768 | 362 | 15 | 有损 | 3037.842 | 3001.314 | 61.24 | 49 | 2.278 | 20.64 | 0.145 | 12.1 | 0.806 | 0.001 | 0.695 | 56.6 | 74.76 |
<!-- end id5 -->

以上表格中的测试数据，无损优化均采用对应机型的推荐无损优化配置、有损优化均采用对应机型的推荐有损优化配置，并在vllm-omni 0.28.0 版本下测试得出。

### Atlas 650服务器（单机8卡 Ascend 950PR）

vllm-omni版本:v0.32.0-a5。

部署采用USP8、Ring1、文本编码TP8、VAE tile并行8，关闭Cache-DiT，不使用LoRA。测试请求采用24 FPS、50个去噪步骤，覆盖T2VA、Ref2VA在标称5秒和15秒下的三种配置，每项预热后执行正式测试：

- 无损：DLO与FLASH_ATTN。
- 有损（BF16）：MXFP8在线权重量化与RainFusion BF16稀疏注意力，不开启DLO。
- 有损（mix）：MXFP8在线权重量化与RainFusion mix混合精度稀疏注意力，不开启DLO。

两种有损配置均采用`sparsity=0.8`、`start_step=8`、`end_step=12`。

| 任务类型 | 输出分辨率 | 输出帧数 | 输出时长(s) | 优化配置 | E2E总时延(s) | DiT总时延(s) | DiT单步时延(s) | 实际步数 | 文本编码(s) | 参考音视频编码总时延(s) | 解码(s) | 参考视频预处理(s) | 后处理(s) | CPU MP4(s) | worker常驻显存/卡（GB） | 峰值显存（GB） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| t2va | 1344×768 | 124 | 5.17 | 无损 | 89.389 | 84.331 | 1.687 | 50 | 0.329 | NA | 3.207 | NA | 0.000 | 0.888 | 3.22 | 21.726 |
| t2va | 1344×768 | 124 | 5.17 | 有损（BF16） | 57.934 | 53.627 | 1.073 | 50 | 0.078 | NA | 2.636 | NA | 0.000 | 1.153 | 51.20 | 61.967 |
| t2va | 1344×768 | 124 | 5.17 | 有损（mix） | 57.025 | 52.907 | 1.058 | 50 | 0.079 | NA | 2.646 | NA | 0.000 | 0.954 | 51.20 | 61.994 |
| t2va | 1344×768 | 362 | 15.08 | 无损 | 464.657 | 452.738 | 9.055 | 50 | 0.354 | NA | 8.776 | NA | 0.000 | 1.532 | 3.22 | 23.673 |
| t2va | 1344×768 | 362 | 15.08 | 有损（BF16） | 276.564 | 265.398 | 5.308 | 50 | 0.072 | NA | 8.127 | NA | 0.000 | 1.968 | 51.20 | 63.890 |
| t2va | 1344×768 | 362 | 15.08 | 有损（mix） | 265.239 | 248.834 | 4.977 | 50 | 0.077 | NA | 8.143 | NA | 0.000 | 1.930 | 51.20 | 63.890 |
| ref2va | 1344×768 | 124 | 5.17 | 无损 | 257.325 | 242.801 | 4.856 | 50 | 0.789 | 4.646 | 3.304 | — | 0.001 | 0.836 | 3.22 | 21.890 |
| ref2va | 1344×768 | 124 | 5.17 | 有损（BF16） | 170.461 | 157.469 | 3.149 | 50 | 0.645 | 4.153 | 2.652 | — | 0.000 | 0.657 | 50.93 | 62.095 |
| ref2va | 1344×768 | 124 | 5.17 | 有损（mix） | 159.724 | 146.356 | 2.927 | 50 | 0.635 | 4.147 | 2.663 | — | 0.000 | 1.068 | 50.93 | 62.153 |
| ref2va | 1344×768 | 362 | 15.08 | 无损 | 1757.931 | 1719.482 | 34.390 | 50 | 2.517 | 12.730 | 8.732 | — | 0.003 | 1.572 | 3.22 | 23.926 |
| ref2va | 1344×768 | 362 | 15.08 | 有损（BF16） | 1091.935 | 1054.333 | 21.087 | 50 | 2.237 | 12.388 | 8.117 | — | 0.001 | 2.358 | 50.93 | 65.565 |
| ref2va | 1344×768 | 362 | 15.08 | 有损（mix） | 956.927 | 919.068 | 18.381 | 50 | 2.384 | 12.364 | 8.158 | — | 0.000 | 2.429 | 50.93 | 65.691 |

## 后续计划

针对Minimax-H3的性能优化近期持续更新中，敬请关注。
