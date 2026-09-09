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

4 步 FlashGen 推理使用在线 LoRA 加载，无需合并权重；LoRA 权重见[FlashGen 4 步在线 LoRA（T2VA）](#flashgen-4-步在线-lorat2va)。

### 部署环境

#### 1）官方 Docker 镜像

您可以通过[vllm-omni镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)下载vllm-omni官方发布的docker镜像来进行部署。官方镜像中已安装好配套的vllm/vllm-ascend/vllm-omni。具体流程如下，以 Atlas 800I A3 为例：

```bash
# 拉取Atlas 800I A3镜像，建议拉取最新版本的镜像，截至目前最新发布的版本是0.28.0
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

#### 2）安装MindIE-SD【可选】

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

### Atlas 800I A2 / Atlas 800I A3

以下是Atlas 800I A2 / Atlas 800I A3 适配的 8 卡启动命令，均使用了 8 卡 USP、8 卡文本编码器 TP、逐层卸载、VAE `tile` 并行。
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

以下小节只列出相对 [推荐无损优化配置（Atlas 800I A2 / A3）](#推荐无损优化配置) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

##### 使能cache-dit

在启动命令末尾追加以下参数，启用 DiT block 缓存与 TaylorSeer 外推：

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

### Ascend 950PR / Ascend 950DT

以下是Ascend 950PR / Ascend 950DT 适配的 4 卡启动命令，均使用了 4 卡 USP、4 卡文本编码器 TP、VAE `tile` 并行。
推荐无损优化配置章节只启动了无损优化（融合算子、高性能fa后端）。
推荐有损优化配置章节在推荐无损优化配置的基础上叠加了Dit的稀疏fa和在线mxfp8量化。另外还提供了其他可选的有损优化配置。
推荐无损优化配置使用了DLO权重逐层卸载策略来避免较大时长情况下出现的激活OOM问题，推荐有损优化配置在minimax-h3支持的最大规格下不存在OOM问题，因此为了性能考虑不开启DLO。

#### 推荐无损优化配置（Ascend 950PR / 950DT）

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

#### 推荐有损优化配置（Ascend 950PR / 950DT）

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

#### 其他可选性能优化配置（Ascend 950PR / 950DT）

以下小节只列出相对 [推荐无损优化配置（Ascend 950PR / 950DT）](#推荐无损优化配置ascend-950pr--950dt) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

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

##### 使能cache-dit（Ascend 950PR / 950DT）

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
| `num_inference_steps` | `int` | 否 | 去噪推理步数；基座默认 50 步；FlashGen 在线 LoRA 须为 `4`，见[FlashGen 4 步在线 LoRA（T2VA）](#flashgen-4-步在线-lorat2va) |
| `flow_shift` | `float` | 否 | 视频流偏移参数 |
| `audio_flow_shift` | `float` | 否 | 音频流偏移参数 |
| `seed` | `int` | 否 | 随机种子 |
| `extra_params.task` | `str` | 是 | 任务类型：`t2va` / `fl2va` / `ref2va` |
| `extra_params.duration` | `float` | 否 | 生成音视频时长（秒），最高 15 秒 |
| `input_references` | `file` | 否 | 参考输入文件：FL2VA 传首帧图像（`image/png` 等），Ref2VA 传参考视频/音频（`video/mp4` 等） |
| `lora` | `str` (JSON) | 否 | 在线 LoRA 配置；FlashGen 4 步须传 `name` / `path` / `scale`，见[FlashGen 4 步在线 LoRA（T2VA）](#flashgen-4-步在线-lorat2va) |

## FlashGen 4 步在线 LoRA（T2VA）

FlashGen 4 步权重为 native-layout LoRA 单文件（`key_format=minimax-h3-native`），通过 vLLM-Omni **运行时加载**，无需 merge 进基座。

权重发布于 ModelScope：[FlashGen/Minimax-H3-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen)

### 下载 LoRA

```bash
pip install modelscope
export FLASHGEN_DIR=/path/to/minimax-h3-flashgen-lora
export FLASHGEN_FILE=minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

### 启动服务（在推荐配置基础上追加）

相对 [推荐有损优化配置（Atlas 800I A2 / A3）](#推荐有损优化配置)，在 `--trust-remote-code` 之后追加 LoRA 参数，并将该配置中的

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

Atlas 800I A2 / A3 完整示例：

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

说明：

- `--lora-backend peft` 启用运行时 LoRA manager；native 布局由 LoRA 文件 metadata 自动识别，无需单独 backend。
- 仅支持 **T2VA**；不支持 FL2VA / Ref2VA。
- 不支持 `--enable-cpu-offload` 或 `--enable-layerwise-offload`（普通逐层卸载）。

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

参数要点：

| 参数 | 取值 | 说明 |
| ------ | ------ | ------ |
| `extra_params.task` | `t2va` | native LoRA 仅支持 T2VA |
| `num_inference_steps` | `4` | interval 数（4 次 denoise），不是 5 |
| `lora.scale` | `1.0` | 激活 LoRA；`0.0` 可对比基座输出 |
| `flow_shift` / `audio_flow_shift` | 可省略 | 调度由 LoRA metadata 中的 `base_schedule` 提供 |

## 当前已适配的优化点

1. 昇腾亲和的注意力算子：MindIE-SD的ascend_laser_attention算子，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
2. 昇腾亲和的融合算子：RMSNorm、AddRMSNorm、GQA、SwiGLU、rotary_position_embedding，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
3. 稀疏fa算子，详情见 [sparse.md](../../docs/zh/features/sparse.md)
4. 在线量化：int8、mxfp8、mxfp4
5. USP\TP\Tile 并行
6. 后训练：少步蒸馏

## Benchmark

### Atlas 800I A2

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

### Atlas 800I A3

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

### Ascend 950PR

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

以上表格中的测试数据，无损优化均采用对应机型的推荐无损优化配置、有损优化均采用对应机型的推荐有损优化配置，并在vllm-omni 0.28.0 版本下测试得出。

## 后续计划

针对Minimax-H3的性能优化近期持续更新中，敬请关注。
