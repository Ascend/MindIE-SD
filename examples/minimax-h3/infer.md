# MiniMax-H3

## 简介

2026年8月3日，MiniMax正式开源通用全模态生成模型 MiniMax-H3，有33B参数量，覆盖文本生成音视频（T2VA）、首帧/末帧生成音视频（FL2VA）及多模态参考生成音视频（Ref2VA）等工作流。该模型统一理解文本、图像、视频和声音等多模态上下文，并生成原生双声道音视频，最高支持 15 秒 2K 分辨率输出，在指令遵循、品牌与文字呈现、视频动作迁移等场景展现出较强的可控生成能力。

## 环境准备

### 模型权重

MiniMax-H3: [下载模型权重](https://huggingface.co/MiniMaxAI/MiniMax-H3)

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL_ROOT}"
```

下载完成后，权重目录结构应如下：

```text
MiniMax-H3/
├── FL2VA/          # T2VA/FL2VA 任务共用权重
├── Ref2VA/         # Ref2VA 任务权重
└── model_index.json
```

启动服务时通过 `MODEL` 环境变量指向对应子目录（见[启动服务](#启动服务)）。

### 部署环境

#### 1）官方 Docker 镜像

您可以通过[镜像链接](https://quay.io/repository/atlas-ci/vllm-ascend?tab=tags)下载镜像压缩包来进行部署，具体流程如下，以 Atlas 800I A3 为例：

```bash
# 拉取Atlas 800I A3镜像
docker pull quay.io/atlas-ci/vllm-ascend:v0.26.0-a3

# 创建容器
export IMAGE=vllm-ascend:v0.26.0-a3
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

#### 2）安装MindIE-SD

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

#### 3）安装最新的vllm-Omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple
```

#### 4）安装其他依赖

```bash
apt update

# 安装ffmpeg
apt install -y ffmpeg

# 安装decord
git clone --depth 1 --recursive https://github.com/dmlc/decord.git
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DUSE_FFMPEG=1 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
pip3 install . --no-build-isolation
```

[vllm_omni镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=info)已提供Atlas 800I A2 / Atlas 800I A3的基础镜像。如需在Ascend 950PR / Ascend 950DT 中部署，需从零构建，推荐使用配套版本如下：

| 产品名称 | 版本 |
| -------- | -------- |
| CANN | 9.1.0 |
| TorchNPU | 2.10.0 |
| vLLM | 0.26.0 |
| vLLM Ascend | 最新主分支 |
| vLLM Omni | 最新主分支 |
| MindIE SD | 最新dev分支 |

## 启动服务

T2VA/FL2VA共享同一份模型权重，Ref2VA的模型权重与之不同。在启动服务时通过设置**MODEL**环境变量为对应的模型权重路径来分别拉起不同的服务。其他环境变量和配置均为T2VA/FL2VA/Ref2VA通用。

启动 T2VA / FL2VA服务：

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
```

启动 Ref2VA 服务：

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
```

### Atlas 800I A2 / Atlas 800I A3

以下是Atlas 800I A2 / Atlas 800I A3 适配的 8 卡启动命令，均使用了 8 卡 USP、8 卡文本编码器 TP、逐层卸载、VAE `tile` 并行。

#### 推荐配置（Atlas 800I A2 / A3）

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

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-distributed-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

注意：在启动Ref2VA服务时，将

```bash
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

替换成

```bash
  --diffusion-attention-backend FLASH_ATTN
```

#### 其他可选性能优化配置

以下小节只列出相对 [推荐配置（Atlas 800I A2 / A3）](#推荐配置atlas-800i-a2--a3) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

##### 使能cache-dit

在启动命令末尾追加以下参数，启用 DiT block 缓存与 TaylorSeer 外推：

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

##### 使能在线int8量化

在 `--text-encoder-tp-size 8` 之后追加 `--diffusion-quantization-config '{"transformer":{"method":"int8"}}`，并为逐层卸载追加 `--dlo-no-use-allgather`：

```bash
  --diffusion-quantization-config '{"transformer":{"method":"int8"}}' \
  --dlo-no-use-allgather
```

### Ascend 950PR / Ascend 950DT

以下是Ascend 950PR / Ascend 950DT 适配的 8 卡启动命令，均使用了 8 卡 USP、8 卡文本编码器 TP、VAE `tile` 并行。

#### 推荐配置（Ascend 950PR / 950DT）

与 Atlas 800I A2 / A3 的差异：不使用逐层卸载（无 `--enable-distributed-layerwise-offload`）；默认启用 `mxfp8` 在线量化；不设置 `MINDIE_SD_FA_TYPE`（该环境变量不适用于 950PR / 950DT）。

```bash
export PORT=9098
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
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
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'
```

#### 其他可选性能优化配置

以下小节只列出相对 [推荐配置（Ascend 950PR / 950DT）](#推荐配置ascend-950pr--950dt) 的**增量或替换参数**，环境变量与其余启动参数均保持不变。

##### 使能cache-dit

在启动命令末尾追加以下参数，启用 DiT block 缓存与 TaylorSeer 外推：

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
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
|------|------|------|------|
| `prompt` | `str` | 是 | 文本提示词 |
| `width` / `height` | `int` | 与 `short_edge` 二选一 | 输出分辨率（像素），T2VA 示例采用此方式 |
| `short_edge` | `int` | 与 `width`/`height` 二选一 | 输出短边长度（像素），FL2VA / Ref2VA 示例采用此方式 |
| `aspect_ratio` | `str` | 否 | 输出宽高比（如 `16:9`），与 `short_edge` 搭配使用 |
| `fps` | `int` | 否 | 输出帧率 |
| `num_inference_steps` | `int` | 否 | 去噪推理步数 |
| `flow_shift` | `float` | 否 | 视频流偏移参数 |
| `audio_flow_shift` | `float` | 否 | 音频流偏移参数 |
| `seed` | `int` | 否 | 随机种子 |
| `extra_params.task` | `str` | 是 | 任务类型：`t2va` / `fl2va` / `ref2va` |
| `extra_params.duration` | `float` | 否 | 生成音视频时长（秒），最高 15 秒 |
| `input_references` | `file` | 否 | 参考输入文件：FL2VA 传首帧图像（`image/png` 等），Ref2VA 传参考视频/音频（`video/mp4` 等） |

## 当前已适配的优化点

1. 昇腾亲和的注意力算子：MindIE-SD的ascend_laser_attention算子，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
2. 昇腾亲和的融合算子：RMSNorm、AddRMSNorm、GQA、SwiGLU、rotary_position_embedding，详情见 [core_layers.md](../../docs/zh/features/core_layers.md)
3. 稀疏fa算子，详情见 [sparse.md](../../docs/zh/features/sparse.md)

## Benchmark

### Atlas 800I A3

| Npus | Workload | Parallelism | Precision | Input | Output | E2E (s) | Per step (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 5s | 97.84 | 1.87 |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 10s | 245.77 | 4.75 |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 15s | 448.19 | 8.75 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + 首帧图像 | 768P, 5s | 117.23 | 2.18 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + 首帧图像 | 768P, 10s | 278.94 | 5.43 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + 首帧图像 | 768P, 15s | 489.05 | 9.61 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 时长为5s的参考视频 | 768P, 5s | 343.95 | 6.55 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 时长为10s的参考视频 | 768P, 10s | 1088.93 | 21.40 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 时长为15s的参考视频 | 768P, 15s | 2333.15 | 46.60 |

以上表格中的数据均使用[推荐配置（Atlas 800I A2 / A3）](#推荐配置atlas-800i-a2--a3)测试所得。

### Ascend 950PR

| Npus | Workload | Parallelism | Precision | Input | Output | E2E (s) | Per step (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | T2VA | TP8 + USP8 | mxfp8 | prompt | 768P, 5s | 93.31 | 1.58 |
| 8 | T2VA | TP8 + USP8 | mxfp8 | prompt | 768P, 10s | 320.17 | 3.93 |
| 8 | T2VA | TP8 + USP8 | mxfp8 | prompt | 768P, 15s | 680.46 | 7.30 |
| 8 | FL2VA | TP8 + USP8 | mxfp8 | prompt + 首帧图像 | 768P, 5s | 106.68 | 1.77 |
| 8 | FL2VA | TP8 + USP8 | mxfp8 | prompt + 首帧图像 | 768P, 10s | 344.46 | 4.24 |
| 8 | FL2VA | TP8 + USP8 | mxfp8 | prompt + 首帧图像 | 768P, 15s | 707.65 | 7.67 |
| 8 | Ref2VA | TP8 + USP8 | mxfp8 | prompt + 时长为5s的参考视频 | 768P, 5s | 331.77 | 5.06 |
| 8 | Ref2VA | TP8 + USP8 | mxfp8 | prompt + 时长为10s的参考视频 | 768P, 10s | 1264.21 | 16.96 |
| 8 | Ref2VA | TP8 + USP8 | mxfp8 | prompt + 时长为15s的参考视频 | 768P, 15s | 2648.94 | 34.30 |

以上表格中的数据均使用[推荐配置（Ascend 950PR / 950DT）](#推荐配置ascend-950pr--950dt)测试所得。

## 后续计划

针对Minimax-H3的性能优化近期持续更新中，敬请关注。
