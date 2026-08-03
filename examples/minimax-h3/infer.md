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

### 部署环境

#### 1）官方 Docker 镜像

您可以通过[镜像链接](https://quay.io/repository/atlas-ci/vllm-ascend?tab=tags)下载镜像压缩包来进行部署，具体流程如下：

```bash
# 拉取镜像
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
  -v /data:/data \
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

#### 3）安装vllm-Omni

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
git fetch origin pull/5699/head
git cherry-pick FETCH_HEAD
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

### 启动服务

#### T2VA / FL2VA：启动服务与发起请求

T2VA 与 FL2VA 共用 `FL2VA` 分区和同一个服务进程。以下为 8 卡启动命令：

```bash
export PORT=9098
export MODEL="${MODEL_ROOT}/FL2VA"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
```

该配置使用 8 卡 USP、8 卡文本编码器 TP、逐层卸载、VAE `tile` 并行和 FlashAttention。服务启动后，可分别发起以下请求：

```bash
# 文本生成音视频（T2VA）
hdr="$(mktemp)"
curl -sS -m 3600 -D "${hdr}" \
  -o video_out.mp4 \
  "http://127.0.0.1:9098/v1/videos/sync" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}'

# 首帧生成音视频（FL2VA）
export FIRST_FRAME=/path/to/first_frame.jpeg
hdr="$(mktemp)"
curl -sS -m 3600 -D "${hdr}" \
  -o fl2va_out.mp4 \
  "http://127.0.0.1:9098/v1/videos/sync" \
  -F 'prompt=A little girl grows up.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/jpeg"
```

FL2VA 不传 `width`、`height` 时，会保持首帧宽高比，并使用 768 像素短边。

#### Ref2VA：启动服务与发起请求

Ref2VA 使用独立的 `Ref2VA` 分区，需单独启动服务。

```bash
export PORT=9098
export MODEL="${MODEL_ROOT}/Ref2VA"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN

# 文本 + 参考视频生成音视频
export DATA_DIR=/path/to/sample2_text_video
export REFERENCE_VIDEO="${DATA_DIR}/reference_video.mp4"
export PROMPT="$(tr -d '\r' < "${DATA_DIR}/prompt.txt")"
hdr="$(mktemp)"
curl -sS -m 3600 -D "${hdr}" \
  -X POST "http://127.0.0.1:9098/v1/videos/sync" \
  -F "prompt=${PROMPT}" \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":5.0,"audio_flow_shift":3.0}' \
  -F "input_references=@${REFERENCE_VIDEO};type=video/mp4" \
  -o ref2va_video.mp4
```

`DATA_DIR` 需包含 `prompt.txt` 与 `reference_video.mp4`。
