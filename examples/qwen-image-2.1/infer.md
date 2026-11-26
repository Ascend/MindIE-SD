# Qwen-Image-2.1

## 简介

Qwen-Image-2.1 是图像生成模型，覆盖文生图（T2I）及参考图像编辑（I2I）等工作流。该模型使用 Qwen3-VL 编码文本与参考图像，通过单流 DiT 生成图像，使用 VAE 完成参考图像编码和生成图像解码。文生图与参考图像编辑共用同一套模型权重和推理服务。

## 典型规格与推荐配置

使用 Qwen-Image-2.1 基础权重，统一启用 `FLASH_ATTN` 和 eager。典型用例为 1024×1024 文生图、1024×1024 单参考图编辑、2048×2048 单参考图编辑和 2048×2048 十参考图编辑。请求固定 50 步、`true_cfg_scale=1.0`、`seed=42`，每次输出一张 PNG。

| 平台 | 卡数 | 并行配置 | 精度 | CPU offload | 适用场景 |
| --- | ---: | --- | --- | --- | --- |
| Ascend 950PR / 950DT | 1 | TP1、SP1 | BF16 / Linear MXFP8 / W8A8F8 | 关闭 | 单卡文生图和编辑，十参考图按对应平台的实测显存选型 |
| Ascend 950PR / 950DT | 4 | TP1、SP4 | BF16 / Linear MXFP8 | 关闭 | 对时延有要求的文生图和编辑，每卡均需容纳完整 DiT 权重 |
| Atlas 850 超节点（Ascend 950PR） | 1 | TP1、SP1 | BF16 / Linear MXFP8 / W8A8F8 | 关闭 | 单卡文生图和编辑，十参考图按实测峰值显存选型 |
| Atlas 850 超节点（Ascend 950PR） | 4 | TP1、SP4 | BF16 / Linear MXFP8 | 关闭 | 单图和十参考图编辑，四卡配置的实测数据见 Benchmark |
| Atlas 800I A2推理服务器 | 1 | TP1、SP1 | BF16 / INT8 | 关闭 | 1024×1024 文生图及单参考图编辑，2048×2048 单参考图需核对峰值显存 |
| Atlas 800I A2推理服务器 | 4 | TP1、SP4 | BF16 | 关闭 | 对时延有要求的文生图和单参考图编辑 |
| Atlas 800I A2推理服务器 | 1 | TP1、SP1 | BF16 | 开启 | 显存受限时使用，已完成 2048×2048 十参考图编辑 |
| Atlas 800I A3超节点服务器 | — | — | — | — | 本轮在线性能待测，部署示例见启动服务章节 |

Atlas 850 超节点采用下文 Ascend 950PR / 950DT 的镜像和启动示例，显存按 [Benchmark](#benchmark) 中独立记录的整机测试结果评估。Atlas 800I A2推理服务器的 2048×2048 十参考图编辑在 BF16 SP1、BF16 SP4 和 INT8 SP1 关闭 CPU offload 时均遇到 OOM。

## 环境准备

### 模型权重

Qwen-Image-2.1：[下载模型权重](https://huggingface.co/Qwen/Qwen-Image-2.1)。

下载完整模型目录：

```bash
export MODEL=/path/to/Qwen-Image-2.1
hf download Qwen/Qwen-Image-2.1 --local-dir "${MODEL}"
```

下载后的主要目录结构如下：

```text
Qwen-Image-2.1/
├── text_encoder/       # Qwen3-VL 文本与参考图像编码器权重
├── transformer/        # 图像生成 DiT 权重
├── vae/                # 图像编解码权重
├── processor/          # 文本与图像预处理配置
├── scheduler/          # 去噪调度配置
└── model_index.json    # 模型组件配置
```

服务启动时，`MODEL` 指向上述根目录，文本编码器从其中的 `text_encoder/` 加载。使用完整模型包时，无需另行下载通用 Qwen3-VL 权重。BF16 和本文在线量化配置均使用这份原始权重。

### 部署环境

#### 1）官方 Docker 镜像

宿主机使用 Linux，安装与设备配套的昇腾驱动。通过 [vLLM-Omni 镜像仓库](https://quay.io/repository/ascend/vllm-omni?tab=tags)获取包含配套 CANN、PyTorch、torch_npu、vLLM、vLLM-Ascend 和 vLLM-Omni 的镜像。

以下以 v0.33.0 镜像为例，该版本可运行 Qwen-Image-2.1。推荐使用与目标设备配套的最新版本，以获取更多优化；镜像安装方式参见本仓[安装指导](../../docs/zh/installation.md#镜像安装vllm-omni)。

| 平台 | 镜像 |
| --- | --- |
| Ascend 950PR / 950DT | `quay.io/ascend/vllm-omni:v0.33.0-a5` |
| Atlas 800I A2推理服务器 | `quay.io/ascend/vllm-omni:v0.33.0` |
| Atlas 800I A3超节点服务器 | `quay.io/ascend/vllm-omni:v0.33.0-a3` |

以 Ascend 950PR / 950DT 为例，先拉取镜像，再创建单卡容器：

```bash
export IMAGE=quay.io/ascend/vllm-omni:v0.33.0-a5
export CONTAINER_NAME=qwen21
export OUTPUT_DIR="${HOME}/qwen21-results"
mkdir -p "${OUTPUT_DIR}" "${HOME}/.cache"
docker pull "${IMAGE}"

docker run -it -u root --name "${CONTAINER_NAME}" \
  -e MODEL="${MODEL}" \
  --privileged=true \
  --shm-size=200g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v "${HOME}/.cache":/root/.cache \
  -v "${MODEL}":"${MODEL}":ro \
  -v "${OUTPUT_DIR}":/outputs \
  -w /outputs \
  "${IMAGE}" /bin/bash
```

使用 Ascend 950PR / 950DT 芯片时，创建容器需将宿主机的驱动拓扑目录 `/usr/local/Ascend/driver/topo` 挂载到容器内的相同路径。Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器使用对应镜像，按实际驱动目录调整挂载。

四卡场景在 `docker run` 的镜像名称前追加以下设备参数，服务启动时再通过 `ASCEND_RT_VISIBLE_DEVICES` 指定使用的设备：

```bash
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3
```

设备编号按宿主机实际情况调整。CPU offload 使用主机内存，需为模型卸载和推理预留足够空间。

#### 2）安装最新的 MindIE SD【可选】

在容器内获取 MindIE SD `dev` 分支的最新源码，编译并安装。编译使用容器内配套的 PyTorch、torch_npu 和 CANN Toolkit；其余依赖参见[MindIE SD 安装指导](../../docs/zh/installation.md)。

```bash
# 准备源码目录和编译工具
export SOURCE_DIR="${HOME}/qwen21-src"
mkdir -p "${SOURCE_DIR}"
cd "${SOURCE_DIR}"
python -m pip install setuptools wheel pybind11 cmake ninja

# 获取源码
git clone --branch dev https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD

# 加载 CANN 环境并编译 wheel
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py bdist_wheel

# 安装编译产物，沿用镜像中已配套的运行依赖
python -m pip install --no-deps --force-reinstall dist/mindiesd-*.whl
```

#### 3）安装最新的 vLLM-Omni【可选】

使用镜像内置版本时可跳过本步骤。需要使用新版模型适配或优化功能时，在配套的 vLLM、vLLM-Ascend 环境中安装 vLLM-Omni 源码，版本关系参见 [vLLM-Omni NPU 安装说明](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/npu/)。

```bash
export SOURCE_DIR="${HOME}/qwen21-src"
mkdir -p "${SOURCE_DIR}"
cd "${SOURCE_DIR}"

git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu python -m pip install -e . --no-build-isolation

export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

可选优化所需的软件版本与启用方式见[当前已适配的优化点](#当前已适配的优化点)。

#### 4）安装其他依赖【可选】

下文请求示例使用 Bash、curl、jq 和 base64。在发送请求的 Ubuntu 环境中安装：

```bash
apt-get update
apt-get install -y curl jq coreutils
```

## 启动服务

以下命令在容器内执行。按机型和使用卡数选择启动配置，T2I 和 I2I 共用一个服务。

无损优化配置使用 BF16，结合昇腾注意力算子和并行计算；有损优化配置在此基础上使用在线量化。本文命令统一使用 eager 模式。

### Ascend 950PR / 950DT

Atlas 850 超节点（Ascend 950PR）使用本节的单卡和四卡启动配置。

#### 推荐无损优化配置

**单卡 BF16**：使用一张 NPU，通过 `FLASH_ATTN` 后端调用昇腾注意力实现。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN
```

**四卡 SP BF16**：使用 Ulysses 序列并行，TP1、SP4。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 4 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --diffusion-attention-backend FLASH_ATTN
```

Ulysses 分摊序列计算，不切分 DiT 权重。四卡仍需为每个设备上的模型权重及文本编码、VAE 等阶段预留显存。

#### 推荐有损优化配置

MXFP8 对 DiT 中支持量化的线性层进行在线量化，文本编码器、VAE 和 Attention 保持 BF16。

**单卡 MXFP8**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"},"text_encoder":null,"vae":null}'
```

**四卡 SP MXFP8**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 4 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 4 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"},"text_encoder":null,"vae":null}'
```

#### 其他可选优化配置

**单卡 W8A8F8**：DiT 线性层使用 MXFP8，Attention 在第 0 步 prefill 保持 BF16，后续 decode 步使用 MXFP8。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"mxfp8"},"text_encoder":null,"vae":null}' \
  --diffusion-kv-cache-dtype mxfp8 \
  --diffusion-kv-cache-skip-steps "0"
```

`--diffusion-kv-cache-skip-steps "0"` 跳过第 0 步的 Attention 量化，prefill 正常执行；DiT 线性层在各步仍使用 MXFP8。

**四卡 TP**：需要分摊 DiT 权重时，将四卡 SP 命令中的并行参数替换为：

```bash
  --tensor-parallel-size 4 \
  --ulysses-degree 1
```

**MXFP4 在线量化**：在支持该量化方式的 Ascend 950PR / 950DT 软件环境中，将 MXFP8 命令中的量化配置替换为：

```bash
  --diffusion-quantization-config '{"transformer":{"method":"mxfp4"},"text_encoder":null,"vae":null}'
```

MXFP4 的适用版本、生成效果与性能数据待补充。

**VAE tiling / Tile 并行**：高分辨率输出可使用 `--vae-use-tiling` 分块编解码，降低单块计算的显存需求。在支持 Qwen-Image-2.1 Tile 并行的版本中，可在四卡启动命令末尾追加：

```bash
  --vae-use-tiling \
  --vae-parallel-mode tile \
  --vae-patch-parallel-size 4
```

Tile 并行分配 VAE 解码块，使用同一组四张设备。分块后的输出可能与未分块输出存在差异，需结合图像效果和 VAE 峰值显存选择。对应 Ascend 950PR / 950DT 版本与实测数据待补充。

**CPU offload**：显存不足时，在所选服务命令末尾增加 `--enable-cpu-offload`。

### Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器

#### 推荐无损优化配置（Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器）

Atlas 800I A2推理服务器关闭 CPU offload 时，使用上文单卡 BF16 或四卡 SP BF16 命令，容器采用本机型镜像。四卡 SP 使用 `--num-gpus 4 --tensor-parallel-size 1 --ulysses-degree 4`。

**单卡 BF16＋CPU offload**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN \
  --enable-cpu-offload
```

**四卡 TP BF16**：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 4 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 4 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN
```

四卡 TP 默认关闭 CPU offload。显存不足时，可增加 `--enable-cpu-offload` 降低显存占用。

#### 推荐有损优化配置（Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器）

使用 INT8 对 DiT 线性层进行在线量化，文本编码器、VAE 和 Attention 保持 BF16。以下保留单卡启动示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export PORT=8091

vllm serve "${MODEL}" \
  --omni \
  --served-model-name Qwen/Qwen-Image-2.1 \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --num-gpus 1 \
  --max-num-seqs 1 \
  --cfg-parallel-size 1 \
  --dtype bfloat16 \
  --enforce-eager \
  --tensor-parallel-size 1 \
  --ulysses-degree 1 \
  --diffusion-attention-backend FLASH_ATTN \
  --diffusion-quantization-config '{"transformer":{"method":"int8"},"text_encoder":null,"vae":null}'
```

四卡 TP 使用同一量化配置，将可见设备设为 `0,1,2,3`，并将 `--num-gpus` 和 `--tensor-parallel-size` 均设为 `4`，`--ulysses-degree` 保持 `1`。

#### 其他可选优化配置（Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器）

BF16 和 INT8 均可结合 CPU offload 使用。在所选命令末尾追加：

```bash
  --enable-cpu-offload
```

offload 降低设备显存占用，同时引入 CPU 与 NPU 间的数据搬运。开启后的峰值显存和 E2E 应作为独立配置记录。

### CFG 配置

`true_cfg_scale` 是请求中的引导强度，`--cfg-parallel-size` 是服务启动时的 CFG 并行度，两者含义不同：

- 不传 `negative_prompt`，或 `true_cfg_scale <= 1` 时，不启用 true CFG，应使用 `--cfg-parallel-size 1`。配置为 2 不会自动退回单路，会产生冗余计算。
- 需要 true CFG 时，提供 `negative_prompt` 并设置 `true_cfg_scale > 1`。CFG 并行度为 1 时顺序执行两个分支；四卡 SP 服务也可使用 `--cfg-parallel-size 2 --ulysses-degree 2`，同时保持 `--tensor-parallel-size 1`，并行执行两个分支。

下文请求示例统一使用 `true_cfg_scale=1.0`，不传负向提示词。

## 发送请求

服务启动后，单卡与多卡、BF16 与量化服务使用相同的请求格式。在客户端设置服务地址和参考图路径：

```bash
set -o pipefail
export BASE_URL=http://127.0.0.1:8091
export REF_IMAGE=/path/to/qwen_bear.png
```

从其他机器发送请求时，将 `127.0.0.1` 替换为服务所在机器的 IP 地址。参考图路径为请求客户端上的文件路径。

典型用例使用以下输入和输出规格。

| 用例 | 请求 | 输入 | 预期输出 |
| --- | --- | --- | --- |
| 1K T2I | [文生图](#文生图) | 茶壶提示词 | 1024×1024 PNG |
| 1K I2I | [单参考图编辑](#单参考图编辑) | `qwen_bear.png` 和编辑提示词，`size=1024x1024` | 1024×1024 PNG |
| 2K I2I | [单参考图编辑](#单参考图编辑) | 与 1K I2I 相同 | 2048×2048 PNG |
| 2K I2I 十参考图 | [多参考图编辑](#多参考图编辑) | 按顺序上传十张 1024×1024 的人物图 | 2048×2048 PNG |

运行对应 curl 命令，检查 HTTP 状态为 200、PNG 可解码且尺寸与表中一致。十参考图用例还需核对服务端收到并处理了十张参考图。每条命令打印的 E2E 时间是从客户端发起请求到完整接收响应的耗时。

### 文生图

生成一张 1024×1024 的 PNG，提示词为“木桌上的一个陶瓷茶壶”，沿用 [vLLM-Omni Qwen-Image-2.1 文生图示例](https://github.com/vllm-project/vllm-omni/blob/0be14932ba0f823709c16d7f23ded32c32bdb138/recipes/Qwen/Qwen-Image-2.1.md#text-to-image)。

```bash
curl -fsS "${BASE_URL}/v1/images/generations" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "A ceramic teapot on a wooden table",
    "size": "1024x1024",
    "n": 1,
    "num_inference_steps": 50,
    "true_cfg_scale": 1.0,
    "seed": 42,
    "response_format": "b64_json",
    "output_format": "png"
  }' \
  -o t2i_1024_response.json \
  -w '\nHTTP: %{http_code}\nE2E: %{time_total} s\n' \
&& jq -er '.data[0].b64_json' t2i_1024_response.json \
  | base64 --decode > t2i_1024.png
```

### 参考图像编辑

#### 单参考图编辑

使用吉祥物图片 `qwen_bear.png`，要求它在月光下跳舞，周围环绕漂浮的星星。参考图名称与提示词沿用 [vLLM-Omni Qwen-Image-2.1 编辑示例](https://github.com/vllm-project/vllm-omni/blob/0be14932ba0f823709c16d7f23ded32c32bdb138/recipes/Qwen/Qwen-Image-2.1.md#image-conditioned-generation-editing)。示例输出 2048×2048。运行 1K 用例时，将 `size` 改为 `1024x1024`。

```bash
curl -fsS "${BASE_URL}/v1/images/edits" \
  -F "model=Qwen/Qwen-Image-2.1" \
  -F "image=@${REF_IMAGE}" \
  -F "prompt=Let this mascot dance under the moon, surrounded by floating stars" \
  -F "size=2048x2048" \
  -F "n=1" \
  -F "num_inference_steps=50" \
  -F "true_cfg_scale=1.0" \
  -F "seed=42" \
  -F "response_format=b64_json" \
  -F "output_format=png" \
  -o i2i_2048_response.json \
  -w '\nHTTP: %{http_code}\nE2E: %{time_total} s\n' \
&& jq -er '.data[0].b64_json' i2i_2048_response.json \
  | base64 --decode > i2i_2048.png
```

#### 多参考图编辑

多图编辑通过重复的 `image` 字段按顺序上传多张参考图。本例使用 `person-01.png` 至 `person-10.png`。该用例需要包含 [#9770](https://github.com/vllm-project/vllm-omni/pull/9770) 的 vLLM-Omni 版本。测试时每张参考图为 1024×1024，输出为 2048×2048。

```bash
REF_DIR=/path/to/reference-images
images=()
for image in "${REF_DIR}"/person-{01..10}.png; do
  images+=(-F "image=@${image};type=image/png")
done

curl -fsS "${BASE_URL}/v1/images/edits" \
  -F 'model=Qwen/Qwen-Image-2.1' \
  "${images[@]}" \
  --form-string "prompt=Create a group portrait with all ten people from the reference images. Arrange them in two rows of five and preserve each person's distinct appearance." \
  -F 'size=2048x2048' \
  -F 'n=1' \
  -F 'num_inference_steps=50' \
  -F 'true_cfg_scale=1.0' \
  -F 'seed=42' \
  -F 'response_format=b64_json' \
  -F 'output_format=png' \
  -o i2i_10_response.json \
  -w '\nHTTP: %{http_code}\nE2E: %{time_total} s\n' \
&& jq -er '.data[0].b64_json' i2i_10_response.json \
  | base64 --decode > i2i_10.png
```

### 请求参数说明

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `model` | `Qwen/Qwen-Image-2.1` | 与服务启动时的 `--served-model-name` 一致 |
| `prompt` | — | 文本提示词或图像编辑指令 |
| `negative_prompt` | blurry, low quality | 可选，配合 `true_cfg_scale > 1` 启用 true CFG |
| `true_cfg_scale` | 1.0 | true CFG 引导强度；不传负向提示词时不启用 true CFG |
| `num_inference_steps` | 50 | 去噪步数 |
| `size` | 1024x1024 | 输出图像尺寸，宽高须为 32 的倍数；建议显式指定 |
| `seed` | 42 | 随机种子 |
| `n` | 1 | 每次请求输出的图片数量 |
| `image` | 客户端图像文件 | 编辑请求的参考图像文件。包含 [#9770](https://github.com/vllm-project/vllm-omni/pull/9770) 的版本单次最多 10 张 |
| `response_format` | b64_json | 返回 Base64 编码图像 |
| `output_format` | png | 输出图像格式 |

I2I 的参考图尺寸和数量会影响编码及 prefill 的计算量；`size` 指定输出尺寸，无需将参考图预先拉伸到该尺寸。

## 当前已适配的优化点

### 注意力后端与融合算子

| 优化项 | 作用 | 配置与版本说明 |
| --- | --- | --- |
| 昇腾注意力后端 | 加速 DiT Attention 计算 | 推荐命令使用 `--diffusion-attention-backend FLASH_ATTN`，相关算子见 [核心层](../../docs/zh/features/core_layers.md) |
| `flex_attention` | 通过结构化 mask 处理 prefill 中的条件与目标序列注意力 | 对应发布版本及启用方式待补充 |

### 在线量化：INT8、MXFP8、MXFP4

在线量化直接加载原始权重，在推理初始化过程中处理支持量化的线性层。本文默认仅量化 DiT，文本编码器和 VAE 保持 BF16。

| 精度 | 使用位置 | 典型平台 | 配置方式 |
| --- | --- | --- | --- |
| INT8 | DiT 线性层 | Atlas 800I A2推理服务器 / Atlas 800I A3超节点服务器 | `transformer.method=int8` |
| MXFP8 | DiT 线性层 | Ascend 950PR / 950DT | `transformer.method=mxfp8` |
| MXFP4 | DiT 线性层 | Ascend 950PR / 950DT | `transformer.method=mxfp4`，适用版本与效果数据待补充 |
| W8A8F8 | DiT 线性层及 decode Attention | Ascend 950PR / 950DT | Linear MXFP8＋Attention MXFP8，跳过 prefill Attention 量化 |

以上 `transformer.method` 均通过 `--diffusion-quantization-config` 的 JSON 配置。量化后的输出效果需结合业务场景选择；权重占用下降不一定带来整次请求峰值显存同比下降。

### USP / TP / Tile 并行

| 方式 | 作用对象 | 主要参数 | 显存与 E2E 的关系 |
| --- | --- | --- | --- |
| USP（本文采用 Ulysses SP） | DiT 序列计算 | `--ulysses-degree`，别名 `--usp` | 分摊序列计算，不切分 DiT 权重；加速收益受序列长度及通信开销影响 |
| TP | 支持张量并行的 DiT 层 | `--tensor-parallel-size` | 分摊相关权重和计算，降低这部分单设备占用，同时引入通信 |
| Tile | VAE 分块解码 | `--vae-use-tiling`、`--vae-parallel-mode tile`、`--vae-patch-parallel-size` | 降低单块计算的显存需求，并将解码块分配到多设备；具体收益取决于输出尺寸与分块配置 |

SP/TP 的参数不会自动开启文本编码器并行或 VAE Tile 并行。若整次请求的峰值来自文本编码器或 VAE，增加 DiT 并行度后，峰值显存仍可能变化较小。

选择四卡方案时，先确认目标任务能在单设备容量内完成，再比较相同精度、offload 和请求条件下 SP4 与 TP4 的 E2E。Tile 的显存和时延需随分块参数单独记录。

### CPU offload

通过 `--enable-cpu-offload` 将暂不参与计算的模型组件卸载到主机内存。它可以降低设备显存占用，但会引入数据搬运开销。

选择 offload 时，应同时查看峰值显存和 E2E：容量充足时优先减少搬运，容量不足时使用 offload 完成部署。开启与关闭 offload 的结果分别列出。

## Benchmark

各平台分别列出在线服务的测试结果。测试使用 Qwen-Image-2.1 基础权重，启用 `FLASH_ATTN` 和 eager，固定 TP1、CFG 并行 1、50 步、`true_cfg_scale=1.0`、`seed=42`。服务以 `--max-num-seqs 1` 串行处理请求。

HTTP E2E 取同宿主机回环请求输出的 `curl time_total`。DiT 耗时独立采集，按每次请求中累计耗时最长的 rank 统计 50 步总时延、平均单步、第 0 步 prefill 和第 1～49 步 decode 的平均单步时延。两类时延均在预热 2 次后各取 5 次正式测量的中位数。

常驻权重显存为服务就绪后设备上的权重存储大小，共享存储只计一次。推理峰值显存为请求期间整卡 HBM 的最高观测值。多卡结果均取单卡最大值，单位为 GB（10⁹ 字节）。`OOM` 表示实测显存不足，`—` 表示暂无结果。

Ascend 950PR 的 HBM 由 `npu-smi` 采样。Ascend 950DT、Atlas 850 超节点和 Atlas 800I A2推理服务器使用驱动 DCMI 接口。

### 测试环境

2026-11-17 至 2026-11-18 的在线测试加载 [vLLM-Omni #9770 的合入提交](https://github.com/vllm-project/vllm-omni/commit/9ab4fabc74f271074d77485560a3599649cc9853)。已测平台均使用 PyTorch `2.10.0+cpu`、torch_npu `2.10.0.post7`、vLLM `0.33.0+empty`、vLLM-Ascend `0.19.1rc2.dev3349+g221fb096c.d20261104` 和 MindIE SD `3.2.0`。

| 平台 | CPU 架构 | 单设备 HBM 容量 | 驱动 | CANN | 实测镜像 |
| --- | --- | ---: | --- | --- | --- |
| Atlas 350（Ascend 950PR） | x86_64 | 112 GiB | 25.7.rc1.b999 | 9.2.1 | `quay.io/ascend/vllm-omni:v0.33.0rc1-a5` |
| Atlas 850E 服务器（Ascend 950DT） | aarch64 | 84 GiB | 25.6.rc2.b001 | 9.2.1 | `quay.io/ascend/vllm-omni:v0.33.0rc1-a5` |
| Atlas 850 超节点（Ascend 950PR） | aarch64 | 84 GiB | 25.1.rc1.b097 | 9.2.1 | `quay.io/ascend/vllm-omni:v0.33.0rc1-a5` |
| Atlas 800I A2推理服务器（910B4-1） | aarch64 | 64 GiB | 25.2.0 | 9.2.1 | `quay.io/ascend/vllm-omni:v0.33.0rc1` |

单图 I2I 的原始参考图 `qwen_bear.png` 为 514×556。提示词和其他输入见[发送请求](#发送请求)。

### Atlas 350（Ascend 950PR）性能数据

| 任务 | 输出分辨率 | 参考图数 | 并行配置 | 精度 | CPU offload | HTTP E2E（s） | DiT 总时延（s） | DiT 平均单步（s） | Prefill 单步（s） | Decode 平均单步（s） | 常驻权重显存（GB） | 推理峰值显存（GB） | 状态 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 关闭 | 12.632 | 12.435 | 0.249 | 0.239 | 0.249 | 32.44 | 42.27 | 完成（时延复测，显存为初测值） |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 关闭 | 16.712 | 16.288 | 0.326 | 0.603 | 0.320 | 32.44 | 42.29 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 关闭 | 91.544 | 90.517 | 1.810 | 2.070 | 1.805 | 32.44 | 58.01 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 关闭 | 176.452 | 171.650 | 3.433 | 8.234 | 3.335 | 32.44 | 89.40 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | BF16 | 关闭 | 3.822 | 3.672 | 0.073 | 0.076 | 0.073 | 32.44 | 43.26 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | BF16 | 关闭 | 5.209 | 4.797 | 0.096 | 0.346 | 0.091 | 32.44 | 43.29 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | BF16 | 关闭 | 26.706 | 25.807 | 0.516 | 0.788 | 0.510 | 32.44 | 57.86 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | BF16 | 关闭 | 55.362 | 51.455 | 1.029 | 4.508 | 0.958 | 32.44 | 73.42 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | Linear MXFP8 | 关闭 | 9.741 | 9.631 | 0.193 | 0.177 | 0.193 | 25.68 | 35.52 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | Linear MXFP8 | 关闭 | 12.671 | 12.349 | 0.247 | 0.448 | 0.243 | 25.68 | 35.74 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | Linear MXFP8 | 关闭 | 77.175 | 75.927 | 1.519 | 1.705 | 1.515 | 25.68 | 52.64 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | Linear MXFP8 | 关闭 | 159.724 | 156.690 | 3.134 | 7.207 | 3.051 | 25.68 | 85.47 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | Linear MXFP8 | 关闭 | 2.896 | 2.739 | 0.055 | 0.057 | 0.055 | 25.68 | 36.55 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | Linear MXFP8 | 关闭 | 4.007 | 3.620 | 0.072 | 0.245 | 0.069 | 25.68 | 36.57 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | Linear MXFP8 | 关闭 | 22.126 | 21.463 | 0.429 | 0.620 | 0.425 | 25.68 | 51.55 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | Linear MXFP8 | 关闭 | — | — | — | — | — | — | — | 待补齐 |
| T2I | 1024×1024 | 0 | SP1 | W8A8F8 | 关闭 | 9.401 | 9.337 | 0.187 | 0.182 | 0.187 | 25.68 | 35.54 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | W8A8F8 | 关闭 | 11.571 | 11.203 | 0.224 | 0.444 | 0.220 | 25.68 | 35.76 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | W8A8F8 | 关闭 | 75.959 | 74.748 | 1.495 | 1.685 | 1.491 | 25.68 | 52.66 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | W8A8F8 | 关闭 | 164.203 | 160.959 | 3.219 | 7.110 | 3.140 | 25.68 | 85.49 | 完成 |

Ascend 950PR x86_64 的 Linear MXFP8 SP4 十参考图场景因四卡设备组占用，正式计时待补齐。

### Atlas 850E 服务器（Ascend 950DT）性能数据

| 任务 | 输出分辨率 | 参考图数 | 并行配置 | 精度 | CPU offload | HTTP E2E（s） | DiT 总时延（s） | DiT 平均单步（s） | Prefill 单步（s） | Decode 平均单步（s） | 常驻权重显存（GB） | 推理峰值显存（GB） | 状态 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 关闭 | 11.309 | 11.190 | 0.224 | 0.226 | 0.224 | 32.44 | 42.00 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 关闭 | 13.657 | 13.214 | 0.264 | 0.490 | 0.260 | 32.44 | 42.02 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 关闭 | 72.812 | 71.882 | 1.438 | 1.673 | 1.433 | 32.44 | 57.76 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 关闭 | 140.773 | 137.845 | 2.757 | 6.613 | 2.678 | 32.44 | 73.31 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | BF16 | 关闭 | 3.481 | 3.314 | 0.066 | 0.078 | 0.066 | 32.44 | 42.94 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | BF16 | 关闭 | 4.429 | 4.004 | 0.080 | 0.288 | 0.076 | 32.44 | 42.98 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | BF16 | 关闭 | 20.540 | 19.601 | 0.392 | 0.619 | 0.387 | 32.44 | 57.57 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | BF16 | 关闭 | 41.855 | 38.406 | 0.768 | 3.304 | 0.717 | 32.44 | 73.13 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | Linear MXFP8 | 关闭 | 8.068 | 7.927 | 0.159 | 0.146 | 0.159 | 25.68 | 35.24 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | Linear MXFP8 | 关闭 | 10.303 | 9.950 | 0.199 | 0.353 | 0.196 | 25.68 | 35.46 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | Linear MXFP8 | 关闭 | 59.970 | 59.160 | 1.183 | 1.337 | 1.180 | 25.68 | 52.38 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | Linear MXFP8 | 关闭 | 127.280 | 124.195 | 2.484 | 5.668 | 2.419 | 25.68 | 73.59 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | Linear MXFP8 | 关闭 | 2.615 | 2.501 | 0.050 | 0.067 | 0.050 | 25.68 | 36.23 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | Linear MXFP8 | 关闭 | 3.505 | 3.053 | 0.061 | 0.198 | 0.058 | 25.68 | 36.26 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | Linear MXFP8 | 关闭 | 17.207 | 16.250 | 0.325 | 0.474 | 0.322 | 25.68 | 51.26 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | Linear MXFP8 | 关闭 | 37.888 | 34.355 | 0.687 | 2.533 | 0.650 | 25.68 | 69.05 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | W8A8F8 | 关闭 | 7.511 | 7.384 | 0.148 | 0.146 | 0.148 | 25.68 | 35.27 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | W8A8F8 | 关闭 | 9.327 | 8.960 | 0.179 | 0.351 | 0.176 | 25.68 | 35.49 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | W8A8F8 | 关闭 | 58.601 | 57.676 | 1.154 | 1.335 | 1.150 | 25.68 | 52.41 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | W8A8F8 | 关闭 | 120.540 | 116.948 | 2.339 | 5.546 | 2.273 | 25.68 | 73.61 | 完成 |

### Atlas 850 超节点（Ascend 950PR） 性能数据

| 任务 | 输出分辨率 | 参考图数 | 并行配置 | 精度 | CPU offload | HTTP E2E（s） | DiT 总时延（s） | DiT 平均单步（s） | Prefill 单步（s） | Decode 平均单步（s） | 常驻权重显存（GB） | 推理峰值显存（GB） | 状态 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 关闭 | 11.583 | 11.417 | 0.228 | 0.254 | 0.228 | 32.44 | 42.51 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 关闭 | 13.903 | 13.427 | 0.269 | 0.529 | 0.263 | 32.44 | 42.53 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 关闭 | 71.951 | 70.787 | 1.416 | 1.656 | 1.411 | 32.44 | 58.27 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 关闭 | 134.463 | 131.068 | 2.621 | 6.542 | 2.542 | 32.44 | 73.82 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | BF16 | 关闭 | 3.611 | 3.421 | 0.068 | 0.087 | 0.068 | 32.44 | 43.39 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | BF16 | 关闭 | 4.636 | 4.134 | 0.083 | 0.344 | 0.078 | 32.44 | 43.43 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | BF16 | 关闭 | 20.085 | 18.923 | 0.378 | 0.668 | 0.373 | 32.44 | 58.02 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | BF16 | 关闭 | 39.229 | 35.325 | 0.707 | 3.455 | 0.650 | 32.44 | 73.58 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | Linear MXFP8 | 关闭 | 8.338 | 8.179 | 0.164 | 0.177 | 0.163 | 25.68 | 35.75 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | Linear MXFP8 | 关闭 | 10.495 | 10.024 | 0.200 | 0.392 | 0.197 | 25.68 | 35.97 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | Linear MXFP8 | 关闭 | 59.855 | 58.718 | 1.174 | 1.351 | 1.171 | 25.68 | 52.89 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | Linear MXFP8 | 关闭 | 121.920 | 118.361 | 2.367 | 5.644 | 2.300 | 25.68 | 74.10 | 完成 |
| T2I | 1024×1024 | 0 | SP4 | Linear MXFP8 | 关闭 | 2.908 | 2.673 | 0.053 | 0.079 | 0.053 | 25.68 | 36.68 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | Linear MXFP8 | 关闭 | 3.774 | 3.235 | 0.065 | 0.252 | 0.061 | 25.68 | 36.71 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | Linear MXFP8 | 关闭 | 16.846 | 15.681 | 0.314 | 0.522 | 0.309 | 25.68 | 51.71 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | Linear MXFP8 | 关闭 | 35.760 | 31.775 | 0.635 | 2.731 | 0.593 | 25.68 | 69.49 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | W8A8F8 | 关闭 | 7.951 | 7.799 | 0.156 | 0.179 | 0.156 | 25.68 | 35.78 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | W8A8F8 | 关闭 | 9.971 | 9.539 | 0.191 | 0.393 | 0.187 | 25.68 | 36.00 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | W8A8F8 | 关闭 | 69.430 | 68.273 | 1.365 | 1.342 | 1.366 | 25.68 | 52.92 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | W8A8F8 | 关闭 | 147.190 | 143.751 | 2.875 | 5.642 | 2.818 | 25.68 | 74.12 | 完成 |

### Atlas 800I A2推理服务器 性能数据

| 任务 | 输出分辨率 | 参考图数 | 并行配置 | 精度 | CPU offload | HTTP E2E（s） | DiT 总时延（s） | DiT 平均单步（s） | Prefill 单步（s） | Decode 平均单步（s） | 常驻权重显存（GB） | 推理峰值显存（GB） | 状态 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 关闭 | 21.299 | 20.973 | 0.419 | 0.457 | 0.419 | 32.44 | 45.42 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 关闭 | 26.788 | 25.847 | 0.517 | 1.034 | 0.506 | 32.44 | 45.45 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 关闭 | 159.886 | 157.899 | 3.158 | 4.432 | 3.132 | 32.44 | 64.33 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 关闭 | — | — | — | — | — | — | — | OOM |
| T2I | 1024×1024 | 0 | SP4 | BF16 | 关闭 | 7.699 | 7.332 | 0.147 | 0.174 | 0.146 | 32.44 | 46.26 | 完成 |
| I2I | 1024×1024 | 1 | SP4 | BF16 | 关闭 | 9.552 | 8.671 | 0.173 | 0.543 | 0.166 | 32.44 | 46.29 | 完成 |
| I2I | 2048×2048 | 1 | SP4 | BF16 | 关闭 | 41.734 | 39.412 | 0.788 | 1.186 | 0.780 | 32.44 | 65.17 | 完成 |
| I2I | 2048×2048 | 10 | SP4 | BF16 | 关闭 | — | — | — | — | — | — | — | OOM |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 开启 | 24.414 | 20.387 | 0.408 | 0.455 | 0.407 | 14.91 | 28.07 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 开启 | 29.813 | 25.232 | 0.505 | 1.026 | 0.494 | 14.91 | 30.98 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 开启 | 162.637 | 156.543 | 3.131 | 4.328 | 3.106 | 14.91 | 57.57 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 开启 | 346.985 | 333.908 | 6.678 | 19.731 | 6.406 | 14.91 | 68.72 | 完成 |
| T2I | 1024×1024 | 0 | SP1 | INT8 | 关闭 | 16.445 | 16.170 | 0.323 | 0.366 | 0.323 | 25.47 | 38.44 | 完成 |
| I2I | 1024×1024 | 1 | SP1 | INT8 | 关闭 | 21.566 | 20.718 | 0.414 | 0.870 | 0.405 | 25.47 | 38.47 | 完成 |
| I2I | 2048×2048 | 1 | SP1 | INT8 | 关闭 | 137.712 | 135.814 | 2.716 | 3.983 | 2.690 | 25.47 | 62.62 | 完成 |
| I2I | 2048×2048 | 10 | SP1 | INT8 | 关闭 | — | — | — | — | — | — | — | OOM |

2048×2048 十参考图的三种 OOM 均发生在 DiT 第 0 步。BF16 SP1 的 rank 0 申请 1.32 GiB，剩余 779.64 MiB。BF16 SP4 的四个 rank 各申请 1.04 GiB，剩余 932.74～945.95 MiB。INT8 SP1 的 rank 0 申请 1.32 GiB，剩余 1.07 GiB。开启 CPU offload 的单卡 BF16 完成了该用例，推理峰值为 68.72 GB。

### Atlas 800I A3超节点服务器 性能数据

| 任务 | 输出分辨率 | 参考图数 | 并行配置 | 精度 | CPU offload | HTTP E2E（s） | DiT 总时延（s） | DiT 平均单步（s） | Prefill 单步（s） | Decode 平均单步（s） | 常驻权重显存（GB） | 推理峰值显存（GB） | 状态 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| T2I | 1024×1024 | 0 | SP4 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 1024×1024 | 1 | SP4 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 1 | SP4 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 10 | SP4 | BF16 | 关闭 | — | — | — | — | — | — | — | 待测 |
| T2I | 1024×1024 | 0 | SP1 | BF16 | 开启 | — | — | — | — | — | — | — | 待测 |
| I2I | 1024×1024 | 1 | SP1 | BF16 | 开启 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 1 | SP1 | BF16 | 开启 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 10 | SP1 | BF16 | 开启 | — | — | — | — | — | — | — | 待测 |
| T2I | 1024×1024 | 0 | SP1 | INT8 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 1024×1024 | 1 | SP1 | INT8 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 1 | SP1 | INT8 | 关闭 | — | — | — | — | — | — | — | 待测 |
| I2I | 2048×2048 | 10 | SP1 | INT8 | 关闭 | — | — | — | — | — | — | — | 待测 |
