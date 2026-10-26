# MindIE SD

## 快速参考

| 项目 | 值 |
|------|-----|
| **镜像** | `mindiesd` |
| **Tags** | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` |
| | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` |
| | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` |
| **基础镜像** | Atlas 800I A2推理服务器：`quay.io/ascend/vllm-omni:v0.28.0` |
| | Atlas 800I A3超节点服务器：`quay.io/ascend/vllm-omni:v0.28.0-a3` |
| | Ascend 950PR&950DT系列产品：`quay.io/ascend/vllm-omni:v0.28.0-a5` |
| **架构** | `linux/amd64`（x86_64） |
| **操作系统** | Ubuntu 22.04 |
| **Python** | 3.12 |
| **CANN** | 9.1.0 |
| **TorchNPU** | 2.10.0.post4 |
| **许可证** | 木兰宽松许可证 第2版（Mulan PSL v2） |

本镜像由 [MindIE community](https://www.hiascend.com/cn/developer/software/mindie) 维护。

获取帮助：

- [MindIE 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
- [MindIE SD 文档](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/index.md)
- [昇腾开发者社区](https://www.hiascend.com/developer)
- [问题反馈](https://gitcode.com/Ascend/MindIE-SD/issues)

## 镜像介绍

本镜像将 **vLLM-Omni** 与 **MindIE SD**（Mind Inference Engine Stable Diffusion）集成在单一容器中，支持在昇腾 NPU 上同时进行多模态大语言模型推理和 Stable Diffusion 图像生成。

镜像基于 `quay.io/ascend/vllm-omni` 基础镜像构建（已包含 CANN 9.1.0、torch、TorchNPU、vllm 及 vllm_ascend），提供三个版本以适配不同的 NPU 系列：

- **Atlas 800I A2推理服务器**：基于 `quay.io/ascend/vllm-omni:v0.28.0`，适用于Atlas 800I A2推理服务器
- **Atlas 800I A3超节点服务器**：基于 `quay.io/ascend/vllm-omni:v0.28.0-a3`，适用于Atlas 800I A3超节点服务器
- **Ascend 950PR&950DT系列产品**：基于 `quay.io/ascend/vllm-omni:v0.28.0-a5`，适用于Ascend 950PR&950DT系列产品

三个基础镜像标签仅提供 `linux/amd64`。

三个版本均额外安装了以下昇腾调优与调试工具：

| 组件 | 版本 | 说明 |
|------|------|------|
| mindiesd | 3.1.0 | MindIE Stable Diffusion 推理引擎 |
| msprobe | 0.1.4 | 精度调试工具 |
| msmodelslim | 8.2.1 | 模型压缩/量化工具 |
| msprof-analyze | 26.0.0 | MindStudio Profiler 分析工具 |
| msprof | *（随 CANN 捆绑）* | NPU 性能Profiling工具 |

## 镜像Tag说明及Dockerfile归档路径

### Tag 命名规则

```text
{版本号}-{CANN版本}-{torch_npu版本}-{适用产品信息}-{操作系统}-{Python版本}
```

| 系列 | 示例 Tag | 基础镜像 |
|------|----------|----------|
| Atlas 800I A2推理服务器 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0` |
| Atlas 800I A3超节点服务器 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a3` |
| Ascend 950PR&950DT系列产品 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` | `quay.io/ascend/vllm-omni:v0.28.0-a5` |

### v3.1.0 版本 Dockerfile 目录

每个系列均有独立 Dockerfile，存放于 MindIE SD 源码仓库的 `docker/omni` 目录：

| 系列 | 示例 Tag | Dockerfile |
|------|----------|------------|
| Atlas 800I A2推理服务器 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a2.ubuntu) |
| Atlas 800I A3超节点服务器 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a3.ubuntu) |
| Ascend 950PR&950DT系列产品 | `v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/omni/Dockerfile.a5.ubuntu) |

已发布镜像的 Tag（包括 v3.0.0）见 [Supported Tags](https://gitcode.com/Ascend/MindIE-SD/blob/dev/docker/supported_tags.md)。

## 快速开始

### 拉取基础镜像

浏览所有可用 Tag：[quay.io/ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags)。

<!-- npu="910b" id1 -->
Atlas 800I A2推理服务器：

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0
```
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas 800I A3超节点服务器：

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0-a3
```
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR&950DT系列产品：

```bash
docker pull quay.io/ascend/vllm-omni:v0.28.0-a5
```
<!-- end id3 -->

> **提示：** 如果使用 Podman 作为容器运行时，可将 `docker pull` 替换为 `podman pull`。

### 运行容器

<!-- npu="910b" id4 -->
Atlas 800I A2推理服务器：

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    bash
```
<!-- end id4 -->

<!-- npu="A3" id5 -->
Atlas 800I A3超节点服务器：

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    bash
```
<!-- end id5 -->

<!-- npu="950" id6 -->
Ascend 950PR&950DT系列产品：

```bash
docker run -it --rm --name=mindiesd \
    --privileged \
    --shm-size=1g \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci_manager \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12 \
    bash
```
<!-- end id6 -->

> **注意：** `--privileged` 和设备映射是访问 NPU 的必要条件。需要将宿主机的驱动库（`/usr/local/Ascend/driver/lib64`）、驱动版本信息、DCMI、`npu-smi` 及昇腾安装信息挂载到容器中。

### 本地构建

克隆 MindIE SD 仓库后，进入 `docker/omni` 目录执行构建：

<!-- npu="910b" id7 -->
Atlas 800I A2推理服务器：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12 \
    -f Dockerfile.a2.ubuntu .
```
<!-- end id7 -->

<!-- npu="A3" id8 -->
Atlas 800I A3超节点服务器：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-a3-ubuntu22.04-py3.12 \
    -f Dockerfile.a3.ubuntu .
```
<!-- end id8 -->

<!-- npu="950" id9 -->
Ascend 950PR&950DT系列产品：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
git checkout dev
cd docker/omni

docker build --platform linux/amd64 -t mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-950-ubuntu22.04-py3.12 \
    -f Dockerfile.a5.ubuntu .
```
<!-- end id9 -->

### 二次开发

如需添加自定义依赖或应用代码，可基于本镜像创建新的 Dockerfile：

```dockerfile
FROM mindiesd:v3.1.0-cann9.1.0-torch_npu2.10.0.post4-910b-ubuntu22.04-py3.12

# 安装自定义依赖包
RUN pip install --no-cache-dir your-package

# 拷贝应用代码
COPY ./your-app /workspace/your-app
WORKDIR /workspace/your-app
```

## 基础验证流程（Ascend 950PR&950DT系列产品）

容器启动后，按以下三步完成基础验证，全部命令在容器内执行。以下流程已在 Ascend 950PR 实机验证（宿主机 Ubuntu 24.04，驱动 25.7.rc1.10）。

> **验证所用镜像 tag 说明**：实测镜像为昇腾镜像仓库实际发布的 `v3.1.0-A5-ubuntu22.04-py3.11`（`swr.cn-south-1.myhuaweicloud.com/ascendhub/mindiesd`），与本仓上方表格中的正式 tag 命名暂不一致；且该 tag 中的 `py3.11` 与镜像内实际 Python 3.12.13 存在出入。差异已在 [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267) 中反馈维护者确认，拉取镜像请以[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f)页面为准。

### 1. NPU 环境检查

```bash
npu-smi info
```

期望输出（Ascend 950PR 实测摘录）：

```text
| NPU ID | Name        | Health | Power(W) | Temp(C) | HBM-Usage(MB)  |
|--------|-------------|--------|----------|---------|----------------|
| 0      | Ascend950PR | OK     | 211.7    | 55      | 5236 / 131072  |
```

### 2. CANN / TorchNPU 加速库自检

```bash
python3 -c "import torch, torch_npu; print('torch:', torch.__version__); print('torch_npu:', torch_npu.__version__); print('npu available:', torch.npu.is_available())"
```

实测输出（Ascend 950PR）：

```text
torch: 2.10.0+cpu
torch_npu: 2.10.0.post4
npu available: True
```

### 3. MindIE-SD 无权重基础推理

使用仓库 `examples/dummy_run` 的空权重验证入口，无需下载模型权重（仅自动拉取 KB 级配置文件）。以 Wan2.2 为例（FLUX.1-dev 为 gated 仓库，需要 HuggingFace 授权后使用 `flux_infer.py`，其余入口同参数）：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
pip install -r examples/dummy_run/requirements.txt
python examples/dummy_run/wan_infer.py --device_id 0 --num_layers 2
```

实测输出摘录（Ascend 950PR，含各阶段耗时与显存统计）：

```text
INFERENCE                  Time(ms)   Mem(GB)   Peak(GB)
-- Timed --
text_encoder                  5.36      8.63      12.08
transformer                 1051.80      8.65      12.08
OVERALL TOTAL               2718.37                12.08
Wan2.2 dummy weight verification PASSED
```

## 已知限制（Ascend 950PR&950DT系列产品）

- 基础镜像标签仅提供 `linux/amd64`（x86_64）架构。
- 实测 Ascend 950PR 环境不存在 `/dev/devmm_svm` 设备（910B/A3 环境存在该设备）。上文运行命令已相应移除该设备映射；若自行编写启动命令，请勿挂载该设备，否则 `docker run` 会因设备缺失直接报错。
- 运行命令中的 `/dev/davinciN` 设备映射需按实际卡数调整：实测单卡 950PR 环境仅存在 `/dev/davinci0`，直接照抄多卡示例中的 `davinci1`~`davinci3` 会因设备不存在而启动失败。
- 基础镜像默认 `SOC_VERSION=ascend950dt_9582`（对应 950DT），在 Ascend 950PR 实机上经运行时自动适配可正常使用。
- 宿主机需安装与 CANN 9.1.0 配套的 NPU 驱动（实测 25.7.rc1.10）及 Ascend Docker Runtime，并按上文挂载驱动库与设备文件。
- 本仓正式 tag 命名（如 `v3.1.0-...-950-...-py3.12`）与昇腾镜像仓库实际发布 tag（`v3.1.0-A5-ubuntu22.04-py3.11`，镜像内 Python 实为 3.12.13）存在差异，镜像拉取请以[昇腾镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f)页面为准，两者对应关系待维护者确认（见 [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267)）。
- 在无法运行容器 runtime 的受限环境（例如禁止创建 namespace 的开发容器）中，可通过 `skopeo copy` 拉取镜像、`umoci unpack` 解包并 `chroot` 进入 rootfs 的方式完成镜像内容与基础功能的等价验证，完整记录参见 [issue #267](https://gitcode.com/Ascend/MindIE-SD/issues/267)。

## 兼容性变更说明

请参考 [MindIE SD 文档](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/index.md) 获取最新的版本发布说明和兼容性信息。

## 许可证/免责声明

本镜像采用 **木兰宽松许可证 第2版（Mulan PSL v2）** 授权。完整许可文本请参见 [LICENSE](https://gitcode.com/Ascend/MindIE-SD/blob/master/LICENSE.md) 文件。

拉取并使用本容器镜像即表示您接受华为容器许可协议的条款和条件。许可副本可通过以下地址获取：<https://www.hiascend.com/legal/ascendhub-download>

您同意并承诺，在使用本镜像中的华为或第三方软件时，遵守相应华为或第三方软件的许可协议。
