# MindIE-SD

## 快速参考

| 项目 | 值 |
|------|-----|
| **镜像** | `mindiesd` |
| **Tags** | `v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64` |
| | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` |
| **基础镜像** | Atlas 800I A2 推理服务器：`quay.io/ascend/vllm-omni:v0.20.0` |
| | Atlas 800I A3 超节点服务器：`quay.io/ascend/vllm-omni:v0.20.0-a3` |
| **架构** | `linux/arm64`（aarch64） |
| **操作系统** | Ubuntu 22.04 |
| **Python** | 3.11 |
| **CANN** | 8.5.1 |
| **TorchNPU** | 2.9.0 |
| **许可证** | 木兰宽松许可证 第2版（Mulan PSL v2） |

本镜像由 [MindIE community](https://www.hiascend.com/cn/developer/software/mindie) 维护。

获取帮助：

- [MindIE 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
- [MindIE-SD 文档](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/index.md)
- [昇腾开发者社区](https://www.hiascend.com/developer)
- [问题反馈](https://gitcode.com/Ascend/MindIE-SD/issues)

## 镜像介绍

本镜像将 **vLLM-Omni** 与 **MindIE-SD**（Mind Inference Engine Stable Diffusion）集成在单一容器中，支持在昇腾 NPU 上同时进行多模态大语言模型推理和 Stable Diffusion 图像生成。

镜像基于 `quay.io/ascend/vllm-omni` 基础镜像构建（已包含 CANN 8.5.1、torch、TorchNPU、vllm 及 vllm_ascend），提供两个版本以适配不同的 NPU 系列：

- **Atlas 800I A2 推理服务器**：基于 `quay.io/ascend/vllm-omni:v0.20.0`，适用于Atlas 800I A2 推理服务器
- **Atlas 800I A3 超节点服务器**：基于 `quay.io/ascend/vllm-omni:v0.20.0-a3`，适用于Atlas 800I A3 超节点服务器

两个版本均额外安装了以下昇腾调优与调试工具：

| 组件 | 版本 | 说明 |
|------|------|------|
| mindiesd | 最新版 | MindIE Stable Diffusion 推理引擎 |
| msprobe | 0.1.4 | 精度调试工具 |
| msmodelslim | 8.2.1 | 模型压缩/量化工具 |
| msprof-analyze | 26.0.0 | MindStudio Profiler 分析工具 |
| msprof | *（随 CANN 捆绑）* | NPU 性能Profiling工具 |

## 镜像Tag说明及Dockerfile归档路径

### Tag 命名规则

```text
{版本号}-{CANN版本}-{torch_npu版本}-{适用产品信息}-{操作系统}-{Python版本}-{架构类型}-{其他字段}
```

| 系列 | 示例 Tag | 基础镜像 |
|------|----------|----------|
| Atlas 800I A2 推理服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0` |
| Atlas 800I A3 超节点服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0-a3` |

### v3.0.0 版本 Dockerfile 目录

每个系列均有独立 Dockerfile，存放于 MindIE-SD 源码仓库的 `docker/omni` 目录：

| 系列 | 示例 Tag | Dockerfile |
|------|----------|------------|
| Atlas 800I A2 推理服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/master/docker/omni/Dockerfile.a2.ubuntu) |
| Atlas 800I A3 超节点服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | [Dockerfile](https://gitcode.com/Ascend/MindIE-SD/blob/master/docker/omni/Dockerfile.a3.ubuntu) |

## 快速开始

### 拉取基础镜像

浏览所有可用 Tag：[quay.io/ascend/vllm-omni](https://quay.io/repository/ascend/vllm-omni?tab=tags)。

Atlas 800I A2 推理服务器：

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0
```

Atlas 800I A3 超节点服务器：

```bash
docker pull quay.io/ascend/vllm-omni:v0.20.0-a3
```

> **提示：** 如果使用 Podman 作为容器运行时，可将 `docker pull` 替换为 `podman pull`。

### 运行容器

Atlas 800I A2 推理服务器：

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
    mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64 \
    bash
```

Atlas 800I A3 超节点服务器：

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
    mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 \
    bash
```

> **注意：** `--privileged` 和设备映射是访问 NPU 的必要条件。需要将宿主机的驱动库（`/usr/local/Ascend/driver/lib64`）、驱动版本信息、DCMI、`npu-smi` 及昇腾安装信息挂载到容器中。

### 本地构建

克隆 MindIE-SD 仓库后，进入 `docker/omni` 目录执行构建：

Atlas 800I A2 推理服务器：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a2.ubuntu .
```

Atlas 800I A3 超节点服务器：

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD/docker/omni

docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 \
    -f Dockerfile.a3.ubuntu .
```

### 二次开发

如需添加自定义依赖或应用代码，可基于本镜像创建新的 Dockerfile：

```dockerfile
FROM mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64

# 安装自定义依赖包
RUN pip install --no-cache-dir your-package

# 拷贝应用代码
COPY ./your-app /workspace/your-app
WORKDIR /workspace/your-app
```

## 硬件支持信息

| 项目 | 要求 |
|------|------|
| **NPU** | Atlas 800I A2 推理服务器 |
| | Atlas 800I A3 超节点服务器 |
| **驱动** | 宿主机需安装昇腾 NPU 驱动 |
| **宿主机挂载** | `/usr/local/dcmi`、`/usr/local/bin/npu-smi`、`/usr/local/Ascend/driver/lib64/`、`/usr/local/Ascend/driver/version.info`、`/etc/ascend_install.info`、`/root/.cache` |

## 兼容性变更说明

请参考 [MindIE-SD 文档](https://gitcode.com/Ascend/MindIE-SD/blob/master/docs/zh/index.md) 获取最新的版本发布说明和兼容性信息。

## 许可证/免责声明

本镜像采用 **木兰宽松许可证 第2版（Mulan PSL v2）** 授权。完整许可文本请参见 [LICENSE](https://gitcode.com/Ascend/MindIE-SD/blob/master/LICENSE.md) 文件。

拉取并使用本容器镜像即表示您接受华为容器许可协议的条款和条件。许可副本可通过以下地址获取：https://www.hiascend.com/legal/ascendhub-download

您同意并承诺，在使用本镜像中的华为或第三方软件时，遵守相应华为或第三方软件的许可协议。
