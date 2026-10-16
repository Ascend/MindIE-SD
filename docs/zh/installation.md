# 安装指导

## Python包安装

MindIE SD是一个 Python 包，它基于PyTorch构建，可以轻松集成到 Python 应用程序中。

Python 包安装适用于大多数使用场景，但需要手动安装 CANN。如果希望免去手动安装 CANN 的步骤，也可以选择镜像安装：直接从昇腾社区拉取镜像并启动容器即可。

### 安装依赖

* OS: Linux
* Python: >=3.10
* Pytorch：2.6, 2.7, 2.8, 2.9, 2.10
* torch-npu: 2.6, 2.7, 2.8, 2.9, 2.10
* CANN: 9.0.1

#### CANN 安装

MindIE SD 依赖 CANN Toolkit开发套件包和 CANN ops 算子包，请参考 <a href="https://gitcode.com/cann/ops-cv/blob/master/docs/zh/install/quick_install.md" target="_blank" rel="noopener noreferrer">CANN 软件安装指南</a>

安装完成后，执行以下命令设置环境变量（以默认安装路径为例）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 注意事项

1. MindIE SD主要依赖torch-npu的版本，会尽力满足其要求的CANN以及Python版本要求。
2. CANN版本安装后，安装路径下提供进程级环境变量设置脚本“set_env.sh“，以自动完成环境变量设置，该脚本包含如[表1 环境变量](#table_environment0001)所示中的LD_LIBRARY_PATH和ASCEND_CUSTOM_OPP_PATH，用户进程结束后自动失效。

**表 1**  环境变量<a id="table_environment0001"></a>

|环境变量|说明|
|--|--|
|LD_LIBRARY_PATH|动态库的查找路径。|
|ASCEND_CUSTOM_OPP_PATH|推理引擎自定义算子包安装路径。|
|ASCEND_RT_VISIBLE_DEVICES|指定当前进程所用的昇腾AI处理器的逻辑ID，如有需要请自行配置。<br>配置示例："0,1,2"或"0-2"；昇腾AI处理器的逻辑ID间使用“,”表示分割，使用“-”表示连续。|

### 快速安装

现在最简单的方式是通过 pip 源安装，我们的软件包名字叫 mindiesd，与仓库名有些不一样。在安装 mindiesd 之前，需要先安装相关 python 依赖包：

> `requirements.txt` 位于本仓库根目录，执行下面的命令前需先获取该文件：克隆仓库（`git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD`）或从仓库单独下载 `requirements.txt`。

```bash
pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
```

然后安装 mindiesd：

```bash
pip install mindiesd
```

### 源码安装

在某些情况下，您可能需要从源代码安装 MindIE SD，以便尝试最新功能，或者根据您的特定需求自定义库。

您可以按照以下步骤从源代码安装 MindIE SD：

1. 克隆仓库&进入项目：

   ```bash
   git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD && git checkout dev
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
   ```

   说明：MindIE SD 的部分算子依赖triton-ascend==3.2.1，该版本目前仅在 <https://triton-ascend.osinfra.cn/pypi/simple> 中提供

3. 编译并安装：

   ```bash
   python setup.py bdist_wheel
   cd dist
   pip install mindiesd-*.whl
   ```

> **依赖分层说明**
>
> 仓库将依赖按用途拆分，按需安装即可：
>
> * `requirements.txt`：核心编译构建和运行依赖（最小安装）。
> * `examples/dummy_run/requirements.txt`：`dummy_run` 模型推理示例依赖（diffusers/transformers 等）。
> * `examples/service/requirements.txt`：服务化示例依赖（ray、fastapi、uvicorn、pydantic、Pillow）。
> * 测试、Lint、文档构建依赖分别见 `requirements-test.txt`、`requirements-lint.txt`、`docs/requirements-docs.txt`（详见开发者指南）。

## 镜像安装（vLLM-Omni）

除 Python 包安装外，我们还提供集成 **vLLM-Omni + MindIE-SD** 的 Docker 镜像，支持在昇腾 NPU 上同时进行多模态大模型推理与 Stable Diffusion 图像生成。镜像基于 `quay.io/ascend/vllm-omni` 基础镜像构建，提供以下两个版本：

| 适用产品 | 镜像 Tag | 基础镜像 |
|--|--|--|
| Atlas 800I A2 推理服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0` |
| Atlas 800I A3 超节点服务器 | `v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64` | `quay.io/ascend/vllm-omni:v0.20.0-a3` |

**获取镜像（二选一）：**

* 从 [MindIE 镜像仓库](https://www.hiascend.com/developer/ascendhub/detail/7c3b1b7c5151469a98ac08b868dab45f) 拉取已构建好的 `mindiesd` 镜像（推荐）。
* 本地构建：克隆仓库后进入 `docker/omni` 目录，使用对应产品的 Dockerfile 构建：

  ```bash
  git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD/docker/omni
  # Atlas 800I A2 推理服务器
  docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a2-ubuntu22.04-py3.11-aarch64 -f Dockerfile.a2.ubuntu .
  # Atlas 800I A3 超节点服务器
  docker build -t mindiesd:v3.0.0-cann8.5.1-torch_npu2.9.0-a3-ubuntu22.04-py3.11-aarch64 -f Dockerfile.a3.ubuntu .
  ```

镜像的运行参数、硬件要求与二次开发等详细说明，请参考 [vLLM-Omni 镜像说明](../../docker/omni/OVERVIEW.zh.md)。
