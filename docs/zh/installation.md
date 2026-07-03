# 安装指导

## Python包安装

MindIE SD是一个 Python 包，它基于PyTorch构建，可以轻松集成到 Python 应用程序中。

### 安装依赖

* OS: Linux
* Python: >=3.10
* Pytorch：2.6, 2.7, 2.8, 2.9, 2.10
* torch-npu: 2.6, 2.7, 2.8, 2.9, 2.10
* CANN: 9.0.0
* triton: 3.5.0
* triton-ascend: 3.2.1

> **说明**：`triton-ascend 3.2.1` 不在 PyPI 发布，请从 [GitCode 发布页](https://gitcode.com/Ascend/triton-ascend/releases) 下载与 Python 版本/架构匹配的 wheel 安装；该 wheel 会锁定 `triton==3.5.0`。未安装 3.2.1 时 SparseLinearAttention（FA 稀疏）特性会被禁用。

#### CANN 安装

MindIE SD 依赖 CANN Toolkit开发套件包和 CANN ops 算子包，请参考 [CANN 软件安装指南](https://gitcode.com/cann/ops-cv/blob/master/docs/zh/install/quick_install.md) 完成安装。根据安装方式、操作系统等选择安装场景，选择完成后单击“开始阅读”，按“安装CANN”章节进行安装。

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

现在最简单的方式是通过pip源安装，我们的软件包名字叫mindiesd，与仓库名有些不一样。

```bash
pip install mindiesd
```

### 源码安装

在某些情况下，您可能需要从源代码安装 MindIE SD，以便尝试最新功能，或者根据您的特定需求自定义库。

您可以按照以下步骤从源代码安装 MindIE SD：

1. 克隆仓库&进入项目：

   ```bash
   git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
   ```

2. [可选] 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

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
> - `requirements.txt`：核心运行依赖（最小安装，仅 torch/torch_npu 钉版本，其余宽松）。
> - `examples/dummy_run/requirements.txt`：`dummy_run` 模型推理示例依赖（diffusers/transformers 等）。
> - `examples/service/requirements.txt`：服务化示例依赖（ray、fastapi、uvicorn、pydantic、Pillow）。
> - 测试、Lint、文档构建依赖分别见 `requirements-test.txt`、`requirements-lint.txt`、`docs/requirements-docs.txt`（详见开发者指南）。

### 每日构建安装

每日构建版本可供测试最新功能：

待提供...
