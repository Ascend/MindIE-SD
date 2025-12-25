# README

## 简介

**MindIE SD**（Mind Inference Engine Stable Diffusion）是 MindIE 的视图生成推理模型套件，它的目标是为稳定扩散（**Stable Diffusion**, SD）系列大模型提供在昇腾硬件及其软件栈上的端到端推理解决方案。该软件系统内部集成了各功能模块，并对外提供统一的编程接口。

以下是两个 MindIE-SD 代码仓库**智能体**，只需点击 "**Ask AI**" 徽章，即可进入其专属页面，有效缓解源码阅读的困难，开启智能代码学习与问答体验！它们将帮助您更深入地理解 MindIE-SD 的运行原理，并协助解决使用过程中遇到的问题与错误。

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-SD)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-SD)

</div>

## Latest News

-   12/25/2025：vLLM Omni x MindIE SD 实现 Qwen-Image-Edit-2511 / Qwen-Image-Layered 昇腾原生高性能推理
-   11/30/2025：MindIE SD 正式宣布开源并面向公众开放！[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-SD)

## 架构介绍及关键特性

详见[架构介绍](docs/architecture.md)（包含：关键特性，目录设计等）

现已支持主流扩散模型，对于部分 diffusers 模型进行了昇腾硬件亲和的加速改造，模型归档在 [Modelers](https://modelers.cn/models?name=MindIE&page=1&size=16)/[ModelZoo](https://www.hiascend.com/software/modelzoo)，模型列表详见 [List of Supported Models](docs/architecture.md/table1198934616167)，也支持手动改造，详见 examples。


## 快速开始

本章节以 **Wan2.1** 模型为例，展示如何使用 MindIE SD 进行文本生成视频，关于该模型的更多推理内容请参见 [Modelers - MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1)。

1.  源码编译安装 MindIE SD（镜像 / 软件包安装方式详见 [developer_guide](docs/developer_guide.md)）
    ```bash
    git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD
    python setup.py bdist_wheel

    cd dist
    pip install mindiesd-*.whl
    ```

2.  安装模型所需依赖并执行推理。

    在任意路径下载 Wan2.1 模型仓，并安装所需依赖，进行推理。用户可根据需要自行设置权重路径（例：/home/{用户名}/Wan2.1-T2V-14B）和推理脚本中的模型参数，参数解释详情请参见[参数配置](./examples/wan/parameter_config.md)。

    ```bash
    git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
    pip install -r requirements.txt

    # Wan2.1-T2V-14B 8 卡推理
    bash examples/wan/infer_t2v.sh --model_base="/home/{用户名}/Wan2.1-T2V-14B"
    ```

## 加速特性效果展示

下面以 Wan2.1 模型为例，展示在 Atlas 800I A2 (1\*64G) 机器上单卡和多卡实现不同加速特性的加速效果。

其中Cache表示使用[AttentionCache](./docs/features/cache.md#attentioncache)特性, TP表示使用[Tensor Parallel](./docs/features/parallelism.md#张量并行)特性, FA稀疏表示使用FA稀疏中的[RainFusion特性](./docs/features/sparse_quantization.md#fa稀疏)，CFG表示使用[CFG并行](./docs/features/parallelism.md#cfg并行)特性，Ulysses表示使用[Ulysses并行](./docs/features/parallelism.md#ulysses-sequence-parallel)加速特性，模型生成的视频的H\*W为832\*480, sample_steps为50。

### 单卡加速效果

#### cache 加速效果

| Baseline | + Cache 加速比1.6 | + Cache 加速比2.0 | + Cache 加速比2.4 |
|:---:|:---:|:---:|:---:|
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](./docs/figures/单卡base%20+%20高性能FA算子.gif) | ![](./docs/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为1.6.gif) | ![](./docs/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.0.gif) | ![](./docs/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.4.gif) |

### 并行策略效果

#### 双卡单个并行策略效果

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化| FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 2 | VAE | 832\*480 | √ | √ | √ | 548.8 | 1.02x|
| Wan2.1 | 2 | TP | 832\*480 | √ | √ | √ | 502.8 | 1.12x|
| Wan2.1 | 2 | CFG | 832\*480 | √ | √ | √ | 332.6 | 1.69x|
| Wan2.1 | 2 | Ulysses | 832\*480 | √ | √ | √ | 327.6 | ***1.71x**|

注：\* 号表示最优加速效果

#### 多卡并行策略组合效果

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化| FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 4 | TP=4, VAE | 832\*480 | √ | √ | √ | 204.0 | 2.754x|
| Wan2.1 | 4 | CFG=2, TP=2, VAE | 832\*480 | √ | √ | √ | 175.8 | 3.19x|
| Wan2.1 | 4 | Ulysses=4, VAE | 832\*480 | √ | √ | √ | 151.1 | 3.71x|
| Wan2.1 | 4 | CFG=2, Ulysses=2, VAE | 832\*480 | √ | √ | √ | 147.9 | ***3.79x**|
| Wan2.1 | 8 | TP=8, VAE | 832\*480 | √ | √ | √ | 141.5| 3.96x|
| Wan2.1 | 8 | CFG=2, TP=4, VAE | 832\*480 | √ | √ | √ | 102.9 | 5.45x|
| Wan2.1 | 8 | Ulysses=8, VAE | 832\*480 | √ | √ | √ | 78.1 | 7.18x|
| Wan2.1 | 8 | CFG=2, Ulysses=4, VAE | 832\*480 | √ | √ | √ | 76.4 | ***7.34x**|

注：\* 号表示最优加速效果

## 联系我们
![](./docs/figures/contact-us.jpg)
