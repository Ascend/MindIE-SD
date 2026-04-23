# MindIE SD

## 📢 Latest News

- 12/31/2025: MindIE SD提供稀疏Attention计算能力
- 12/25/2025：vLLM Omni x MindIE SD 实现 Qwen-Image-Edit-2511 / Qwen-Image-Layered 昇腾原生高性能推理
- 11/30/2025：MindIE SD 正式宣布开源并面向公众开放！[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-SD)

## 🚀 简介

**MindIE SD**（Mind Inference Engine Stable Diffusion）是 MindIE 的视图生成推理模型套件，它的目标是为稳定扩散（**Stable Diffusion**, SD）系列大模型提供在昇腾硬件及其软件栈上的端到端推理解决方案。该软件系统内部集成了各功能模块，并对外提供统一的编程接口。

以下是两个 MindIE-SD 代码仓库**智能体**，只需点击 "**Ask AI**" 徽章，即可进入其专属页面，有效缓解源码阅读的困难，开启智能代码学习与问答体验！它们将帮助您更深入地理解 MindIE-SD 的运行原理，并协助解决使用过程中遇到的问题与错误。

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-SD)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-SD)

</div>

## 🔍目录结构

```txt
|- benchmarks         // 提供核心kernel的性能看护和compilation的加速效果看护
|- build              // 编译脚本
|- csrc               // 昇腾kernel代码位置
|- docs               // 项目文档
|- examples
  |- cache            // cache特性样例：使能cache进行模型加速
  |- service          // 服务化样例：将命令行模式改造成服务化方式
  |- wan              // 模型推理样例：模型推理命令以及参数配置
|- mindiesd
  |- cache_agent      // 高阶特性：提供cache能力
  |- compilation      // 提供编译能力，基于fx graph实现自动改图（可依旧保持单算子下发）。
  |- eplb             // 高阶特性：提供专家并行负载均衡能力
  |- layers           // 提供基础的pytorch的layer接口
  |- quantization     // 高阶特性：提供量化能力
  |- utils            // 核心工具模块，提供共享的基础设施服务和通用功能
|- tests              // 测试用例
```

## 🚀 架构介绍及关键特性

MindIE SD 架构和关键特性详见[架构介绍](docs/zh/architecture.md)。
MindIE SD 支持[魔乐社区](https://modelers.cn/models?name=MindIE&page=1&size=16)、vLLM Omni、Cache Dit等框架/社区，现已支持主流扩散模型，对于部分 diffusers 模型进行了昇腾硬件亲和的加速改造，详见[模型/框架支持情况](docs/zh/features/supported_matrix.md)，模型也支持手动改造，详见 examples。

## 📝 Paper Citations

```txt
@misc{RainFusion2.0@2025,
    title = {RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention},
    url = {https://gitcode.com/Ascend/MindIE-SD.git},
    note = {Open-source software available at https://gitcode.com/Ascend/MindIE-SD.git},
    author = {Aiyue Chen and others},
    year = {2025}
    }
```

## 💖 联系我们

![](./docs/zh/figures/contact-us.jpg)
![](./docs/zh/figures/contact-us-MindIESD.jpg)
