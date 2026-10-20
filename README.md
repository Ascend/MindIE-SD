<!-- markdownlint-disable-next-line MD041 -->
<p align="center">
    <img alt="MindIE SD" src="docs/figures/MindIE-SD-logo-ascend-red-square-dot-v5.png">
</p>

<p align="center">
    <a href="https://mindie-sd.readthedocs.io/en/latest/zh/">
        <img alt="Documentation" src="https://img.shields.io/badge/Docs-Read%20the%20Docs-8CA1AF">
    </a>
    <a href="https://pypi.org/project/mindiesd/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/mindiesd?logo=pypi&logoColor=white">
    </a>
    <a href="#wechat-community">
        <img alt="Community QR Code" src="https://img.shields.io/badge/Community-WeChat-07C160">
    </a>
    <a href="./LICENSE.md">
        <img alt="License" src="https://img.shields.io/badge/License-Mulan-blue">
    </a>
    <a href="https://meeting.ascend.osinfra.cn/?sig=sig-MindIE-SD">
        <img alt="SIG Meetings" src="https://img.shields.io/badge/Meetings-SIG-0A7B83">
    </a>
    <a href="https://zread.ai/verylucky01/MindIE-SD">
        <img alt="Zread" src="https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff">
    </a>
    <a href="https://deepwiki.com/verylucky01/MindIE-SD">
        <img alt="DeepWiki" src="https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==">
    </a>
</p>

简体中文 | [English](./README.en.md)

## 🚀 简介

**MindIE SD**（Mind Inference Engine Stable Diffusion）是 MindIE 的视图生成推理模型套件，它的目标是为稳定扩散（**Stable Diffusion**, SD）系列大模型提供在昇腾硬件及其软件栈上的端到端推理解决方案。该软件系统内部集成了各功能模块，并对外提供统一的编程接口。

## 📢 Latest News

- 09/23/2026: [MXAttention 论文](docs/tech_report/MXAttention.pdf) 已投稿 ICLR（[arXiv:2607.24377](https://arxiv.org/abs/2607.24377)）📄
- 09/20/2026: 完成 [Qwen-Image-2.1 昇腾高性能推理适配与优化](examples/qwen-image-2.1/infer.md) 🚀
- 09/04/2026: MindIE SD支持Lightx2v框架实现MiniMax H3在Ascend 950PR&950DT系列产品上的加速
- 08/03/2026: 完成 [MiniMax-H3 昇腾高性能推理适配与优化](examples/minimax-h3/infer.md) 🚀
- 07/2026: [RotateAttention](docs/tech_report/RotateAttention.pdf) 已被 ECCV 2026 录用 🏆
- 06/2026: 完成低精度推理、多模态 MoE、稀疏注意力和部署生态增强 ⚡
- 05/28/2026: [Cache-DiT 完成 MindIE SD 接入](https://github.com/vipshop/cache-dit/pull/1004) 🤝
- 05/2026: 完成多模态 MoE 推理能力建设和多模型适配验证 🧠
- 04/2026: 完成图编译、算子融合与稀疏计算能力增强 ⚙️
- 03/2026: 扩展 W4A4 与 MXFP4 量化能力 ⚡
- 02/2026: 新增稀疏注意力算子与 Dynamic EPLB 🧩
- 01/2026: 新增内存优化能力与服务化样例 💾
- 12/31/2025: MindIE SD提供稀疏Attention计算能力 ⚡
- 12/25/2025：vLLM Omni x MindIE SD 实现 Qwen-Image-Edit-2511 / Qwen-Image-Layered 昇腾原生高性能推理 🚀
- 11/30/2025：MindIE SD 正式宣布开源并面向公众开放！ 🎉

## 📦 安装指南

```bash
pip install mindiesd
```

其他安装方式和环境依赖详见[安装指导](docs/zh/installation.md)。

## 🚀 架构介绍

MindIE SD 架构和关键特性详见[架构介绍](docs/zh/architecture.md)。

![MindIE SD架构图](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/architecture_overview.png)

## ⚡️ 快速入门

[快速开始](docs/zh/quick_start.md): 以wan2.1为例，介绍使用mindiesd后的整体加速效果。

## 📚 开发文档

- [贡献指南](docs/zh/developer_guide/contribution_guide.md)：说明如何提交 Issue、Pull Request 以及代码规范。
- [AI Agent 支持](.agents/README.md)：介绍仓内默认使用的一些 skills 及使用 Agent 辅助开发流程。
- [测试验证](docs/zh/developer_guide/test.md)：介绍单元测试样例运行流程。

## 🌐 生态支持

以下项目已集成MindIE SD作为多模态推理后端：

- [vLLM Omni](https://github.com/vllm-project/vllm-omni)
- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [Lightx2v](https://github.com/ModelTC/lightx2v)

## 🙏 Acknowledgment

感谢 [CANN ops-transformer](https://gitcode.com/cann/ops-transformer) 和 [CANN ops-nn](https://gitcode.com/cann/ops-nn) 社区提供的算子源码。

- **迁移与多模态性能优化**：本仓的 `QuantFlashAttn`、`EagleFusedInferAttentionScore`、`EagleBlockSparseAttention` 和 `EagleFfn` 分别迁移自 CANN 的 [QuantFlashAttn](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_flash_attn)、[FusedInferAttentionScore](https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_infer_attention_score)、[BlockSparseAttention](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_sparse_attention) 和 [FFN](https://gitcode.com/cann/ops-transformer/tree/master/ffn/ffn)，并针对多模态推理场景进行性能优化。
- **基于 CANN 算子开发**：`EagleQuantBlockSparseAttention` 基于 CANN [BlockSparseAttention](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_sparse_attention) 开发混合精度块稀疏注意力；`quant_four_over_six_a5` 基于 CANN [动态 MX 量化](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_mx_quant) 基础代码开发 Four Over Six 量化扩展。

感谢以下贡献者在相关 CANN 算子上的工作：[cardiac_index](https://gitcode.com/cardiac_index), [XiyueL](https://gitcode.com/XiyueL), [jiang-lirui](https://gitcode.com/jiang-lirui), [monologue815](https://gitcode.com/monologue815), [chenyizhou](https://gitcode.com/chenyizhou), [liwenguihw](https://gitcode.com/liwenguihw)。

<a id="paper-citations"></a>

## 📝 Paper Citations

<pre><details><summary>Show more
<code>@misc{yu2026mxattentiondatafreeoptimalscaling,
      title={MXAttention: Data-Free Optimal Scaling and Pre-Normalization Quantization for MXFP4 Attention},
      author={Jianlin Yu and Jing Lin and Linghui Kong and Aiyue Chen and Weiyi Sun and Chenyu Zeng and Wangli Lan and Jinxi Li and Zhuo Zheng and Ziyang Yue and Danning Ke and Fei Yi and Tianchi Hu and Yuan Ding and Yiwu Yao and Junsong Wang},
      year={2026},
      eprint={2607.24377},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.24377},
}

@misc{liu2026rotateattentionropeawarerotationrange,
      title={RotateAttention: RoPE-Aware Rotation and Range Rectification for INT4 Quantized Attention in Video Generation},
      author={Yaofu Liu and Wanli Lan and Jinxi Li and Binhang Yuan and Harry Yang},
      year={2026},
      eprint={2607.02584},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.02584},
}</code></summary><code>

@misc{RainFusion2.0@2025,
    title = {RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention},
    author = {Aiyue Chen and Yaofu Liu and Junjian Huang and Guang Lian and Yiwu Yao and Wangli Lan and Jing Lin and Zhixin Ma and Tingting Zhou and Harry Yang},
    year = {2025},
    url = {https://gitcode.com/Ascend/MindIE-SD/blob/dev/docs/tech_report/RainFusion2.0.pdf}
}

@misc{zhang2025slasparsitydiffusiontransformers,
      title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
      author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
      year={2025},
      eprint={2509.24006},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.24006},
}

@misc{zhang2025sageattentionaccurate8bitattention,
      title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration},
      author={Jintao Zhang and Jia Wei and Haofeng Huang and Pengle Zhang and Jun Zhu and Jianfei Chen},
      year={2025},
      eprint={2410.02367},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02367},
}

@misc{ashkboos2024quarotoutlierfree4bitinference,
      title={QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs},
      author={Saleh Ashkboos and Amirkeivan Mohtashami and Maximilian L. Croci and Bo Li and Pashmina Cameron and Martin Jaggi and Dan Alistarh and Torsten Hoefler and James Hensman},
      year={2024},
      eprint={2404.00456},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2404.00456},
}

@misc{cook2026sixaccuratenvfp4quantization,
      title={Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling},
      author={Jack Cook and Junxian Guo and Guangxuan Xiao and Yujun Lin and Keith Wyss and Mahdi Nazemi and Asit Mishra and Carlo del Mundo and Tijmen Blankevoort and Song Han},
      year={2026},
      eprint={2512.02010},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.02010},
}</code></details></pre>

<a id="wechat-community"></a>

## 💬 联系我们

欢迎通过 [GitCode Issues](https://gitcode.com/Ascend/MindIE-SD/issues) 直接参与讨论与反馈。

为了更便捷地交流并获得更快的响应，我们建立了 MindIE SD 微信交流群。您可以扫描左侧的微信群二维码加入交流群；若群二维码已失效，请扫描右侧的个人微信二维码，添加好友后由其邀请您加入交流群。

<div align="center">

| 微信交流群 | 个人微信（群二维码失效时添加） |
| --- | --- |
| ![微信群聊二维码](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/wx_sd.jpg) | ![个人微信二维码](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/wx_lwl.jpg) |

</div>
