<!-- markdownlint-disable-next-line MD041 -->
<p align="center">
    <img alt="MindIE SD" src="https://raw.gitcode.com/Ascend/MindIE-SD/raw/2169b12abd36eb7f65498de94dd143df6fcfc63f/docs/figures/MindIE-SD-logo-2k.png">
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
</p>

English | [简体中文](./README.md)

## Latest News

- 08/03/2026: Completed [high-performance MiniMax-H3 inference adaptation and optimization on Ascend](examples/minimax-h3/infer.md)
- 07/2026: [RotateAttention](docs/tech_report/RotateAttention.pdf) was accepted to ECCV 2026
- 06/2026: Enhanced low-precision inference, multimodal MoE, sparse attention, and the deployment ecosystem
- 05/28/2026: [Cache-DiT completed its MindIE-SD integration](https://github.com/vipshop/cache-dit/pull/1004)
- 05/2026: Completed multimodal MoE inference capability development and multi-model adaptation validation
- 04/2026: Enhanced graph compilation, operator fusion, and sparse computation capabilities
- 03/2026: Extended W4A4 and MXFP4 quantization capabilities
- 02/2026: Added sparse attention operators and Dynamic EPLB
- 01/2026: Added memory optimization capabilities and serving examples
- 12/31/2025: MindIE SD provides sparse Attention computation capability
- 12/25/2025: vLLM Omni x MindIE SD achieves native high-performance Ascend inference for Qwen-Image-Edit-2511 / Qwen-Image-Layered
- 11/30/2025: MindIE SD officially open-sourced and available to the public!

## Introduction

**MindIE SD** (Mind Inference Engine Stable Diffusion) is the visual generation inference model suite of MindIE. Its goal is to provide an end-to-end inference solution for Stable Diffusion (SD) series large models on Ascend hardware and its software stack. The software system internally integrates various functional modules and provides a unified programming interface externally.

Below are two **AI Agents** for the MindIE-SD code repositories. Simply click the "**Ask AI**" badge to enter their dedicated pages, effectively alleviating the difficulty of source code reading and starting an intelligent code learning and Q&A experience! They will help you understand the operating principles of MindIE-SD more deeply and assist in resolving issues and errors encountered during use.

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/verylucky01/MindIE-SD)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/verylucky01/MindIE-SD)

</div>

## Installation Guide

**Quick Start**:

```bash
pip install mindiesd
```

For other installation methods and environment dependencies, see [Installation Guide](docs/zh/installation.md).

## Architecture Overview

For MindIE SD architecture and key features, see [Architecture Overview](docs/zh/architecture.md).

![MindIE SD Architecture Diagram](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/architecture_overview.png)

## Quick Start

[Quick Start](docs/zh/quick_start.md): Using Wan2.1 as an example, introduces the overall acceleration effect with MindIE SD.

<a id="wechat-community"></a>

## Contact Us

Join the MindIE-SD WeChat community by scanning the group QR code below. If the group QR code has expired, scan the personal WeChat QR code to request an invitation.

<div align="center">

| WeChat Group | Personal WeChat (if the group QR code has expired) |
| --- | --- |
| ![WeChat Group QR Code](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/wx_sd.jpg) | ![Personal WeChat QR Code](https://raw.gitcode.com/Ascend/MindIE-SD/raw/dev/docs/figures/wx_lwl.jpg) |

</div>

## Developer Documentation

- [Contribution Guide](docs/zh/developer_guide/contribution_guide.md): Explains how to submit Issues, Pull Requests, and coding standards.
- [AI Agent Support](.agents/README.md): Introduces the default skills used in the repository and the agent-assisted development workflow.
- [Test Verification](docs/zh/developer_guide/test.md): Introduces the unit test example execution workflow.

## Ecosystem Support

The following projects have integrated MindIE SD as a multimodal inference backend:

- [vLLM Omni](https://github.com/vllm-project/vllm-omni):
- [Cache-DiT](https://github.com/vipshop/cache-dit):

## Paper Citations

```text
@misc{RainFusion2.0@2025,
    title = {RainFusion2.0: Temporal-Spatial Awareness and Hardware-Efficient Block-wise Sparse Attention},
    url = {https://gitcode.com/Ascend/MindIE-SD.git},
    note = {Open-source software available at https://gitcode.com/Ascend/MindIE-SD.git},
    author = {Aiyue Chen and others},
    year = {2025}
    }
```
