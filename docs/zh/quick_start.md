# 快速开始

本章节以 **Wan2.1** 模型为例，展示如何使用 MindIE SD 进行文本生成视频，关于该模型的更多推理内容请参见 [Modelers - MindIE](https://modelers.cn/models/MindIE/Wan2.1)。

> 开始推理前，请先按 [安装指导](./installation.md) 完成环境准备和 MindIE SD 安装。

## 模型下载与运行

### 1. 获取推理脚本

从魔乐社区克隆 Wan2.1 推理脚本仓库，并安装依赖：

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt
```

### 2. 获取模型权重

上述仓库包含推理脚本，**不包含模型权重文件**。权重需要单独下载，以 Wan2.1 为例，支持以下模型：

| 模型 | 说明 | 权重下载 |
|------|------|----------|
| Wan2.1-T2V-14B | 文生视频 | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-I2V-14B-480P | 图生视频（480P） | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1-I2V-14B-720P | 图生视频（720P） | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |

下载完成后，权重目录结构应如下（以 Wan2.1-T2V-14B 为例）：

```text
Wan2.1-T2V-14B/
├── config.json
├── model_index.json
├── models/
│   ├── dit/
│   ├── vae/
│   └── text_encoder/
└── ...
```

> **说明**
>
> - 除 HuggingFace 外，也可从 [modelscope](https://modelscope.cn/models) 获取模型权重。
> - 如需下载其他模型的权重（FLUX.1-dev、HunyuanVideo 等），请参见[模型/框架支持情况](features/supported_matrix.md)中的链接。

### 3. 运行推理

将权重路径设置到 `model_base` 参数，运行推理脚本。参数解释详情请参见[参数配置](../../examples/wan/parameter_config.md)。

```bash
# Wan2.1-T2V-14B 8 卡推理
cp MindIE-SD/examples/wan/infer_t2v.sh ./
export model_base="/path/to/Wan2.1-T2V-14B"
bash infer_t2v.sh
```

## 加速效果展示

下面以 Wan2.1 模型为例，展示在 Atlas 800I A2 推理服务器（1*64G）上单卡和多卡实现不同加速特性的加速效果。

其中：

- Cache：表示使用[AttentionCache](./features/cache.md#attentioncache)特性；
- TP：表示使用[Tensor Parallel](./features/parallelism.md)特性；
- FA 稀疏：表示使用 FA 稀疏中的[RainFusion 特性](./features/sparse.md)；
- CFG：表示使用[CFG 并行](./features/parallelism.md#cfg-parallel)特性；
- Ulysses：表示使用[Ulysses 并行](./features/parallelism.md#ulysses-sequence-parallel)加速特性，模型生成的视频的 H*W 为 832*480，`sample_steps` 为 50。

### 单卡加速效果

**Cache 加速效果**

| Baseline | + Cache 加速比1.6 | + Cache 加速比2.0 | + Cache 加速比2.4 |
|:---:|:---:|:---:|:---:|
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](../figures/single_card_base_fa.gif) | ![](../figures/single_card_fa_attentioncache_speedup_1_6.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_0.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_4.gif) |

### 并行策略效果

**双卡单个并行策略效果**

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化 | FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 2 | VAE | 832*480 | √ | √ | √ | 548.8 | 1.02x |
| Wan2.1 | 2 | TP | 832*480 | √ | √ | √ | 502.8 | 1.12x |
| Wan2.1 | 2 | CFG | 832*480 | √ | √ | √ | 332.6 | 1.69x |
| Wan2.1 | 2 | Ulysses | 832*480 | √ | √ | √ | 327.6 | ***1.71x** |

注：`*` 表示最优加速效果。

**多卡并行策略组合效果**

| 模型 | 卡数 | 并行策略 | 视频输出分辨率 | 算子优化 | cache 算法优化 | FA 稀疏 | 50 步 E2E 耗时(s) | 加速比 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 4 | TP=4, VAE | 832*480 | √ | √ | √ | 204.0 | 2.754x |
| Wan2.1 | 4 | CFG=2, TP=2, VAE | 832*480 | √ | √ | √ | 175.8 | 3.19x |
| Wan2.1 | 4 | Ulysses=4, VAE | 832*480 | √ | √ | √ | 151.1 | 3.71x |
| Wan2.1 | 4 | CFG=2, Ulysses=2, VAE | 832*480 | √ | √ | √ | 147.9 | ***3.79x** |
| Wan2.1 | 8 | TP=8, VAE | 832*480 | √ | √ | √ | 141.5 | 3.96x |
| Wan2.1 | 8 | CFG=2, TP=4, VAE | 832*480 | √ | √ | √ | 102.9 | 5.45x |
| Wan2.1 | 8 | Ulysses=8, VAE | 832*480 | √ | √ | √ | 78.1 | 7.18x |
| Wan2.1 | 8 | CFG=2, Ulysses=4, VAE | 832*480 | √ | √ | √ | 76.4 | ***7.34x** |

注：`*` 表示最优加速效果。
