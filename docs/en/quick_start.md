# Quick Start

This page uses **Wan2.1** as an example to show how to run text-to-video inference with MindIE SD. For more model-specific inference details, see [Modelers - MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1).

## Prerequisites

Before running inference, complete the environment preparation and install MindIE SD by following the [Installation Guide](installation.md).

## Downloading and Running a Model

### 1. Obtaining the Inference Script

Clone the Wan2.1 inference script repository from the Modelers community and install the dependency.

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt
```

### 2. Downloading Model Weights

The preceding repository contains the inference script but **does not contain the model weight file**. The weights need to be downloaded separately. The following models are supported (Example: Wan2.1):

| Model| Description| Weight Download|
|------|------|----------|
| Wan2.1-T2V-14B | Text-to-video| [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-I2V-14B-480P | Image-to-video (480p)| [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1-I2V-14B-720P | Image-to-video (720p)| [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |

After the download is complete, the weight directory structure is as follows (using Wan2.1-T2V-14B as an example):

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

> **NOTE**
>
> - In addition to HuggingFace, you can also obtain the model weights from [ModelScope](https://modelscope.cn/models).
> - To download the weights of other models (such as FLUX.1-dev and HunyuanVideo), click the links in [Model/Framework Support](features/supported_matrix.md).

### 3. Running Inference

Set the weight path to the `model_base` parameter and run the inference script. For details about the parameters, see [Parameter Configuration](../../examples/wan/parameter_config.md).

```bash
# Wan2.1-T2V-14B 8-device inference
cp MindIE-SD/examples/wan/infer_t2v.sh ./
export model_base="/path/to/Wan2.1-T2V-14B"
bash infer_t2v.sh
```

## Acceleration results

The following Wan2.1 example shows the effect of different acceleration features on an Atlas 800I A2 inference server (1*64GB), including both single-card and multi-card runs.

Where:

- Cache refers to the [AttentionCache](./features/cache.md#attentioncache) feature.
- TP refers to the [Tensor Parallel](./features/parallelism.md) feature.
- FA sparse refers to the [RainFusion](./features/sparse.md) optimization under FA sparsity.
- CFG refers to the [CFG Parallel](features/parallelism.md#cfg-parallel) feature.
- Ulysses refers to the [Ulysses Sequence Parallel](./features/parallelism.md#USP) feature. The generated video resolution is 832*480 and `sample_steps` is 50.

### Single-card acceleration

**Cache acceleration**

| Baseline | + Cache ratio 1.6 | + Cache ratio 2.0 | + Cache ratio 2.4 |
|:---:|:---:|:---:|:---:|
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](../figures/single_card_base_fa.gif) | ![](../figures/single_card_fa_attentioncache_speedup_1_6.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_0.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_4.gif) |

### Parallel strategy results

**Two-card single-strategy results**

| Model | Cards | Parallel strategy | Output resolution | Operator optimization | Cache optimization | FA sparse | 50-step E2E time (s) | Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 2 | VAE | 832*480 | √ | √ | √ | 548.8 | 1.02x |
| Wan2.1 | 2 | TP | 832*480 | √ | √ | √ | 502.8 | 1.12x |
| Wan2.1 | 2 | CFG | 832*480 | √ | √ | √ | 332.6 | 1.69x |
| Wan2.1 | 2 | Ulysses | 832*480 | √ | √ | √ | 327.6 | ***1.71x** |

Note: `*` marks the best acceleration result.

**Multi-card combined-strategy results**

| Model | Cards | Parallel strategy | Output resolution | Operator optimization | Cache optimization | FA sparse | 50-step E2E time (s) | Speedup |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Wan2.1 | 4 | TP=4, VAE | 832*480 | √ | √ | √ | 204.0 | 2.754x |
| Wan2.1 | 4 | CFG=2, TP=2, VAE | 832*480 | √ | √ | √ | 175.8 | 3.19x |
| Wan2.1 | 4 | Ulysses=4, VAE | 832*480 | √ | √ | √ | 151.1 | 3.71x |
| Wan2.1 | 4 | CFG=2, Ulysses=2, VAE | 832*480 | √ | √ | √ | 147.9 | ***3.79x** |
| Wan2.1 | 8 | TP=8, VAE | 832*480 | √ | √ | √ | 141.5 | 3.96x |
| Wan2.1 | 8 | CFG=2, TP=4, VAE | 832*480 | √ | √ | √ | 102.9 | 5.45x |
| Wan2.1 | 8 | Ulysses=8, VAE | 832*480 | √ | √ | √ | 78.1 | 7.18x |
| Wan2.1 | 8 | CFG=2, Ulysses=4, VAE | 832*480 | √ | √ | √ | 76.4 | ***7.34x** |

Note: `*` marks the best acceleration result.
