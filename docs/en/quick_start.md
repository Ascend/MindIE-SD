# Quick Start

This page uses **Wan2.1** as an example to show how to run text-to-video inference with MindIE SD. For more model-specific inference details, see [Modelers - MindIE/Wan2.1](https://modelers.cn/models/MindIE/Wan2.1).

## Prerequisites

Before running inference, complete the environment preparation and install MindIE SD by following the [Installation Guide](installing_guide.md).

## Run inference

Install the model-specific dependencies and then run inference.

Clone the Wan2.1 model repository anywhere, install its requirements, and run the inference script from the MindIE SD workspace. Adjust the weight path as needed, for example `/home/{user}/Wan2.1-T2V-14B`. Parameter details are documented in [parameter_config.md](../../examples/wan/parameter_config.md).

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt

# 8-card inference for Wan2.1-T2V-14B
cp MindIE-SD/examples/wan/infer_t2v.sh ./
bash infer_t2v.sh --model_base="/home/{user}/Wan2.1-T2V-14B"
```

## Acceleration results

The following Wan2.1 example shows the effect of different acceleration features on an Atlas 800I A2 inference server (1*64G), including both single-card and multi-card runs.

Where:

- Cache refers to the [AttentionCache](features/cache.md#attentioncache) feature.
- TP refers to the [Tensor Parallel](features/parallelism.md#tensor-parallel) feature.
- FA sparse refers to the [RainFusion](features/sparse_quantization.md#fa-sparsity) optimization under FA sparsity.
- CFG refers to the [CFG Parallel](features/parallelism.md#cfg-parallel) feature.
- Ulysses refers to the [Ulysses Sequence Parallel](features/parallelism.md#ulysses-sequence-parallel) feature. The generated video resolution is 832*480 and `sample_steps` is 50.

### Single-card acceleration

**Cache acceleration**

| Baseline | + Cache ratio 1.6 | + Cache ratio 2.0 | + Cache ratio 2.4 |
|:---:|:---:|:---:|:---:|
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](../zh/figures/单卡base%20+%20高性能FA算子.gif) | ![](../zh/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为1.6.gif) | ![](../zh/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.0.gif) | ![](../zh/figures/单卡%20+%20高性能FA算子%20+%20开启attentioncache+加速比为2.4.gif) |

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
