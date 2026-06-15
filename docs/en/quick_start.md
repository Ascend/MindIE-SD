# Quick Start

This section uses the **Wan2.1** model as an example to demonstrate how to use MindIE SD for text-to-video generation. For more inference details about this model, see [Modelers - MindIE](https://modelers.cn/models/MindIE).

> Before starting inference, complete the environment setup and MindIE SD installation as described in [Installation Guide](./installation.md).

## Model Download and Execution

### 1. Obtain the Inference Script

Clone the Wan2.1 inference script repository from Modelers and install dependencies:

```bash
git clone https://modelers.cn/MindIE/Wan2.1.git && cd Wan2.1
pip install -r requirements.txt
```

### 2. Obtain Model Weights

The repository above contains inference scripts but **does not include model weight files**. Weights must be downloaded separately. Using Wan2.1 as an example, the following models are supported:

| Model | Description | Weight Download |
| ------ | ------ | ---------- |
| Wan2.1-T2V-14B | Text-to-Video | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) |
| Wan2.1-I2V-14B-480P | Image-to-Video (480P) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P) |
| Wan2.1-I2V-14B-720P | Image-to-Video (720P) | [HuggingFace](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P) |

After downloading, the weight directory structure should be as follows (using Wan2.1-T2V-14B as an example):

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

> **Note**
>
> - In addition to HuggingFace, model weights can also be obtained from [modelscope](https://modelscope.cn/models).
> - For weights of other models (FLUX.1-dev, HunyuanVideo, etc.), see the links in [Model/Framework Support Matrix](features/supported_matrix.md).

### 3. Run Inference

Set the weight path in the `model_base` parameter and run the inference script. For detailed parameter explanations, see [Parameter Configuration](../../examples/wan/parameter_config.md).

```bash
# Wan2.1-T2V-14B 8-card inference
cp MindIE-SD/examples/wan/infer_t2v.sh ./
export model_base="/path/to/Wan2.1-T2V-14B"
bash infer_t2v.sh
```

## Acceleration Results

Below, using Wan2.1 as an example, we show the acceleration effects of different features on Atlas 800I A2 inference servers (1*64G) for single-card and multi-card configurations.

Where:

- Cache: Uses the [AttentionCache](./features/cache.md#attentioncache) feature;
- TP: Uses the [Tensor Parallel](./features/parallelism.md) feature;
- FA Sparse: Uses the [RainFusion](./features/sparse.md) feature in FA Sparse;
- CFG: Uses the [CFG Parallel](./features/parallelism.md) feature;
- Ulysses: Uses the [Ulysses Parallel](./features/parallelism.md#ulysses-sequence-parallel) acceleration feature. The generated video resolution is H*W 832*480, with `sample_steps` of 50.

### Single-Card Acceleration

**Cache Acceleration**

| Baseline | + Cache Speedup 1.6 | + Cache Speedup 2.0 | + Cache Speedup 2.4 |
| :---: | :---: | :---: | :---: |
| 860.2s | 631.7s 1.36x | 541.8s 1.59x | 516.9s ***1.66x** |
| ![](../figures/single_card_base_fa.gif) | ![](../figures/single_card_fa_attentioncache_speedup_1_6.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_0.gif) | ![](../figures/single_card_fa_attentioncache_speedup_2_4.gif) |

### Parallel Strategy Results

**Dual-Card Single Parallel Strategy**

| Model | Cards | Parallel Strategy | Video Output Resolution | Operator Optimization | Cache Optimization | FA Sparse | 50-Step E2E Time(s) | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wan2.1 | 2 | VAE | 832*480 | Yes | Yes | Yes | 548.8 | 1.02x |
| Wan2.1 | 2 | TP | 832*480 | Yes | Yes | Yes | 502.8 | 1.12x |
| Wan2.1 | 2 | CFG | 832*480 | Yes | Yes | Yes | 332.6 | 1.69x |
| Wan2.1 | 2 | Ulysses | 832*480 | Yes | Yes | Yes | 327.6 | ***1.71x** |

Note: `*` indicates the best acceleration result.

**Multi-Card Combined Parallel Strategies**

| Model | Cards | Parallel Strategy | Video Output Resolution | Operator Optimization | Cache Optimization | FA Sparse | 50-Step E2E Time(s) | Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wan2.1 | 4 | TP=4, VAE | 832*480 | Yes | Yes | Yes | 204.0 | 2.754x |
| Wan2.1 | 4 | CFG=2, TP=2, VAE | 832*480 | Yes | Yes | Yes | 175.8 | 3.19x |
| Wan2.1 | 4 | Ulysses=4, VAE | 832*480 | Yes | Yes | Yes | 151.1 | 3.71x |
| Wan2.1 | 4 | CFG=2, Ulysses=2, VAE | 832*480 | Yes | Yes | Yes | 147.9 | ***3.79x** |
| Wan2.1 | 8 | TP=8, VAE | 832*480 | Yes | Yes | Yes | 141.5 | 3.96x |
| Wan2.1 | 8 | CFG=2, TP=4, VAE | 832*480 | Yes | Yes | Yes | 102.9 | 5.45x |
| Wan2.1 | 8 | Ulysses=8, VAE | 832*480 | Yes | Yes | Yes | 78.1 | 7.18x |
| Wan2.1 | 8 | CFG=2, Ulysses=4, VAE | 832*480 | Yes | Yes | Yes | 76.4 | ***7.34x** |

Note: `*` indicates the best acceleration result.
