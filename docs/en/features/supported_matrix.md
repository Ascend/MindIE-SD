# Model/Framework Support

Currently, MindIE SD supports the vLLM Omni framework, the vLLM-Ascend framework, and the Modelers community, among others. In theory, MindIE SD supports inference acceleration for any multimodal model. This page only lists the feature stacking status for our supported typical models.

> [!NOTE] Note
>
> - **A2**: Atlas 800I A2 inference server, default to 313T compute and 64 GB memory.
> - **A3**: Atlas 800I A3 SuperPod server, default to 560T compute and 64 GB memory.
> - **A5**: 950PR&950DT products.
> - **300I DUO**: Atlas 300I DUO Inference Card, default to 140T (FP16) compute and 48 GB memory.

## Model Support

| Model | Model Type | Hardware | vLLM Omni | vLLM-Ascend | Modelers |
| :--------- | :--------- | :--------- | :-----: | :-----: | :-----: |
| MiniMax H3 | Audio-video generation | A2/A3/A5 | Yes | No | No |
| Wan2.2 | Video generation | A2/A3/A5 | Yes | No | Yes |
| HunyuanVideo-1.5 | Video generation | A2/A3/A5 | No | No | Yes |
| HunyuanVideo | Video generation | A2/A3/A5 | No | No | Yes |
| Qwen-Image-2.1 | Image generation/editing | A2/A3/A5 | Yes | No | No |
| Qwen Image Edit Plus 2511 | Image editing | A2/A3/A5 | Yes | No | Yes |
| Qwen Image Edit Plus 2509 | Image editing | A2/A3/A5 | Yes | No | Yes |
| Qwen Image Edit | Image editing | A2/A3/A5 | Yes | No | Yes |
| Qwen-Image | Image generation | A2/A3/A5 | Yes | No | Yes |
| HunyuanImage 3.0 | Image generation/editing/COT | A2/A3/A5 | Yes | No | No |
| FLUX.1-dev | Image generation | A2/A3/A5 | Yes | No | Yes |
| Qwen3-Omni | Omni-modal | A2/A3/A5 | Yes | No | No |
| Qwen3-Omni-Thinking | Omni-modal | A2/A3/A5 | No | Yes | No |
| Qwen2.5-Omni | Omni-modal | A2/A3/A5 | Yes | No | No |
| Qwen3-TTS | Speech synthesis | A2/A3/A5 | Yes | No | No |
| CosyVoice3 | Speech synthesis | A2/A3/A5 | Yes | No | No |
| Qwen3-ASR | Speech recognition | A2/A3/A5 | No | Yes | No |
| Cosmos3 | World model/COT | A2/A3/A5 | Yes | No | No |
| Z-Image | Image generation | A2/A3 | No | No | Yes |
| Z-Image Turbo | Image generation | A2/A3 | Yes | No | Yes |
| FLUX.2-dev | Image generation/editing | A2/A3 | No | No | Yes |
| Hunyuan3D-2.1 | 3D generation | A2/A3 | No | No | Yes |

> [!NOTE] Note
>
> vLLM Omni, vLLM-Ascend, and Modelers each maintain the supported features of their own models separately. See the community-specific sections below.

## vLLM Omni Features and Model Performance

| Model | Hardware | Parallel-Text Encoder | Parallel-DiT | Parallel-Vae | FA Optimization-Quantization | FA Optimization-Sparse | Linear-layer Quantization | Cache | Fused Ops | Compile |
| :--------- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| MiniMax H3 | A2/A3 | TP | USP | Tile | No | Yes | int8/int4 | Yes | Yes | No |
| MiniMax H3 | A5 | TP | USP | Tile | fp8 | Yes | mxfp8/mxfp4 | Yes | Yes | No |
| Wan2.2 | A2/A3 | No | USP/CFG/TP | Tile | No | Yes | int8 | No | Yes | No |
| Wan2.2 | A5 | No | USP/CFG/TP | Tile | fp8/mxfp8/mxfp4 | Yes | mxfp8/mxfp4 | No | Yes | No |
| Qwen-Image-2.1 | A2/A3 | No | TP/USP/CFG | Tile | No | No | int8 | Yes | Yes | No |
| Qwen-Image-2.1 | A5 | No | TP/USP/CFG | Tile | mxfp8/mxfp4 | No | mxfp8/mxfp4 | Yes | Yes | No |
| Qwen Image Edit Plus 2511 | A2/A3 | No | TP/USP/CFG | No | No | No | int8 | Yes | Yes | No |
| Qwen Image Edit Plus 2511 | A5 | No | TP/USP/CFG | No | fp8/mxfp8/mxfp4 | No | mxfp8/mxfp4 | Yes | Yes | No |
| Qwen Image Edit Plus 2509 | A2/A3 | No | TP/USP/CFG | No | No | No | int8 | Yes | Yes | No |
| Qwen Image Edit Plus 2509 | A5 | No | TP/USP/CFG | No | fp8/mxfp8/mxfp4 | No | mxfp8/mxfp4 | Yes | Yes | No |
| Qwen Image Edit | A2/A3 | No | TP/USP/CFG | No | No | No | int8 | Yes | Yes | No |
| Qwen Image Edit | A5 | No | TP/USP/CFG | No | fp8/mxfp8/mxfp4 | No | mxfp8/mxfp4 | Yes | Yes | No |
| Qwen-Image | A2/A3 | No | TP/USP/CFG | Tile | No | No | int8 | Yes | Yes | No |
| Qwen-Image | A5 | No | TP/USP/CFG | Tile | fp8/mxfp8/mxfp4 | No | mxfp8/mxfp4 | Yes | Yes | No |
| HunyuanImage 3.0 | A2/A3 | No | TP/EP/USP/CFG | Tile | No | No | int8/int4 | Yes | Yes | No |
| HunyuanImage 3.0 | A5 | No | TP/EP/USP/CFG | Tile | fp8/mxfp8 | No | mxfp8/mxfp4 | Yes | Yes | No |
| Qwen3-Omni | A2/A3 | TP/EP | No | No | No | No | w8a8 | Yes | Yes | Yes |
| Qwen3-Omni | A5 | TP/EP | No | No | No | No | No | Yes | Yes | Yes |
| Qwen3-Omni-Thinking | A2/A3 | TP/EP | No | No | No | No | w8a8 | Yes | Yes | Yes |
| Qwen3-Omni-Thinking | A5 | TP/EP | No | No | No | No | No | Yes | Yes | Yes |
| Qwen2.5-Omni | A2/A3 | TP | TP/USP | No | No | No | w8a8 | Yes | Yes | Yes |
| Qwen2.5-Omni | A5 | TP | TP/USP | No | No | No | No | Yes | Yes | Yes |
| Qwen3-TTS | A2/A3 | TP | No | No | No | No | No | Yes | Yes | Yes |
| Qwen3-TTS | A5 | TP | No | No | No | No | No | Yes | Yes | Yes |
| CosyVoice3 | A2/A3 | No | TP/USP/CFG | No | No | No | No | Yes | Yes | Yes |
| CosyVoice3 | A5 | No | TP/USP/CFG | No | No | No | No | Yes | Yes | Yes |
| Qwen3-ASR | A2/A3 | TP | No | No | No | No | No | Yes | Yes | Yes |
| Qwen3-ASR | A5 | TP | No | No | No | No | No | Yes | Yes | Yes |
| Cosmos3 | A2/A3 | TP | TP/USP/CFG | Tile | No | No | No | No | Yes | No |
| Cosmos3 | A5 | TP | TP/USP/CFG | Tile | No | No | mxfp4/mxfp8 | No | Yes | No |

> [!NOTE] Note
>
> - Only features verified on the corresponding hardware are listed; unlisted hardware means not yet adapted or verified.
> - Column prefixes indicate groups: "Parallel-" columns describe parallelism support; "FA Optimization-" columns describe FA quantization and sparse-attention precision.
> - The "Compile" column indicates ACLGraph acceleration support.
> - Yes/No indicates supported/not supported.

## Modelers Community Feature Stacking and Model Performance

| Model | Hardware | Parallel-Text Encoder | Parallel-DiT | Parallel-Vae | FA Optimization-Quantization | FA Optimization-Sparse | Linear-layer Quantization | Cache | Fused Ops | Compile |
| :--------- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | A2/A3 | No | USP | No | No | No | int8 | Yes | Yes | No |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | A5 | No | USP | No | fp8 | No | mxfp8 | Yes | Yes | No |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | A2/A3 | Yes | USP/CFG | Yes | int8 | Yes | int8 | Yes | Yes | No |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | A5 | Yes | USP/CFG | Yes | fp8/mxfp8/mxfp4 | Yes | mxfp8/mxfp4 | Yes | Yes | No |
| [Qwen Image Edit Plus 2511](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | No | USP/CFG | No | int8 | No | int8 | Yes | Yes | No |
| [Qwen Image Edit Plus 2509](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | No | USP/CFG | No | int8 | No | int8 | Yes | Yes | No |
| [Qwen Image Edit](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | No | USP/CFG | No | int8 | No | int8 | Yes | Yes | No |
| [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | No | USP/CFG | No | int8 | No | int8 | Yes | Yes | No |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | A2/A3 | Yes | USP | Yes | No | Yes | No | Yes | Yes | No |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | A5 | Yes | USP | Yes | fp8 | Yes | mxfp8 | Yes | Yes | No |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | A2/A3 | Yes | USP | Yes | No | No | int8 | Yes | Yes | No |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | A5 | Yes | USP | Yes | mxfp8 | No | int8/mxfp8 | Yes | Yes | No |

> [!NOTE] Note
>
> - Only features verified on the corresponding hardware are listed; unlisted hardware means not yet adapted or verified.
> - Column prefixes indicate groups: "Parallel-" columns describe parallelism support; "FA Optimization-" columns describe FA quantization and sparse-attention precision.
> - The "Compile" column indicates ACLGraph acceleration support.
> - Yes/No indicates supported/not supported.

## Models No Longer Maintained

The following models are no longer maintained with new releases. Only historical support information is retained, and they are not recommended for new services.

| Model | Original Community | Hardware |
| :--------- | :--------- | :--------- |
| Stable Diffusion 1.5 | Modelers | A2/300I DUO |
| Stable Diffusion 2.1 | Modelers | A2/300I DUO |
| Stable Diffusion XL | Modelers | A2/A3/300I DUO |
| Stable Diffusion XL_inpainting | Modelers | A2/A3 |
| Stable Diffusion XL_lighting | Modelers | A2/A3 |
| Stable Diffusion XL_controlnet | Modelers | A2/A3 |
| Stable Diffusion XL_prompt_weight | Modelers | A2/A3 |
| Stable Diffusion 3 | Modelers | A2/300I DUO |
| Stable Video Diffusion | Modelers | A2 |
| Stable Audio Open v1.0 | Modelers | A2/300I DUO |
| OpenSora v1.2 | Modelers | A2/A3 |
| OpenSoraPlan v1.2 | Modelers | A2/A3 |
| OpenSoraPlan v1.3 | Modelers | A2 |
| CogView3-Plus-3B | Modelers | A2/A3 |
| CogVideoX-2B | Modelers | A2/A3 |
| CogVideoX-5B | Modelers | A2/A3 |
| HunyuanDit | Modelers | A2/A3 |
| Wan2.1 | Modelers | A2/A3 |
