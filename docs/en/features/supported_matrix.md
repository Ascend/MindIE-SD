# Model and Framework Support Matrix

MindIE SD currently supports the vLLM Omni framework, the Cache DiT framework, and the Modelers community. In theory, MindIE SD can accelerate inference for any multimodal model, but the matrix below lists the representative models and feature combinations that are currently supported.

## Model support

| Model | vLLM Omni | Cache DiT + diffusers | Modelers community |
|:----------:|:---------:|:---------------------:|:------:|
| Stable Diffusion 1.5 | ✖️ | ✖️ | ✅ |
| Stable Diffusion 2.1 | ✖️ | ✖️ | ✅ |
| Stable Diffusion XL | ✖️ | ✖️ | ✅ |
| Stable Diffusion XL_inpainting | ✖️ | ✖️ | ✅ |
| Stable Diffusion XL_lighting | ✖️ | ✖️ | ✅ |
| Stable Diffusion XL_controlnet | ✖️ | ✖️ | ✅ |
| Stable Diffusion XL_prompt_weight | ✖️ | ✖️ | ✅ |
| Stable Diffusion 3 | ✖️ | ✖️ | ✅ |
| Stable Video Diffusion | ✖️ | ✖️ | ✅ |
| Stable Audio Open v1.0 | ✖️ | ✖️ | ✅ |
| OpenSora v1.2 | ✖️ | ✖️ | ✅ |
| OpenSoraPlan v1.2 | ✖️ | ✖️ | ✅ |
| OpenSoraPlan v1.3 | ✖️ | ✖️ | ✅ |
| CogView3-Plus-3B | ✖️ | ✖️ | ✅ |
| CogVideoX-2B | ✖️ | ✖️ | ✅ |
| CogVideoX-5B | ✖️ | ✖️ | ✅ |
| HunyuanDit | ✖️ | ✖️ | ✅ |
| HunyuanVideo | ✖️ | ✖️ | ✅ |
| HunyuanVideo-1.5 | ✖️ | ✖️ | ✅ |
| Hunyuan3D-2.1 | ✖️ | ✖️ | ✅ |
| Wan2.1 | ✖️ | ✖️ | ✅ |
| Wan2.2 | ✖️ | ✖️ | ✅ |
| FLUX.1-dev | ✅ | ✅ | ✅ |
| FLUX.2-dev | ✖️ | ✅ | ✅ |
| Qwen-Image | ✅ | ✖️ | ✅ |
| Qwen-Image-Edit | ✅ | ✖️ | ✅ |
| Qwen-Image-Edit-2509 | ✅ | ✖️ | ✅ |
| Z-Image | ✖️ | ✖️ | ✅ |
| Z-Image-Turbo | ✅ | ✖️ | ✅ |

## vLLM Omni features and model performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused operators |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
| FLUX.1-dev | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✅ | ✅ |
| Qwen-Image | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✖️ | ✅ |
| Qwen-Image-Edit | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✖️ | ✅ |
| Qwen-Image-Edit-2509 | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✖️ | ✅ |
| Z-Image-Turbo | Atlas 800I A2 server | ✅ | ✖️ | ✖️ | ✖️ | ✅ |

> **Note**
> Atlas 800I A2 servers use 313T default compute and 64 GB of memory.

## Cache DiT + diffusers features and model performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused operators |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
| FLUX.1-dev | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✅ | ✅ |
| FLUX.2-dev | Atlas 800I A2 server | ✖️ | ✅ | ✖️ | ✖️ | ✅ |

## Modelers community feature combinations and model performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused operators | Notes |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:|
| [Stable Diffusion 1.5](https://modelers.cn/models/MindIE/stable_diffusion_v1.5) | <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [Stable Diffusion 2.1](https://modelers.cn/models/MindIE/stable_diffusion_2.1) | <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [Stable Diffusion XL](https://modelers.cn/models/MindIE/stable-diffusion-xl) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li><li>Atlas 300I DUO inference card</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [Stable Diffusion XL_inpainting](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_inpainting) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | Functional integration complete |
| [Stable Diffusion XL_lighting](https://modelers.cn/models/MindIE/SDXL-Lighting) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | Functional integration complete |
| [Stable Diffusion XL_controlnet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_controlnet) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | Functional integration complete |
| [Stable Diffusion XL_prompt_weight](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_prompt_weight) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | Functional integration complete |
| [Stable Diffusion 3](https://modelers.cn/models/MindIE/stable_diffusion3) | <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [Stable Video Diffusion](https://modelers.cn/models/MindIE/stable-video-diffusion) | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [Stable Audio Open v1.0](https://modelers.cn/models/MindIE/stable_audio_open_1.0) | <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | None |
| [OpenSora v1.2](https://modelers.cn/models/MindIE/opensora_v1_2) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [OpenSoraPlan v1.2](https://modelers.cn/models/MindIE/open_sora_planv1_2) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [OpenSoraPlan v1.3](https://modelers.cn/models/MindIE/open_sora_planv1_3) | Atlas 800I A2 server | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [CogView3-Plus-3B](https://modelers.cn/models/MindIE/CogView3-Plus-3B) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [CogVideoX-2B](https://modelers.cn/models/MindIE/CogVideoX) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [CogVideoX-5B](https://modelers.cn/models/MindIE/CogVideoX) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✖️ | ✅ | None |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [FLUX.2-dev](https://modelers.cn/models/MindIE/FLUX.2-dev) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [HunyuanDit](https://modelers.cn/models/MindIE/hunyuan_dit) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✖️ | ✖️ | ✖️ | ✅ | None |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✅ | ✅ | ✅ | None |
| [Hunyuan3D-2.1](https://modelers.cn/models/MindIE/Hunyuan3D-2.1) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [Wan2.1](https://modelers.cn/models/MindIE/Wan2.1) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✅ | ✅ | ✅ | None |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✅ | ✅ | ✅ | None |
| [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [Qwen-Image-Edit](https://modelers.cn/models/MindIE/Qwen-Image-Edit) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✅ | ✅ | ✖️ | ✅ | ✅ | None |
| [Z-Image](https://modelers.cn/models/MindIE/Z-Image) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | None |
| [Z-Image-Turbo](https://modelers.cn/models/MindIE/Z-Image-Turbo) | <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 supernode server</li></ul> | ✖️ | ✖️ | ✖️ | ✖️ | ✅ | None |

> **Note**
>
> - Atlas 300I DUO inference cards use 280T default compute and 48 GB of memory.
> - Atlas 800I A2 servers use 313T default compute and 64 GB of memory.
> - Atlas 800I A3 supernode servers use 560T default compute and 64 GB of memory.
