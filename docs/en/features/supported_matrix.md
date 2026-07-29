# Model/Framework Support Matrix

Currently, MindIE SD supports the vLLM Omni framework, Cache DiT framework, and Modelers community, among others. In theory, MindIE SD supports inference acceleration for any multimodal model. This page only lists the feature stacking status for our supported typical models.

## Model Support

| Model | vLLM Omni | Cache DiT + diffusers | Modelers |
|:----------:|:---------:|:---------------------:|:------:|
| Stable Diffusion 1.5 | No | No | Yes |
| Stable Diffusion 2.1 | No | No | Yes |
| Stable Diffusion XL | No | No | Yes |
| Stable Diffusion XL_inpainting | No | No | Yes |
| Stable Diffusion XL_lighting | No | No | Yes |
| Stable Diffusion XL_controlnet | No | No | Yes |
| Stable Diffusion XL_prompt_weight | No | No | Yes |
| Stable Diffusion 3 | No | No | Yes |
| Stable Video Diffusion | No | No | Yes |
| Stable Audio Open v1.0 | No | No | Yes |
| OpenSora v1.2 | No | No | Yes |
| OpenSoraPlan v1.2 | No | No | Yes |
| OpenSoraPlan v1.3 | No | No | Yes |
| CogView3-Plus-3B | No | No | Yes |
| CogVideoX-2B | No | No | Yes |
| CogVideoX-5B | No | No | Yes |
| HunyuanDit | No | No | Yes |
| HunyuanVideo | No | No | Yes |
| HunyuanVideo-1.5 | No | No | Yes |
| Hunyuan3D-2.1 | No | No | Yes |
| Wan2.1 | No | No | Yes |
| Wan2.2 | No | No | Yes |
| FLUX.1-dev | Yes | Yes | Yes |
| FLUX.2-dev | No | Yes | Yes |
| Qwen-Image | Yes | No | Yes |
| Qwen-Image-Edit | Yes | No | Yes |
| Qwen-Image-Edit-2509 | Yes | No | Yes |
| Z-Image | No | No | Yes |
| Z-Image-Turbo | Yes | No | Yes |

## vLLM Omni Features and Model Performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused Ops |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
| FLUX.1-dev | Atlas 800I A2 Server | Yes | Yes | No | Yes | Yes |
| Qwen-Image | Atlas 800I A2 Server | Yes | Yes | No | No | Yes |
| Qwen-Image-Edit | Atlas 800I A2 Server | Yes | Yes | No | No | Yes |
| Qwen-Image-Edit-2509 | Atlas 800I A2 Server | Yes | Yes | No | No | Yes |
| Z-Image-Turbo | Atlas 800I A2 Server | Yes | No | No | No | Yes |

> **Note:**
> Atlas 800I A2 servers default to 313T compute and 64 GB memory.

## Cache DiT + diffusers Features and Model Performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused Ops |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
| FLUX.1-dev | Atlas 800I A2 Server | Yes | Yes | No | Yes | Yes |
| FLUX.2-dev | Atlas 800I A2 Server | No | Yes | No | No | Yes |

## Modelers Community Feature Stacking and Model Performance

| Model | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused Ops | Notes |
|:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:|
| [Stable Diffusion 1.5](https://modelers.cn/models/MindIE/stable_diffusion_v1.5) | Atlas 800I A2 Server / Atlas 300I DUO Inference Card | Yes | Yes | No | No | Yes | N/A |
| [Stable Diffusion 2.1](https://modelers.cn/models/MindIE/stable_diffusion_2.1) | Atlas 800I A2 Server / Atlas 300I DUO Inference Card | Yes | Yes | No | No | Yes | N/A |
| [Stable Diffusion XL](https://modelers.cn/models/MindIE/stable-diffusion-xl) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server / Atlas 300I DUO Inference Card | Yes | Yes | No | No | Yes | N/A |
| [Stable Diffusion XL_inpainting](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_inpainting) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | No | No | No | Yes | Feature enabled |
| [Stable Diffusion XL_lighting](https://modelers.cn/models/MindIE/SDXL-Lighting) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | No | No | No | Yes | Feature enabled |
| [Stable Diffusion XL_controlnet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_controlnet) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | No | No | No | Yes | Feature enabled |
| [Stable Diffusion XL_prompt_weight](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_prompt_weight) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | No | No | No | Yes | Feature enabled |
| [Stable Diffusion 3](https://modelers.cn/models/MindIE/stable_diffusion3) | Atlas 800I A2 Server / Atlas 300I DUO Inference Card | Yes | Yes | No | No | Yes | N/A |
| [Stable Video Diffusion](https://modelers.cn/models/MindIE/stable-video-diffusion) | Atlas 800I A2 Server | Yes | Yes | No | No | Yes | N/A |
| [Stable Audio Open v1.0](https://modelers.cn/models/MindIE/stable_audio_open_1.0) | Atlas 800I A2 Server / Atlas 300I DUO Inference Card | Yes | No | No | No | Yes | N/A |
| [OpenSora v1.2](https://modelers.cn/models/MindIE/opensora_v1_2) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | No | Yes | N/A |
| [OpenSoraPlan v1.2](https://modelers.cn/models/MindIE/open_sora_planv1_2) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | No | Yes | N/A |
| [OpenSoraPlan v1.3](https://modelers.cn/models/MindIE/open_sora_planv1_3) | Atlas 800I A2 Server | Yes | Yes | No | No | Yes | N/A |
| [CogView3-Plus-3B](https://modelers.cn/models/MindIE/CogView3-Plus-3B) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | No | Yes | N/A |
| [CogVideoX-2B](https://modelers.cn/models/MindIE/CogVideoX) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | No | Yes | N/A |
| [CogVideoX-5B](https://modelers.cn/models/MindIE/CogVideoX) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | No | Yes | N/A |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [FLUX.2-dev](https://modelers.cn/models/MindIE/FLUX.2-dev) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [HunyuanDit](https://modelers.cn/models/MindIE/hunyuan_dit) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | No | No | No | Yes | N/A |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | Yes | Yes | Yes | N/A |
| [Hunyuan3D-2.1](https://modelers.cn/models/MindIE/Hunyuan3D-2.1) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [Wan2.1](https://modelers.cn/models/MindIE/Wan2.1) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | Yes | Yes | Yes | N/A |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | Yes | Yes | Yes | N/A |
| [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [Qwen-Image-Edit](https://modelers.cn/models/MindIE/Qwen-Image-Edit) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | Yes | Yes | No | Yes | Yes | N/A |
| [Z-Image](https://modelers.cn/models/MindIE/Z-Image) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | No | No | No | No | No | N/A |
| [Z-Image-Turbo](https://modelers.cn/models/MindIE/Z-Image-Turbo) | Atlas 800I A2 Server / Atlas 800I A3 Supernode Server | No | No | No | No | Yes | N/A |

> [!NOTE] Note
>
> - Atlas 300I DUO inference cards default to 280T compute and 48 GB memory.
> - Atlas 800I A2 servers default to 313T compute and 64 GB memory.
> - Atlas 800I A3 supernode servers default to 560T compute and 64 GB memory.
