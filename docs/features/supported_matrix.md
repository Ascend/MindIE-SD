# 模型/框架支持情况

当前，MindIE SD 支持🤗魔乐社区 🤗vLLM Omni 🤗Cache Dit 等框架/社区，不同社区的具体使能方法请参考 example。

理论上，MindIE SD 支持任何多模态模型的推理加速，此处仅列出了我们支持的典型模型的特性叠加情况。
我们将会持续刷新相关数据，如果你将 MindIE SD 适配到了新模型，欢迎更新 example，刷新列表。

## 📚 模型支持情况
 |  模型       |  vLLM Omni | Cache DiT + diffusers |  魔乐社区  |
 |:----------:|:---------:|:---------------------:|:------:|
 | Stable Diffusion 1.5 |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion 2.1 |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion XL  |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion XL_inpainting |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion XL_lighting |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion XL_controlnet |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion XL_prompt_weight |     ✖️    |          ✖️           |  ✅️    |
 | Stable Diffusion 3 |     ✖️    |          ✖️           |  ✅️    |
 | Stable Video Diffusion |     ✖️    |          ✖️           |  ✅️    |
 | Stable Audio Open v1.0 |     ✖️    |          ✖️           |  ✅️    |
 | OpenSora v1.2 |     ✖️    |          ✖️           |  ✅️    |
 | OpenSoraPlan v1.2 |     ✖️    |          ✖️           |  ✅️    |
 | OpenSoraPlan v1.3 |     ✖️    |          ✖️           |  ✅️    |
 | CogView3-Plus-3B |     ✖️    |          ✖️           |  ✅️    |
 | CogVideoX-2B |     ✖️    |          ✖️           |  ✅️    |
 | CogVideoX-5B |     ✖️    |          ✖️           |  ✅️    |
 | HunyuanDit |     ✖️    |          ✖️           |  ✅️    |
 | HunyuanVideo |     ✖️    |          ✖️           |  ✅️    |
 | HunyuanVideo-1.5 |     ✖️    |          ✖️           |  ✅️    |
 | Hunyuan3D-2.1 |     ✖️    |          ✖️           |  ✅️    |
 | Wan2.1 |     ✖️    |          ✖️           |  ✅️    |
 | Wan2.2 |     ✖️    |          ✖️           |  ✅️    |
 | FLUX.1-dev |     ✅️    |          ✅️           |  ✅️    |
 | FLUX.2-dev |     ✖️    |          ✅️           |  ✅️    |
 | Qwen-Image |     ✅️    |          ✖️           |  ✅️    |
 | Qwen-Image-Edit |     ✅️    |          ✖️           |  ✅️    |
 | Qwen-Image-Edit-2509 |     ✅️    |          ✖️           |  ✅️    |
 | Z-Image |     ✖️    |          ✖️           |  ✅️    |
 | Z-Image-Turbo |     ✅️    |          ✖️           |  ✅️    |

## 📚 vLLM Omni 特性&模型性能
 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 | 说明 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:| 
 | FLUX.1-dev |  A2  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | Qwen-Image |  A2  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | Qwen-Image-Edit |  A2  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | Qwen-Image-Edit-2509 |  A2  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | Z-Image-Turbo |  A2  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |    |

说明：
A2代表Atlas 800I A2，默认使用的版本算力313T，内存64 GB。

## 📚 Cache DiT + diffusers 特性&模型性能
 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 | 说明 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:| 
 | FLUX.1-dev |  A2  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | FLUX.2-dev |  A2  |    ✖️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |

## 📚 魔乐社区的特性叠加&模型性能
 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 | 说明 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:| 
 | [Stable Diffusion 1.5](https://modelers.cn/models/MindIE/stable_diffusion_v1.5) |  A2 DUO  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [Stable Diffusion 2.1](https://modelers.cn/models/MindIE/stable_diffusion_2.1) |  A2 DUO  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [Stable Diffusion XL](https://modelers.cn/models/MindIE/stable-diffusion-xl)  |  A2 A3 DUO  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [Stable Diffusion XL_inpainting](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_inpainting) |  A2 A3  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_lighting](https://modelers.cn/models/MindIE/SDXL-Lighting) |  A2 A3  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_controlnet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_controlnet) |  A2 A3  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_prompt_weight](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_prompt_weight) |  A2 A3  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion 3](https://modelers.cn/models/MindIE/stable_diffusion3) |  A2 DUO  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [Stable Video Diffusion](https://modelers.cn/models/MindIE/stable-video-diffusion) |  A2  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [Stable Audio Open v1.0](https://modelers.cn/models/MindIE/stable_audio_open_1.0) |  A2 DUO  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |    |
 | [OpenSora v1.2](https://modelers.cn/models/MindIE/opensora_v1_2) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [OpenSoraPlan v1.2](https://modelers.cn/models/MindIE/open_sora_planv1_2) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [OpenSoraPlan v1.3](https://modelers.cn/models/MindIE/open_sora_planv1_3) |  A2  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [CogView3-Plus-3B](https://modelers.cn/models/MindIE/CogView3-Plus-3B) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [CogVideoX-2B](https://modelers.cn/models/MindIE/CogVideoX) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [CogVideoX-5B](https://modelers.cn/models/MindIE/CogVideoX) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |    |
 | [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [FLUX.2-dev](https://modelers.cn/models/MindIE/FLUX.2-dev) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [HunyuanDit](https://modelers.cn/models/MindIE/hunyuan_dit) |  A2 A3  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |    |
 | [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) |  A2 A3  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |    |
 | [Hunyuan3D-2.1](https://modelers.cn/models/MindIE/Hunyuan3D-2.1) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [Wan2.1](https://modelers.cn/models/MindIE/Wan2.1) |  A2 A3  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |    |
 | [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) |  A2 A3  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |    |
 | [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [Qwen-Image-Edit](https://modelers.cn/models/MindIE/Qwen-Image-Edit) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) |  A2 A3  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |    |
 | [Z-Image](https://modelers.cn/models/MindIE/Z-Image) |  A2 A3  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✖️    |    |
 | [Z-Image-Turbo](https://modelers.cn/models/MindIE/Z-Image-Turbo) |  A2 A3  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✅️    |    |

 说明：
 A2代表Atlas 800I A2，默认使用的版本算力313T，内存64 GB。
 A3代表Atlas 800I A3，默认使用的版本算力560T，内存64 GB。
 DUO代表300I DUO，默认使用的版本算力280T，内存48 GB。
 