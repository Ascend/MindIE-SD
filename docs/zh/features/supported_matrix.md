# 模型/框架支持介绍

当前，MindIE SD支持vLLM Omni框架、Cache DiT框架和魔乐社区等。理论上，MindIE SD支持任何多模态模型的推理加速，此处仅列出了我们支持的典型模型的特性叠加情况。

## 模型支持情况

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

## vLLM Omni 特性&模型性能

 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
 | FLUX.1-dev |  Atlas 800I A2服务器  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |
 | Qwen-Image |  Atlas 800I A2服务器  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Qwen-Image-Edit |  Atlas 800I A2服务器  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Qwen-Image-Edit-2509 |  Atlas 800I A2服务器  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Z-Image-Turbo |  Atlas 800I A2服务器  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |

>**说明：**
>Atlas 800I A2服务器默认使用的版本算力313T，内存64 GB。

## Cache DiT + diffusers 特性&模型性能

 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
 | FLUX.1-dev |  Atlas 800I A2服务器  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |
 | FLUX.2-dev |  Atlas 800I A2服务器  |    ✖️    | ✅️  |  ✖️   | ✖️ |   ✅️    |

## 魔乐社区的特性叠加&模型性能

 |   模型     |  硬件  | Cache   | 并行 | 稀疏FA | 量化 | 融合算子 | 说明 |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:|
 | [Stable Diffusion 1.5](https://modelers.cn/models/MindIE/stable_diffusion_v1.5) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 300I DUO 推理卡</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [Stable Diffusion 2.1](https://modelers.cn/models/MindIE/stable_diffusion_2.1) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 300I DUO 推理卡</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [Stable Diffusion XL](https://modelers.cn/models/MindIE/stable-diffusion-xl)  |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li><li>Atlas 300I DUO 推理卡</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [Stable Diffusion XL_inpainting](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_inpainting) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_lighting](https://modelers.cn/models/MindIE/SDXL-Lighting) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_controlnet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_controlnet) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion XL_prompt_weight](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_prompt_weight) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  功能打通  |
 | [Stable Diffusion 3](https://modelers.cn/models/MindIE/stable_diffusion3) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 300I DUO 推理卡</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [Stable Video Diffusion](https://modelers.cn/models/MindIE/stable-video-diffusion) |  Atlas 800I A2 服务器  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [Stable Audio Open v1.0](https://modelers.cn/models/MindIE/stable_audio_open_1.0) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 300I DUO 推理卡</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [OpenSora v1.2](https://modelers.cn/models/MindIE/opensora_v1_2) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [OpenSoraPlan v1.2](https://modelers.cn/models/MindIE/open_sora_planv1_2) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [OpenSoraPlan v1.3](https://modelers.cn/models/MindIE/open_sora_planv1_3) |  Atlas 800I A2 服务器  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [CogView3-Plus-3B](https://modelers.cn/models/MindIE/CogView3-Plus-3B) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [CogVideoX-2B](https://modelers.cn/models/MindIE/CogVideoX) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [CogVideoX-5B](https://modelers.cn/models/MindIE/CogVideoX) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [FLUX.2-dev](https://modelers.cn/models/MindIE/FLUX.2-dev) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [HunyuanDit](https://modelers.cn/models/MindIE/hunyuan_dit) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  无  |
 | [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  无  |
 | [Hunyuan3D-2.1](https://modelers.cn/models/MindIE/Hunyuan3D-2.1) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [Wan2.1](https://modelers.cn/models/MindIE/Wan2.1) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  无  |
 | [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  无  |
 | [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [Qwen-Image-Edit](https://modelers.cn/models/MindIE/Qwen-Image-Edit) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  无  |
 | [Z-Image](https://modelers.cn/models/MindIE/Z-Image) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✖️    |  无  |
 | [Z-Image-Turbo](https://modelers.cn/models/MindIE/Z-Image-Turbo) |  <ul><li>Atlas 800I A2 服务器</li><li>Atlas 800I A3 超节点服务器</li></ul>  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  无  |

>[!NOTE]说明
>
>- Atlas 300I DUO 推理卡默认使用算力280T，内存48 GB。
>- Atlas 800I A2 服务器默认使用算力313T，内存64 GB。
>- Atlas 800I A3 超节点服务器默认使用算力560T，内存64 GB。
