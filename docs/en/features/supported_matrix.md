# Model/Framework Support Introduction

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:16:13.002Z pushedAt=2026-06-08T09:18:16.283Z -->

MindIE SD currently supports vLLM-Omni and Cache DiT, and is available in Modelers community. In principle, it accelerates inference for any multimodal model; the following highlights features for popular models.

## Model Support

 |  Model       |  vLLM-Omni | Cache-DiT + diffusers |  Modelers Community  |
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

## vLLM-Omni Features and Model Performance

 |   Model   | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused Operator |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
 | FLUX.1-dev | Atlas 800I A2 server |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |
 | Qwen-Image | Atlas 800I A2 server |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Qwen-Image-Edit | Atlas 800I A2 server |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Qwen-Image-Edit-2509 | Atlas 800I A2 server |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |
 | Z-Image-Turbo | Atlas 800I A2 server |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |

>**NOTE:**
>Atlas 800I A2 server defaults to 313T computing power and 64GB memory.

## Cache-DiT + diffusers Features and Model Performance

 |   Model   | Hardware | Cache | Parallelism | Sparse FA | Quantization | Fused Operator |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|
 | FLUX.1-dev | Atlas 800I A2 server |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |
 | FLUX.2-dev | Atlas 800I A2 server |    ✖️    | ✅️  |  ✖️   | ✖️ |   ✅️    |

## Feature Stack and Model Performance Provided in the Modelers Community

 |   Model     |  Hardware  | Cache   | Parallelism | Sparse FA | Quantization | Fused Operator | Description |
 |:----------:|:----:|:-------:|:--:|:----:|:--:|:---------:|:--:|
 | [Stable Diffusion 1.5](https://modelers.cn/models/MindIE/stable_diffusion_v1.5) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [Stable Diffusion 2.1](https://modelers.cn/models/MindIE/stable_diffusion_2.1) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [Stable Diffusion XL](https://modelers.cn/models/MindIE/stable-diffusion-xl)  |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li><li>Atlas 300I DUO inference card</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [Stable Diffusion XL_inpainting](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_inpainting) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  Feature Enabled  |
 | [Stable Diffusion XL_lighting](https://modelers.cn/models/MindIE/SDXL-Lighting) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD Server</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  Feature Enabled  |
 | [Stable Diffusion XL_controlnet](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_controlnet) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  Feature Enabled  |
 | [Stable Diffusion XL_prompt_weight](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/MindIE/MindIE-Torch/built-in/foundation/stable_diffusion_xl_prompt_weight) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  Feature Enabled  |
 | [Stable Diffusion 3](https://modelers.cn/models/MindIE/stable_diffusion3) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [Stable Video Diffusion](https://modelers.cn/models/MindIE/stable-video-diffusion) |  Atlas 800I A2 server  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [Stable Audio Open v1.0](https://modelers.cn/models/MindIE/stable_audio_open_1.0) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 300I DUO inference card</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [OpenSora v1.2](https://modelers.cn/models/MindIE/opensora_v1_2) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [OpenSoraPlan v1.2](https://modelers.cn/models/MindIE/open_sora_planv1_2) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [OpenSoraPlan v1.3](https://modelers.cn/models/MindIE/open_sora_planv1_3) |  Atlas 800I A2 server  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [CogView3-Plus-3B](https://modelers.cn/models/MindIE/CogView3-Plus-3B) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [CogVideoX-2B](https://modelers.cn/models/MindIE/CogVideoX) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [CogVideoX-5B](https://modelers.cn/models/MindIE/CogVideoX) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [FLUX.2-dev](https://modelers.cn/models/MindIE/FLUX.2-dev) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [HunyuanDit](https://modelers.cn/models/MindIE/hunyuan_dit) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  None  |
 | [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD Server</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  None  |
 | [Hunyuan3D-2.1](https://modelers.cn/models/MindIE/Hunyuan3D-2.1) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [Wan2.1](https://modelers.cn/models/MindIE/Wan2.1) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  None  |
 | [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✅️   | ✅️ |   ✅️    |  None  |
 | [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [Qwen-Image-Edit](https://modelers.cn/models/MindIE/Qwen-Image-Edit) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [Qwen-Image-Edit-2509](https://modelers.cn/models/MindIE/Qwen-Image-Edit-2509) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✅️    | ✅️  |  ✖️   | ✅️ |   ✅️    |  None  |
 | [Z-Image](https://modelers.cn/models/MindIE/Z-Image) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✖️    |  None  |
 | [Z-Image-Turbo](https://modelers.cn/models/MindIE/Z-Image-Turbo) |  <ul><li>Atlas 800I A2 server</li><li>Atlas 800I A3 SuperPoD server</li></ul>  |    ✖️    | ✖️  |  ✖️   | ✖️ |   ✅️    |  None  |

>[!NOTE]Note
>
>- Atlas 300I DUO inference card defaults to 280T computing power and 48GB memory.
>- Atlas 800I A2 server defaults to 313T computing power and 64GB memory.
>- Atlas 800I A3 SuperPoD server defaults to 560T computing power and 64GB memory.
