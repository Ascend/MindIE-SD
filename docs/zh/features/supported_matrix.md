# 模型/框架支持介绍

当前，MindIE SD支持vLLM Omni框架、vLLM-Ascend框架和魔乐社区等。理论上，MindIE SD支持任何多模态模型的推理加速，此处仅列出了我们支持的典型模型的特性叠加情况。

>[!NOTE]说明
>
>- **A2**：Atlas 800I A2推理服务器，默认使用算力313T，内存64 GB。
>- **A3**：Atlas 800I A3超节点服务器，默认使用算力560T，内存64 GB。
>- **A5**：Ascend 950PR&950DT系列产品。
>- **300I DUO**：Atlas 300I DUO推理卡，默认使用算力140T（FP16），内存48 GB。

## 模型支持情况

| 模型 | 模型类型 | 硬件 | vLLM Omni | vLLM-Ascend | 魔乐社区 |
| :--------- | :--------- | :--------- | :-----: | :-----: | :-----: |
| MiniMax H3 | 音视频生成 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Wan2.2 | 视频生成 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| HunyuanVideo-1.5 | 视频生成 | A2/A3/A5 | ✖️ | ✖️ | ✅️ |
| HunyuanVideo | 视频生成 | A2/A3/A5 | ✖️ | ✖️ | ✅️ |
| Qwen-Image-2.1 | 图像生成/编辑 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Qwen Image Edit Plus 2511 | 图像编辑 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| Qwen Image Edit Plus 2509 | 图像编辑 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| Qwen Image Edit | 图像编辑 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| Qwen-Image | 图像生成 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| HunyuanImage 3.0 | 图像生成/编辑/COT | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| FLUX.1-dev | 图像生成 | A2/A3/A5 | ✅️ | ✖️ | ✅️ |
| Qwen3-Omni | 全模态 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Qwen3-Omni-Thinking | 全模态 | A2/A3/A5 | ✖️ | ✅️ | ✖️ |
| Qwen2.5-Omni | 全模态 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Qwen3-TTS | 语音合成 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| CosyVoice3 | 语音合成 | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Qwen3-ASR | 语音识别 | A2/A3/A5 | ✖️ | ✅️ | ✖️ |
| Cosmos3 | 世界模型/COT | A2/A3/A5 | ✅️ | ✖️ | ✖️ |
| Z-Image | 图像生成 | A2/A3 | ✖️ | ✖️ | ✅️ |
| Z-Image Turbo | 图像生成 | A2/A3 | ✅️ | ✖️ | ✅️ |
| FLUX.2-dev | 图像生成/编辑 | A2/A3 | ✖️ | ✖️ | ✅️ |
| Hunyuan3D-2.1 | 3D生成 | A2/A3 | ✖️ | ✖️ | ✅️ |

>[!NOTE]说明
>
>vLLM Omni、vLLM-Ascend和魔乐社区各自维护其支持模型的具体特性，详见下文各社区章节。

## vLLM Omni 特性&模型性能

| 模型 | 硬件 | 并行-Text Encoder | 并行-DiT | 并行-Vae | FA优化-量化 | FA优化-稀疏 | 线性层量化 | Cache | 融合算子 | Compile |
| :--------- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| MiniMax H3 | A2/A3 | TP | USP | Tile | ✖️ | ✅️ | int8/int4 | ✅️ | ✅️ | ✖️ |
| MiniMax H3 | A5 | TP | USP | Tile | fp8 | ✅️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Wan2.2 | A2/A3 | ✖️ | USP/CFG/TP | Tile | ✖️ | ✅️ | int8 | ✖️ | ✅️ | ✖️ |
| Wan2.2 | A5 | ✖️ | USP/CFG/TP | Tile | fp8/mxfp8/mxfp4 | ✅️ | mxfp8/mxfp4 | ✖️ | ✅️ | ✖️ |
| Qwen-Image-2.1 | A2/A3 | ✖️ | TP/USP/CFG | Tile | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| Qwen-Image-2.1 | A5 | ✖️ | TP/USP/CFG | Tile | mxfp8/mxfp4 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit Plus 2511 | A2/A3 | ✖️ | TP/USP/CFG | ✖️ | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit Plus 2511 | A5 | ✖️ | TP/USP/CFG | ✖️ | fp8/mxfp8/mxfp4 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit Plus 2509 | A2/A3 | ✖️ | TP/USP/CFG | ✖️ | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit Plus 2509 | A5 | ✖️ | TP/USP/CFG | ✖️ | fp8/mxfp8/mxfp4 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit | A2/A3 | ✖️ | TP/USP/CFG | ✖️ | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| Qwen Image Edit | A5 | ✖️ | TP/USP/CFG | ✖️ | fp8/mxfp8/mxfp4 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Qwen-Image | A2/A3 | ✖️ | TP/USP/CFG | Tile | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| Qwen-Image | A5 | ✖️ | TP/USP/CFG | Tile | fp8/mxfp8/mxfp4 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| HunyuanImage 3.0 | A2/A3 | ✖️ | TP/EP/USP/CFG | Tile | ✖️ | ✖️ | int8/int4 | ✅️ | ✅️ | ✖️ |
| HunyuanImage 3.0 | A5 | ✖️ | TP/EP/USP/CFG | Tile | fp8/mxfp8 | ✖️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| Qwen3-Omni | A2/A3 | TP/EP | ✖️ | ✖️ | ✖️ | ✖️ | w8a8 | ✅️ | ✅️ | ✅️ |
| Qwen3-Omni | A5 | TP/EP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen3-Omni-Thinking | A2/A3 | TP/EP | ✖️ | ✖️ | ✖️ | ✖️ | w8a8 | ✅️ | ✅️ | ✅️ |
| Qwen3-Omni-Thinking | A5 | TP/EP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen2.5-Omni | A2/A3 | TP | TP/USP | ✖️ | ✖️ | ✖️ | w8a8 | ✅️ | ✅️ | ✅️ |
| Qwen2.5-Omni | A5 | TP | TP/USP | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen3-TTS | A2/A3 | TP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen3-TTS | A5 | TP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| CosyVoice3 | A2/A3 | ✖️ | TP/USP/CFG | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| CosyVoice3 | A5 | ✖️ | TP/USP/CFG | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen3-ASR | A2/A3 | TP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Qwen3-ASR | A5 | TP | ✖️ | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✅️ | ✅️ |
| Cosmos3 | A2/A3 | TP | TP/USP/CFG | Tile | ✖️ | ✖️ | ✖️ | ✖️ | ✅️ | ✖️ |
| Cosmos3 | A5 | TP | TP/USP/CFG | Tile | ✖️ | ✖️ | mxfp4/mxfp8 | ✖️ | ✅️ | ✖️ |

>[!NOTE]说明
>
>- 表中仅列出对应硬件上已验证的特性；未列出的硬件表示暂未适配或验证。
>- 列名前缀表示分组：“并行-”列为并行支持情况；“FA优化-”列为 FA 量化与稀疏注意力精度。
>- “Compile”列为 ACLGraph 加速支持情况。
>- ✅️/✖️表示支持/不支持。

## 魔乐社区的特性叠加&模型性能

| 模型 | 硬件 | 并行-Text Encoder | 并行-DiT | 并行-Vae | FA优化-量化 | FA优化-稀疏 | 线性层量化 | Cache | 融合算子 | Compile |
| :--------- | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :-----: |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | A2/A3 | ✖️ | USP | ✖️ | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [FLUX.1-dev](https://modelers.cn/models/MindIE/FLUX.1-dev) | A5 | ✖️ | USP | ✖️ | fp8 | ✖️ | mxfp8 | ✅️ | ✅️ | ✖️ |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | A2/A3 | ✅️ | USP/CFG | ✅️ | int8 | ✅️ | int8 | ✅️ | ✅️ | ✖️ |
| [Wan2.2](https://modelers.cn/models/MindIE/Wan2.2) | A5 | ✅️ | USP/CFG | ✅️ | fp8/mxfp8/mxfp4 | ✅️ | mxfp8/mxfp4 | ✅️ | ✅️ | ✖️ |
| [Qwen Image Edit Plus 2511](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | ✖️ | USP/CFG | ✖️ | int8 | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [Qwen Image Edit Plus 2509](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | ✖️ | USP/CFG | ✖️ | int8 | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [Qwen Image Edit](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | ✖️ | USP/CFG | ✖️ | int8 | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [Qwen-Image](https://modelers.cn/models/MindIE/Qwen-Image-series) | A2/A3 | ✖️ | USP/CFG | ✖️ | int8 | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | A2/A3 | ✅️ | USP | ✅️ | ✖️ | ✅️ | ✖️ | ✅️ | ✅️ | ✖️ |
| [HunyuanVideo-1.5](https://modelers.cn/models/MindIE/HunyuanVideo-1.5) | A5 | ✅️ | USP | ✅️ | fp8 | ✅️ | mxfp8 | ✅️ | ✅️ | ✖️ |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | A2/A3 | ✅️ | USP | ✅️ | ✖️ | ✖️ | int8 | ✅️ | ✅️ | ✖️ |
| [HunyuanVideo](https://modelers.cn/models/MindIE/hunyuan_video) | A5 | ✅️ | USP | ✅️ | mxfp8 | ✖️ | int8/mxfp8 | ✅️ | ✅️ | ✖️ |

>[!NOTE]说明
>
>- 表中仅列出对应硬件上已验证的特性；未列出的硬件表示暂未适配或验证。
>- 列名前缀表示分组：“并行-”列为并行支持情况；“FA优化-”列为 FA 量化与稀疏注意力精度。
>- “Compile”列为 ACLGraph 加速支持情况。
>- ✅️/✖️表示支持/不支持。

## 已停止维护的模型

以下模型已不再随版本维护，仅保留历史支持信息，不建议在新业务中使用。

| 模型 | 原支持社区 | 硬件 |
| :--------- | :--------- | :--------- |
| Stable Diffusion 1.5 | 魔乐社区 | A2/300I DUO |
| Stable Diffusion 2.1 | 魔乐社区 | A2/300I DUO |
| Stable Diffusion XL | 魔乐社区 | A2/A3/300I DUO |
| Stable Diffusion XL_inpainting | 魔乐社区 | A2/A3 |
| Stable Diffusion XL_lighting | 魔乐社区 | A2/A3 |
| Stable Diffusion XL_controlnet | 魔乐社区 | A2/A3 |
| Stable Diffusion XL_prompt_weight | 魔乐社区 | A2/A3 |
| Stable Diffusion 3 | 魔乐社区 | A2/300I DUO |
| Stable Video Diffusion | 魔乐社区 | A2 |
| Stable Audio Open v1.0 | 魔乐社区 | A2/300I DUO |
| OpenSora v1.2 | 魔乐社区 | A2/A3 |
| OpenSoraPlan v1.2 | 魔乐社区 | A2/A3 |
| OpenSoraPlan v1.3 | 魔乐社区 | A2 |
| CogView3-Plus-3B | 魔乐社区 | A2/A3 |
| CogVideoX-2B | 魔乐社区 | A2/A3 |
| CogVideoX-5B | 魔乐社区 | A2/A3 |
| HunyuanDit | 魔乐社区 | A2/A3 |
| Wan2.1 | 魔乐社区 | A2/A3 |
