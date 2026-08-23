# 空权重运行验证

在 NPU 设备上**不依赖真实权重**（仅下载配置文件，几十 KB）构造生成类模型并完成前向推理验证。

## 目录结构

```text
examples/dummy_run/
├── model/
│   ├── __init__.py                       # check_npu(), resolve_config_path(), _PhaseTimer
│   ├── wan_model.py                      # build_wan_pipeline()
│   ├── qwen_image_model.py               # build_qwen_image_pipeline()
│   ├── flux_model.py                     # build_flux_pipeline()
│   └── minimax_h3_model.py               # build_minimax_h3_pipeline()
├── wan_infer.py                          # Wan2.2 入口脚本
├── qwen_image_infer.py                   # Qwen-Image 入口脚本
├── flux_infer.py                         # FLUX.1-dev 入口脚本
├── minimax_h3_infer.py                   # MiniMax-H3 入口脚本
├── requirements.txt                      # 依赖声明
└── README.md
```

## 前置准备

```shell
pip install -r examples/dummy_run/requirements.txt
```

| 依赖 | 最低版本 |
|---|---|
| Python | 3.10 |
| torch / TorchNPU | 与 CANN 版本匹配 |
| diffusers | >= 0.40.0（MiniMax-H3 需要 0.40.0+，其余模型 0.34.0 即可） |
| transformers | >= 4.56.0（Qwen3-VL） |
| huggingface_hub | >= 0.23.0 |

确保 NPU 可用：`npu-smi info -l`

## CLI 参数（四模型统一）

```shell
python <model>_infer.py --device_id <N> --num_layers <N>
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--device_id` | 0 | NPU 设备索引 |
| `--config_cache` | 无 | 离线模式，指定本地配置文件目录 |
| `--num_layers` | 2 | Transformer 层数 |
| `--compile` | False | 使能 MindieSDBackend 编译 |
| `--profile` | False | 使能 NPU profiling (level=l1) |
| `--skip-vae` / `--no-skip-vae` | True | 跳过 VAE decode（默认）。`--no-skip-vae` 开启（Wan2.2 / Qwen-Image / FLUX.1-dev） |
| `--compute-precision` | bf16 | 计算精度 `bf16` / `fp32`（Wan2.2 / MiniMax-H3）。bf16 = 模型层权重 cast + 激活原生 bf16，编译侧零隐式精度转换（GEMM ~15× 加速） |

MiniMax-H3 额外参数（见 [MiniMax-H3](#minimax-h3) 小节）：`--height`、`--width`、`--num_frames`、`--num_inference_steps`，无 `--skip-vae`（固定输出 latent，不解码）。

## 配置缓存

配置文件（JSON、tokenizer，KB 级）首次运行时自动下载到 HuggingFace Hub 缓存，后续运行免联网。
通过 `--config_cache /path/to/config` 可指定离线缓存目录。

对于 **gated model**（如 FLUX.1-dev），设 `HF_TOKEN` 或从 modelscope 下载后
通过 `--config_cache` 加载。

## 验证记录（910B, 64GB HBM, NPU 直连, 2 layers）

| Model | Build (ms) | Timed (ms) | Peak Mem | Status |
|---|---|---|---|---|
| Wan2.2 | 1,200 | 7,000 | 10.18 GB | PASSED |
| Qwen-Image | 7,000 | 100 | 6.26 GB | PASSED |
| FLUX.1-dev | 20,500 | 900 | 24.20 GB | PASSED |
| MiniMax-H3 (bf16) | 800 | 51 | 13.50 GB | PASSED |

> MiniMax-H3 默认 bf16（`--compute-precision bf16`）：eager fp32 下 Timed 338ms / 21.90GB。

---

## Wan2.2

### 模型组件

| 组件 | 类 | 层数 |
|---|---|---|
| Transformer | `WanTransformer3DModel` | 2 (原始 40) |
| Transformer_2 | `WanTransformer3DModel` | 2 |
| Text Encoder | `UMT5EncoderModel` | 2 (原始 28) |
| VAE | `AutoencoderKLWan` | — |
| Scheduler | `UniPCMultistepScheduler` | — |

### 使用方式

```shell
python wan_infer.py --device_id 0
python wan_infer.py --device_id 0 --num_layers 4
python wan_infer.py --device_id 0 --no-skip-vae      # 输出视频帧
python wan_infer.py --device_id 0 --config_cache /path/to/config
python wan_infer.py --device_id 0 --compile
python wan_infer.py --device_id 0 --profile
```

### 内嵌默认值

- height: 720, width: 1280, num_frames: 81
- num_inference_steps: 1（warmup 1, timed 1）
- guidance_scale: 1.0, prompt: "test"

---

## Qwen-Image

### 模型组件

| 组件 | 类 | 层数 |
|---|---|---|
| Transformer | `QwenImageTransformer2DModel` | 2 (原始 60) |
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | 2 (原始 28) |
| VAE | `AutoencoderKLQwenImage` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |
| Tokenizer | `Qwen2Tokenizer` | — |

### 使用方式

```shell
python qwen_image_infer.py --device_id 0
python qwen_image_infer.py --device_id 0 --num_layers 4
python qwen_image_infer.py --device_id 0 --no-skip-vae    # 输出图像
python qwen_image_infer.py --device_id 0 --config_cache /path/to/config
python qwen_image_infer.py --device_id 0 --compile
python qwen_image_infer.py --device_id 0 --profile
```

### 内嵌默认值

- height: 1024, width: 1024
- num_inference_steps: 1（warmup 1, timed 1）
- true_cfg_scale: 1.0, prompt: "test"

---

## FLUX.1-dev

### 模型组件

| 组件 | 类 | 层数 |
|---|---|---|
| Transformer | `FluxTransformer2DModel` | 2 |
| Text Encoder (CLIP) | `CLIPTextModel` | 1 (原始 12) |
| Text Encoder (T5) | `T5EncoderModel` | 2 (原始 24) |
| VAE | `AutoencoderKL` | — |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |

### Gated model 配置

FLUX.1-dev 需鉴权。二选一：

```shell
# 方式 A: 设置 HF_TOKEN
export HF_TOKEN=hf_xxx
python flux_infer.py --device_id 0

# 方式 B: modelscope 离线下载后指定缓存
python flux_infer.py --device_id 0 --config_cache /home/lb/workspace/flux_configs
```

### 使用方式

```shell
python flux_infer.py --device_id 0
python flux_infer.py --device_id 0 --num_layers 4
python flux_infer.py --device_id 0 --no-skip-vae     # 输出图像
python flux_infer.py --device_id 0 --config_cache /path/to/config
python flux_infer.py --device_id 0 --compile
python flux_infer.py --device_id 0 --profile
```

### 内嵌默认值

- height: 1024, width: 1024
- num_inference_steps: 1（warmup 1, timed 1）
- guidance_scale: 1.0, max_sequence_length: 512, prompt: "test"

---

## MiniMax-H3

### 模型组件

MiniMax-H3（33B 全模态生成模型，T2VA/FL2VA/Ref2VA 工作流）以 diffusers `MiniMaxH3ModularPipeline` 接入，
dummy run 覆盖 `t2va`（文生音视频）工作流：文本编码 + packed 序列去噪（视频+音频）。参考权重为
[Hugging Face MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)（根目录为 diffusers 格式，
`FL2VA/`、`Ref2VA/` 为 vLLM-Omni 格式）。

| 组件 | 类 | 层数 |
|---|---|---|
| Transformer | `MiniMaxH3Transformer3DModel` | 2 (原始 50) |
| Text Encoder | `Qwen3VLForConditionalGeneration` | 2 (原始 64) |
| Video VAE | `AutoencoderKLMiniMaxH3` | —（仅统计参数量，不参与前向） |
| Audio VAE | `AutoencoderKLMiniMaxH3Audio` | —（仅统计参数量，不参与前向） |
| Scheduler | `MiniMaxH3Scheduler` (shift=12.0) | — |
| Audio Scheduler | `MiniMaxH3Scheduler` (shift=3.0) | — |
| Tokenizer / Processor | `Qwen2TokenizerFast` / `Qwen3VLProcessor` | — |

### 配置文件获取（modelscope）

MiniMax-H3 在 HF 上为 **gated 模型**，需要审批；modelscope 镜像 `MiniMax/MiniMax-H3` 无需鉴权。
脚本优先通过 modelscope 下载配置文件（KB 级，只拉取 json/py/txt/tokenizer，不含 safetensors）：

```shell
# 方式 A: 自动从 modelscope 下载配置（无需任何 token）
python minimax_h3_infer.py --device_id 0

# 方式 B: 手动离线下载后指定缓存目录
pip install modelscope
python -c "from modelscope import snapshot_download; \
    print(snapshot_download('MiniMax/MiniMax-H3', allow_patterns=['*.json','*.txt','*.py','tokenizer*'], \
    ignore_patterns=['*.safetensors','*.bin']))"
python minimax_h3_infer.py --device_id 0 --config_cache /path/to/MiniMax-H3-configs
```

> 注意：模型权重目录（`hf download MiniMaxAI/MiniMax-H3` 或 vLLM-Omni 部署目录）中的 `FL2VA/` 是
> vLLM-Omni 格式，不能作为 `--config_cache`；请使用仓库**根目录**的 diffusers 格式配置。

### 使用方式

```shell
python minimax_h3_infer.py --device_id 0                    # 默认 bf16
python minimax_h3_infer.py --device_id 0 --num_layers 4
python minimax_h3_infer.py --device_id 0 --height 512 --width 768
python minimax_h3_infer.py --device_id 0 --compute-precision fp32   # 对照 fp32（慢 ~10x）
python minimax_h3_infer.py --device_id 0 --config_cache /path/to/config
python minimax_h3_infer.py --device_id 0 --compile
python minimax_h3_infer.py --device_id 0 --profile
```

### 内嵌默认值

- compute precision: **bf16（默认）**——MiniMax-H3 DiT 无 fp32 强制岛（dtype 全由参数决定），
  权重 cast bf16 后整个 DiT block stack 原生 bf16 计算，编译侧零隐式精度转换
- height: 256, width: 384（小画布：MiniMax-H3 对 packed 序列做全自注意力 O(seq²)，768×1344 在单卡不可行）
- num_frames: 124（17×7+5，最短 5s @ 24fps，范围 5–15s）
- num_inference_steps: 2（`MiniMaxH3Scheduler` 要求 ≥2，2 步 = 1 次 transformer 前向）
- prompt: "test"

### 与其余 dummy run 的差异

- **固定输出 latent**：为规避自定义 remote-code 的 Video/Audio VAE decode（NPU 兼容性未验证），
  dummy run 裁剪掉 pipeline 的 decode 块，`t2va` 去噪后直接返回视频/音频 latents，无 `--skip-vae` 参数。
- **文本编码计时**：MiniMax-H3 的文本编码块直接驱动 `text_encoder.model` 子模块，`_PhaseTimer` 对该
  子模块单独挂 hook 计时。
- **text_encoder_layer**：完整模型在 Qwen3-VL 第 50 层 hidden state 上做条件；截断为 2 层后改为第 1 层。
- **bf16 机制**：与 Wan 不同，MiniMax-H3 DiT 没有 `.float()` 岛，无需源码级改写；`--compute-precision bf16`
  仅 cast 权重即可（详见 model-verification skill 的 minimax-h3-notes.md）。
- **--compile**：对 transformer 应用 MindieSDBackend（`_CompiledDiT` wrapper 保留原始 forward 签名，
  因 denoise 块按 `inspect.signature` 过滤行索引参数）；编译图验证无 fp32 计算节点。

---

## 已知限制

| 问题 | 说明 |
|---|---|
| tokenizer 兼容性 | diffusers `Pipeline.from_config()` 对 tokenizer 存在 bug，改为手动逐组件构造 |
| `expandable_segments:True` | 部分 NPU 环境中可能锁池导致 OOM。Wan2.2 使用该配置但不影响；Qwen/FLUX 移除后正常分配 |
| `torch.compile` + CPU offload | 不兼容（`InternalTorchDynamoError`），仅在 NPU 直连模式下可用 |
| modelscope 离线配置 | FLUX.1-dev 的 spiece.model 为 protobuf 文件，上传时禁止 CRLF→LF 转换 |
