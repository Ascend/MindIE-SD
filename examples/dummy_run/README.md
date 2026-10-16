# 空权重运行验证

在 NPU 设备上**不依赖真实权重**（仅下载配置文件，几十 KB）构造生成类模型并完成前向推理验证。

## 目录结构

```text
examples/dummy_run/
├── model/
│   ├── __init__.py                       # check_npu(), resolve_config_path(), _PhaseTimer
│   ├── wan_model.py                      # build_wan_pipeline()
│   ├── qwen_image_model.py               # build_qwen_image_pipeline()
│   └── flux_model.py                     # build_flux_pipeline()
├── wan_infer.py                          # Wan2.2 入口脚本
├── qwen_image_infer.py                   # Qwen-Image 入口脚本
├── flux_infer.py                         # FLUX.1-dev 入口脚本
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
| torch / torch_npu | 与 CANN 版本匹配 |
| diffusers | >= 0.34.0 |
| transformers | >= 4.44.0 |
| huggingface_hub | >= 0.23.0 |

确保 NPU 可用：`npu-smi info -l`

## CLI 参数（三模型统一）

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
| `--skip-vae` / `--no-skip-vae` | True | 跳过 VAE decode（默认）。`--no-skip-vae` 开启 |

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

## 已知限制

| 问题 | 说明 |
|---|---|
| tokenizer 兼容性 | diffusers `Pipeline.from_config()` 对 tokenizer 存在 bug，改为手动逐组件构造 |
| `expandable_segments:True` | 部分 NPU 环境中可能锁池导致 OOM。Wan2.2 使用该配置但不影响；Qwen/FLUX 移除后正常分配 |
| `torch.compile` + CPU offload | 不兼容（`InternalTorchDynamoError`），仅在 NPU 直连模式下可用 |
| modelscope 离线配置 | FLUX.1-dev 的 spiece.model 为 protobuf 文件，上传时禁止 CRLF→LF 转换 |
