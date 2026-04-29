# 空权重运行验证

以 Wan2.2 为例，展示如何在 NPU 设备上**不依赖真实权重**（仅下载配置文件，几十 KB）构造生成类模型并完成前向推理验证。

## 目录结构

```shell
examples/dummy_run/
├── model/
│   ├── __init__.py              # check_npu(), resolve_config_path(), StageTracker
│   └── wan_model.py             # build_wan_pipeline()
├── wan_infer.py                 # 入口脚本
└── README.md
```

## 前置准备

| 依赖 | 最低版本 |
|---|---|
| Python | 3.10 |
| torch / torch_npu | 与 CANN 版本匹配 |
| diffusers | >= 0.33.0 |
| transformers | >= 4.44.0 |

确保 NPU 设备可用：

```shell
npu-smi info -l
```

## 使用方式

使用 2 个 Transformer block 构造轻量版 Wan2.2，单卡 910B (64GB) 可直接加载。默认跳过 VAE decode 以减少推理耗时，配置文件首次下载后自动缓存：

```shell
# 默认模式（跳过 VAE decode，推理最快）
python wan_infer.py --device_id 0

# 完整模式（含 VAE decode，输出视频帧）
python wan_infer.py --device_id 0 --no-skip-vae

# 离线模式（使用本地配置缓存）
python wan_infer.py --device_id 0 --config_cache /path/to/cache

# 使用 HF 镜像（首次下载时）
HF_ENDPOINT=https://hf-mirror.com python wan_infer.py --device_id 0

# 使能 MindieSDBackend 编译
python wan_infer.py --device_id 0 --compile

# 使能 NPU profiling
python wan_infer.py --device_id 0 --profile
```

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--device_id` | 0 | NPU 设备索引 |
| `--config_cache` | 无 | 本地配置文件目录（离线模式） |
| `--skip-vae` | True | 跳过 VAE decode。`--no-skip-vae` 启用解码输出视频帧 |
| `--compile` | False | 使能 MindieSDBackend 编译 |
| `--profile` | False | 使能 NPU profiling (level=l1) |

## 各阶段回显

运行时自动追踪各阶段耗时与内存，即时回显并输出最终汇总：

```bash
Warmup inference (1 step) ...
  [text_encoder] 0.4s
  [transformer] 7.1s

Timed inference (1 step) ...
  [text_encoder] 0.1s
  [transformer] 7.0s
Inference time: 7.1 s

Module                          Calls    Time(s)
--------------------------------------------------
text_encoder                        1        0.1
transformer                         1        7.0
--------------------------------------------------
TOTAL                                       7.1
============================================================
  Stage                           Time(s)  Mem Delta(GB)       Peak(GB)
  ----------------------------------------------------------
  Transformer                         0.1            3.49           3.49
  Transformer_2                       0.1            3.49           6.97
  VAE                                 0.1            0.47           7.44
  Text encoder + scheduler + tokenizer  1.5           10.58          30.56
  Build pipeline                      1.7           18.03          30.56
  Move to device                      0.0            0.00          30.56
  Inference                           7.1            0.00          38.33
  ----------------------------------------------------------
  TOTAL                              25.2
============================================================
```

## 内嵌默认配置

- height: 720, width: 1280, num_frames: 81
- num_inference_steps: 1（warmup 1 步，timed 1 步）
- guidance_scale: 1.0（关闭 CFG）
- prompt: "test"

## 配置缓存

配置文件（JSON、tokenizer）首次运行时自动下载并缓存到 HuggingFace Hub 目录
（`~/.cache/huggingface/hub/`），后续运行直接读本地缓存，无需联网：

```bash
~/.cache/huggingface/hub/models--Wan-AI--Wan2.2-T2V-A14B-Diffusers/snapshots/<hash>/
├── model_index.json
├── transformer/config.json
├── transformer_2/config.json
├── vae/config.json
├── text_encoder/config.json
├── tokenizer/tokenizer_config.json, spiece.model, ...
└── scheduler/scheduler_config.json
```

通过 `--config_cache` 可指定自定义缓存目录。

## 验证记录（910B, 64GB HBM）

| blocks | steps | skip VAE | 构建 | 推理 | 峰值显存 | 结果 |
|---|---|---|---|---|---|---|
| 2 | 1 | 是 | 1.7s | 7.1s | 38.33 GB | PASSED |

## 已知限制

| 问题 | 说明 |
|---|---|
| tokenizer 兼容性 | diffusers `Pipeline.from_config()` 对 tokenizer 存在 bug，改为手动逐组件构造 |
