# 参数说明

> **目录** · 完整参数列表(必需/可选/LLM专用/Diffusion专用) · 量化方式说明 · 并行策略说明 · 日志文件命名规范 · config.json 字段说明

## 完整参数列表

### 必需参数（无默认值，必须明确指定）

| 参数类别 | 参数名 | 说明 | 用户输入示例 | 约束 |
|---------|--------|------|-------------|------|
| **模型规格** | `model_spec` | 模型的具体规格版本 | `14B`, `7B`, `T2V-14B` | 如果模型有多个规格，必须列出供用户选择 |
| **设备类型** | `--device` | 目标硬件设备名称 | `ATLAS_800_A2_376T_64G` | 必须在支持列表中或用户提供完整规格 |
| **分辨率** | `--height`, `--width` | 多模态输入分辨率（图像/视频） | `480`, `832` | 无默认值，必须明确 |
| **量化方式** | `--quantize-linear-action` | 模型量化策略 | `DISABLED`, `W8A8_DYNAMIC`, `INT8` | 无默认值，必须明确 |
| **视频帧数** | `--frame-num` | 视频模型必须明确（仅限视频模型） | `81`, `121` | 视频模型必须指定 |

### 可选参数（有默认值）

#### 通用参数

| 参数 | 默认值 | 说明 | 可选值 |
|------|--------|------|--------|
| `--seq-len` | **64** | 文本输入序列长度（tokens） | 32, 64, 128, 256, 512 |
| `--dtype` | `bfloat16` | 数据类型（推荐BF16） | float16, bfloat16, float32 |
| `--batch-size` | `1` | 批次大小 | 1, 2, 4, 8 |
| `--sample-step` | `28` | 采样步数（Diffusion模型） | 20, 28, 50 |

#### 多卡并行参数

| 参数 | 默认值 | 说明 | 约束 |
|------|--------|------|------|
| `--world-size` | **1** | 使用的卡数（默认1卡） | 1, 2, 4, 8 |
| `--ulysses-size` | **1** | Ulysses并行大小 | 多卡时必须明确 |
| `--cfg-parallel` | **false** | CFG并行 | 支持CFG的模型可选 |
| `--tp-size` | **1** | Tensor Parallelism大小 | LLM多卡时使用 |
| `--dp-size` | auto | Data Parallelism大小 | 可选 |
| `--ep-size` | **1** | Expert Parallelism大小(MoE) | MoE模型使用 |

#### 其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--context-length` | 0 | 上下文长度(tokens) |
| `--decode` | False | 启用解码模式 |
| `--compile` | False | 启用torch.compile |
| `--chrome-trace` | - | 生成Chrome trace文件 |
| `--quantize-attention-action` | `DISABLED` | Attention量化 |
| `--num-mtp-tokens` | 0 | MTP token数量(DeepSeek) |
| `--use-cfg` | False | 启用CFG(Classifier-Free Guidance) |
| `--dit-cache` | False | 启用DiT缓存 |
| `--cache-step-range` | - | 缓存步数范围 |

### 文本生成评估参数 (LLM)

| 参数 | 说明 | 默认值 | 用户必须指定 |
|------|------|--------|-------------|
| `model_id` | HuggingFace模型ID或本地路径 | - | ✅ 是 |
| `model_spec` | 模型规格 | - | ✅ **必须** |
| `--device` | 硬件设备名 | - | ✅ **必须** |
| `--num-queries` | 并行查询数量 | 1 | ❌ 否 |
| `--query-length` | 输入序列长度(tokens) | **64** | ❌ 否（有默认值） |
| `--context-length` | 上下文长度(tokens) | 0 | ❌ 否 |
| `--decode` | 启用解码模式 | False | ❌ 否 |
| `--dtype` | 数据类型 | `bfloat16` | ❌ 否 |
| `--tp-size` | Tensor Parallelism大小 | 1 | 多卡时✅**必须** |
| `--dp-size` | Data Parallelism大小 | auto | ❌ 否 |
| `--ep-size` | Expert Parallelism大小(MoE) | 1 | ❌ 否 |
| `--compile` | 启用torch.compile | False | ❌ 否 |
| `--chrome-trace` | 生成Chrome trace文件 | - | ❌ 否 |
| `--quantize-linear-action` | 线性层量化 | - | ✅ **必须** |
| `--quantize-attention-action` | Attention量化 | `DISABLED` | ❌ 否 |
| `--num-mtp-tokens` | MTP token数量(DeepSeek) | 0 | ❌ 否 |

### 视频生成评估参数 (Diffusion)

| 参数 | 说明 | 默认值 | 用户必须指定 |
|------|------|--------|-------------|
| `model_path` | 模型路径或HuggingFace ID | - | ✅ 是 |
| `model_spec` | 模型规格 | - | ✅ **必须** |
| `--device` | 硬件设备名 | - | ✅ **必须** |
| `--batch-size` | 批次大小 | 1 | ❌ 否 |
| `--seq-len` | 文本序列长度 | **64** | ❌ 否（有默认值） |
| `--height` | 图像高度 | - | ✅ **必须** |
| `--width` | 图像宽度 | - | ✅ **必须** |
| `--frame-num` | 视频帧数 | - | ✅ **必须** |
| `--sample-step` | 采样步数 | 28 | ❌ 否 |
| `--dtype` | 数据类型 | `bfloat16` | ❌ 否 |
| `--use-cfg` | 启用CFG(Classifier-Free Guidance) | False | ❌ 否 |
| `--world-size` | 设备总数 | **1** | ❌ 否 |
| `--ulysses-size` | Ulysses并行大小 | **1** | 多卡时✅**必须** |
| `--cfg-parallel` | CFG并行 | **false** | 支持CFG时可选 |
| `--dit-cache` | 启用DiT缓存 | False | ❌ 否 |
| `--cache-step-range` | 缓存步数范围 | - | ❌ 否 |
| `--quantize-linear-action` | 线性层量化 | - | ✅ **必须** |
| `--chrome-trace` | 生成Chrome trace | - | ❌ 否 |

## 量化方式说明

| 量化方式 | 说明 | 适用场景 |
|---------|------|---------|
| `DISABLED` | 无量化，使用原始精度 | 质量敏感场景 |
| `W8A8_DYNAMIC` | 动态INT8量化 | 推理加速，轻微质量损失 |
| `W8A8_STATIC` | 静态INT8量化 | 推理加速，预校准 |
| `FP8` | FP8量化 | 新一代GPU/NPU |
| `MXFP4` | 4-bit量化 | 极端压缩场景 |

## 并行策略说明

### 单卡（默认）

```bash
--world-size 1 --ulysses-size 1
```

### 多卡（4卡，Ulysses并行）

```bash
--world-size 4 --ulysses-size 4
```

### 多卡+CFG（4卡，CFG并行+Ulysses）

```bash
--world-size 4 --cfg-parallel --ulysses-size 2
```

### 自动推荐策略

如果没有明确指定并行策略，系统会：

1. **多模态生成模型**：推荐 `ulysses-size = world-size`
2. **支持CFG的模型**：推荐 `cfg-parallel` + `ulysses-size = world-size / 2`

## 日志文件命名规范

| 文件类型 | 命名格式 | 示例 |
|---------|---------|------|
| 执行日志 | `iteration_{N}.log` | `iteration_1.log` |
| 配置信息 | `config.json` | `config.json` |
| Chrome Trace | `trace_{N}.json` | `trace_1.json` |
| 通信日志 | `communication_{N}.log` | `communication_1.log` |
| 汇总数据 | `summary.json` | `summary.json` |
| 最终报告 | `evaluation_report.md` | `evaluation_report.md` |

## config.json 字段说明

```json
{
  "config_name": "配置名称",
  "model_name": "模型名称",
  "model_spec": "模型规格",
  "model_params": "参数量",
  "device": "设备名称",
  "device_spec": "设备规格",
  "model_path": "模型路径",
  "height": "图像高度",
  "width": "图像宽度",
  "frame_num": "视频帧数",
  "sample_step": "采样步数",
  "seq_len": "序列长度",
  "dtype": "数据类型",
  "quantization": "量化方式",
  "world_size": "卡数",
  "ulysses_size": "Ulysses并行大小",
  "cfg_parallel": "是否CFG并行",
  "parallel_strategy": "并行策略",
  "iterations": "迭代次数",
  "timestamp": "时间戳",
  "user_specified": {
    "model_spec": "用户指定的模型规格",
    "resolution": "用户指定的分辨率",
    "frame_num": "用户指定的帧数",
    "device": "用户指定的设备",
    "quantization": "用户指定的量化方式"
  },
  "default_used": {
    "seq_len": "使用的默认值",
    "dtype": "使用的默认值",
    "batch_size": "使用的默认值"
  }
}
```
