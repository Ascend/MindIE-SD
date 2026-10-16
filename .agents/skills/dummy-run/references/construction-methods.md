# 模型构造方式与配置获取

## 1. 模型构造方式

有两种构造方式，各有适用场景和已知陷阱：

### 方式 A：meta→to_empty（推荐用于已知兼容的模型）

```python
with torch.device("meta"):
    model = ModelClass.from_config(cfg, torch_dtype=torch.bfloat16)
model.to_empty(device=torch.device("cpu"))  # 或 npu_device
```

- **优点**：构造时不分配 CPU/NPU 内存，极快
- **适用范围**：`WanTransformer3DModel`、`AutoencoderKLWan`、`AutoencoderKL` 等经过验证的模型

### 方式 B：CPU 直接构造（兜底方案）

```python
model = ModelClass.from_config(cfg, torch_dtype=torch.bfloat16)  # 分配 CPU 内存
model.to(npu_device)
```

- **优点**：100% 可靠，不依赖 meta 设备
- **缺点**：`from_config` 时占用 CPU RAM（大模型可能 OOM）

### 已知陷阱

| 陷阱 | 表现 | 解决 |
|---|---|---|
| meta→to_empty 后未注册 buffer 残留 meta 设备 | 运行时 `NotImplementedError: Cannot copy out of meta tensor; no data!` | 降级为方式 B（CPU 直接构造）。常见于含位置编码预计算的模型（如 `QwenImageTransformer2DModel`） |
| meta→to_empty→cpu 后与 accelerate hooks 不兼容 | NPU 硬件错误 `SUSPECT REMOTE ERROR, error code 507057` | 降级为方式 B。常见于 `CLIPTextModel` 等通过 accelerate 管理的轻量模型 |
| VL 模型未裁剪 vision tower | NPU OOM（vision tower 通常 32 层） | 同步设置 `vision_config.depth`（见模型特定注意事项） |

### 模型特定注意事项

**Qwen2_5_VLForConditionalGeneration**（Qwen-Image 的 text_encoder）：

- 含独立 vision tower（`vision_config.depth = 32`），裁剪 `num_hidden_layers` 时必须同步裁剪 `vision_config.depth`
- 推荐始终用方式 B（CPU 构造），避免 VL 模型复杂性

**FluxTransformer2DModel**：

- 层数参数可能有多个名称：`num_layers`、`num_single_transformer_blocks`、`num_joint_transformer_blocks`
- 为兼容性，三者同时设置相同值

## 2. Gated Model 配置获取

对于 gated model（如 FLUX.1-dev）：

| 方案 | 说明 |
|---|---|
| `HF_TOKEN` 环境变量 | 接受 license 后设置 token，`snapshot_download` 自动鉴权 |
| **modelscope 离线下载**（推荐） | 从 modelscope 下载 KB 级配置文件，`--config_cache` 指定路径，无需 HF_TOKEN |
| 公开组件拆分 | 仅适用于 text_encoder/tokenizer 来自公开 repo 的模型（如 FLUX 的 CLIP + T5） |

**modelscope 离线流程**：

```python
# 本地下载配置
from modelscope import snapshot_download
config_dir = snapshot_download(
    "black-forest-labs/FLUX.1-dev",
    allow_patterns=["*.json", "*.txt", "*.model", "tokenizer*", "merges*", "vocab*"],
    ignore_patterns=["*.safetensors", "*.bin", "*.ckpt", "*.pth"],
)
# 上传到远端后使用 --config_cache
```

## 维护与更新

当模型构造方式或 gated config 获取方法变化时，按 dev-workflow 的复盘流程更新本文件。
