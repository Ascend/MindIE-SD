# FLUX.1-dev 模型推理优化指南

使能MindIE-SD的编译优化功能和Cache-DiT的DBCache功能，实现FLUX.1-dev模型的推理加速。

> 本文面向部署 FLUX.1-dev 推理服务的终端用户，是可直接运行的端到端案例。
> 特性实现原理与接口说明（CompilationConfig、CacheConfig 等）请参见开发者的[编译特性](../../docs/zh/features/compilation.md)与[以存代算](../../docs/zh/features/cache.md)。

---

## 环境信息

本指南的实验环境如下：

| 组件 | 版本 |
|------|------|
| 服务器 | Atlas 910B4（HBM 64GB） |
| Python | 3.11 |
| diffusers | 0.36.0 |
| torch | 2.9.0 |
| torch_npu | 2.9.0 |
| mindiesd | 3.0.0 |
| cache_dit | 1.2.3 |

---

## 1. 前置准备

### 1.1 基础环境安装（昇腾 NPU）

请参考 [MindIE-SD 安装指导](../../docs/zh/installation.md) 完成基础环境搭建，包括：

- 驱动固件安装
- CANN 安装
- PyTorch 和 Torch NPU 安装
- MindIE-SD 安装

### 1.2 安装 Diffusers 和 Cache-DiT

在完成基础环境安装后，还需要安装以下 Python 包：

```bash
# 安装 Diffusers（支持 FLUX.1-dev 的版本）
pip install diffusers

# 安装 Cache-DiT
pip install cache-dit
```

### 1.3 下载模型权重

- 原始权重来自 [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev)（gated model，需鉴权）
- 国内权重可使用 [ModelScope](https://modelscope.cn/models/AI-ModelScope/FLUX.1-dev)

---

## 2. 模型推理示例

### 2.1 最简单的验证（基于 cache-dit CLI）

无需编写推理代码，使用 cache-dit 自带的 `generate` 命令即可快速验证 MindIE-SD 编译优化效果：

```bash
# 设置 FLUX.1-dev 权重路径
export FLUX_PATH=/path/to/FLUX.1-dev

# 非编译（基线）
python3 -m cache_dit.generate flux --model-path $FLUX_PATH

# 使能 MindIE-SD 编译优化
python3 -m cache_dit.generate flux --model-path $FLUX_PATH --compile
```

对比两次运行的耗时即可验证编译优化的加速效果。`--compile` 在 NPU 上会自动使用 `MindieSDBackend` 进行编译。

### 2.2 基于diffusers的标准推理

以下是基于diffusers库的标准推理代码，未使能任何优化

```python
#!/usr/bin/env python3
import torch
import torch_npu
from diffusers import FluxPipeline

# 初始化npu环境
DEVICE_ID = 0
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"

# 加载模型权重
model_path = ""  # 模型权重的保存路径
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# 准备输入
prompt = "A cat holding a sign that says hello world"

# 执行模型推理
with torch.inference_mode():
    output = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=28,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    output.save("flux_out.png")
```

### 2.3 使能编译优化（cache-dit 原生支持 MindIE-SD）

cache-dit 在昇腾 NPU 环境检测到已安装 mindiesd 时，会自动使用 MindIE-SD 的 `MindieSDBackend()` 对 transformer 进行编译优化（与 `torch.compile(pipe.transformer, backend=MindieSDBackend())` 等价），RMSNorm、RoPE、AdaLayerNorm、fastGELU 等算子会被自动替换为融合算子，**无需修改推理代码**。首次推理因包含编译预热，耗时较久，从第二次起才是真实推理速度。

```bash
python3 -m cache_dit.generate flux --model-path $FLUX_PATH --compile
```

> [!NOTE]说明
> torch_npu 2.9 起（MR 30358）移除了 `Tensor.to` 的 NPU 过适配，编译图中的 dtype cast 由 `torch.ops.npu._npu_dtype_cast` 变为 `torch.ops.aten._to_copy`。MindIE-SD 3.0.0 已按 torch 版本自动适配，无需额外配置；若在 torch 2.9+ 环境观察到 RMSNorm / RoPE 融合未生效，请确认 MindIE-SD 版本已包含该版本适配。

### 2.4 使能Cache-DiT的DBCache功能

**注意**：使能DBCache功能前，需要设置环境变量`export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`，否则可能会出现内存溢出。

FLUX.1-dev 的 transformer 包含双流（`transformer_blocks`）与单流（`single_transformer_blocks`）两种 block，需分别配置缓存参数。本环境验证可用的推荐参数组合见 2.5 完整脚本：核心设置为 `residual_diff_threshold=0.4`，单流 block 的误差会从双流 block 累积，阈值取三倍（1.2）；每次推理前需调用 `cache_dit.refresh_context()` 刷新缓存上下文。`DBCacheConfig`、`BlockAdapter`、`ParamsModifier` 等各参数含义与更多配置方式请参考 [Cache-DiT CACHE_API 文档](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/)。

### 2.5 完整单卡模型推理示例

结合编译优化（CLI 自动启用，见 2.3）与 Cache-DiT DBCache 细粒度配置的完整推理脚本 `flux_optimized_infer.py` 如下所示：

```python
#!/usr/bin/env python3
import torch
import torch_npu
import cache_dit
from diffusers import FluxPipeline
from cache_dit import (
    BlockAdapter,
    ForwardPattern,
    ParamsModifier,
    DBCacheConfig,
)

# 初始化npu环境
DEVICE_ID = 0
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"

# 加载模型权重
model_path = ""  # 模型权重的保存路径
pipe = FluxPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to(device)

# 使能Cache-DiT的DBCache功能
# （MindIE-SD 编译优化由 cache-dit 在 CLI --compile 模式下自动启用，无需在此配置）
cache_dit.enable_cache(
    BlockAdapter(
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_1,
        ],
    ),
    cache_config=DBCacheConfig(
        Fn_compute_blocks=2,
        Bn_compute_blocks=1,
        max_warmup_steps=4,
        max_cached_steps=8,
        max_continuous_cached_steps=4,
        residual_diff_threshold=0.4,
    ),
    params_modifiers=[
        ParamsModifier(
            cache_config=DBCacheConfig().reset(residual_diff_threshold=0.4),
        ),
        ParamsModifier(
            cache_config=DBCacheConfig().reset(residual_diff_threshold=0.4 * 3),
        ),
    ],
)

# 准备输入
prompt = "A cat holding a sign that says hello world"

# 执行模型推理
with torch.inference_mode():
    for i in range(2):
        # 每次推理前刷新缓存上下文
        cache_dit.refresh_context(
            pipe.transformer,
            num_inference_steps=28,
        )
        output = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            num_inference_steps=28,
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]
        output.save(f"flux_out_{i}.png")
```

---

## 参考链接

- [MindIE-SD 编译特性文档](../../docs/zh/features/compilation.md)
- [Cache-DiT 使用说明](https://gitcode.com/vipshop/cache-dit)
- [FLUX.1-dev 模型](https://huggingface.co/black-forest-labs/FLUX.1-dev)
