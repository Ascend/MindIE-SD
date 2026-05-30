# Pattern 代码模板

三种常见模型的 pattern 代码模板，Phase 2 创建 pattern 时直接参考。

---

## 1. Wan 风格模板

Wan 使用 `FP32LayerNorm` + `scale_shift_table` 进行 modulation，
使用 `torch.nn.RMSNorm` (内置) 进行 QK normalization，
使用自定义 `apply_rotary_emb` (even/odd interleaving) 进行 RoPE。

### Wan RMSNorm

```python
def create(dtype, epsilon=1e-6):
    class WanRmsNormPattern(PatternBase):
        @staticmethod
        def name(): return __class__.__name__ + "-%s" % dtype

        @staticmethod
        def inputs():
            # 模型 graph 中输入已是 fp32 (Wan 在之前做了 .float())
            x = torch.empty(1, 75600, 5120, dtype=torch.float32, device="meta")
            weight = torch.empty(5120, dtype=dtype, device="meta")
            return [x, weight]

        @staticmethod
        def pattern(x, weight):
            def func(x, weight):
                # 分解形式 (匹配 Dynamo 的 decomposition)
                variance = x.pow(2).mean(-1, keepdim=True)
                result = x * torch.rsqrt(variance + epsilon)
                return result * weight
            return func(x, weight)

        @staticmethod
        def replacement(x, weight):
            def func(x, weight):
                return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]
            return func(x, weight)

    return WanRmsNormPattern
```

**关键点**: 输入是 fp32（Wan 的 `hidden_states.float()` 已在上游执行），
pattern 不应包含 dtype cast。使用分解形式（`pow → mean → rsqrt → mul → mul`）匹配 Dynamo 的 decomposition。

### Wan AdaLayerNorm

```python
def create(dtype, epsilon=1e-6):
    class WanAdaLayerNormPattern(PatternBase):
        @staticmethod
        def name(): return __class__.__name__ + "-%s" % dtype

        @staticmethod
        def inputs():
            x = torch.empty(1, 75600, 5120, dtype=torch.float32, device="meta")
            scale = torch.empty(1, 1, 5120, dtype=torch.float32, device="meta")
            shift = torch.empty(1, 1, 5120, dtype=torch.float32, device="meta")
            return [x, scale, shift]

        @staticmethod
        def pattern(x, scale, shift):
            def func(x, scale, shift):
                # native_layer_norm 直接调用 (atten.layernorm 被移出分解)
                # 注意参数顺序: scale + 1 (不是 1 + scale!)
                ln_out = torch.ops.aten.native_layer_norm(
                    x, [x.shape[-1]], None, None, epsilon)[0]
                return ln_out * (scale + 1) + shift
            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(
                x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)
            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(
                    layernorm=norm, x=x, scale=scale, shift=shift, fused=True)
            return func(x, scale, shift)

    return WanAdaLayerNormPattern
```

**关键点**: 使用 `torch.ops.aten.native_layer_norm` 直接调用（而非 `nn.LayerNorm` 模块），
因为 `aten.native_layer_norm` 被移出分解表，会保留单节点。
注意 `scale + 1` 的参数顺序——模型 graph 中为 `add(scale, 1)`。

### Wan RoPE

```python
def create(dtype):
    class WanRopePattern(PatternBase):
        @staticmethod
        def name(): return __class__.__name__ + "-%s" % dtype

        @staticmethod
        def inputs():
            x = torch.empty(1, 75600, 40, 128, dtype=dtype, device="meta")
            freqs_cos = torch.empty(1, 75600, 1, 128, dtype=dtype, device="meta")
            freqs_sin = torch.empty(1, 75600, 1, 128, dtype=dtype, device="meta")
            return [x, freqs_cos, freqs_sin]

        @staticmethod
        def pattern(x, freqs_cos, freqs_sin):
            def func(x, freqs_cos, freqs_sin):
                # 包含 freq slice 操作 (模型图中内联)
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)
            return func(x, freqs_cos, freqs_sin)

        @staticmethod
        def replacement(x, freqs_cos, freqs_sin):
            def func(x, freqs_cos, freqs_sin):
                cos_sliced = freqs_cos[..., 0::2]
                sin_sliced = freqs_sin[..., 1::2]
                cos_full = cos_sliced.repeat_interleave(2, dim=-1)
                sin_full = sin_sliced.repeat_interleave(2, dim=-1)
                return mindiesd.rotary_position_embedding(
                    x, cos_full, sin_full,
                    rotated_mode="rotated_interleaved",
                    head_first=False, fused=True)
            return func(x, freqs_cos, freqs_sin)

    return WanRopePattern
```

**关键点**: 输入包含完整的 `freqs_cos` / `freqs_sin`（未切片），
pattern 内部包含 slice 操作（`[..., 0::2]` / `[..., 1::2]`），
因为模型 graph 中这些 slice 是内联的。
replacement 中 `repeat_interleave(2)` 将半长 cos/sin 恢复为全长，以适配 `rotary_position_embedding` 的输入要求。

---

## 2. Qwen-Image 风格模板

Qwen-Image 使用 `nn.LayerNorm` + `img_mod`/`txt_mod` MLP 进行 modulation，
使用 diffusers `RMSNorm` 类进行 QK normalization，
使用 `apply_rotary_emb_qwen` (complex number path) 进行 RoPE。

### Qwen RMSNorm (diffusers RMSNorm fallback)

```python
def create(dtype, epsilon=1e-6):
    class QwenRmsNormPattern(PatternBase):
        @staticmethod
        def name(): return __class__.__name__ + "-%s" % dtype

        @staticmethod
        def inputs():
            x = torch.empty(1, 4096, 3584, dtype=dtype, device="meta")
            weight = torch.empty(3584, dtype=dtype, device="meta")
            return [x, weight]

        @staticmethod
        def pattern(x, weight):
            def func(x, weight):
                # diffusers RMSNorm fallback path (非 NPU 路径)
                variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
                result = x * torch.rsqrt(variance + epsilon)
                result = result.to(weight.dtype)
                return result * weight
            return func(x, weight)

        @staticmethod
        def replacement(x, weight):
            def func(x, weight):
                return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]
            return func(x, weight)

    return QwenRmsNormPattern
```

### Qwen AdaLayerNorm

```python
def create(dtype, epsilon=1e-6):
    class QwenAdaLayerNormPattern(PatternBase):
        @staticmethod
        def name(): return __class__.__name__ + "-%s" % dtype

        @staticmethod
        def inputs():
            x = torch.empty(1, 4096, 3584, dtype=dtype, device="meta")
            scale = torch.empty(1, 3584, dtype=dtype, device="meta")
            shift = torch.empty(1, 3584, dtype=dtype, device="meta")
            return [x, scale, shift]

        @staticmethod
        def pattern(x, scale, shift):
            def func(x, scale, shift):
                ln_out = torch.nn.LayerNorm(
                    x.shape[-1], elementwise_affine=False,
                    eps=epsilon, dtype=x.dtype, device=x.device)(x)
                return ln_out * (1 + scale) + shift
            return func(x, scale, shift)

        @staticmethod
        def replacement(x, scale, shift):
            norm = torch.nn.LayerNorm(
                x.shape[-1], eps=epsilon, dtype=x.dtype, device=x.device)
            def func(x, scale, shift):
                return mindiesd.layernorm_scale_shift(
                    layernorm=norm, x=x, scale=scale, shift=shift, fused=True)
            return func(x, scale, shift)

    return QwenAdaLayerNormPattern
```

**关键点**: Qwen 的 `_modulate()` 不做 `[:, None]` unsqueeze，
scale/shift 通过隐式广播与 norm output 相乘。
pattern 中的 `(1 + scale)` 直接使用隐式广播。

### Qwen RoPE (复杂数路径 — 当前已知问题)

Qwen 的 RoPE 使用 `use_real=False`（complex number）路径，
导致 `torch.compile` 过程中 `torch.view_as_complex` 后的广播失败和 inductor 警告:
`Torchinductor does not support code generation for complex operators.`

当前状态: **已禁用** (`enable_qwen_rope: bool = False`)。
修复方向: 提取 `freqs_cis` 的 real/imag 部分 → `repeat_interleave` → 走实数 RoPE 路径。

---

## 3. 通用 diffusers 模板

适用于 FLUX.1-dev 等使用标准 diffusers AdaLayerNormZero + RMSNorm + apply_rotary_emb 的模型。
详见 `mindiesd/compilation/patterns/rms_norm_pattern.py`、`adalayernorm_pattern.py`、`rope_pattern.py`。

---

## 4. Pattern 创建规范

### ABCMeta isinstance 陷阱

`PatternBase` 继承自 `ABC`（有 ABCMeta 元类），其 `isinstance` 会对实现抽象方法的子类返回 `True`。
判断实例 vs 类时必须额外排除 `type`：

```python
if not isinstance(pat, type) and isinstance(pat, PatternBase):
    # 实例路径
else:
    # 类路径
```

### 去重注册

使用模块级 `_registered_pattern_names: set[str]` 记录已注册 pattern。
测试 setUp 须同时清理该集合和 `patterns.pattern_replacements`。

### 外接逻辑融合优先于接口照搬

从外部项目（如 vllm-ascend）引入融合逻辑时，逻辑层直接采用外部项目的 pattern 形状、
replacement 目标、kernel 调用方式，接口层保持本地 `PatternBase` 的 `@staticmethod` 接口约定。
外部项目通过构造函数注入的参数用工厂函数 + 闭包桥接。

### 融合 Operator 速查表

| 原语 | 融合 operator | 调用方式 |
|------|-------------|---------|
| RMSNorm | `torch_npu.npu_rms_norm` | `torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]` |
| RoPE | `mindiesd.rotary_position_embedding` | `rotary_position_embedding(x, cos, sin, rotated_mode="rotated_interleaved", head_first=False, fused=True)` |
| AdaLayerNorm | `mindiesd.layernorm_scale_shift` | `layernorm_scale_shift(layernorm=norm, x=x, scale=scale, shift=shift, fused=True)` |
| GELU | `torch_npu.npu_fast_gelu` | `torch_npu.npu_fast_gelu(hidden_states)` |

---

## 维护与更新

当新模型有与上述模板不同的结构（新的 norm 实现、不同的 RoPE 算法等），或发现现有模板存在问题时更新此文件。
