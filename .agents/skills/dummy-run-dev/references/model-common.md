# dummy run `model/common` 通用能力

> **定位**：`examples/dummy_run/model/common/` 承载 dummy run **跨模型共享**的能力，
> 按职责分类（精度 / 编译性能补丁 / 量化），与模型特有的 builder
> （`model/flux_model.py`、`model/qwen_image_model.py` 等）分离。

## 目录结构

```text
examples/dummy_run/model/
├── __init__.py            # 基础设施: check_npu / resolve_config_path / _PhaseTimer
├── flux_model.py ...      # 各模型 builder（模型特有）
└── common/
    ├── __init__.py        # 统一导出面（7 个符号）
    ├── precision.py       # 精度类: bf16/fp32 模型级机制（--quant bf16/fp32）
    ├── compile_patches.py # 性能补丁类: compile 图层性能问题的模型层替换
    └── quantization.py    # 量化类: W8A8 在线量化，设备感知（--quant w8a8）
```

## 模块职责与 API

### `precision.py` — 模型级 bf16/fp32 精度机制

| 函数 | 职责 |
|---|---|
| `compute_dtype_from_precision("bf16"/"fp32")` | 精度字符串 → torch dtype |
| `apply_compute_precision(pipe, precision)` | 组件权重 cast + 触发精度岛改写（bf16）；fp32 原行为 |
| `_rewrite_apply_rotary_emb(dtype)` | 源码级改写 diffusers `apply_rotary_emb` 的 fp32 岛（FLUX/Qwen 共用） |
| `_rewrite_apply_rotary_emb_qwen(dtype)` | Qwen complex 旋转改实数域等价（qwen_rope pattern 前置） |
| `verify_compute_precision_graph()` | 遍历编译图标记 fp32/int32 计算节点（验证图真 bf16） |

**要点**：bf16 必须"图真正 bf16"（源码级改写精度岛），否则编译侧隐式转换会打断 pattern 匹配。

### `compile_patches.py` — compile 图层性能补丁（纯性能，与精度/量化正交）

| 函数 | 解决的问题 |
|---|---|
| `replace_zero_dropout(module)` | `dropout(p=0)` → `nn.Identity`（compile 图残留 `aten.dropout` → DropoutV3 kernel 空跑 ~1.43ms/step） |
| `replace_pos_embed_with_buffers(...)` | qwen pos_embed 输出预计算为 buffer（AiCpu freqs 生成链 ~1.70ms/step；qwen rope pattern 的 freqs 绑定 buffer 不受影响） |

**共性**：eager 无开销、compile 图保留无意义节点/热点 → 用模型层模块替换解决（曾尝试 pattern matcher 方案触发死循环，故走模型层）。

### `quantization.py` — W8A8 在线量化（设备感知）

| 函数 | 职责 |
|---|---|
| `apply_w8a8_quant(pipe, attrs, dtype, fallback_layers, algorithm=None)` | 主入口：只量化 `nn.Linear`（Matmul），其余向量运算保持 bf16；`algorithm=None` 按设备自动选择 |
| `_resolve_w8a8_algorithm()` | `NPUDevice.A5` → `W8A8_MXFP8`；A2/A3/Duo → `W8A8_DYNAMIC`（INT8） |
| `apply_mxfp8_quant(...)` | 兼容别名（强制 MXFP8，历史脚本使用） |
| `report_quant_layers(pipe, attrs)` | 汇总量化命中（quant linear / remaining nn.Linear） |
| `_align_bias_dtype(module, dtype)` | 兜底：量化层 bias 对齐 bf16（防 Dynamo guard 失败重编译） |

**设备映射**（`--quant w8a8`）：A5（950PR）→ MXFP8；A2/A3（910B/910C）→ INT8。
INT8 路径（`W8A8_DYNAMIC` → `W8A8OnlineQuantLinear`）实现完毕但需在真实 910B/910C 上验证。

**量化范围**（kernel 实证）：只有 `nn.Linear` 被替换；GroupMatmul 仅走 MoE 路径（dummy 无 MoE）；
FA 不量化；图中无 fp32 计算节点。

## 脚本接入方式

各 `*_infer.py` 统一 `from model.common import ...`（7 个符号：`apply_compute_precision`、
`apply_w8a8_quant`、`compute_dtype_from_precision`、`replace_pos_embed_with_buffers`、
`replace_zero_dropout`、`report_quant_layers`、`verify_compute_precision_graph`）。

`--quant` 分发（main 内）：

```python
if args.quant == "w8a8":
    apply_compute_precision(pipe, "bf16")          # bf16 基座
    apply_w8a8_quant(pipe, attrs=("transformer",)) # 设备感知: A5->MXFP8 / A2,A3->INT8
    report_quant_layers(pipe, attrs=("transformer",))
else:
    apply_compute_precision(pipe, args.quant)      # fp32 / bf16
```

> Wan 需传 `fallback_layers={"*time_embedder*": QuantAlgorithm.W16A16}`（time_embedder
> 用 `next(iter(parameters()))` 探测 dtype，量化后参数为空会 StopIteration）。

## 与 mindiesd 框架的关系

- 复用 `mindiesd.quantize()` 在线路径，**mindiesd quantization 模块零改动**（唯一例外：修复
  guard 稳定性 bug 时改了 `mindiesd/quantization/layer.py` 的 bias 就地变异，见
  compilation-dev/references/pattern-dev.md §4）
- 设备识别用 `mindiesd.utils.get_platform.get_npu_device()`（soc 版本映射）

## 维护与更新

当新增共享能力、量化算法映射或模型接入方式变化时，按 dev-workflow 的复盘流程更新本文件。
