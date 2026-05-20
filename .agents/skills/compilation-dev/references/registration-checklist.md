# 三段注册检查清单

Phase 3 创建 pattern 后，必须在这 3 个文件中追加注册条目。

---

## 1. `mindiesd/compilation/patterns/__init__.py`

```python
# __all__ 列表追加
__all__ = [
    # ... existing ...
    'ModelOpPatternGroup',          # ← 追加
]

# import 追加 (底部)
from .model_op_pattern import ModelOpPatternGroup   # ← 追加
```

**核对**:

- [ ] `__all__` 列表中的名称与 import 的 PatternGroup 名称一致
- [ ] import 路径正确: `.model_op_pattern` = 同目录下的 `model_op_pattern.py`
- [ ] 按字母序排序（与其他 import 保持一致性）

---

## 2. `mindiesd/compilation/passes/__init__.py`

```python
pattern_registry = {
    # ... existing ...
    "enable_model_op": ("ModelOpPatternGroup", "..patterns"),  # ← 追加
}
```

**核对**:

- [ ] key (`enable_model_op`) 与 Phase 3 的 `FusionPatterns` dataclass 字段名完全一致
- [ ] value 的 tuple 中第一个元素 = Phase 1 `__all__` 中的 PatternGroup 名称
- [ ] `"..patterns"` 是相对于 `passes/` 目录的 module path（通常不变）

---

## 3. `mindiesd/compilation/compiliation_config.py`

```python
@dataclasses.dataclass(frozen=False)
class FusionPatterns:
    # ... existing ...
    enable_model_op: bool = True          # ← 追加
```

**核对**:

- [ ] 字段名与 registry key 完全一致
- [ ] 默认值为 `True`（默认启用新 pattern）
- [ ] 若 pattern 处于实验阶段或已知有问题，可设为 `False`

---

## ⚠️ Decomp Table 默认参数一致性

`mindiesd/compilation/passes/pattern_match_pass.py` 中 `fwd_only_with_custom_decomp` 函数
的 `get_decomp_fn` 默认参数**必须与 import 语句一致**。

```python
# 正确: import 与默认参数指向同一函数
from .._custom_decomposition import select_pattern_decomp_table   # line 21

def fwd_only_with_custom_decomp(..., get_decomp_fn=select_pattern_decomp_table): # line 107
```

**常见 bug**: import 改了但默认参数未同步 → pattern trace 使用错误的分解表 → 全部 pattern 失效。

**核对**:

- [ ] line 21 import 与 line 107 默认参数指向同一函数
- [ ] 该函数返回的表包含 `aten.rms_norm` 的分解规则（否则 RMSNorm 不会被分解）

---

## 命名规范

| 组件 | 格式 | 示例 |
|------|------|------|
| config key | `enable_<model>_<op>` | `enable_wan_rope`, `enable_qwen_rms_norm` |
| PatternGroup | `<Model><Op>PatternGroup` | `WanRopePatternGroup`, `QwenRmsNormPatternGroup` |
| Pattern class | `<Model><Op>Pattern` | `WanRopePattern`, `QwenRmsNormPattern` |
| File name | `<model>_<op>_pattern.py` | `wan_rope_pattern.py`, `qwen_rms_norm_pattern.py` |

**缩写原则**: model 名用小写简称（`wan`, `qwen`, `flux`），op 名用完整小写（`rms_norm`, `adalayernorm`, `rope`），避免歧义。

---

## 维护与更新

当注册接口变更（如 `pattern_registry` 结构变化）、命名规范调整或新增独立的编译配置维度时更新此文件。
