# NPU 算子验证经验

## 已验证可用的 TorchNPU 算子

远端环境（910B, torch 2.8.0, TorchNPU 2.8.0）：

| 算子 | 状态 |
|------|------|
| `npu_add_rms_norm` | ✓ 可用，返回 3 元元组 `(out, rstd, residual)` |
| `npu_dynamic_quant` | ✓ 可用，返回 2 元元组 |
| `npu_rms_norm` | ✓ 可用 |
| `npu_fast_gelu` | ✓ 可用 |
| `npu_add_rms_norm_dynamic_quant` | ✗ crash（core dump） |
| `npu_add_rms_norm_quant` | ✗ ACL 错误 161001 |

## triton vs triton-ascend 包名混淆

**问题**：`pip install triton` 安装 3.6.0，`import triton` 成功但 `driver.active` 报告 `0 active drivers`，无法在 Ascend NPU 上运行。

**根因**：标准 triton 仅支持 CUDA/ROCm 后端，Ascend 需要 `triton-ascend`（PyPI 包名 `triton-ascend`，import 名 `triton`）。

**规则**：

- Ascend 环境安装 triton 时必须用 `pip install triton-ascend`，不能安装标准 `triton`
- 需同时安装 `pybind11` 作为隐式依赖
- 代码中通过 `_TRITON_ON_ASCEND` 标志区分 triton 是否真正可用（而非仅可 import）

## 维护与更新

当NPU 算子 API 或环境诊断方法变化时，按 dev-workflow 的复盘流程更新本文件。
