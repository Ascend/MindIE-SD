# NPU 算子可用性快照（**环境绑定 · 用前先复核**）

> **作用域（强制读）**：本文件是**某次环境快照**（远端 910B / torch 2.8.0 / TorchNPU 2.8.0，2026-08 记录），
> **不是普适结论**。换芯片代际 / CANN / torch / torch_npu 版本后，下表结论可能**整表失效**（曾经 crash 的
> 可能已修好，曾经可用的可能改名或行为变更）。
>
> **使用纪律**：① 先按每行的「复核方法」在本机跑一次；② 复核通过才套用下表的返回约定；
> ③ **不再复现的条目请删除**（留着会误导后来者绕行已修好的原生路径）；④ 需要稳定结论时以
> `docs/zh/features/*` 与 `framework-integration/references/framework-support-matrix.md` 为准。

## 算子可用性与返回约定（快照）

| 算子 | 快照结论 | 复核方法（先跑这个） |
|---|---|---|
| `npu_add_rms_norm` | ✓ 可用，返回 3 元元组 `(out, rstd, residual)` | `python -c "import torch,torch_npu;o=torch_npu.npu_add_rms_norm(torch.randn(2,8).npu(),torch.randn(2,8).npu(),torch.randn(8).npu());print(type(o), len(o) if isinstance(o,tuple) else 1)"` |
| `npu_dynamic_quant` | ✓ 可用，返回 2 元元组 | 同上模式，换算子名并打印返回元组长度 |
| `npu_rms_norm` | ✓ 可用 | 同上 |
| `npu_fast_gelu` | ✓ 可用 | 同上 |
| `npu_add_rms_norm_dynamic_quant` | ✗ crash（core dump） | 单卡最小输入试跑；**能跑通即说明该问题已修复**，删除本行 |
| `npu_add_rms_norm_quant` | ✗ ACL 错误 161001 | 同上；错误码消失即删除本行 |

## triton vs triton-ascend 包名混淆（快照）

**现象**：`pip install triton` 装上的是 **标准 triton**，`import triton` 成功但 `driver.active` 报
`0 active drivers`，无法在 Ascend NPU 上运行。

**根因**：标准 triton 无昇腾后端；Ascend 需要 `triton-ascend`（PyPI 包名 `triton-ascend`，import 名仍是 `triton`）。

**处理**：

- Ascend 环境安装 triton 时必须 `pip install triton-ascend`，不要装标准 `triton`
- 需同时安装 `pybind11`（隐式依赖）
- 代码中用 `_TRITON_ON_ASCEND` 标志判定 triton 是否**真正可用**（而非仅可 import）

**复核方法（换版本后先跑）**：

```bash
pip show triton-ascend 2>/dev/null | head -3     # 有输出 = 装的是 ascend 版
python -c "import triton; print(triton.__version__); import triton.runtime.driver as d; print(d.active)"
# active 非 None 且不报 '0 active drivers' ⇒ 本问题不存在，无需按上面的规则绕行
```

## 维护与更新

- **触发**：远端 torch / torch_npu / CANN / triton-ascend 版本变化，或复核命令结论与快照不一致时。
- **失效信号**：复核命令报错消失（crash/错误码不再出现）、算子改名、返回约定变化、`driver.active` 正常 ——
  出现任一即**先改本文件**再谈绕行；**已修复条目一律删除，不留"以防万一"**。
- 与 `dev-workflow` 的复盘流程联动：复盘时若发现本文件条目过期，按上述纪律删改。
