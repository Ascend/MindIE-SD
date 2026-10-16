---
name: compilation-dev
description: >
  MindIE-SD 编译后端适配与分析。覆盖 Pattern 创建/注册/调试、Copy 算子消减、
  四后端选择 (default / torchair_ge / npugraph_ex / aclgraph) 的全生命周期。
  触发: "pattern", "compile", "Copy", "InplaceCopy", "fusion", "backend",
  "torchair_ge", "--npugraph", "register_replacement".
---

# 编译开发（Pattern + Copy 消减 + 后端选择）

## 生命周期

```text
Phase 1: 模型分析 → 提取每个原语的完整代码 + 判断参数来源
Phase 2: 创建 Pattern → 先检查现有 pattern → 决定路径:
  ├─ 参数来自 functional API (placeholder) → register_replacement
  └─ 参数来自 nn.Module (get_attr) → 自定义 Graph Pass（见 custom-graph-pass.md）
Phase 3: 三段注册
Phase 4: 单元验证 → ⚠️ 仅验证代码正确性，不等同于全模型命中
Phase 5: Debug Mismatch → graph dump → 逐节点对齐 → 修正/回退
Phase 6: 集成验证 → kernel diff 确认 → 全模型回归
Phase 7: Copy 消减 → 检测 ViewCopy 翻倍 → 后端选择
```

每阶段有明确的工具和产出物。Phase 5 可回路到 Phase 2（当 `register_replacement` 框架性失败时）。

---

## Phase 1: 分析模型结构

在 diffusers 源码中定位目标模型的 RMSNorm / RoPE / AdaLN 的实际实现代码，
提取完整代码片段作为 pattern 和测试 model 的依据。

**产出物**: 每个原语的完整代码片段。**额外记录**: 判断参数来源——`nn.Module` 参数（`self.weight`）为 `get_attr`，
函数输入为 `placeholder`。这直接决定 Phase 2 的路径选择。

---

## Phase 2: 创建 Pattern + 路径选择

**原则**: 始终创建新文件（非侵入式），不修改现有 pattern 文件。
先检查现有 pattern 是否匹配（复用决策），Phase 5 确认不匹配后才新建。

**路径选择**: 由 Phase 1 的「参数来源」决定：

| 目标算子参数来源 | Pattern 路径 | 执行位置 |
|-----------------|-------------|---------|
| functional API (`F.rms_norm` 等) | `register_replacement` | `register_pattern_to_pass` |
| `nn.Module` (`self.weight` 等) | 自定义 Graph Pass | `graph_rewrite_after_freezing` |

- **register_replacement 路径**: 创建 `PatternBase` 子类（工厂+闭包），注册到 `pattern_registry`。
  代码模板和融合 op 速查见 `references/pattern-templates.md`。
- **自定义 Graph Pass 路径**: 直接跳至 Phase 5 → 实现见 `references/custom-graph-pass.md`。

---

## Phase 3: 三段注册

**总是 3 个文件**（全部是代码追加）：

1. `patterns/__init__.py` — `__all__` + `from .xxx_pattern import XxxPatternGroup`
2. `passes/__init__.py` — `pattern_registry` 字典追加
3. `compiliation_config.py` — `FusionPatterns` dataclass 追加 `enable_xxx: bool = True`

命名规范: config key 使用 `enable_<model>_<op>` 格式。检查清单见 `references/registration-checklist.md`。

---

## Phase 4: 单元验证

Test Model 的 forward 与 Phase 1 提取的代码完全一致。
验证标准: `cosine_similarity(compiled, original) > 2^-7`。

> ⚠️ **单元测试通过 ≠ pattern 命中了模型**。测试 model 与 pattern 共享相同代码 → 必然匹配。
> 全模型匹配需 Phase 6 的 kernel diff 最终确认。
> 若 pattern 涉及 `nn.Module` 参数（`get_attr`），单元测试通过但全模型静默失败——见 mismatch 类型 7。

测试组织与模板见 `references/test-templates.md`。

---

## Phase 5: Debug Mismatch

若 Phase 4 通过但全模型未命中：

1. Dump 模型 traced FX graph → 保存为文本
2. 定位目标算子子图 → 逐节点对齐 pattern → 定位第一个不匹配节点
3. 对照 mismatch 类型修正

### 路径 A: 修正 `register_replacement` pattern（类型 1-5）

修正 pattern 代码使之与模型 graph 一致，重新部署。详见 `references/mismatch-catalog.md` 类型 1-5。

### 路径 B: 改用自定义 Graph Pass（类型 6）

当 mismatch 原因为 `placeholder` vs `get_attr`（mismatch 类型 6），
`register_replacement` 框架无法修复。回退到 Phase 2，实现自定义 FX graph traversal pass。

**关键注意**: 自定义 pass 中插入的 NPU custom op（如 `npu_rms_norm`）必须在
`graph_rewrite_after_freezing` 中调用，不能在 freeze 之前。否则 `torch._inductor.freezing.freeze()` 的
`node_copy` 会因不识别 NPU op 而 Crash。

完整实现指南见 `references/custom-graph-pass.md`。

操作指南: `references/graph-comparison-guide.md`。

---

## Phase 6: 集成验证

依次通过三层验证，**顺序不可跳**:

| 层级 | 方法 | 确认内容 |
|------|------|---------|
| 1. Pattern 命中 | graph dump + 日志 | `PatternMatchPass replace N` 增加 |
| 2. Fusion kernel | profiling → `kernel_details.csv` | 融合 kernel 出现 + 原始 kernel 消失 |
| 3. 全模型回归 | `dummy run --compile` | 推理正常完成且无 crash |

**Kernel 级 diff 验证**（比 pattern 日志更可靠）:

1. 分别采集 eager 和 compile 的全模型 profiling（`--profile`）
2. 从 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 按 kernel 名称聚合耗时
3. 对比 `eager_only`（被融合的原始算子）和 `compile_only`（新增的融合 kernel）
4. 同名 kernel 耗时差排序 → 定位编译开销

**Kernel 名称确认**:

- RMSNorm → `rms_norm` / `RmsNorm`
- RoPE → `npu_rotary_mul` / `RotaryMul`
- AdaLN → `adaln` / `adln`
- GELU → `FastGelu`

---

## Phase 7: Copy 算子消减

Pattern 匹配成功后检查 Copy 算子膨胀。`default` 后端的 `aot_autograd` functionalization
将 view/reshape 转为 `_to_copy` → Inductor codegen → `InplaceCopy` NPU kernel。

**检测**: 在 `kernel_details.csv` 中搜索 `InplaceCopy/ViewCopy/TensorMove/StridedSlice`。
同时对比 eager vs compile 的同名 kernel 耗时差，定位膨胀源。

**消减方案**:

- **方案 A** (推荐 3D attn): torchair_ge 绕过 aot_autograd
- **方案 B** (2D attn): 修复 pattern 匹配，减少 Copy 膨胀
- **方案 C**: 混合模式（VAE eager + transformer compile）
- **方案 D** (不推荐): npugraph_ex（与 default 等价）

完整流程见 `references/copy-elimination-guide.md`。后端对比见 `references/backend-comparison.md`。

---

## 参考文件

| 文件 | 加载时机 |
|------|---------|
| `references/pattern-templates.md` | Phase 2: 创建 pattern 代码模板 + 融合 op 速查 |
| `references/registration-checklist.md` | Phase 3: 注册核对清单 |
| `references/test-templates.md` | Phase 4: 测试组织 + 双层测试模板 |
| `references/mismatch-catalog.md` | Phase 5: 7 类 mismatch 目录 |
| `references/graph-comparison-guide.md` | Phase 5: Graph dump + 节点对齐方法 |
| `references/custom-graph-pass.md` | Phase 5: 自定义 Graph Pass 实现指南 |
| `references/copy-elimination-guide.md` | Phase 7: Copy 消减全流程 |
| `references/backend-comparison.md` | Phase 7: 四后端架构对比 + 决策树 |

## 维护与更新

当发现新的 mismatch 类型、Copy 膨胀场景或后端支持矩阵变化时更新本 skill。
各 reference 文件的更新触发条件见各自的"维护与更新"章节。
