# Pattern 开发指南

本文档介绍 MindIE SD 编译模块中 Pattern（融合算子模式）的完整开发流程。

## 开发流程

```text
模型分析 → Pattern 创建 → 三段注册 → 单元验证 → 集成验证
```

## 阶段一：模型分析

在模型源码中定位目标算子的实际实现代码（如 RMSNorm、RoPE、AdaLayerNorm），提取完整代码片段作为 Pattern 和测试的依据。

同时判断参数来源：

| 参数来源 | Pattern 路径 | 适用场景 |
|----------|-------------|----------|
| functional API（如 `F.rms_norm`） | `register_replacement` | 参数由函数参数传入 |
| nn.Module（如 `self.weight`） | 自定义 Graph Pass | 参数来自模块成员变量 |

## 阶段二：创建 Pattern

先检查现有 pattern 是否已覆盖目标算子。确认不匹配后新建，遵循非侵入原则（始终创建新文件，不修改现有 pattern 文件）。

根据阶段一的参数来源选择路径：

- **register_replacement 路径**：创建 `PatternBase` 子类（工厂+闭包），注册到 `pattern_registry`
- **自定义 Graph Pass 路径**：实现自定义 FX graph traversal pass

## 阶段三：三段注册

涉及 3 个文件的修改，全部是代码追加（不修改已有代码）：

1. `patterns/__init__.py` — `__all__` 追加 + `from .xxx_pattern import XxxPatternGroup`
2. `passes/__init__.py` — `pattern_registry` 字典追加
3. `compiliation_config.py` — `FusionPatterns` dataclass 追加 `enable_xxx: bool = True`

命名规范：config key 使用 `enable_<model>_<op>` 格式。

## 阶段四：单元验证

验证标准：`cosine_similarity(compiled, original) > 2^-7`。

> 注意：单元测试通过不保证 Pattern 命中了全模型。测试 model 与 pattern 共享相同代码，必然匹配。全模型匹配需要阶段五集成验证最终确认。

## 阶段五：集成验证

依次通过三层验证，顺序不可跳：

| 层级 | 方法 | 确认内容 |
|------|------|---------|
| 1. Pattern 命中 | graph dump + 日志 | `PatternMatchPass replace N` 数量增加 |
| 2. Fusion kernel | profiling → `kernel_details.csv` | 融合 kernel 出现 + 原始 kernel 消失 |
| 3. 全模型回归 | dummy run + compile | 推理正常完成无 crash |

Kernel 名称对应：

| 算子 | kernel 名称 |
|------|------------|
| RMSNorm | `rms_norm` / `RmsNorm` |
| RoPE | `npu_rotary_mul` / `RotaryMul` |
| AdaLN | `adaln` / `adln` |
| GELU | `FastGelu` |
