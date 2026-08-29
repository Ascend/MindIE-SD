---
name: code-standards
compatibility: ruff, pre-commit（含 codespell/typos 钩子）
description: MindIE-SD Python 代码格式与 lint 规则。当编写、格式化、lint 检查或审查 MindIE-SD 项目的 Python 代码时使用此 skill。
             即使用户只提到"提个MR"或"代码好像有 lint 问题"而未明确说格式化，也应触发。
             通常由 dev-workflow 在编码阶段指引加载。
---

# MindIE-SD 代码格式规范

本 skill 汇总 MindIE-SD 项目的 Python 代码格式与 lint 规则，适用于代码格式化、lint 修复和代码审查。

事实来源：

- `pyproject.toml`（Ruff 配置）
- `.pre-commit-config.yaml`（pre-commit 钩子）
- `mindiesd/compilation/` 目录下既有代码风格

---

## 1. Ruff 格式化配置

以下配置项来自 `pyproject.toml` `[tool.ruff]` 段：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `line-length` | 100 | 单行最大字符数 |
| `target-version` | py310 | 目标 Python 版本 |
| `exclude` | build, dist | 排除目录 |
| `docstring-code-format` | true | 格式化 docstring 中的代码块 |

### 1.1 启用的 Lint 规则

`select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM", "G"]`

| 规则组 | 含义 |
|--------|------|
| E | pycodestyle 错误（空白行、缩进、行长度等） |
| F | Pyflakes（未使用导入、未定义名称等） |
| I | isort 导入排序 |
| N | pep8-naming 命名规范 |
| W | pycodestyle 警告 |
| UP | pyupgrade 语法现代化 |
| B | flake8-bugbear 常见错误 |
| C4 | flake8-comprehensions 推导式优化 |
| SIM | flake8-simplify 简化建议 |
| G | flake8-logging-format 日志格式 |

### 1.2 忽视的规则

| 规则 | 原因 |
|------|------|
| B007 | 循环变量未使用（允许不使用的循环变量） |
| B905 | `zip()` 无 `strict=`（允许省略 strict） |
| E731 | lambda 赋值（允许 lambda 赋值给变量） |
| F403 / F405 | `import *` 相关（允许通配导入） |
| UP009 | UTF-8 编码声明（允许 `# coding=utf-8` 头） |
| UP032 | `.format()` 转 f-string（允许保留 .format） |

### 1.3 文件级例外

`"mindiesd/__init__.py" = ["E402", "I001"]`

- E402：允许 `__init__.py` 中模块级导入不在文件顶部
- I001：允许 `__init__.py` 中导入顺序不标准

---

## 2. Pre-commit 钩子

来自 `.pre-commit-config.yaml`：

| 钩子 | 行为 |
|------|------|
| `ruff-check` | `ruff check --output-format github --fix`，自动修复 Python lint 问题 |
| `ruff-format` | `ruff format`，自动格式化 Python 代码 |
| `codespell` | 注释/文档拼写检查（跳过 `.py/.cpp/.hpp/.c/.h` 代码文件） |
| `typos` | 代码标识符拼写检查 |
| `trailing-whitespace` | 删除行尾空白字符 |
| `end-of-file-fixer` | 文件末尾添加换行符 |
| `check-yaml` | YAML 格式检查 |
| `check-added-large-files` | 拦截 >50MB 文件 |
| `check-merge-conflict` | 检测未解决的合并冲突标记 |
| `detect-private-key` | 检测硬编码密钥 |
| `check-json` | JSON 格式检查 |
| `markdownlint` | Markdown 格式检查（CI 手动执行），含 MD040（代码块必须指定语言），详见 `markdown-lint` skill |
| `no-commit-to-branch` | 禁止直接提交到 main/master |

---

## 3. 代码风格约定

### 3.1 文件头

所有 `.py` 文件必须包含：

```python
#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
```

### 3.2 导入顺序

isort 排序规则（Ruff I 规则强制执行）：

1. 标准库（如 `contextlib`, `dataclasses`, `logging`, `typing`）
2. 第三方库（如 `torch`）
3. 本地模块（如 `from .compiliation_config import ...`）

每组内部按字母序排列。

### 3.3 空白行

- 模块级函数和类定义前必须有 **2 个空白行**（PEP 8 E302）
- 类内方法定义前 **1 个空白行**
- 过多空白行会被 E303 规则自动修正
- 章节注释块（`# ----...----`）与代码之间保持阅读友好的间距，由 Ruff formatter 自动处理

### 3.4 行长度

单行不超过 **100 字符**。Ruff formatter 会自动将超长行折叠为多行。

### 3.5 命名规范

由 pep8-naming（N 规则）检查：

- 函数/变量：`snake_case`
- 类：`CamelCase`
- 私有成员：前缀 `_`
- 常量：`UPPER_CASE`

### 3.6 引号

统一使用 **双引号**。Ruff formatter 默认使用双引号并自动统一。

### 3.7 类型注解

- 目标 Python 3.10，可使用 `X | Y` 联合语法（PEP 604）
- 函数签名建议包含参数和返回值类型注解
- 前向引用使用字符串注解（如 `"torch.npu.NPUGraph"`）

### 3.8 日志

使用标准 `logging` 模块（G 规则检查）：

```python
logger = logging.getLogger(__name__)
logger.debug("...")
logger.warning("...")
```

避免在日志字符串中使用 `.format()` 或 f-string，优先使用 `%` 风格参数（logging 延迟求值）。

### 3.9 受保护成员访问（protected-access）

门禁检查中的 pylint `protected-access` 规则：

- 禁止从类外部访问以 `_` 前缀命名的受保护成员（如 `obj._method()`、`obj._field`）
- 如果类本身已经用 `_` 前缀标记为模块私有（如 `_ACLGraphEntry`），则其内部成员不应再使用 `_` 前缀，避免嵌套函数或同模块代码访问时触发保护警告
- **正例**：`_ACLGraphEntry` 类中使用 `copy_stream`、`ensure_copy_stream()`（无 `_` 前缀）
- **反例**：类外部调用 `entry._copy_stream`、`entry._ensure_copy_stream()`

### 3.10 避免 `__import__`（avoid-import-method）

门禁检查禁止直接使用 `__import__` 内置函数。应改为：

```python
# 反例
mod = __import__(mod_name, fromlist=[attr])

# 正例
import importlib
mod = importlib.import_module(mod_name)
```

注意：此规则同时被 Ruff UP015 覆盖（见 5.3 节），但门禁 `avoid-import-method` 独立检查，两个检查需同时通过。`examples/` 目录下的示例脚本同样适用此规则。

### 3.11 禁止格式化未修改代码行

修改现有文件时，**绝对禁止**对未涉及功能变更的代码行做任何格式化调整：

- 禁止 `ruff format` 批量格式化整个文件
- 禁止改变未修改行的引号风格（如 `'` → `"`）
- 禁止调整未修改行的空行数量
- 禁止将单行表达式折叠为多行（如 `@unittest.skipIf`、`raise` 语句）
- 禁止改变已有变量命名（如 `B,S,N,D` → `b,s,n,d`）
- 禁止改变已有导入别名（如 `import torch.nn.functional as F` 保持不变）
- 禁止删除已有的注释（如行尾 `# FLux.1-dev`）
- 禁止保留注释掉的代码行（comment-out-code 检测）。已废弃的代码行应直接删除，不应以注释形式保留
- 描述性注释（自然语言说明）与注释掉代码的区别：前者是 `# Step 1: do X`，后者是 `# old_func(arg)`

**原因**：这些无关 diff 增加 review 负担、引入合并冲突风险，且不改善功能。

**仅允许**：在需要新增代码的行上使用与周围代码一致的风格。

- 仅对被改动的行运行 `ruff check --fix` 修复 lint 问题
- 优先使用精确 `edit` 替换而非整文件 rewrite

### 3.12 禁止无意义的自动修复重命名

`ruff check --fix` 的自动修复可能引发不必要的重命名，必须人工审查：

- 禁止改变已有变量名（如 `B,S,N,D` → `b,s,n,d`）（N806）
- 禁止改变已有导入别名（如 `import torch.nn.functional as F` → `as nn_functional`）（N812）
- 如果 lint 规则与现有代码风格冲突，**优先保持现有风格**，而非修改代码
- `ruff check --fix` 的输出 diff 必须逐行审查，不自动放行

### 3.13 禁止 try/except/pass（bare-except-pass）

门禁检查禁止无日志的异常吞没（`except Exception: pass`）。捕获异常后至少记录日志：

```python
# 反例
try:
    result = local_fetch()
except Exception:
    pass

# 正例
try:
    result = local_fetch()
except Exception:
    logger.debug("Local fetch failed, trying fallback")
```

### 3.14 禁止 sys.exit() 在非入口函数中使用（avoid-using-exit）

门禁检查禁止在函数体内使用 `sys.exit()` 或 `raise SystemExit()`。应改为抛出标准异常：

```python
# 反例
def check_npu():
    if not available:
        sys.exit(1)

# 正例
def check_npu():
    if not available:
        raise RuntimeError("NPU is not available")
```

如果是 `__main__` 入口调用栈的顶层 `main()` 函数且确实需要控制退出码，可在 `if __name__ == "__main__"` 块中捕获异常后调用 `sys.exit()`。

异常类型转换时必须使用 `raise NewError(...) from original_exc` 保留原始调用栈（G.ERR.04 / raise-missing-from）：

```python
# 反例
except Exception as exc:
    raise RuntimeError("Failed to download config: %s" % exc)

# 正例
except Exception as exc:
    raise RuntimeError("Failed to download config: %s" % exc) from exc
```

### 3.15 函数参数数量限制（too-many-arguments）

门禁限制函数/方法参数 ≤ 5 个（不含 `self`/`cls`）。超过时选择以下重构方案之一：

| 方案 | 适用场景 | 示例 |
|------|----------|------|
| 移除未使用参数 | 调用方从未传入非默认值 | 删除 `torch_dtype` 参数，内联常量 |
| 合并相关参数 | 语义关联的参数对/组 | `num_layers, num_layers_2` → `layer_cfg: dict` |
| 提取配置对象 | 多项可选配置 | dataclass / TypedDict 替代多参数 |

```python
# 反例（6 参数）
def build(config, a=None, b=None, c=None, d=None, e=None):
    ...

# 正例：移除调用方未使用的参数（5 参数）
def build(config, a=None, b=None, c=None, d=None):
    ...
```

### 3.16 类方法排序规范（function-order）

门禁检查类方法定义的排列顺序。推荐顺序：

1. `__init__` / `__new__`
2. 类属性 / 常量
3. 公共方法（按调用链逻辑顺序）
4. 私有方法（`_` 前缀，集中放置在公共方法之后）

私有辅助方法应与引用它们的公共方法就近，或在全部公共方法之后集中存放。

### 3.17 重复字符串字面量（duplicate-string）

门禁检测同一作用域内重复出现的字符串字面量。应提取为常量：

```python
# 反例
logger.warning("  " + "-" * 58)   # 第 1 次
logger.warning("  " + "-" * 58)   # 第 2 次 — 重复

# 正例
_DIVIDER_58 = "  " + "-" * 58
logger.warning(_DIVIDER_58)
logger.warning(_DIVIDER_58)
```

字符串常量应定义为类级属性（如果作用域在类内）或模块级变量（如果作用域在模块内）。

### 3.18 可执行文件使用完整路径（full-path-executable）

门禁检查子进程调用中直接使用裸命令名（如 `"npu-smi"`）。应使用 `shutil.which()` 解析完整路径：

```python
import shutil

# 反例
subprocess.run(["npu-smi", "info", "-l"], ...)

# 正例
npu_smi = shutil.which("npu-smi")
if npu_smi is None:
    raise RuntimeError("npu-smi not found in PATH")
subprocess.run([npu_smi, "info", "-l"], ...)
```

---

## 4. 最小格式化命令

```bash
# 格式化
ruff format mindiesd/compilation/target_file.py

# Lint + 自动修复
ruff check --fix mindiesd/compilation/target_file.py

# 全部检查（提交前）
pre-commit run --all-files
```

---

## 5. Ruff 自动修复速查

| 规则组 | 自动修复内容 |
|--------|------------|
| E302/E303 | 空白行：模块级定义前恰 2 行 |
| F401 | 删除未使用的 `import` |
| UP015 | `__import__` → `importlib.import_module`（同时受 §3.10 门禁检查） |
| C4/SIM | 简化冗余推导式 |
| Formatter | 多行表达式自动添加尾部逗号 |

## 6. 跨文件一致性

> Markdown 文件的格式规范由独立的 `markdown-lint` skill 承接，本 skill 仅覆盖 Python 代码格式。

修改代码时注意与其他编译模块保持一致，参考文件：

- `mindiesd/compilation/aclgraph_backend.py`
- `mindiesd/compilation/mindie_sd_backend.py`
- `mindiesd/compilation/compiliation_config.py`
- `mindiesd/compilation/_custom_decomposition.py`

## Reference Files

- 📋 `references/gate-check-rules.md` — 加载时机: 遇到门禁违规排查或需要查看具体规则的代码示例时

## 维护与更新

当 Ruff 版本升级、pre-commit 配置变更、新 lint 规则启用或发现新的代码风格约定时，
按 dev-workflow 的复盘流程更新本 skill。
