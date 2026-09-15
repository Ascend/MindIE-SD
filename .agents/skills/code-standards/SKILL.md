---
name: code-standards
compatibility: ruff, pre-commit（含 ruff/codespell/typos/pylint/bandit/clang-format/clang-tidy/gitleaks 钩子）
description: MindIE-SD Python 代码格式与 lint 规则。当编写、格式化、lint 检查或审查 MindIE-SD 项目的 Python 代码时使用此 skill。
             即使用户只提到"提个MR"或"代码好像有 lint 问题"而未明确说格式化，也应触发；Markdown 格式问题见 markdown-lint，提交/PR 规范见 mindie-sd-community-governance。
             通常由 dev-workflow 在编码阶段指引加载。
---

# MindIE-SD 代码格式规范

本 skill 汇总 MindIE-SD 项目的 Python 代码格式与 lint 规则，适用于代码格式化、lint 修复和代码审查。

事实来源（**取值只认 §0/§1 的记录，不要从历史描述或直觉推断**）：

- `pre-commit/pyproject.toml` —— **门禁实际读取的那一份**（`.pre-commit-config.yaml` 用 `--config` / `--rcfile` 指定）
- `.pre-commit-config.yaml` —— 17 个钩子
- 根 `pyproject.toml` —— **不是** pre-commit 用的那份；IDE / 手动 ruff 可能读它，取值不同（见 §0）
- `mindiesd/compilation/` 目录下既有代码风格
- 规则正文与示例：`references/gate-check-rules.md`（**单点**，本文件只留摘要）

---

## 0. 先确认门禁读哪一份配置（真实陷阱）

同一条规则在两份文件里取值不同，**门禁按左边一列判**；右边一列只在本地 IDE / 手动 ruff 时生效：

| 配置项 | 门禁生效值（`pre-commit/pyproject.toml`） | 根 `pyproject.toml` | 后果 |
|--------|------------------------------------------|---------------------|------|
| `line-length` | **120** | 100 | 本地按根配置跑 ruff 会在 100 列报 `E501`，**CI 门禁不报**（§3.4） |
| `format.quote-style` | `"preserve"`（formatter **不改**引号） | 未设置 → ruff 默认 `double` | 本地会把单引号统一成双引号 → 无关 diff（§3.6） |
| `format.docstring-code-format` | 未设置 → **disabled** | `true` | 本地会格式化 docstring 内代码块，门禁不会 |
| `[tool.ruff.lint] select` | **未声明** → ruff 默认规则集（E4/E7/E9 + F）+ `extend-select` 的 `D209`/`SIM115`（实测 enabled 共 61 条） | `E,F,I,N,W,UP,B,C4,SIM,G` | **`I`/`N`/`UP`/`B`/`C4`/`G`/`W`/`E501`/`E302` 门禁侧均未启用**（§1.1、§3.2、§3.5、§3.8） |
| `[tool.ruff.lint] ignore` | 未声明 | B007/B905/E731/F403/F405/UP009/UP032 | 这些规则门禁侧本就没启用，差异无实际影响 |
| `[tool.ruff.lint.per-file-ignores]` | `"tests/**/*" = ["F401","I","E402"]` | `mindiesd/__init__.py = ["E402","I001"]` 等 | 门禁侧 `tests/` 不报 F401；`mindiesd/__init__.py` 的例外在门禁侧**不生效** |
| `exclude` | 未声明 → ruff 内置默认排除（含 `dist`、`.venv`，**不含 `build`**） | `build`、`dist` | 按根配置跑会跳过 `build/`，门禁侧不跳过（实际以改动文件为输入，影响很小） |

**自查口径**：想复现 CI 门禁用 `--config pre-commit/pyproject.toml`；想解释"本地 IDE 为什么报错"再用根配置。
两侧取值不必互相迁就：**超过 100 列不会让门禁失败**，但评审仍按 **120 显示列**要求排版（§3.4）。

---

## 1. 配置取值（以门禁实际读取的 `pre-commit/pyproject.toml` 为准）

### 1.1 Ruff（`ruff-check` / `ruff-format` 都带 `--config pre-commit/pyproject.toml`）

| 配置项 | 生效值 | 说明 |
|--------|--------|------|
| `line-length` | **120** | 与 pylint `max-line-length = 120` 同值；由 `ruff format` 重排可折叠行 |
| `target-version` | `py310` | 实测 `linter.unresolved_target_version = 3.10` |
| `format.quote-style` | `preserve` | formatter **保持原引号**，不统一 |
| `format.docstring-code-format` | **disabled**（未设置） | 不格式化 docstring 内代码块 |
| `[tool.ruff.lint] extend-select` | `D209`、`SIM115` | 段落末尾空行 / `open()` 未用 `with` |
| `select` / `ignore` | **未声明** | 用 ruff 默认规则集，**不是**根配置那份 E/F/I/N/W/UP/B/C4/SIM/G |
| `per-file-ignores` | `"tests/**/*" = ["F401","I","E402"]` | 门禁侧 `tests/` 不报未使用导入 |

实测 enabled 规则集合（61 条）= pycodestyle `E4xx`/`E7xx`/`E902` + pyflakes `Fxxx` + `D209` + `SIM115`：
**不含** `I`（isort）、`N`（命名）、`UP`（pyupgrade）、`B`、`C4`、`G`、`W`，也**不含 `E501`**。

### 1.2 pylint（`--rcfile=pre-commit/pyproject.toml`）

门禁是否拦下某条消息，**以 `disable` 列表为准**：列在其中的不会导致失败。

- `enable` 显式列出：`E0100`、`E0601`、`E0602`、`E0603`、`E0611`、`E0632`（原配置重复列了一次）、`E1101`、`E1120`、`W0632`、`W1514`
- `disable` 含（与本 skill 相关的关键项）：`protected-access`、`raise-missing-from`、
  `too-many-arguments`、`too-many-positional-arguments`、`too-many-branches`、`too-many-statements`、
  `too-many-locals`、`too-many-return-statements`、`invalid-name`、`missing-docstring` 系列、
  `line-too-long`、`broad-except`、`bare-except`、`duplicate-code`、`unused-import`/`unused-variable`/`unused-argument`
- 阈值（多数因对应消息被 `disable` 而**不产生失败**）：`max-line-length = 120`、`max-args = 15`、
  `max-branches = 50`、`max-statements = 200`、`max-locals = 50`、`max-positional-arguments = 20`
- 除 `disable` 覆盖掉的消息外，pylint 其余默认消息中未被 `disable` 的仍会报（复核见 §5）

### 1.3 bandit（`--config=pre-commit/pyproject.toml`）

- `severity_level = LOW`、`confidence_level = LOW` → 门限很低，LOW 级问题也报
- `tests` 含 `B110`（try/except/pass）、`B607`（子进程不完整路径）等，两条都**不在 `skips` 内** → 门禁项
- `exclude_dirs` 含 `tests`/`test`/`venv`/`build`/`dist` 等

---

## 2. Pre-commit 钩子（共 17 个；CI 默认 stage 跑 16 个）

来自 `.pre-commit-config.yaml`（`default_stages: [pre-commit]`）：

| 钩子 | 行为 | 备注 |
|------|------|------|
| `trailing-whitespace` | 删除行尾空白字符 | |
| `end-of-file-fixer` | 文件末尾补换行符 | |
| `check-yaml` | YAML 检查（`--allow-multiple-documents`） | |
| `check-added-large-files` | 拦截超大文件 | |
| `check-merge-conflict` | 检测未解决的合并冲突标记 | |
| `detect-private-key` | 检测硬编码密钥 | |
| `check-json` | JSON 格式检查 | |
| `ruff-check` | `ruff check --config pre-commit/pyproject.toml --output-format github --fix`（仅 `*.py`） | Python lint |
| `ruff-format` | `ruff format --config pre-commit/pyproject.toml`（仅 `*.py`） | Python 格式化 |
| `codespell` | 拼写检查，`--skip *.toml,*.py,*.cpp,*.hpp,*.c,*.h` | 跳过代码文件 |
| `pylint` | `--rcfile=pre-commit/pyproject.toml`（仅 `*.py`） | 见 §1.2 |
| `bandit` | `--config=pre-commit/pyproject.toml --quiet`（仅 `*.py`） | 见 §1.3 |
| `typos` | `--force-exclude --config pre-commit/typos.toml` | 标识符拼写 |
| `markdownlint` | `-c .markdownlint.json`，**`stages: [manual]`** | **默认不跑**，需显式触发（§4）；详见 `markdown-lint` skill |
| `clang-format` | `--style=file --verbose -i`，`files: \.(c\|h\|cpp\|hpp\|cc\|hh\|cxx\|hxx)$` | C++ |
| `clang-tidy` | `repo: local`，`files: ^csrc/.*\.(cpp\|cc\|cxx\|c)$`（排除 `csrc/ops/`） | 依赖宿主机已装 `clang-tidy` |
| `gitleaks-offline-scan` | `repo: local`，`entry: ./gitleaks`（本地离线二进制） | 密钥扫描 |

**不存在** `no-commit-to-branch`（本仓不禁止直接提交到 main/master）。
`markdownlint` 带 `stages: [manual]`，不随默认 stage 运行，所以"本地 `pre-commit run --all-files` 通过"
**不等于** Markdown 合规；`clang-tidy`/`gitleaks-offline-scan` 是 local 钩子，缺少对应二进制时会失败。

---

## 3. 代码风格约定

### 3.1 文件头（评审约定）

所有 `.py` 文件包含：

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

门禁侧无"文件头"钩子（`detect-private-key`/`codespell`/`typos` 不检查版权头），本条按评审约定执行。
注意 `# coding=utf-8` 被 ruff `UP009` 允许（且门禁侧 `UP` 组未启用），不要为了"现代化"删掉它。

### 3.2 导入顺序（评审约定）

顺序要求：标准库（如 `contextlib`, `dataclasses`, `logging`, `typing`）→ 第三方库（如 `torch`）→
本地模块（如 `from .compiliation_config import ...`），组内按字母序。

门禁侧**未启用** `I`（isort）：`ruff check --config pre-commit/pyproject.toml` 不会报 `I001`；
根 `pyproject.toml` 的 `select` 含 `I`，本地按根配置跑才会报。因此**两侧都不要求为对方重排既有导入块**
（重排会制造无关 diff，§3.10）。

### 3.3 空白行

- 模块级函数/类定义前 2 个空行、类内方法前 1 个空行：**评审约定**（门禁侧 `E302`/`E303` 未启用），
  由 `ruff format` 归整
- 过多空白行由 `ruff format` 处理，不要手工去改未修改行（§3.10）

### 3.4 行长度（门禁取值 **120**；不要再用 100 判门禁）

- **门禁取值 `line-length = 120`**（`pre-commit/pyproject.toml`，ruff 与 pylint `max-line-length` 同值）。
  `ruff format` 会把**可折叠**的代码行重排到 120 以内；**中文长行、长字符串 formatter 无法折行**，
  需要人工排版。
- **门禁侧 `E501` 未启用**（`select` 未声明，enabled 集合里没有 `line-too-long`）。
  实测（2026-09，本地 ruff 0.15.12 + 本仓配置）：同一文件含 102 显示列的中文注释行，
  `ruff check --config pre-commit/pyproject.toml` → `All checks passed!`。
- **只有在根 `pyproject.toml` / 本地 IDE 下**才会报 `E501`，且那里阈值是 `line-length = 100`、
  按**显示列宽**计（CJK/全角按 2 列）。同一实测：注释符 `#` 加 1 个空格再跟 50 个汉字 = **102 列** →
  `E501 Line too long (102 > 100)`；跟 49 个汉字 = 100 列 → 通过。
  这就是"本地报错、CI 不报"的成因，**不要把本地 100 列报错当成门禁结论**。
- 评审口径：按 **120 显示列**排版（中文注释约 60 个汉字即到上限，还要留出缩进与行首注释符的余量）；
  按"字符数"估长度会把超限行误判为合规。
- 把中文串写成相邻字面量（`"…前半" "后半…"`）只要仍在**同一行**，该行列宽**不减**；
  必须把相邻字面量**分到不同行**（隐式拼接）才真正降宽。
- 读报错：`E501 Line too long (102 > 100)` 括号里的数字是检测值（**显示列宽**），
  `file:line:col` 的 `col` 是字符位置——中文行两者不一致，回查时按显示列算。

### 3.5 命名规范（评审约定）

`snake_case`（函数/变量）、`CamelCase`（类）、`_` 前缀（私有成员）、`UPPER_CASE`（常量）：
门禁侧 `N`（pep8-naming）未启用，pylint 的 `invalid-name` 也在 `disable` 内，不会让门禁失败。

### 3.6 引号（`quote-style = "preserve"`）

**formatter 不统一引号**：门禁读取的 `pre-commit/pyproject.toml` 里 `format.quote-style = "preserve"`，
`ruff format` 保持文件既有引号。**不要**为了"统一双引号"去改未修改行——改引号风格会制造无关 diff（§3.10）。
（根 `pyproject.toml` 未设置该项 → 本地按根配置跑 formatter 会向双引号统一，这就是本地/门禁差异。）

### 3.7 类型注解（评审约定）

- 目标 Python 3.10（`target-version = "py310"`），可使用 `X | Y` 联合语法（PEP 604）
- 函数签名建议包含参数和返回值类型注解
- 前向引用使用字符串注解（如 `"torch.npu.NPUGraph"`）

### 3.8 日志（评审约定）

使用标准 `logging` 模块：

```python
logger = logging.getLogger(__name__)
logger.debug("...")
logger.warning("...")
```

避免在日志字符串中使用 `.format()` 或 f-string，优先使用 `%` 风格参数（logging 延迟求值）。
门禁侧 `G`（flake8-logging-format）未启用，本条按评审约定执行。

### 3.9 门禁专项规则摘要（正文在 `references/gate-check-rules.md`）

| 条目 | 门禁强制？ | 说明 |
|------|-----------|------|
| `except …: pass` | **是**（bandit `B110`） | 见 §5 复核 |
| 子进程裸命令名 | **是**（bandit `B607`，仅覆盖该类场景） | 见 §5 复核 |
| `open()` 未用 `with` / 段落末尾空行 | **是**（ruff `SIM115` / `D209`） | `extend-select` |
| 受保护成员访问（`protected-access`） | **否**（pylint 已 `disable`） | 评审约定，给出判定命令 |
| 异常转译 `raise … from`（`raise-missing-from`） | **否**（pylint 已 `disable`） | 评审约定，给出判定命令 |
| 函数参数 ≤5 | **否**（`max-args = 15` 且 `too-many-arguments` 已 `disable`） | **评审约定，不由 pre-commit 强制** |
| 禁止 `__import__`（`avoid-import-method`） | **否**（无对应钩子；与 `UP015` 无关） | 历史记录·未能复核，按评审约定 |
| 禁止非入口 `sys.exit()` / 方法排序 / 重复字符串 | **否**（无对应钩子） | 历史记录·未能复核，按评审约定 |

> **规则号纠正**：Ruff `UP015` 的真实语义是 **`redundant-open-modes`**（`open(f, "r")` → `open(f)`），
> **不是** `__import__` → `importlib.import_module`；本仓门禁侧 `UP` 组也未启用。
> `__import__` 相关规范属**评审约定**，不要引用规则号来"强制"它。

### 3.10 禁止格式化未修改代码行

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

### 3.11 禁止无意义的自动修复重命名

`ruff check --fix` 的自动修复可能引发不必要的重命名，必须人工审查：

- 禁止改变已有变量名（如 `B,S,N,D` → `b,s,n,d`）（N806）
- 禁止改变已有导入别名（如 `import torch.nn.functional as F` → `as nn_functional`）（N812）
- 如果 lint 规则与现有代码风格冲突，**优先保持现有风格**，而非修改代码
- `ruff check --fix` 的输出 diff 必须逐行审查，不自动放行

> 上两条引用的 N806/N812 属 `N` 组：门禁侧未启用（§1.1），风险来自本地按根配置跑 `--fix`。

---

## 4. 最小格式化与自查命令

```bash
# 按门禁配置格式化 / lint 单个文件（目标文件路径按需替换）
ruff format --config pre-commit/pyproject.toml mindiesd/compilation/target_file.py
ruff check --config pre-commit/pyproject.toml --fix mindiesd/compilation/target_file.py

# 门禁等价：默认 stage（16 个钩子，不含 markdownlint）
pre-commit run --all-files
pre-commit run --files mindiesd/compilation/target_file.py

# Markdown（stages: [manual]，必须显式指定 hook stage）
pre-commit run --all-files --hook-stage manual
pre-commit run markdownlint --hook-stage manual --files docs/zh/README.md
```

适用范围（容易误判的两点）：

- `pre-commit run --all-files` 只跑**默认 stage** 的钩子；`markdownlint` 带 `stages: [manual]`，
  **不在其中**，需按上面的 `--hook-stage manual` 单独跑
- `--all-files` 只覆盖 **git 跟踪的文件**：新建但未 `git add` 的文件不在扫描范围内
- `clang-tidy` / `gitleaks-offline-scan` 是 `repo: local` 钩子，依赖宿主机存在对应二进制

---

## 5. 复核：如何判定本技能记录仍然成立（仓规强制）

> 依据 `.agents/README.md`「必写复核方法（强制）」：每条问题/陷阱都要写清"**如何判定它仍存在**"，
> 先复核、再套用；升级软件栈后逐条复核，**不再复现的条目必须删除**。
> 本 skill 的取值全部来自两份配置文件，**升级 ruff / pylint / bandit / 改动 pre-commit 配置后必须重跑本节**。

命令在仓库根执行（POSIX；Windows 无 `grep` 时用 `Select-String` 等价替代）。
多条模式一律写成多个 `-e`，避免管道符与表格语法冲突：

| 要复核什么 | 命令 | 期望 |
|-----------|------|------|
| 行长 / 引号 / pylint 阈值 | `grep -n -e line-length -e quote-style -e max-line-length -e max-args pre-commit/pyproject.toml` | `line-length = 120`、`max-line-length = 120`、`max-args = 15`、`quote-style = "preserve"` |
| 某条 pylint 消息是否仍被禁用 | `grep -n -e protected-access -e raise-missing-from -e too-many-arguments pre-commit/pyproject.toml` | 命中位于 `disable = [...]` 内 → 仍不作为门禁失败项；若挪进 `enable` → 必须删除 §3.9 对应行与 `gate-check-rules.md` 对应档位 |
| ruff 实际生效取值与规则集 | `ruff check --config pre-commit/pyproject.toml --show-settings mindiesd/__init__.py` 后接 `grep -e "Settings path" -e "^linter.line_length" -e "^formatter.quote_style" -e "^formatter.docstring_code_format"` | `Settings path` 指向 `pre-commit/pyproject.toml`；`linter.line_length = 120`；`quote_style = preserve`；`docstring_code_format = disabled` |
| 门禁侧是否仍不启用 `E501` | 同上 `--show-settings` 输出接 `grep -c "line-too-long (E501)"` | 输出 `0`（enabled 列表里没有它）→ §3.4 结论成立 |
| 钩子清单与 manual stage | `grep -n -e "^- id:" -e "stages:" .pre-commit-config.yaml`，再跑 `pre-commit validate-config .pre-commit-config.yaml` | 17 个 `id`；`markdownlint` 带 `stages: [manual]`；无 `no-commit-to-branch`；配置校验退出码 0 |
| 两份配置是否仍不同 | `grep -n -e line-length -e quote-style -e "^select" -e "^ignore" pyproject.toml pre-commit/pyproject.toml` | 有差异属预期（100 vs 120、`select` 有无）；若某天合并成一份，§0 的差异表要重写 |
| 门禁专项规则是否仍无强制来源 | `grep -n -e avoid-import-method -e avoid-using-exit -e function-order -e duplicate-string .pre-commit-config.yaml pre-commit/pyproject.toml` | 无命中（退出码 1）→ `gate-check-rules.md` 的"历史记录·未能复核"档位仍正确 |
| bandit 是否仍拦 `except: pass` / 裸命令名 | `grep -n -e B110 -e B607 -e skips pre-commit/pyproject.toml` | `B110`、`B607` 在 `tests` 内且不在 `skips` 内 |

**维护纪律**：复核后

1. 取值变了（如 `line-length` 改动）→ 同步 `SKILL.md` §0/§1/§3.4 与 `gate-check-rules.md`；
2. 某条不再复现（如 pylint 重新启用 `protected-access`）→ **删除**对应条目或改写档位，不留"以防万一"；
3. 新增条目必须同时给出复核命令，否则不予回填（仓规）。

---

## 6. 跨文件一致性

> Markdown 文件的格式规范由独立的 `markdown-lint` skill 承接，本 skill 仅覆盖 Python 代码格式。

修改代码时注意与其他编译模块保持一致，参考文件：

- `mindiesd/compilation/aclgraph_backend.py`
- `mindiesd/compilation/mindie_sd_backend.py`
- `mindiesd/compilation/compiliation_config.py`
- `mindiesd/compilation/_custom_decomposition.py`

## Reference Files

- 📋 `references/gate-check-rules.md` — 加载时机: 需要门禁专项规则的正文/示例、或要判断某条规则是否真由 pre-commit 强制时（本文件 §3.9 只留摘要）

## 维护与更新

当 ruff / pylint / bandit 版本升级、`pre-commit/pyproject.toml` 或 `.pre-commit-config.yaml` 取值变更、
新 lint 规则启用或发现新的代码风格约定时：先按 §5 复核，再更新 §0/§1 与
`references/gate-check-rules.md`（单点），最后按 dev-workflow 的复盘流程登记。
