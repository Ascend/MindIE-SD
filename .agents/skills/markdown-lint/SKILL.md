---
name: markdown-lint
description: MindIE-SD 仓库 Markdown 格式 lint 规则。当编写、修改或审查 Markdown 文件（README、文档、
              变更日志等）、或 CI 门禁报出 markdownlint 违规时使用此 skill。
              即使用户只提到"格式问题"或"MD040报错"而未说 markdownlint，也应触发。
              通常由 dev-workflow 和 code-standards 在编码/审查阶段指引加载。
---

# Markdown 格式 Lint 规范

本 skill 汇总 MindIE-SD 仓库的 Markdown 格式 lint 规则，适用于 `.md` 文件的编写、修复与审查。

事实来源：

- `.pre-commit-config.yaml`（`markdownlint` 钩子配置）
- `markdownlint-cli` v0.46.0 规则集

---

## 1. 核心规则

### 1.1 MD040 — Fenced code blocks should have a language specified

所有围栏代码块（` ``` `）必须指定编程语言或内容类型。

**正例：**

```markdown
```shell
npu-smi info -l
```

```markdown
```python
def foo():
    pass
```

```text

**反例（触发 MD040）：**

```markdown
```

examples/dummy_run/
├── model/
├── wan_infer.py
└── README.md

```text
```

**代码块语言选用指南：**

| 内容类型 | 语言标记 |
|----------|----------|
| Python 代码 | `python` |
| Shell 命令 | `shell` 或 `bash` |
| 目录 / 文件树 | `text` |
| 终端输出 / 日志 | `text` |
| 纯文本 / 伪代码 | `text` |
| YAML 配置 | `yaml` |
| Markdown 示例 | `markdown` |
| JSON 数据 | `json` |

### 1.2 其他常见规则

| 规则 | 说明 |
|------|------|
| MD009 | 禁止行尾空格（由 `trailing-whitespace` 钩子覆盖） |
| MD012 | 禁止连续多个空白行 |
| MD031 | 围栏代码块前后需有空行 |
| MD047 | 文件末尾需有换行符（由 `end-of-file-fixer` 钩子覆盖） |

---

## 2. 验证命令

### 2.1 全量检查

```shell
# 检查单个文件
pre-commit run markdownlint --files path/to/file.md

# 检查所有文件（CI 模式）
pre-commit run markdownlint --all-files
```

> `markdownlint` 钩子在 `.pre-commit-config.yaml` 中配置为 `stages: [manual]`，需显式触发。

### 2.2 提交范围专项检查

对 commit 新增或修改的 `.md` 文件做独立检查，避免全量噪音：

```shell
git diff --name-only HEAD~1..HEAD -- '*.md' | xargs markdownlint -c .markdownlint.json
```

### 2.3 配置优先

- 始终使用 `-c .markdownlint.json` 配置文件而非 `--disable` 命令行参数
- PowerShell 中 `--disable MDxxx` 的数组传参可能不生效，导致规则未真正禁用

---

## 3. 修复模板

### 3.1 目录树 / 终端输出

```markdown
<!-- 修改前 -->
 ```

 examples/
 ├── a.py
 └── b.py

 ```text

<!-- 修改后 -->
 ```text
 examples/
 ├── a.py
 └── b.py
 ```

```text

### 3.2 无明确代码语言的内容

```markdown
<!-- 修改前 -->
 ```

 Warmup inference (1 step) ...
   [transformer] 7.1s
 Inference time: 7.1 s

 ```text

<!-- 修改后 -->
 ```text
 Warmup inference (1 step) ...
   [transformer] 7.1s
 Inference time: 7.1 s
 ```

```text

---

## 4. 与 dev-workflow 的关系

本 skill 作为 `dev-workflow` 的补充模块，在以下时机触发：

- 新建或修改 `.md` 文件后，提交前
- CI 门禁报出 `markdownlint` 违规时
- 模板文件（PR/Issue）变更时

`dev-workflow` 的 Test-First 流程中，编码阶段需确保 Python 代码通过 Ruff 检查，同时确保 Markdown 文件通过 `markdownlint` 检查。

---

## 5. 批量修复注意事项

### 5.1 编码安全

PowerShell 5.1 下用 `Get-Content` / `Set-Content` 读写含中文的 UTF-8 文件会导致编码破坏（BOM 注入 + 字节重解释）。

**安全方式：**

- Python：`open(f, 'r', encoding='utf-8')` / `open(f, 'w', encoding='utf-8')`
- Node.js：`fs.readFileSync(f, 'utf-8')` / `fs.writeFileSync(f, content, 'utf-8')`

**避免：** PowerShell `Get-Content` / `Set-Content` / `Out-File` 直接读写 UTF-8 含中文文件。

### 5.2 MD040 修复模式

替换 `` ``` `` → `` ```text `` 时需覆盖两种变体：

- **顶格** `` ``` ``（行首无空白）
- **缩进** ``   ``` ``、``     ``` `` 等（保留前导空白，缩进代码块）

### 5.3 修复后验证

1. 重跑 `markdownlint -c .markdownlint.json <files>` 确认 0 违规
2. `git diff` 检查只有预期替换行变化，无编码污染（BOM、乱码）
3. 确认中文等非 ASCII 字符未损坏

## 6. 维护与更新

当 `markdownlint-cli` 版本升级、`.pre-commit-config.yaml` 中 markdownlint 配置变更、
或新 MD 规则启用时，按 `dev-workflow` 的复盘流程更新本 skill。
