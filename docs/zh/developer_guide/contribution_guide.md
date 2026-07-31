# 贡献指南

## 前置准备

1. Fork 仓库到个人账号下，克隆到本地：

   ```bash
   git clone https://gitcode.com/<你的用户名>/MindIE-SD.git
   ```

2. 按 [build_guide.md](build_guide.md) 完成源码编译。

3. 安装 lint 依赖和 pre-commit hook：

   ```bash
   pip install -r requirements-lint.txt
   pre-commit install
   ```

   pre-commit 详细说明见 [dev_setup.md](dev_setup.md) 的「Lint 与提交前检查」章节。

## 编码规范

### Ruff 配置

Ruff 配置来自 `pyproject.toml` 的 `[tool.ruff]` 段：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| 行长度 | 100 | 单行最大字符数 |
| Python 版本 | py310 | 目标版本 |
| docstring 格式化 | true | 格式化 docstring 中的代码块 |

启用的 Lint 规则组：

| 规则组 | 含义 |
|--------|------|
| E | pycodestyle 错误（空白行、缩进等） |
| F | Pyflakes（未使用导入、未定义名称等） |
| I | isort 导入排序 |
| N | pep8-naming 命名规范 |
| W | pycodestyle 警告 |
| UP | pyupgrade 语法现代化 |
| B | flake8-bugbear 常见错误 |
| C4 | flake8-comprehensions 推导式优化 |
| SIM | flake8-simplify 简化建议 |
| G | flake8-logging-format 日志格式 |

### Pre-commit 钩子

`pre-commit install` 安装的钩子在每次 `git commit` 时自动运行：

| 钩子 | 行为 |
|------|------|
| `ruff-check` | 自动修复 Python lint 问题 |
| `ruff-format` | 自动格式化 Python 代码 |
| `pylint` | Python 代码质量检查 |
| `bandit` | Python 安全漏洞扫描 |
| `codespell` | 注释/文档拼写检查（跳过代码文件） |
| `typos` | 代码标识符拼写检查 |
| `clang-format` | C++ 代码格式化 |
| `markdownlint` | Markdown 格式检查（手动模式） |
| `trailing-whitespace` | 删除行尾空白字符 |
| `end-of-file-fixer` | 文件末尾添加换行符 |
| `check-yaml` | YAML 格式检查 |
| `check-added-large-files` | 拦截大于 50MB 文件 |
| `check-merge-conflict` | 检测未解决的合并冲突 |
| `detect-private-key` | 检测硬编码密钥 |

### Python 编码约定

**文件头模板**：所有 `.py` 文件必须包含 Mulan PSL v2 许可证头。

**导入顺序**（Ruff I 规则强制执行）：

1. 标准库（如 `contextlib`, `dataclasses`, `typing`）
2. 第三方库（如 `torch`）
3. 本地模块（如 `from .compiliation_config import ...`）

每组内部按字母序排列。

**空白行**：模块级函数/类定义前 2 个空白行，类内方法前 1 个空白行。

**引号**：统一使用双引号。

**类型注解**：目标 Python 3.10，可使用 `X | Y` 联合语法（PEP 604）。函数签名建议包含参数和返回值类型注解。

**日志**：使用标准 `logging` 模块，避免在日志字符串中使用 `.format()` 或 f-string，优先使用 `%` 风格参数：

```python
logger = logging.getLogger(__name__)
logger.debug("...")
logger.warning("...")
```

### C/C++

使用 **clang-format** 做代码格式化，配置位于 `.clang-format`。

### Markdown

使用 **markdownlint** 做文档格式检查。手动运行方式见 [dev_setup.md](dev_setup.md)。

核心规则：

| 规则 | 说明 |
|------|------|
| MD040 | 围栏代码块必须指定语言标记 |
| MD009 | 禁止行尾空格 |
| MD012 | 禁止连续多个空白行 |
| MD031 | 围栏代码块前后需有空行 |

代码块语言选用：

| 内容类型 | 语言标记 |
|----------|----------|
| Python 代码 | `python` |
| Shell 命令 | `bash` |
| 目录/文件树 | `text` |
| 终端输出/日志 | `text` |

### 拼写与命名检查

**codespell** + **typos** 对代码和文档做拼写检查。

## 提交前检查

确保所有 pre-commit 检查通过后再提交：

```bash
pre-commit run --all-files
```

如需临时绕过（仅紧急情况）：

```bash
git commit --no-verify
```

## 运行测试

### CPU 友好单元测试

```bash
bash tests/run_UT_test.sh
```

### NPU 全量测试

```bash
bash tests/run_test.sh --all
```

可选参数：`--cpu_only`、`--npu_only`。

测试详细说明见 [test.md](test.md)。

## 分支与提交规范

- 分支命名：`feat/<功能名>`、`fix/<问题>`、`docs/<主题>`、`refactor/<范围>`
- 提交信息：首行简明（不超过72字符），可附带详细说明段落

## Pull Request 流程

1. 从 `master` 创建新分支。
2. 完成修改并确保 pre-commit + 测试通过。
3. 提交 PR，关联 Issue（如有），描述变更内容与动机。
4. Reviewer 给出技术评审意见。
5. Approver 确认变更已具备合入条件。
6. Branch keeper 或授权 maintainer 将变更合入受保护分支。

角色定义和决策流程详见 [../community/governance.md](../community/governance.md)。

## 文档同步要求

- 用户可见的变更须同步更新 `docs/zh/` 和 `docs/en/` 下的对应文档。
- 重要变更须更新 `release_note.md` 的版本说明。
- 重大接口变更、行为变更应事先提交 RFC。

## 新手上路

新手可从浏览仓库中所有开放的 Issue 开始，选择与自己能力匹配的任务。阅读 [repo_structure.md](repo_structure.md) 了解代码布局，有助于快速上手。

## 编码注意事项

- 避免重复代码，超过5行的重复片段应提取为公共函数。
- 减少 NPU-CPU 同步操作（如 `.item()`、`.cpu()`），优先向量化。
- 模型前向中的运行时检查应缓存为成员变量，避免每层重复计算。
- 纯函数优先，避免对参数原地修改。
- 单文件超过2000行时应拆分。
- 工具类函数放在文件底部，核心数据结构放在顶部。
- 禁止使用 `pickle.loads()` / `pickle.load()` 反序列化不可信数据。
- 新增功能或适配新硬件时，尽量不改造现有代码，优先创建新文件引入独立模块。
- 多分支判断时，确保默认路径/公共路径作为第一个分支。
