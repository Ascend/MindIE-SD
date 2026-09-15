# 门禁专项规则

> 本文件是「门禁专项规则 + 示例」的**单点**：正文只在这里维护，`SKILL.md` 只留摘要与加载时机（不重复正文）。
> 读每条前先看档位标注：
>
> - **【门禁·pre-commit】**：`.pre-commit-config.yaml` 的钩子会让提交失败（钩子清单见 `SKILL.md` §2）。
> - **【评审约定】**：不产生门禁失败，只是评审/自查要求——门禁**不会**替你拦住它。
> - **【历史记录·未能复核】**：本仓 pre-commit 17 个钩子中无对应钩子，pylint 的 `enable`/`disable`
>   与 bandit 的 `tests`/`skips` 也覆盖不到；按【评审约定】对待，**不要**当成"提交必过"的硬约束。
>
> 取值真源：`pre-commit/pyproject.toml`（门禁实际读取的那份）+ `.pre-commit-config.yaml`。
> 复核方法见 `SKILL.md` §5。

## 0. 强制来源速查

| 条目 | 强制来源 | 档位 |
|------|----------|------|
| 行长 120 | ruff `line-length`（`ruff format` 重排可折叠行；**`E501` 门禁侧未启用**） | 门禁·pre-commit |
| 引号 | ruff `format.quote-style = "preserve"` → formatter **不统一**引号 | 门禁·pre-commit（方向是"别乱改"，不是"统一双引号"） |
| `except …: pass` | bandit `B110`（在 `tests` 内、不在 `skips` 内，门限 LOW/LOW） | 门禁·pre-commit |
| 子进程裸命令名 | bandit `B607`（同上） | 门禁·pre-commit（只覆盖"启动进程"这一类） |
| 未使用导入 / 未定义名 / IO 错误 | ruff `F401` / `F821` / `E902` …（默认规则集） | 门禁·pre-commit（`tests/**/*` 免 `F401`） |
| `open()` 未用 `with` | ruff `SIM115`（`extend-select`） | 门禁·pre-commit |
| 段落末尾空行 | ruff `D209`（`extend-select`） | 门禁·pre-commit |
| `protected-access` | pylint `disable` 列表内 | **评审约定**（门禁不禁） |
| `raise-missing-from` | pylint `disable` 列表内 | **评审约定**（门禁不禁） |
| 函数参数 ≤5 | pylint `max-args = 15` 且 `too-many-arguments` 已 `disable` | **评审约定，不由 pre-commit 强制** |
| `avoid-import-method` | 无对应钩子（与 ruff `UP015` 无关，见 §4） | **历史记录·未能复核** |
| `avoid-using-exit` / `function-order` / `duplicate-string` | 无对应钩子 | **历史记录·未能复核** |
| 导入排序（isort）/ 命名（N）/ 日志格式（G） | 只有根 `pyproject.toml` 启用，门禁侧未启用 | **评审约定**（门禁不禁） |

## 1. protected-access — 受保护成员访问

**【评审约定】门禁不强制**：`pre-commit/pyproject.toml` 的 pylint `disable` 列表含 `protected-access`，
所以从类外部访问 `_` 成员**不会**让 CI 失败。判定它是否仍被禁用（在仓库根执行）：

```bash
grep -n "protected-access" pre-commit/pyproject.toml   # 命中在 disable = [...] 内 → 仍不禁
```

评审建议（自洽性考虑，非门禁）：

- 不从类外部访问以 `_` 前缀命名的受保护成员（如 `obj._method()`、`obj._field`）
- 如果类本身已经用 `_` 前缀标记为模块私有（如 `_ACLGraphEntry`），则其内部成员不应再使用 `_` 前缀，
  避免嵌套函数或同模块代码访问时触发保护警告
- **正例**：`_ACLGraphEntry` 类中使用 `copy_stream`、`ensure_copy_stream()`（无 `_` 前缀）
- **反例**：类外部调用 `entry._copy_stream`、`entry._ensure_copy_stream()`

若某天该消息被移到 `enable` 列表内，本节的档位标注必须改为【门禁·pre-commit】。

## 2. raise-missing-from — 异常转译保留原始调用栈

**【评审约定】门禁不强制**：pylint `disable` 列表含 `raise-missing-from`，CI 不会因缺 `from` 失败。
判定：

```bash
grep -n "raise-missing-from" pre-commit/pyproject.toml   # 命中在 disable = [...] 内 → 仍不禁
```

评审建议：异常类型转换时用 `raise NewError(...) from original_exc` 保留调用栈（原 G.ERR.04 口径）：

```python
# 反例（评审不通过，但门禁不报）
except Exception as exc:
    raise RuntimeError("Failed to download config: %s" % exc)

# 正例
except Exception as exc:
    raise RuntimeError("Failed to download config: %s" % exc) from exc
```

## 3. too-many-arguments — 函数参数数量

**【评审约定，不由 pre-commit 强制】**：`pre-commit/pyproject.toml` 里阈值是 `max-args = 15`，
且 `too-many-arguments`、`too-many-positional-arguments` 都在 `disable` 列表内 →
**参数个数不会让门禁失败**。判定：

```bash
grep -nE "max-args|too-many-arguments" pre-commit/pyproject.toml
# 期望：max-args = 15；too-many-arguments 出现在 disable = [...] 内
```

评审建议（≤5 个，不含 `self`/`cls`）——超限时择一重构：

| 方案 | 适用场景 | 示例 |
|------|----------|------|
| 移除未使用参数 | 调用方从未传入非默认值 | 删除 `torch_dtype` 参数，内联常量 |
| 合并相关参数 | 语义关联的参数对/组 | `num_layers, num_layers_2` → `layer_cfg: dict` |
| 提取配置对象 | 多项可选配置 | dataclass / TypedDict 替代多参数 |

```python
# 反例（6 参数，评审建议改，门禁不报）
def build(config, a=None, b=None, c=None, d=None, e=None):
    ...


# 正例：移除调用方未使用的参数（5 参数）
def build(config, a=None, b=None, c=None, d=None):
    ...
```

## 4. avoid-import-method — 禁止直接使用 `__import__`

**【历史记录·未能复核】属评审约定**：本仓 17 个 pre-commit 钩子中没有检查 `__import__` 的钩子
（ruff 门禁侧未启用 `UP` 组，pylint `enable` 列表也不含相关消息）。判定：

```bash
grep -n "__import__\|avoid-import-method" .pre-commit-config.yaml pre-commit/pyproject.toml
# 期望：无命中 → 本条不是门禁项，按评审约定执行
```

评审约定写法：

```python
# 反例
mod = __import__(mod_name, fromlist=[attr])

# 正例
import importlib

mod = importlib.import_module(mod_name)
```

`examples/` 目录下的示例脚本同样适用。

> **规则号纠正**：Ruff `UP015` 的真实语义是 **`redundant-open-modes`**（把 `open(f, "r")` 简化为
> `open(f)`），**与 `__import__` 无关**；`__import__` 相关规范属**评审约定**，不要引用规则号来"强制"它。
> 本仓门禁侧 `UP` 组本身也未启用。

## 5. bare-except-pass — 禁止无日志的异常吞没

**【门禁·pre-commit】**：bandit 的 `tests` 列表含 `B110`（try/except/pass），`skips` 不含它，
且 `severity_level`/`confidence_level` 均为 `LOW` → `except …: pass` 会让 CI 失败。判定：

```bash
grep -nE "B110|tests|skips" pre-commit/pyproject.toml   # B110 在 tests 内、不在 skips 内 → 门禁项
```

正例/反例：

```python
# 反例（门禁报 B110）
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

## 6. avoid-using-exit — 禁止非入口函数中使用 `sys.exit()`

**【历史记录·未能复核】属评审约定**：无对应钩子。判定：

```bash
grep -n "avoid-using-exit\|sys.exit" .pre-commit-config.yaml pre-commit/pyproject.toml   # 期望：无命中
```

评审约定：函数体内不使用 `sys.exit()` / `raise SystemExit()`，改为抛标准异常：

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

如果是 `__main__` 入口调用栈的顶层 `main()` 且确实需要控制退出码，可在 `if __name__ == "__main__"`
块中捕获异常后调用 `sys.exit()`。

## 7. function-order — 类方法排序

**【历史记录·未能复核】属评审约定**：无对应钩子（pylint 无该方法排序检查）。判定：

```bash
grep -n "function-order" .pre-commit-config.yaml pre-commit/pyproject.toml   # 期望：无命中
```

评审约定的推荐顺序：

1. `__init__` / `__new__`
2. 类属性 / 常量
3. 公共方法（按调用链逻辑顺序）
4. 私有方法（`_` 前缀，集中放置在公共方法之后）

私有辅助方法应与引用它们的公共方法就近，或在全部公共方法之后集中存放。

## 8. duplicate-string — 重复字符串字面量

**【历史记录·未能复核】属评审约定**：无对应钩子（pylint 的 `duplicate-code` 是"代码块重复"，
且在 `disable` 列表内，与字符串字面量无关）。判定：

```bash
grep -n "duplicate-string\|duplicate-code" .pre-commit-config.yaml pre-commit/pyproject.toml
# 期望：无 duplicate-string；duplicate-code 出现在 disable = [...] 内
```

评审约定：同一作用域内重复出现的字符串字面量应提取为常量：

```python
# 反例
logger.warning("  " + "-" * 58)   # 第 1 次
logger.warning("  " + "-" * 58)   # 第 2 次 — 重复

# 正例
_DIVIDER_58 = "  " + "-" * 58
logger.warning(_DIVIDER_58)
logger.warning(_DIVIDER_58)
```

字符串常量应定义为类级属性（在类内）或模块级变量（在模块内）。

## 9. full-path-executable — 子进程使用完整路径

**【门禁·pre-commit（部分）】**：bandit `B607`（启动进程时使用不完整路径）在 `tests` 内、
不在 `skips` 内 → `subprocess.run(["npu-smi", …])` 这类**子进程裸命令名会被门禁报出**；
其余"可执行文件用完整路径"的场景（非 subprocess 通路）无钩子覆盖，属评审约定。判定：

```bash
grep -nE "B607|skips" pre-commit/pyproject.toml   # B607 在 tests 内、不在 skips 内 → 门禁项
```

```python
import shutil

# 反例（门禁报 B607）
subprocess.run(["npu-smi", "info", "-l"], ...)

# 正例
npu_smi = shutil.which("npu-smi")
if npu_smi is None:
    raise RuntimeError("npu-smi not found in PATH")
subprocess.run([npu_smi, "info", "-l"], ...)
```

## 维护与更新

- 本文件只维护**规则正文与示例**；摘要与加载时机在 `SKILL.md` §3.9，取值表格在 `SKILL.md` §0/§1。
- 档位标注（门禁 / 评审约定 / 历史记录）必须与 `pre-commit/pyproject.toml`、`.pre-commit-config.yaml` 一致：
  按 `SKILL.md` §5 复核后，**不再复现的条目直接删除**，不要留"以防万一"的旧条目。
- 新增规则前先按 `.agents/README.md` 的「必写复核方法（强制）」补一条判定命令，否则不予回填。
