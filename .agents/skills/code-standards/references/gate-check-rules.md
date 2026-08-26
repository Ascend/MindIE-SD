# 门禁专项规则

以下规则由项目门禁检查（pylint / 自定义规则）强制执行，在提交代码前必须通过。

## protected-access — 受保护成员访问

- 禁止从类外部访问以 `_` 前缀命名的受保护成员（如 `obj._method()`、`obj._field`）
- 如果类本身已经用 `_` 前缀标记为模块私有（如 `_ACLGraphEntry`），则其内部成员不应再使用 `_` 前缀，避免嵌套函数或同模块代码访问时触发保护警告
- **正例**：`_ACLGraphEntry` 类中使用 `copy_stream`、`ensure_copy_stream()`（无 `_` 前缀）
- **反例**：类外部调用 `entry._copy_stream`、`entry._ensure_copy_stream()`

## avoid-import-method — 禁止 `__import__`

```python
# 反例
mod = __import__(mod_name, fromlist=[attr])

# 正例
import importlib
mod = importlib.import_module(mod_name)
```

注意：此规则同时被 Ruff UP015 覆盖（见 SKILL.md §5.3），但门禁 `avoid-import-method` 独立检查，两个检查需同时通过。`examples/` 目录下的示例脚本同样适用此规则。

## bare-except-pass — 禁止 try/except/pass

禁止无日志的异常吞没（`except Exception: pass`）。捕获异常后至少记录日志：

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

## avoid-using-exit — 禁止 sys.exit() 在非入口函数中使用

禁止在函数体内使用 `sys.exit()` 或 `raise SystemExit()`。应改为抛出标准异常：

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

## too-many-arguments — 函数参数数量限制

限制函数/方法参数 ≤ 5 个（不含 `self`/`cls`）。超过时选择以下重构方案之一：

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

## function-order — 类方法排序规范

类方法定义的排列顺序：

1. `__init__` / `__new__`
2. 类属性 / 常量
3. 公共方法（按调用链逻辑顺序）
4. 私有方法（`_` 前缀，集中放置在公共方法之后）

私有辅助方法应与引用它们的公共方法就近，或在全部公共方法之后集中存放。

## duplicate-string — 重复字符串字面量

同一作用域内重复出现的字符串字面量应提取为常量：

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

## full-path-executable — 可执行文件使用完整路径

子进程调用中禁止直接使用裸命令名（如 `"npu-smi"`）。应使用 `shutil.which()` 解析完整路径：

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

## 维护与更新

当门禁规则集更新或新增违规示例时，按 dev-workflow 的复盘流程更新本文件。
