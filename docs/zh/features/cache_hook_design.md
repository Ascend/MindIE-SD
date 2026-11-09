# Cache 功能优化：缓存命中时兼容模型自带 Hook

> 对应 Issue：[#139 cache功能优化-MindIE-SD](https://gitcode.com/Ascend/MindIE-SD/issues/139)

## 1. 背景与问题

MindIE-SD 提供以存代算的缓存加速能力（`AttentionCache` / `DiTBlockCache`），通过在
block / Attention 粒度上复用中间结果，减少扩散模型相邻采样步之间的冗余计算。

**问题**：使能 cache 后，模型原本自带的 hook 操作会被跳过。例如用户通过
`nn.Module.register_forward_pre_hook` / `register_forward_hook` 在 block 上注册的
hook（特征提取、中间结果保存、量化、offload 等），在缓存命中时不再执行，导致模型
行为与未使能 cache 时不一致。

## 2. 现状分析

### 2.1 Cache 机制现状

调用链如下：

```text
CacheAgent.apply(func, *args, **kwargs)
  └─ CacheBase.apply(func, *args, **kwargs)          # 统一计数
       └─ AttentionCache.apply_imp / DiTBlockCache.apply_imp
            ├─ 未命中缓存：func(*args, **kwargs)     # 正常前向，hooks 正常触发
            └─ 命中缓存：直接返回缓存结果             # ❌ 跳过 func，hooks 被跳过
```

缓存命中时直接 `return` 缓存结果，**不再调用 `func`**。而 PyTorch 的 forward hook
是在 `nn.Module.__call__ → _call_impl` 内部触发的，`func`（block 模块）未被调用，
其上的 hooks 自然也不会执行。

### 2.2 PyTorch Hook 触发机制（torch 2.x）

`nn.Module._call_impl` 的执行顺序：

1. **forward pre-hooks**（全局优先于模块）：`hook(module, args)`（`args` 为位置
   参数元组），可返回替换后的 args；`with_kwargs=True` 时为
   `hook(module, args, kwargs)`，可返回 `(new_args, new_kwargs)`。
2. **forward**：`forward(*args, **kwargs)`。
3. **forward hooks**（全局优先于模块）：`hook(module, args, result)`，返回非 None
   时替换结果；`with_kwargs=True` 时为 `hook(module, args, kwargs, result)`。

涉及的关键内部结构：

| 结构 | 说明 |
| ------ | ------ |
| `module._forward_pre_hooks` / `_forward_hooks` | OrderedDict，模块级 hooks |
| `module._forward_pre_hooks_with_kwargs` / `_forward_hooks_with_kwargs` | with_kwargs 标记 |
| `torch.nn.modules.module._global_forward_pre_hooks` / `_global_forward_hooks` | 全局 hooks |
| `_global_forward_hooks_with_kwargs` | 全局 with_kwargs 标记 |

## 3. 设计目标

1. **兼容性**：可缓存 block 命中时，forward pre-hook / forward hook 均被触发
   （模块级 hook 支持 `with_kwargs`；全局 forward hook 也支持 `with_kwargs`，
   全局 pre-hook 不支持）；需要 backward hooks、`always_call` 或 tracing
   的模块始终使用原生调用且不缓存。
2. **零侵入**：不改动用户注册 hook 的方式，`CacheAgent.apply` 接口不变。
3. **低开销**：纯函数 / 无 hooks 时保持原有快速路径。
4. **不重复应用**：可缓存路径的 hook 在每次逻辑前向中恰好触发一次，缓存保存
   forward hook 处理前的原始输出。
5. **语义一致**：命中缓存时 hook 收到的是「缓存模拟输出」，而非真实计算输出（这是
   缓存加速的本质，无法避免）。

## 4. 技术方案

### 4.1 核心思路

选择缓存路径前先检查模块是否需要完整原生调用链；此类模块正常运行且不缓存。对
其他模块，命中时**以缓存结果模拟 forward 的输出，复刻 `Module._call_impl` 的
forward hook 触发链**：

```text
命中缓存时：
  若 func 需要原生处理（backward hooks / always_call / tracing）：
    result = func(*args, **kwargs)            # 完整原生调用，不缓存
  否则若 func 使用自定义调用链：
    result = func(*args, **kwargs)            # 完整原生调用，不缓存
  否则若 func 是 nn.Module、使用标准 PyTorch 调用链且存在 hooks：
    args, kwargs = 执行 forward pre-hooks（全局→模块，含 with_kwargs 分支）
    result = 缓存原始输出（模拟 forward 输出，未经 hooks 处理）
    result = 执行 forward hooks（全局→模块，可修改输出）
    return result
  否则（非 nn.Module / 无 hooks）：
    return 缓存结果的隔离副本
```

缓存未命中时，**缓存未经 forward hooks 处理的原始输出**：

```text
缓存未命中（func 为带 hooks 的 nn.Module）：
  若 func 需要原生处理：
    result = func(*args, **kwargs)       # 完整原生调用，不缓存
    return result
  若 func 使用自定义调用链：
    result = func(*args, **kwargs)       # 完整原生调用，不缓存
    return result
  args, kwargs = 执行 forward pre-hooks
  raw = func.forward(*args, **kwargs)   # 真实前向（无 hooks）
  result = 执行 forward hooks(raw)     # hooks 恰好应用一次
  缓存 raw                             # 缓存无 hooks 的输出
  return result
```

在类或实例上使用自定义 `__call__`、自定义 `_call_impl`，或使用
`torch.compile` 的 `_compiled_call_impl` 的模块不走上述手工链路：每次调用都
执行 `func(*args, **kwargs)` 且不缓存，从而保留完整自定义调用链和 hooks。

由此保证：每个 hook 在每次逻辑前向中**恰好触发一次**，命中时的输出与未命中一致
（输出修改型 hook 不会被重复应用）。

### 4.2 方案对比

| 方案 | 说明 | 结论 |
| ------ | ------ | ------ |
| A. 手动复刻 `_call_impl` hook 链（**采用**） | 在 `CacheBase` 中实现 pre-hooks / hooks 执行逻辑，对齐标准 PyTorch 调用链 | 可控、可测；对自定义调用链显式降级 |
| B. 临时替换 `module.forward` 后调用 `__call__` | 命中缓存时把 forward 换成「返回缓存」，走原生 `__call__` 触发 hooks | 有并发/重入风险；对自定义调用链失效；副作用大 |
| C. 仅提供 CacheAgent 回调接口 | 用户通过新接口注册回调 | 不能覆盖「模型自带 hooks」，需用户改造，不满足需求 |

采用**方案 A**：不依赖替换 forward，只读取 hooks 并手动执行，行为与标准
PyTorch `_call_impl` 调用链保持一致，且可通过单测充分验证。

### 4.3 实现位置

在 `mindiesd/cache_agent/cache.py` 的 `CacheBase` 中新增以下 6 个方法（公共能力，
`AttentionCache` / `DiTBlockCache` 共用）：

```python
class CacheBase(ABC):
    def _call_with_hooks(self, func, cached_result, args, kwargs, protect_result=False):
        """命中缓存时，仍触发 func（若为 nn.Module）上注册的 forward hooks。
        cached_result 须为未经 forward hooks 处理的原始输出。"""

    def _compute_with_hooks(self, func, args, kwargs):
        """计算原始 forward 输出，并恰好应用一次 hooks，返回 (raw, result, cacheable)。"""

    @staticmethod
    def _requires_native_call(func):
        """判断 func 是否需要完整原生调用链。"""

    @staticmethod
    def _clone_result(result):
        """复制输出图，同时保留别名和循环引用。"""

    @staticmethod
    def _run_forward_pre_hooks(module, args, kwargs):
        """执行 forward pre-hooks（全局优先），与 Module._call_impl 保持一致。"""

    @staticmethod
    def _run_forward_hooks(module, args, kwargs, result):
        """执行 forward hooks（全局优先），与 Module._call_impl 保持一致。"""
```

各缓存方法接入：

- **`AttentionCache.apply_imp`**：
  - 命中：`raw = cache[block]; attn = self._call_with_hooks(func, raw, args, kwargs, protect_result=True)`
  - 未命中：`raw, attn, cacheable = self._compute_with_hooks(func, args, kwargs)`；
    仅在 `cacheable` 时缓存 `self._clone_result(raw)`
- **`DiTBlockCache.apply_imp`**：
  - 未命中：`raw, res, cacheable = self._compute_with_hooks(func, call_args, kwargs)`；
    仅在 `cacheable` 时更新缓存；返回 `res`
  - 命中（双输出）：`cached_raw = (hidden_states + res[0], encoder_hidden_states + res[1])`；
    `self._call_with_hooks(func, cached_raw, (hidden_states, encoder_hidden_states) + args, kwargs)`
  - 命中（单输出）：`cached_raw = hidden_states + res[0]`；
    `self._call_with_hooks(func, cached_raw, (hidden_states,) + args, kwargs)`

### 4.4 关键实现细节

- **全局 hooks 优先**：与 `_call_impl` 一致，先执行 `_global_forward_pre_hooks` /
  `_global_forward_hooks`，再执行模块级 hooks。
- **with_kwargs 分支**：pre-hook 的 kwargs 判断只用模块级
  `module._forward_pre_hooks_with_kwargs`（与 torch 源码一致）；forward hook
  的判断同时含模块与全局标记。
- **全局 pre-hook 的限制**：PyTorch 没有全局 pre-hook 的 kwargs 标记，因此不支持
  `register_module_forward_pre_hook(hook, with_kwargs=True)`；这类 hook 会以
  `hook(module, args)` 两参形式被调用。
- **返回值校验**：with_kwargs pre-hook 必须返回 `None` 或二元 tuple，否则抛与 torch
  相同的 `RuntimeError`。
- **非 kwargs pre-hook 参数**：`hook(module, args)`（位置参数元组作为单个参数），
  与 `Module._call_impl` 一致（非 `*args` 解包）。
- **缓存原始输出**：缓存的是未经 forward hooks 处理的原始输出，输出修改型
  forward hook 在每次逻辑前向中恰好应用一次，不会重复应用；交给 hook 的是原始
  输出的副本，原地修改型 hook（如 `out.mul_(2)`）不会改写缓存。输出图使用
  Tensor identity memo 克隆，保留别名、共享 storage、容器子类、循环引用以及
  `torch.return_types.*`；无法安全复制时抛出 `ModelExecError`。
- **自定义调用链**：在类或实例上重写 `__call__` / `_call_impl`，或设置了
  `_compiled_call_impl` 的模块不进入手工 hook 链路且不使用缓存——每次调用都执行
  正常的 `func(*args, **kwargs)`，保留后处理、执行上下文、分发逻辑、编译入口和
  hooks。
- **原生调用回退**：存在 backward hooks、`always_call` forward hook 或处于 tracing
  状态时，始终使用 `func(*args, **kwargs)` 且不缓存，以保留 backward、异常和
  tracing 语义。
- **低开销保护**：`func` 非 `nn.Module` 或模块/全局均无 hooks 时，跳过模拟链路；
  AttentionCache 仍在缓存边界返回隔离值（仍有少量 Python 级开销，非严格零开销）。
- **范围与承诺（重要）**：命中缓存时全局 hooks 仅对被包装的 `func` 触发（其内部
  子模块在命中时未执行故不触发）；pre-hook 对输入（args/kwargs）的修改不影响
  缓存输出（输出为缓存模拟结果）。这是缓存跳过 forward 计算的固有语义；hook
  不应依赖每次前向的真实计算结果。
- **并发**：`CacheAgent.apply` 使用非阻塞状态锁；同一实例的并发或重入调用直接
  抛出 `ModelExecError`。每个请求或执行流应使用独立实例。

## 5. 测试方案

新增 `tests/cache/test_cache_hooks.py`，覆盖：

| 场景 | 验证点 |
| ------ | -------- |
| 命中缓存时 hook 触发 | pre-hook 与 forward hook 调用次数正确，forward hook 收到缓存原始输出 |
| 原地修改型 hook | `out.mul_(2)` 在命中/未命中路径结果一致，缓存不被改写 |
| forward hook 修改输出 | 命中缓存的输出被 hook 替换（AttentionCache / DiTBlockCache 双输出） |
| hook 不重复应用 | 命中缓存输出与未命中一致（回归：输出修改型 hook 不累积） |
| pre-hook 修改输入 | 命中缓存时 pre-hook 返回值格式被正确兼容（不抛异常），修改输入不影响缓存输出 |
| 命中时完整位置参数 | DiT 命中时 hooks 收到完整位置参数元组（含额外 args） |
| with_kwargs 形式 | 模块级 with_kwargs pre-hook / forward hook，以及全局 with_kwargs forward hook，在命中缓存时均被调用 |
| 全局 hook | `register_module_forward_hook` 在命中缓存时同样触发，且不重复执行 |
| 全局 pre-hook | `register_module_forward_pre_hook` 在命中缓存时触发，未命中路径的输入修改生效 |
| backward / `always_call` / tracing | 使用原生调用且不缓存；backward hooks 正常工作，forward 异常时 `always_call` 仍执行；命中路径下前序 hook 抛异常时 `always_call` 同样触发 |
| 重写 `__call__` 的模块 | 每次使用原生自定义链路与 hooks，禁用缓存 |
| 在类上重写 `_call_impl` 的模块 | 每次使用自定义链路，禁用缓存 |
| 在实例上替换 `_call_impl` 的模块 | 每次都执行实例级自定义链路，禁用缓存 |
| `_compiled_call_impl` 模块 | 每次执行编译入口与 hooks，禁用缓存 |
| 输出图复制 | 别名、共享 storage、容器子类、循环和 `torch.return_types.*`；不可复制时显式失败 |
| 纯函数 | 非 nn.Module 命中时复用缓存并返回隔离副本 |
| 无 hooks 模块 | 命中时复用缓存结果；AttentionCache 返回隔离副本 |
| hook-only 未命中 | pre-hook 与 forward hook 恰好触发一次，输出不变 |
| 调用方修改 / 并发 | 缓存输出互相隔离；并发或重入调用抛出 `ModelExecError` |

## 6. 验证结果

- `tests/cache/` 全部 **61 个用例通过**（新增 35 个，原有 26 个无回归）。
- 新增设计文档通过 `markdownlint`（`markdownlint-cli` v0.44.0 + `.markdownlint.json`）；
  `cache.md` 仍存在 MD045（图片缺 alt）告警，属本次改动范围外的问题，需单独跟进。
- `ruff check` 无新增告警（存量 E501 等不在本次范围）。
- `pylint` 与 `torch==2.9.0` NPU 回归仍需由 CI 环境补充。
- 新测试已接入增量测试映射（`tests/incremental_test_finder.py`），纳入 MR 门禁。

## 7. 风险与后续工作

- **风险**：实现依赖 PyTorch 内部 hook 结构（`_forward_pre_hooks` 等），PyTorch
  大版本升级时需回归验证；原生调用回退可避免不支持的 hook 组合破坏语义。输出
  复制隔离值及结构但不保留 autograd 图，缓存复用仍只面向推理。
- **后续**：
  - 在 `torch==2.9.0` + Ascend NPU 环境回归（当前 CPU 2.8 环境验证通过）。
  - 在真实模型上补充无 hooks 快速路径的性能基准（存在少量 Python 级开销，属低开销
    而非严格零开销）。
  - 代码与文档分别通过 #543、#783 推进合入。
