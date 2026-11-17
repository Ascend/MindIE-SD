# Cache Optimization: Keep Model Hooks Fired on Cache Hit

> Related Issue: [#139 cache功能优化-MindIE-SD](https://gitcode.com/Ascend/MindIE-SD/issues/139)

## 1. Background

MindIE-SD provides compute-memory trade-off cache acceleration (`AttentionCache` /
`DiTBlockCache`) that reuses intermediate results at block / attention granularity
to reduce redundant computation between adjacent sampling steps of diffusion
models.

**Problem**: When the cache is enabled, hooks registered on the model (e.g.
`nn.Module.register_forward_pre_hook` / `register_forward_hook` for feature
extraction, intermediate-result capture, quantization, offload, etc.) are skipped
on a cache hit, making model behavior inconsistent with cache disabled.

## 2. Current Status

### 2.1 Cache Mechanism

```text
CacheAgent.apply(func, *args, **kwargs)
  └─ CacheBase.apply(func, *args, **kwargs)          # unified counter
       └─ AttentionCache.apply_imp / DiTBlockCache.apply_imp
            ├─ cache miss: func(*args, **kwargs)     # normal forward, hooks fire
            └─ cache hit: return cached result       # ❌ func skipped, hooks skipped
```

On a cache hit the cached result is returned directly without invoking `func`.
Since PyTorch forward hooks are triggered inside `nn.Module.__call__ → _call_impl`,
when `func` (the block module) is not invoked, its hooks are never executed.

### 2.2 PyTorch Hook Mechanism (torch 2.x)

Execution order in `nn.Module._call_impl`:

1. **Forward pre-hooks** (global first): `hook(module, args)` (`args` is the
   positional-argument tuple), may return modified args; with `with_kwargs=True`:
   `hook(module, args, kwargs)`, may return `(new_args, new_kwargs)`.
2. **Forward**: `forward(*args, **kwargs)`.
3. **Forward hooks** (global first): `hook(module, args, result)`, returning a
   non-None value replaces the result; with `with_kwargs=True`:
   `hook(module, args, kwargs, result)`.

Key internal structures:

| Structure | Description |
| ------ | ------ |
| `module._forward_pre_hooks` / `_forward_hooks` | OrderedDict, module-level hooks |
| `module._forward_pre_hooks_with_kwargs` / `_forward_hooks_with_kwargs` | with_kwargs markers |
| `torch.nn.modules.module._global_forward_pre_hooks` / `_global_forward_hooks` | global hooks |
| `_global_forward_hooks_with_kwargs` | global with_kwargs markers |

## 3. Design Goals

1. **Compatibility**: cache-eligible blocks trigger forward pre-hooks / forward
   hooks on a cache hit (module-level hooks support `with_kwargs`; global forward
   hooks also support `with_kwargs`, global pre-hooks do not). Modules requiring
   backward hooks, `always_call`, or tracing use the native PyTorch call path
   every time and are not cached.
2. **Zero intrusion**: no changes to how users register hooks; the
   `CacheAgent.apply` interface is unchanged.
3. **Low overhead**: for plain functions / hook-free modules the original fast
   path is preserved (with a small Python-level overhead).
4. **No duplicated application**: hooks are applied exactly once per logical
   forward pass — the cache stores the raw forward output *before* forward hooks,
   and hooks are applied once on both cache-miss and cache-hit paths.
5. **Consistent semantics**: on a cache hit hooks receive the cached output as the
   simulated forward output (approximated for DiT), which is inherent to
   compute-memory trade-off.

## 4. Technical Design

### 4.1 Core Idea

Before selecting a cache path, the implementation checks whether the module needs
the complete native call chain. Such modules run normally without caching. For
other modules, a cache hit uses the raw cached output to simulate the forward
output while the `Module._call_impl` forward-hook chain is replicated:

```text
cache hit:
  if func requires native handling (backward hooks / always_call / tracing):
    result = func(*args, **kwargs)            # full native call, do not cache
  elif func uses a custom call chain:
    result = func(*args, **kwargs)            # full native call, do not cache
  elif func is nn.Module, uses the standard PyTorch call chain and has hooks:
    args, kwargs = run forward pre-hooks (global -> module, with_kwargs branches)
    result = raw cached output (simulated forward output, no hooks applied)
    result = run forward hooks (global -> module, may modify output)
    return result
  else (not nn.Module / no hooks):
    return isolated copy of cached result
```

On a cache miss the raw forward output (before forward hooks) is cached:

```text
cache miss (func is nn.Module with hooks):
  if func requires native handling:
    result = func(*args, **kwargs)            # full native call, do not cache
    return result
  if func uses a custom call chain:
    result = func(*args, **kwargs)            # full native call, do not cache
    return result
  args, kwargs = run forward pre-hooks
  raw = func.forward(*args, **kwargs)      # real forward without hooks
  result = run forward hooks(raw)          # hooks applied exactly once
  cache raw                                # cache the hook-free output
  return result
```

Modules overriding `__call__` / `_call_impl` on the class or an instance, or
using a non-null `_compiled_call_impl` from `torch.compile`, do not enter this
manual chain: every invocation uses `func(*args, **kwargs)` and the result is not
cached. This preserves the complete custom call chain and its hooks.

This guarantees each hook is applied exactly once per logical forward pass, and
the output on a cache hit equals the output on a cache miss (no duplicated
application of output-modifying hooks).

### 4.2 Design Alternatives

| Option | Description | Verdict |
| ------ | ------ | ------ |
| A. Replicate the `_call_impl` hook chain (**adopted**) | Implement pre-hook / hook execution in `CacheBase`, aligned with the standard PyTorch call chain | Controllable and testable; custom call chains are explicitly degraded |
| B. Temporarily replace `module.forward` then call `__call__` | Swap forward to "return cache" on cache hit and let native `__call__` fire hooks | Concurrency/reentrancy risk; ineffective for custom call chains; side effects |
| C. Only provide a CacheAgent callback interface | Users register callbacks via a new interface | Does not cover hooks already registered on the model; requires user changes |

Option **A** is adopted: hooks are read and executed manually without replacing
forward, matching the standard PyTorch `_call_impl` call chain and fully
verifiable by unit tests.

### 4.3 Implementation

Six methods are added to `CacheBase` in `mindiesd/cache_agent/cache.py`
(shared by `AttentionCache` / `DiTBlockCache`):

```python
class CacheBase(ABC):
    def _call_with_hooks(self, func, cached_result, args, kwargs, protect_result=False):
        """On a cache hit, still trigger forward hooks registered on func
        (if nn.Module). cached_result must be the raw hook-free output."""

    def _compute_with_hooks(self, func, args, kwargs):
        """Compute the raw forward output and apply hooks exactly once.
        Returns (raw, result, cacheable)."""

    @staticmethod
    def _requires_native_call(func):
        """Return whether func needs the complete native call chain."""

    @staticmethod
    def _clone_result(result):
        """Clone an output graph while preserving aliases and cycles."""

    @staticmethod
    def _run_forward_pre_hooks(module, args, kwargs):
        """Run forward pre-hooks (global first), aligned with Module._call_impl."""

    @staticmethod
    def _run_forward_hooks(module, args, kwargs, result):
        """Run forward hooks (global first), aligned with Module._call_impl."""
```

Integration points:

- **`AttentionCache.apply_imp`**:
  - cache hit: `raw = cache[block]; attn = self._call_with_hooks(func, raw, args, kwargs, protect_result=True)`
  - cache miss: `raw, attn, cacheable = self._compute_with_hooks(func, args, kwargs)`;
    cache `self._clone_result(raw)` only when `cacheable`
- **`DiTBlockCache.apply_imp`**:
  - cache miss: `raw, res, cacheable = self._compute_with_hooks(func, call_args, kwargs)`;
    update the cache only when `cacheable`; return `res`
  - cache hit (two outputs): `cached_raw = (hidden_states + res[0], encoder_hidden_states + res[1])`;
    `self._call_with_hooks(func, cached_raw, (hidden_states, encoder_hidden_states) + args, kwargs)`
  - cache hit (single output): `cached_raw = hidden_states + res[0]`;
    `self._call_with_hooks(func, cached_raw, (hidden_states,) + args, kwargs)`

### 4.4 Key Details

- **Global hooks first**: consistent with `_call_impl`, global pre-hooks / hooks
  run before module-level ones.
- **`with_kwargs` branches**: pre-hook kwargs detection uses only the
  module-level `module._forward_pre_hooks_with_kwargs`, matching torch source;
  forward-hook detection includes both module and global markers.
- **Global pre-hook limitation**: PyTorch has no kwargs marker for global
  pre-hooks, so `register_module_forward_pre_hook(hook, with_kwargs=True)` is not
  supported; such a hook is invoked as `hook(module, args)` and must therefore
  use the two-argument signature.
- **Return-value validation**: `with_kwargs` pre-hooks must return `None` or a
  two-tuple, otherwise a `RuntimeError` matching torch is raised.
- **Non-kwargs pre-hook arguments**: `hook(module, args)` — the positional
  argument tuple is passed as a single argument, matching `Module._call_impl`
  (not `*args` unpacked).
- **Raw output caching**: the cache stores the hook-free raw forward output so
  output-modifying forward hooks are applied exactly once per logical forward.
  Hooks receive a copy of that raw output, so in-place hooks (`out.mul_(2)`) do
  not rewrite the cache. The output graph is cloned with a Tensor identity memo:
  aliases, shared storage, container subtypes, cyclic references and
  `torch.return_types.*` are preserved. If copying fails, `ModelExecError` is
  raised instead of reusing a mutable original object.
- **Custom call chains**: modules overriding `__call__` / `_call_impl` on the
  class or an instance, or using a non-null `_compiled_call_impl`, skip the
  manual hook chain and are never cached; every invocation calls
  `func(*args, **kwargs)` so custom post-processing, execution context, dispatch
  logic, the compiled entry point, and hooks are all preserved.
- **Native-call fallback**: modules with backward hooks, `always_call` forward
  hooks, or tracing always use `func(*args, **kwargs)` and are not cached. This
  preserves backward hook, exception, and tracing semantics.
- **Low-overhead guard**: for non-`nn.Module` funcs or hook-free modules, the
  simulation path is skipped; AttentionCache still isolates the cached value at
  the cache boundary.
- **Notes / scope**: on a cache hit, global hooks fire only for the wrapped
  `func` itself (not its child modules, which are not executed). These are
  inherent to skipping forward computation; hooks should not rely on the real
  result of every forward pass.
- **Concurrency**: `CacheAgent.apply` is guarded by a non-blocking state lock.
  Concurrent or reentrant use of one instance raises `ModelExecError`; use one
  agent per request or execution stream.

## 5. Test Plan

New file `tests/cache/test_cache_hooks.py` covers:

| Scenario | Verification |
| ------ | ------ |
| Hooks fire on cache hit | pre-hook / forward-hook call counts correct; forward hook receives cached raw output |
| In-place output hooks | `out.mul_(2)` yields identical results on hit/miss; the cache is not rewritten |
| Output-modifying forward hook | output replaced on cache hit (AttentionCache / DiTBlockCache two outputs) |
| Hook not applied twice | cache hit output equals cache-miss output (regression for duplicated application) |
| Pre-hook input modification | return-value format compatible (no exception); input modification does not affect cached output |
| Full positional args on cache hit | DiT hooks receive the complete positional-argument tuple (including extra args) |
| `with_kwargs` forms | module-level `with_kwargs=True` pre-hook / forward hook, and global `with_kwargs=True` forward hook, fire on cache hit |
| Global hooks | `register_module_forward_hook` fires on cache hit without duplicate execution |
| Global pre-hooks | `register_module_forward_pre_hook` fires on cache hit, and its input modification takes effect on a cache miss |
| Backward / `always_call` / tracing | native call and no cache reuse; backward hooks keep working and `always_call` runs when forward raises, including when an earlier hook raises on the hit path |
| Modules overriding `__call__` | every call uses the native custom chain and hooks; caching is disabled |
| Modules overriding `_call_impl` on the class | every call uses the custom chain; caching is disabled |
| Modules patching `_call_impl` on an instance | the instance patched chain runs every time; caching is disabled |
| `_compiled_call_impl` modules | compiled entry points and hooks run every time; caching is disabled |
| Output graph cloning | aliases, shared storage, container subtypes, cycles, `torch.return_types.*`, and uncloneable outputs are covered |
| Plain functions | non-`nn.Module` reuses the cache and returns an isolated result on hit |
| Hook-free modules | cache result reused on hit; AttentionCache returns an isolated copy |
| Hook-only cache miss | pre-hooks and forward hooks fire exactly once and output is unchanged |
| Caller mutation / concurrency | cached outputs are isolated; concurrent or reentrant agent use raises `ModelExecError` |

## 6. Verification Results

- `tests/cache/`: all **61 test cases pass** (35 new + 26 existing, no
  regressions).
- `ruff check` reports no new findings (pre-existing E501 and similar are out of
  scope for this change).
- New design documents pass `markdownlint` (`markdownlint-cli` v0.44.0 with
  `.markdownlint.json`); `cache.md` still has MD045 (missing image alt text)
  warnings, which are out of scope here and tracked separately.
- `pylint` and the `torch==2.9.0` NPU regression still need CI coverage.
- The new test file is wired into the incremental test mapping
  (`tests/incremental_test_finder.py`) so MR gates cover it.

## 7. Risks and Follow-ups

- **Risk**: the implementation relies on PyTorch internal hook structures
  (`_forward_pre_hooks`, etc.); verify on PyTorch upgrades. The native-call
  fallback prevents unsupported hook combinations from corrupting semantics.
  Output cloning isolates values and structure but intentionally does not retain
  an autograd graph; cache reuse remains inference-oriented.
- **Follow-ups**:
  - Regression on `torch==2.9.0` with Ascend NPU (CPU torch 2.8 passes locally).
  - Benchmarks on real models for the no-hooks fast path (a small Python-level
    overhead remains; it is low-overhead, not strictly zero-overhead).
  - Land the code and documentation through #543 and #783 respectively.
