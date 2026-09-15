# 自验证实现模板与判据

把"等价"从口头声明变成运行时事实。核心是三件事：**断言粒度**、**参考实现的调用策略**、
**条件性等价（形状相关）的显式处理**。

## 1. 最小可用包装器

```python
import torch


def install(module, new_impl, reference=None, *, strict=True, tag=""):
    """Replace module.forward with new_impl, asserting bit-exactness against reference.

    strict=True  : 每次调用都对拍（开发期/验收期用）
    strict=False : 只对拍前 N 次或抽样（线上用），但仍保留断言
    """
    orig = module.forward
    ref = reference or orig
    hits = {"n": 0}

    def forward(*args, **kwargs):
        y_ref = ref(*args, **kwargs)
        y_new = new_impl(*args, **kwargs)
        if strict or hits["n"] < 3:
            if y_new.shape != y_ref.shape:
                raise AssertionError("%s shape %s vs %s" % (tag, tuple(y_new.shape), tuple(y_ref.shape)))
            if not torch.equal(y_new, y_ref):
                d = (y_new.float() - y_ref.float()).abs()
                raise AssertionError(
                    "%s NOT bit-exact: max|d|=%g, rel=%g, bad=%d/%d" % (
                        tag, float(d.max()), float(d.mean() / (y_ref.float().abs().mean() + 1e-12)),
                        int((d > 0).sum()), d.numel()))
        hits["n"] += 1
        return y_new

    module.forward = forward
    return module
```

**为什么断言要比"平均值"严格**：`torch.equal` 是二值的，不会给你"看起来还行"的错觉。很多重写错误
（相位、索引、顺序）在平均值上只体现为很小的数，但在逐位上是明确的"不等"。

## 2. 参考实现的调用策略

| 阶段 | 策略 | 原因 |
|---|---|---|
| 开发期 | **每次调用都全量对拍** | 错误在第一次运行就暴露，避免把 bug 带到产物对比阶段 |
| 验收期 | 全量对拍，覆盖**所有会出现的形状** | 等价性可能是形状的函数 |
| 线上 | 只保留"形状/前置条件检查"，参考实现按 env 打开 | 对拍有成本；但**前置条件必须每次检查** |

## 3. 条件性等价：把前置条件显式化

实测实例：某分片替换只在片长满足 `T//2 + halo ≥ 359` 潜帧时逐位精确；短片（`T=500`）绝大多数
样本不等。**正确做法不是删掉这条路径，而是把它写清楚**：

```python
def can_use_fast_path(shape) -> bool:
    """逐位等价的前置条件；不满足时走原实现（并打日志说明为什么）。"""
    return pieces_ok(shape) and min_chunk_len(shape) >= MIN_LEN


def forward(self, x):
    if not can_use_fast_path(x.shape):
        log_once("fast path skipped: shape=%s (< MIN_LEN)" % (tuple(x.shape),))
        return self._original_forward(x)
    return self._fast_forward(x)
```

为了让短片也走快路径，可以先**修前置条件**（实测做法：对短片补零到满足条件，实测 ndiff = 0），
但修复本身也要走同一套逐位验收。

## 4. 报告格式（交付时带上）

```text
替换：<模块>.<方法>  ->  <等价实现>           等价层级：L1 逐位 / L2 数值门
自验证：per-shape torch.equal，断言位置 <文件:行>；覆盖形状 <列出>
逐位结果：ndiff=0 / 最大差异=…（阈值=… 理由=…）
端到端：产物 md5 <值>（N/N 请求一致）；字节数 <值>
收益：同窗 A/B <对照 x s> -> <本档 y s>（差 Δ），窗口 <标识>；跨窗口不比较
回退：<env 名>=1 回到原实现；默认 <开/关>
前置条件：<条件表达式>；不满足时行为 <回退并打日志>
否决记录（若曾否决其它写法）：<判据 + 数字>
```

## 5. 常见坑

- **对拍只做了首调用**：有些形状要到长序列/尾部才出现，首调用覆盖不到；
- **参考实现被就地修改**：参考实现若和替换共享缓冲区/概率性内核，会把差异掩盖掉；
- **用平均误差当判据**：见 §1 的"为什么断言要比平均值严格"；
- **忘记中间张量**：只对拍最终输出会漏掉"中间错、最终恰好还原"的情况（少见但存在），
  关键重写建议对拍 1–2 个中间张量；
- **条件性等价被静默忽略**：不满足条件时不打日志，事后无法知道线上走的是哪条路。
