#!/usr/bin/env python3
"""Bit-exact replacement harness: prove "等价" at runtime instead of claiming it.

Why a script: the two failure modes of a rewrite are (a) it is not actually equivalent and nobody
noticed because the平均误差 looked small, and (b) it is equivalent only for SOME shapes.  Both are
mechanical, so both can be enforced by code:

  * assert torch.equal per call (bitwise, not "close enough");
  * record which shapes were actually verified, so "we tested one shape" cannot masquerade as "proved";
  * make conditional equivalence explicit -- when the precondition fails, fall back and LOG it.

Usage (as a library):
    from bit_exact_harness import install, HarnessReport
    rep = HarnessReport(tag="audio-upsample")
    install(module, new_impl, reference=orig_impl, report=rep)
    ...
    rep.dump()          # 覆盖形状、ndiff、最大差异、是否所有调用都通过了断言

Usage (self-demo):
    python bit_exact_harness.py --demo
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import torch


class HarnessReport:
    """Collect per-shape verification evidence so "tested" cannot be inflated into "proved"."""

    def __init__(self, tag: str = "") -> None:
        self.tag = tag
        self.by_shape: dict[tuple, dict] = defaultdict(lambda: {"calls": 0, "max_abs": 0.0, "exact": True})
        self.skips: list[tuple] = []

    def record(self, shape, max_abs: float, exact: bool) -> None:
        e = self.by_shape[tuple(shape)]
        e["calls"] += 1
        e["max_abs"] = max(e["max_abs"], float(max_abs))
        e["exact"] = e["exact"] and bool(exact)

    def record_skip(self, shape, reason: str) -> None:
        self.skips.append((tuple(shape), reason))

    def dump(self) -> None:
        n_shapes = len(self.by_shape)
        all_exact = all(v["exact"] for v in self.by_shape.values())
        print(f"== 自验证报告 tag={self.tag}  覆盖形状 {n_shapes} 个  全部逐位={all_exact}")
        for shape, v in sorted(self.by_shape.items(), key=lambda x: str(x[0])):
            print(f"   shape={shape!s:<34} calls={v['calls']:<5d} max|d|={v['max_abs']:<10g} exact={v['exact']}")
        if self.skips:
            print("   -- 前置条件不满足而回退的调用 --")
            for shape, reason in self.skips:
                print(f"      shape={shape!s:<34} {reason}")
        print(
            f"   结论：{'逐位等价' if all_exact else '存在非逐位形状'}（覆盖形状数 > 1 才能支撑'逐形状验证过'的说法）"
        )


def install(
    module,
    new_impl,
    reference=None,
    *,
    report: HarnessReport | None = None,
    strict: bool = True,
    precondition=None,
    tag: str = "",
    log_once: bool = True,
):
    """Replace module.forward by new_impl with a bitwise assertion against reference.

    precondition(callable) -> (ok: bool, reason: str); when it returns False the original forward is
    used and the skip is logged once per shape (silent skips are how "conditional equivalence" turns
    into a production bug).
    """
    orig = module.forward
    ref = reference or orig
    rep = report or HarnessReport(tag=tag or getattr(module, "__class__", type(module)).__name__)
    seen_skip: set = set()

    def forward(*args, **kwargs):
        x = args[0] if args else next(iter(kwargs.values()))
        if precondition is not None:
            ok, reason = precondition(x)
            if not ok:
                shape = tuple(getattr(x, "shape", ()))
                if log_once and shape not in seen_skip:
                    seen_skip.add(shape)
                    print(f"[harness] fast path skipped for shape={shape}: {reason}", file=sys.stderr)
                rep.record_skip(shape, reason)
                return orig(*args, **kwargs)

        y_new = new_impl(*args, **kwargs)
        if strict:
            y_ref = ref(*args, **kwargs)
            if y_new.shape != y_ref.shape:
                raise AssertionError(f"{tag} shape mismatch {tuple(y_new.shape)} vs {tuple(y_ref.shape)}")
            max_abs = float((y_new.float() - y_ref.float()).abs().max())
            exact = bool(torch.equal(y_new, y_ref))
            rep.record(y_new.shape, max_abs, exact)
            if not exact:
                raise AssertionError(
                    f"{tag} NOT bit-exact: max|d|={max_abs:g} (断言失败即视为不等价；不要用平均误差替代)"
                )
        return y_new

    module.forward = forward
    return module, rep


# --------------------------------------------------------------------------------------
# demo：一个"看起来等价但有形状条件"的替换，用来演示断言与覆盖记录
# --------------------------------------------------------------------------------------
class _Patch:
    """示意模块：把最后一维按 2 倍复制（最近邻 1-D 上采样的等价写法）。"""

    def __init__(self, dim: int = -1) -> None:
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _fast(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(2, dim=x.dim() - 1 if x.dim() else 0) if x.dim() == 1 else x.repeat_interleave(2, dim=-1)


def _precondition(x: torch.Tensor):
    # 演示"条件性等价"：只在长度 ≤ 6 时假定安全（真实场景里这个条件来自实测，不要凭感觉设）
    if x.shape[-1] > 6:
        return False, f"len={x.shape[-1]} > 6 (demo condition)"
    return True, ""


def demo() -> int:
    m = _Patch()
    _, rep = install(m, _fast, reference=_Patch().forward, precondition=_precondition, tag="demo-upsample", strict=True)
    for shape in ((2, 3, 4), (1, 5, 6), (1, 5, 8)):
        x = torch.randn(*shape)
        y = m(x)
        assert y.shape[-1] == shape[-1] * 2
    rep.dump()
    print("\n注意 demo 里 (1,5,8) 走了回退路径 —— 它被记录在 report.skips 里，而不是被静默忽略。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="跑内置演示（含条件性回退与逐形状覆盖记录）")
    a = ap.parse_args()
    if a.demo:
        return demo()
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
