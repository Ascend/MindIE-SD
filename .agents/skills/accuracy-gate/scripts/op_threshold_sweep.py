#!/usr/bin/env python3
"""Operator threshold sweep: is the defect "a fixed index" or "a size threshold"?

Why this matters: it decides the fix.  A fixed index points at a block/tiling boundary; a size
threshold means "keep every single call below N" (split the call, or carry state across slices).
Both look identical in a single end-to-end run -- only a sweep separates them.

It also compares *equivalent API spellings* of the same op, which tells you whether you can fix it by
changing how you call it, or whether you must avoid the underlying kernel/shape entirely (if several
spellings of the same op fail identically, they share one kernel and no call-site change helps).

Usage:
  python op_threshold_sweep.py --op upsample_nearest --batch 8,16,32,64,128,256 --shape C,H,W
  python op_threshold_sweep.py --op interpolate_size --all-ops

Pass the batches and the shape of the *real* workload: thresholds are a function of shape and
version, so they cannot be reused across shapes or releases (re-sweep instead of extrapolating).
Add your own op to OPS with the same signature (x) -> tensor.
"""
from __future__ import annotations

import argparse
import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    sys.exit("需要 torch (%s)" % exc)

TOL = 1e-3


OPS = {
    "upsample_nearest": lambda x: torch.nn.Upsample(scale_factor=2)(x),
    "upsample_nearest_explicit": lambda x: torch.nn.Upsample(scale_factor=2, mode="nearest")(x),
    "interpolate_nearest": lambda x: F.interpolate(x, scale_factor=2, mode="nearest"),
    "interpolate_nearest_exact": lambda x: F.interpolate(x, scale_factor=2, mode="nearest-exact"),
    "interpolate_size": lambda x: F.interpolate(x, size=(x.shape[-2] * 2, x.shape[-1] * 2), mode="nearest"),
    "repeat_interleave": lambda x: x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1),
}


def sweep(op_name: str, batches: list[int], c: int, h: int, w: int, device: str) -> None:
    fn = OPS[op_name]
    print("== op=%s  shape=(B, %d, %d, %d)  device=%s" % (op_name, c, h, w, device))
    print("   %8s %12s %12s %12s" % ("batch", "max|d|", "first_bad", "input_absmax"))
    for b in batches:
        torch.manual_seed(0)
        x_cpu = torch.randn(b, c, h, w, dtype=torch.float32)
        x = x_cpu.to(device)
        y = fn(x).float().cpu()
        y_ref = fn(x_cpu)
        d = (y - y_ref).abs().reshape(b, -1).amax(1)
        bad = (d > TOL).nonzero()
        print("   %8d %12.4f %12s %12.3f" % (
            b, float(d.max()),
            int(bad[0]) if len(bad) else "—",
            float(x_cpu.abs().max())))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", default="upsample_nearest", choices=sorted(OPS))
    ap.add_argument("--batch", default="8,16,32,64,128,256", help="规模档位（用真实负载档位）")
    ap.add_argument("--shape", default="3,64,64", help="C,H,W（不含 batch）；必须用真实形状")
    ap.add_argument("--device", default=None, help="默认 npu（无则 cpu，仅作脚本自检）")
    ap.add_argument("--all-ops", action="store_true", help="对所有等价写法各扫一遍（判断是否同一 kernel）")
    a = ap.parse_args()
    c, h, w = (int(v) for v in a.shape.split(","))
    device = a.device
    if device is None:
        device = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cpu"
    batches = [int(v) for v in a.batch.split(",")]
    ops = sorted(OPS) if a.all_ops else [a.op]
    for op in ops:
        sweep(op, batches, c, h, w, device)
    print("判读：\n"
          "  * first_bad 随 batch 变化但恒等于某个绝对索引 -> 固定索引/分块边界缺陷\n"
          "  * first_bad 随 batch 变化且总在尾部/某个比例  -> 规模阈值缺陷\n"
          "  * max|d| 每次运行都不同（重跑几遍看）        -> 陈旧内存（缓冲区未写入）\n"
          "  * 所有等价写法表现一致                       -> 同一底层 kernel，改调用方式没用\n"
          "  * 注意 repeat_interleave 这类写法会新建整张张量：数值等价但可能 OOM，先算峰值内存")
    return 0


if __name__ == "__main__":
    sys.exit(main())
