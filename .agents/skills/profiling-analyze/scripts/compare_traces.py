#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
Compare two Ascend NPU profiling runs (kernel_details.csv level).

Usage:
    python compare_traces.py \
        --baseline <dir1>/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
        --target <dir2>/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
        --output comparison_report.md
"""
# pylint: disable=duplicate-code

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

_KEY_COUNT = "count"
_KEY_TOTAL_DUR = "total_dur"
_KEY_TOTAL_WAIT = "total_wait"
_KEY_MAX_DUR = "max_dur"


def load_csv(path: str) -> Dict[str, Dict]:
    by_name = defaultdict(lambda: {_KEY_COUNT: 0, _KEY_TOTAL_DUR: 0.0, _KEY_TOTAL_WAIT: 0.0, _KEY_MAX_DUR: 0.0})
    total_dur = 0.0
    total_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = row["Name"]
            dur = float(row["Duration(us)"])
            wait = float(row.get("Wait Time(us)", 0))

            by_name[n][_KEY_COUNT] += 1
            by_name[n][_KEY_TOTAL_DUR] += dur
            by_name[n][_KEY_TOTAL_WAIT] += wait
            by_name[n][_KEY_MAX_DUR] = max(by_name[n][_KEY_MAX_DUR], dur)
            total_dur += dur
            total_count += 1

    return dict(by_name), total_dur / 1000.0, total_count


def render_comparison(
    baseline_path: str,
    target_path: str,
    baseline_label: str = "Baseline",
    target_label: str = "Target",
) -> str:
    base, base_total_ms, base_count = load_csv(baseline_path)
    tgt, tgt_total_ms, tgt_count = load_csv(target_path)

    base_set = set(base.keys())
    tgt_set = set(tgt.keys())

    new_ops = {n: tgt[n] for n in tgt_set - base_set}
    gone_ops = {n: base[n] for n in base_set - tgt_set}
    common_ops = base_set & tgt_set

    lines = []

    def w(line=""):
        lines.append(line)

    w(f"# Profiling Comparison: {baseline_label} vs {target_label}")
    w()
    w("## 1. Overview")
    w()
    w(f"| Metric | {baseline_label} | {target_label} | Delta |")
    w("|---|---|---|---|")
    w(f"| Total kernel count | {base_count} | {tgt_count} | {tgt_count - base_count:+d} |")
    w(f"| Unique operator count | {len(base_set)} | {len(tgt_set)} | {len(tgt_set) - len(base_set):+d} |")
    dur_delta = tgt_total_ms - base_total_ms
    w(f"| Total kernel duration | {base_total_ms:.1f} ms | {tgt_total_ms:.1f} ms | {dur_delta:+.1f} ms |")
    w(f"| Data source | {os.path.basename(baseline_path)} | {os.path.basename(target_path)} |")
    w()

    # New ops
    if new_ops:
        w("## 2. New Operators (Target Only)")
        w()
        w("| Operator | Count | Total Dur (ms) | Max Dur (us) |")
        w("|---|---|---|---|")
        for name in sorted(new_ops.keys(), key=lambda n: new_ops[n][_KEY_TOTAL_DUR], reverse=True):
            info = new_ops[name]
            w(f"| {name} | {info[_KEY_COUNT]} | {info[_KEY_TOTAL_DUR] / 1000:.2f} | {info[_KEY_MAX_DUR]:.0f} |")
        w()

    # Gone ops
    if gone_ops:
        w("## 3. Removed Operators (Baseline Only)")
        w()
        w("| Operator | Count | Total Dur (ms) | Max Dur (us) |")
        w("|---|---|---|---|")
        for name in sorted(gone_ops.keys(), key=lambda n: gone_ops[n][_KEY_TOTAL_DUR], reverse=True):
            info = gone_ops[name]
            w(f"| {name} | {info[_KEY_COUNT]} | {info[_KEY_TOTAL_DUR] / 1000:.2f} | {info[_KEY_MAX_DUR]:.0f} |")
        w()

    # Common ops: sorted by delta abs
    w("## 4. Operator Duration Changes (Common Ops)")
    w()
    w("| Operator | BC Count | BC Dur(ms) | TC Count | TC Dur(ms) | Count Delta | Dur Delta(ms) | Dur % |")
    w("|---|---|---|---|---|---|---|")

    all_diffs = []
    for name in common_ops:
        bi = base[name]
        ti = tgt[name]
        bd_ms = bi[_KEY_TOTAL_DUR] / 1000.0
        td_ms = ti[_KEY_TOTAL_DUR] / 1000.0
        diff_ms = td_ms - bd_ms
        count_diff = ti[_KEY_COUNT] - bi[_KEY_COUNT]

        if bd_ms > 0:
            pct = diff_ms / bd_ms * 100
        elif td_ms > 0:
            pct = 999
        else:
            pct = 0

        all_diffs.append((abs(diff_ms), name, bi, ti, diff_ms, count_diff, pct))

    all_diffs.sort(key=lambda x: x[0], reverse=True)

    for _, name, bi, ti, diff_ms, count_diff, pct in all_diffs:
        bd_ms = bi[_KEY_TOTAL_DUR] / 1000.0
        td_ms = ti[_KEY_TOTAL_DUR] / 1000.0
        bc = bi[_KEY_COUNT]
        tc = ti[_KEY_COUNT]
        marker = ""
        if abs(diff_ms) > 1.0:
            if diff_ms > 0:
                marker = " **REGRESSION**"
            else:
                marker = " *improvement*"
        w(
            f"| {name[:60]} | {bc} | {bd_ms:.2f} | {tc} | {td_ms:.2f} | "
            f"{count_diff:+d} | {diff_ms:+.2f} | {pct:+.0f}% | {marker} |"
        )

    w()
    w("## 5. Summary")
    w()
    pct_change = (tgt_total_ms - base_total_ms) / base_total_ms * 100 if base_total_ms > 0 else 0
    w(f"- Total kernel duration changed by: **{tgt_total_ms - base_total_ms:+.1f} ms ({pct_change:+.1f}%)**")
    w(f"- Kernel count changed by: **{tgt_count - base_count:+d}**")
    w(f"- New operators: {len(new_ops)}, Removed operators: {len(gone_ops)}")

    if new_ops:
        new_total = sum(info[_KEY_TOTAL_DUR] for info in new_ops.values()) / 1000.0
        w(f"- New operators total duration: **{new_total:.1f} ms**")

    if gone_ops:
        gone_total = sum(info[_KEY_TOTAL_DUR] for info in gone_ops.values()) / 1000.0
        w(f"- Removed operators total duration: **{gone_total:.1f} ms**")

    w()
    w("## 6. Auto-Verdict")
    w()
    ctx = VerdictContext(base, tgt, base_total_ms, tgt_total_ms, base_count, tgt_count, new_ops, gone_ops)
    verdict, reasons = _compute_verdict(ctx)
    w(f"**Verdict: {verdict}**")
    w()
    for reason in reasons:
        w(f"- {reason}")

    return "\n".join(lines)


class VerdictContext:
    """Context for computing comparison verdict."""

    def __init__(self, base, tgt, base_total_ms, tgt_total_ms, base_count, tgt_count, new_ops, gone_ops):
        self.base = base
        self.tgt = tgt
        self.base_total_ms = base_total_ms
        self.tgt_total_ms = tgt_total_ms
        self.base_count = base_count
        self.tgt_count = tgt_count
        self.new_ops = new_ops
        self.gone_ops = gone_ops


def _compute_verdict(ctx: VerdictContext) -> Tuple[str, list]:
    reasons = []
    fail_flags = False
    warn_flags = False

    dur_change_pct = (ctx.tgt_total_ms - ctx.base_total_ms) / ctx.base_total_ms * 100 if ctx.base_total_ms > 0 else 0
    reasons.append(f"Total kernel duration: {dur_change_pct:+.1f}%")
    if dur_change_pct >= 5.0:
        reasons.append(f"  => REGRESSION: timed inference slowed by {dur_change_pct:.1f}%")
        fail_flags = True
    elif dur_change_pct <= -5.0:
        reasons.append(f"  => IMPROVEMENT: faster by {abs(dur_change_pct):.1f}%")
    elif abs(dur_change_pct) > 1.0:
        reasons.append(f"  => Minor change (+-{abs(dur_change_pct):.1f}%)")
        warn_flags = True
    else:
        reasons.append("  => Noise-level change")

    count_change_pct = (ctx.tgt_count - ctx.base_count) / ctx.base_count * 100 if ctx.base_count > 0 else 0
    reasons.append(f"Kernel count: {count_change_pct:+.1f}% ({ctx.tgt_count - ctx.base_count:+d})")
    if count_change_pct >= 10.0:
        reasons.append("  => WARNING: kernel count inflated >=10% (functionalization overhead)")
        warn_flags = True

    copy_keywords = ["InplaceCopy", "ViewCopy", "TensorMove", "StridedSlice", "_to_copy", "copy_"]
    base_copy_dur = sum(info[_KEY_TOTAL_DUR] for n, info in ctx.base.items() if any(kw in n for kw in copy_keywords))
    tgt_copy_dur = sum(info[_KEY_TOTAL_DUR] for n, info in ctx.tgt.items() if any(kw in n for kw in copy_keywords))
    if base_copy_dur > 0:
        copy_change_pct = (tgt_copy_dur - base_copy_dur) / base_copy_dur * 100
    elif tgt_copy_dur > 0:
        copy_change_pct = 999
    else:
        copy_change_pct = 0
    reasons.append(
        f"Copy operators: {base_copy_dur / 1000:.1f}ms -> {tgt_copy_dur / 1000:.1f}ms ({copy_change_pct:+.1f}%)"
    )
    if copy_change_pct >= 50.0:
        reasons.append("  => CRITICAL: copy operator overhead >=50%")
        fail_flags = True
    elif copy_change_pct >= 20.0:
        reasons.append("  => WARNING: copy operator inflation >=20%")
        warn_flags = True

    new_total_ms = sum(info[_KEY_TOTAL_DUR] for info in ctx.new_ops.values()) / 1000.0
    gone_total_ms = sum(info[_KEY_TOTAL_DUR] for info in ctx.gone_ops.values()) / 1000.0
    net_new = new_total_ms - gone_total_ms
    reasons.append(f"Net new operator cost: {net_new:+.1f}ms (new={new_total_ms:.1f}ms, gone={gone_total_ms:.1f}ms)")
    if net_new > 10.0:
        reasons.append("  => WARNING: net new operator cost >10ms")
        warn_flags = True

    if fail_flags:
        verdict = "**FAIL** -- compile introduces negative performance impact"
    elif warn_flags:
        verdict = "**WARN** -- compile shows minor regressions but no clear win"
    else:
        verdict = "**PASS** -- compile shows no significant degradation"

    return verdict, reasons


def main():
    parser = argparse.ArgumentParser(description="Compare two Ascend profiler kernel_details.csv files")
    parser.add_argument("--baseline", required=True, help="Path to baseline kernel_details.csv")
    parser.add_argument("--target", required=True, help="Path to target kernel_details.csv")
    parser.add_argument("--baseline-label", default="Baseline", help="Label for baseline in report")
    parser.add_argument("--target-label", default="Target", help="Label for target in report")
    parser.add_argument("--output", default="comparison_report.md", help="Output Markdown report path")
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        logger.error("baseline file not found: %s", args.baseline)
        sys.exit(1)
    if not os.path.exists(args.target):
        logger.error("target file not found: %s", args.target)
        sys.exit(1)

    report = render_comparison(
        args.baseline,
        args.target,
        args.baseline_label,
        args.target_label,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Comparison report saved to: %s", args.output)


if __name__ == "__main__":
    main()
