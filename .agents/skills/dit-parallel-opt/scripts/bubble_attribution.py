#!/usr/bin/env python3
"""Attribute exposed communication and device bubbles from an Ascend kernel capture.

Answers two questions that decide where a masking effort should go next:

1. **Which part of the exposed communication is which collective?**  The compute
   stream's gaps are overlapped against each collective group (e.g. the forward
   Q/K/V all-to-all versus the reverse output all-to-all); a group whose time
   lands mostly *inside* those gaps is the one actually on the critical path.

2. **Are the gaps before each attention kernel a host launch problem or a
   dependency wait?**  Gap time that coincides with a running collective is a
   wait on the wire (a scheduling/payload problem); gap time with nothing
   running anywhere is a genuine bubble (where launch overhead, dispatch
   threading or graph capture could help).

Both are read off one capture -- no NPU needed, stdlib only.

Usage::

    python bubble_attribution.py --profile-dir prof_dit8_v2 --layers 50 \\
        --collective alltoall --group-split 4
    # --group-split N: within each layer's run of collectives, the last 1/N of
    # them are treated as the "reverse" group (4 tensors per chunk: q,k,v,o).

See ``references/comm-masking-method.md`` (what the numbers mean) and
``references/parallel-plan-attribution-method.md`` (how to present them).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

AI_CORE_PREFIXES = ("AI_CORE", "AI_VECTOR", "MIX_")
COMM_CORES = ("COMMUNICATION", "AI_CPU")


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a profiler CSV, dropping rows whose field count does not match."""
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    head = rows[0]
    return head, [r for r in rows[1:] if len(r) == len(head)]


def union_ms(intervals: list[tuple[float, float]]) -> float:
    """Union length of closed-open intervals (inputs in us, result in ms)."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total, cur_s, cur_e = 0.0, ordered[0][0], ordered[0][1]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return (total + cur_e - cur_s) / 1000.0


def overlap_ms(intervals: list[tuple[float, float]], ref: list[tuple[float, float]]) -> float:
    """Total length of ``intervals`` covered by ``ref`` (both in us)."""
    ref = sorted(ref)
    j = 0
    total = 0.0
    for s, e in sorted(intervals):
        while j < len(ref) and ref[j][1] <= s:
            j += 1
        k = j
        while k < len(ref) and ref[k][0] < e:
            lo, hi = max(s, ref[k][0]), min(e, ref[k][1])
            if hi > lo:
                total += hi - lo
            k += 1
    return total / 1000.0


def gaps_of(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Gaps between the union of ``intervals``."""
    ordered = sorted(intervals)
    out = []
    _cs, ce = ordered[0]
    for s, e in ordered[1:]:
        if s > ce:
            out.append((ce, s))
            ce = e
        else:
            ce = max(ce, e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--layers", type=int, default=50, help="layers per step (splits the groups)")
    ap.add_argument(
        "--collective", default="alltoall", help="kernel-name substring of the collective"
    )
    ap.add_argument(
        "--group-split",
        type=int,
        default=4,
        help="tensors per chunk; the last 1/N of each layer's collectives is the reverse",
    )
    ap.add_argument(
        "--attn-name",
        default="Attention",
        help="kernel-name substring identifying the attention kernels",
    )
    ap.add_argument(
        "--release-delta",
        type=float,
        default=100.0,
        help="us tolerance for 'this collective finished exactly at the gap end'",
    )
    args = ap.parse_args()

    root = Path(args.profile_dir)
    hits = sorted(root.rglob("kernel_details.csv"))
    if not hits:
        print(f"no kernel_details.csv under {root}")
        return 2
    head, rows = _read(hits[0])
    idx = {n: i for i, n in enumerate(head)}
    for needed in ("Name", "Duration(us)", "Start Time(us)", "Stream ID", "Accelerator Core"):
        if needed not in idx:
            print(f"missing column {needed!r}")
            return 2

    per_stream: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    coll: list[tuple[float, float]] = []
    attn: list[tuple[float, float]] = []
    for r in rows:
        try:
            s, d = float(r[idx["Start Time(us)"]]), float(r[idx["Duration(us)"]])
        except (TypeError, ValueError):
            continue
        name, core, stream = r[idx["Name"]], r[idx["Accelerator Core"]], r[idx["Stream ID"]]
        per_stream[(stream, core)].append((s, s + d))
        if args.collective in name:
            coll.append((s, s + d))
        if args.attn_name in name:
            attn.append((s, s + d))

    if not per_stream or not coll:
        print("no compute stream or no matching collective in this capture")
        return 2

    # compute stream = the stream carrying the most AI-core work
    candidates = [(len(v), s) for (s, c), v in per_stream.items() if c.startswith(AI_CORE_PREFIXES)]
    if not candidates:
        print("no AI-core kernels found; is this a device capture?")
        return 2
    compute_stream = max(candidates)[1]

    comp = sorted(iv for (s, _c), v in per_stream.items() if s == compute_stream for iv in v)
    comm = sorted(iv for (s, c), v in per_stream.items() if c in COMM_CORES for iv in v)
    gaps = gaps_of(comp)

    coll.sort()
    per_layer = max(len(coll) // max(args.layers, 1), 1)
    rev_n = max(per_layer // max(args.group_split, 1), 1)
    forward = [iv for i, iv in enumerate(coll) if (i % per_layer) < per_layer - rev_n]
    reverse = [iv for i, iv in enumerate(coll) if (i % per_layer) >= per_layer - rev_n]

    span = (
        max(e for v in per_stream.values() for _s, e in v)
        - min(s for v in per_stream.values() for s, _e in v)
    ) / 1000.0
    device_busy = union_ms([iv for v in per_stream.values() for iv in v])
    gap_total = sum(e - s for s, e in gaps) / 1000.0

    print(f"capture            : {hits[0]}")
    print(
        f"step span          : {span:.1f} ms   device busy {device_busy:.1f} ms "
        f"({100 * device_busy / span:.1f}%)   idle {span - device_busy:.1f} ms"
    )
    print(f"compute stream     : {compute_stream}")
    print(f"compute-stream gaps: n={len(gaps)} total={gap_total:.1f} ms")
    running = overlap_ms(gaps, comm)
    print(
        f"  with a collective running : {running:.1f} ms "
        f"({100 * running / gap_total if gap_total else 0:.1f}%)"
        "   (necessary but NOT sufficient for a dependency)"
    )
    print(
        f"  with nothing running      : {gap_total - running:.1f} ms "
        f"({100 * (gap_total - running) / gap_total if gap_total else 0:.1f}%)"
        "   <- real bubble (launch/dispatch/graph lever)"
    )
    # release test: a gap caused by a collective ends when that collective ends
    coll_end = sorted(e for s, e in coll)
    delta = args.release_delta
    released = (
        sum((e - s) for s, e in gaps if any(abs(ce - e) <= delta for ce in coll_end)) / 1000.0
    )
    print(
        f"  released by a collective end: {released:.1f} ms "
        f"({100 * released / gap_total if gap_total else 0:.1f}%)"
        f"   <- true dependency on the wire (delta={delta:g} us)"
    )
    print(
        "  -> only the released part is a wire dependency; the rest resumed while a "
        "transfer was still in flight (it was serving later work)."
    )
    if attn:
        # producer-consumer test, targeted: compare each attention kernel with the
        # collective that carries ITS OWN input (the last forward op of its chunk).
        chunks = max(per_layer // max(args.group_split, 1), 1)
        fwd_per_chunk = max(args.group_split - 1, 1)
        own, generic = [], []
        for i, (s, _e) in enumerate(sorted(attn)):
            prod = (i // chunks) * per_layer + (i % chunks) * fwd_per_chunk + fwd_per_chunk - 1
            if 0 <= prod < len(coll):
                own.append(s - coll[prod][1])
            prev_end = max((e for _s, e in coll if e <= s), default=None)
            if prev_end is not None:
                generic.append(s - prev_end)
        for label, series in (("its own producer", own), ("any collective", generic)):
            if not series:
                continue
            series.sort()
            soon = 100 * sum(1 for b in series if b <= 100) / len(series)
            print(
                f"  consumer start - end({label}) (n={len(series)}): median "
                f"{series[len(series) // 2] / 1000:.2f} ms, <=0.1 ms in {soon:.0f}% of cases"
                "   -> large values mean that collective ran AHEAD (not the bottleneck)"
            )
    print()
    print(
        f"collectives `{args.collective}` : n={len(coll)} per layer={per_layer} "
        f"(reverse group = last {rev_n})"
    )
    for label, grp in (("forward", forward), ("reverse", reverse)):
        total = sum(e - s for s, e in grp) / 1000.0
        in_gap = overlap_ms(grp, gaps)
        print(
            f"  {label:<8} n={len(grp):<5} total={total:8.1f} ms | exposed (inside compute gaps) "
            f"{in_gap:7.1f} ms = {100 * in_gap / total if total else 0:5.1f}%"
            f" | overlapped with attention {overlap_ms(grp, attn):7.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
