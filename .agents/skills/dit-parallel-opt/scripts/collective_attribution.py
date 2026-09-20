#!/usr/bin/env python3
"""Attribute collectives and streams in an Ascend profiler capture.

Answers the three questions that a parallel-plan comparison keeps needing:

1. **Which parallel group does each collective belong to?**  The kernel name
   carries the communicator id (``hcom_allGather_AicpuKernel_527_0_1``), so
   bucketing by ``(family, gid)`` separates e.g. the per-layer weight gather of
   a layerwise-offload from the sequence-parallel all-to-all.  The printed
   ``n/(layers*steps)`` column is the reconciliation check: a family that does
   not divide evenly hides a second same-family collective (offload, TP,
   text encoder, VAE).

2. **Is the measured communication time transfer time or a dependency wait?**
   Per family it computes the union of the ``[start, start+duration)``
   intervals: ``union/sum == 1`` means the family runs strictly one op at a
   time, i.e. the seconds are transfer time and only shrinking the payload can
   help.  The cross-family matrix then shows how much two families actually
   overlap (``inter / min(union)``).

3. **Is any AI-core work leaking onto the communication stream?**  A stream that
   carries a large number of AI-core kernels is the fingerprint of tensor
   transforms being issued inside ``with torch.<backend>.stream(comm)``; the
   masking design requires that stream to carry collectives only.

Stdlib only, no NPU needed.  See ``references/parallel-plan-attribution-method.md``.

Usage::

    python collective_attribution.py --profile-dir prof_dit8_ovl --layers 50
    python collective_attribution.py --profile-dir . --top 5
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

# hcom_allGather_AicpuKernel_527_0_1 -> ("allGather", "527")
_COMM_RE = re.compile(r"hcom_([A-Za-z]+)(?:_AicpuKernel)?_(\d+)_")
_GID_RE = re.compile(r"_(\d+)_")
_FAMILY_HINTS = (
    ("allgather", "allGather"),
    ("alltoall", "allToAll"),
    ("allreduce", "allReduce"),
    ("reducescatter", "reduceScatter"),
    ("broadcast", "broadcast"),
    ("send", "send"),
    ("recv", "recv"),
)


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a profiler CSV, dropping rows whose field count does not match.

    Captures contain a few rows with an embedded newline inside an unquoted
    field, which desynchronises ``csv.reader``; skipping those keeps the rest
    usable instead of failing the whole file.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    header = rows[0]
    return header, [r for r in rows[1:] if len(r) == len(header)]


def _classify(name: str) -> tuple[str, str] | None:
    """Return ``(family, communicator_id)`` for a collective kernel name."""
    match = _COMM_RE.match(name)
    if match:
        return match.group(1), match.group(2)
    low = name.lower()
    if "hcom" not in low and "hccl" not in low:
        return None
    for hint, family in _FAMILY_HINTS:
        if hint in low:
            found = _GID_RE.search(name)
            return family, found.group(1) if found else "-"
    return ("other", "-")


def _union_ms(intervals: list[tuple[float, float]]) -> float:
    """Union length of closed-open intervals, in ms (inputs are us)."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total, cur_start, cur_end = 0.0, ordered[0][0], ordered[0][1]
    for start, end in ordered[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    total += cur_end - cur_start
    return total / 1000.0


def _intersection_ms(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total intersection length of two interval sets, in ms."""
    ua, ub = sorted(a), sorted(b)
    i = j = 0
    total = 0.0
    while i < len(ua) and j < len(ub):
        lo = max(ua[i][0], ub[j][0])
        hi = min(ua[i][1], ub[j][1])
        if hi > lo:
            total += hi - lo
        if ua[i][1] <= ub[j][1]:
            i += 1
        else:
            j += 1
    return total / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--layers", type=float, default=50.0, help="DiT layers per step")
    ap.add_argument("--steps", type=float, default=1.0, help="steps in this capture")
    ap.add_argument("--top", type=int, default=6, help="families in the overlap matrix")
    args = ap.parse_args()

    root = Path(args.profile_dir)
    hits = sorted(root.rglob("kernel_details.csv"))
    if not hits:
        print(f"no kernel_details.csv under {root}", file=sys.stderr)
        return 2
    header, rows = _read_rows(hits[0])
    if not rows:
        print(f"no usable rows in {hits[0]}", file=sys.stderr)
        return 2

    idx = {name: i for i, name in enumerate(header)}
    for needed in ("Name", "Duration(us)", "Start Time(us)"):
        if needed not in idx:
            print(f"missing column {needed!r} in {hits[0]}", file=sys.stderr)
            return 2

    families: dict[tuple[str, str], list[tuple[float, float]]] = {}
    streams: dict[str, list[float]] = {}
    for row in rows:
        try:
            start = float(row[idx["Start Time(us)"]])
            dur = float(row[idx["Duration(us)"]])
        except (TypeError, ValueError):
            continue
        name = row[idx["Name"]]
        core = row[idx["Accelerator Core"]] if "Accelerator Core" in idx else "N/A"
        stream = row[idx["Stream ID"]] if "Stream ID" in idx else "N/A"
        bucket = streams.setdefault(f"{stream}|{core}", [0.0, 0.0])
        bucket[0] += dur / 1000.0
        bucket[1] += 1.0
        found = _classify(name)
        if found is None:
            continue
        families.setdefault(found, []).append((start, start + dur))

    per_layer = max(args.layers * args.steps, 1e-9)
    print(f"capture      : {hits[0]}")
    print(f"denominator  : layers={args.layers:g} steps={args.steps:g} -> {per_layer:g} layer-steps")
    print()
    print(
        f"{'family':<16}{'gid':<8}{'n':>7}{'calls/layer':>13}{'total ms':>11}"
        f"{'mean ms':>10}{'median ms':>11}{'max ms':>10}{'union/sum':>11}"
    )
    ordered = sorted(families.items(), key=lambda kv: -sum(d for _, d in kv[1]))
    for (family, gid), spans in ordered:
        durs = [end - start for start, end in spans]
        total = sum(durs) / 1000.0
        union = _union_ms(spans)
        ratio = union / total if total > 0 else 0.0
        print(
            f"{family:<16}{gid:<8}{len(spans):>7}{len(spans) / per_layer:>13.2f}"
            f"{total:>11.1f}{statistics.mean(durs) / 1000.0:>10.2f}"
            f"{statistics.median(durs) / 1000.0:>11.2f}{max(durs) / 1000.0:>10.2f}"
            f"{ratio:>11.2f}"
        )

    print()
    print("reading: union/sum ~ 1.0 -> the family runs one op at a time, so the time is")
    print("         transfer time (shrink the payload); < 1.0 -> ops are concurrent,")
    print("         so the summed time is occupancy, not wall clock.")
    print("         calls/layer must be an integer multiple of the expected per-layer")
    print("         count; a fractional value hides a second same-family collective.")

    top = [key for key, _ in ordered[: max(args.top, 0)]]
    if len(top) > 1:
        print()
        print("cross-family overlap (inter / min(union), 1.00 = fully concurrent):")
        label = [f"{family}@{gid}" for family, gid in top]
        print(f"{'':<18}" + "".join(f"{name[:11]:>12}" for name in label))
        for i, key_a in enumerate(top):
            cells = []
            for j, key_b in enumerate(top):
                if j <= i:
                    cells.append(f"{'':>12}")
                    continue
                inter = _intersection_ms(families[key_a], families[key_b])
                denom = min(_union_ms(families[key_a]), _union_ms(families[key_b]))
                cells.append(f"{(inter / denom if denom > 0 else 0.0):>12.2f}")
            print(f"{label[i][:17]:<18}" + "".join(cells))

    print()
    print("streams (a masking design requires the communication stream to carry no AI-core work):")
    print(f"{'stream':<10}{'core':<18}{'kernels':>9}{'total ms':>12}")
    for key in sorted(streams, key=lambda k: (k.split("|")[0], k.split("|")[1])):
        stream, core = key.split("|")
        ms, count = streams[key]
        print(f"{stream:<10}{core:<18}{int(count):>9}{ms:>12.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
