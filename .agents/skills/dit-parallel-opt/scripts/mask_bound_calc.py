#!/usr/bin/env python3
"""Compute the communication-masking upper bound from an Ascend profiler capture.

Given a profile directory (one that contains ``step_trace_time.csv`` and
``kernel_details.csv`` anywhere below it), this prints, in the per-layer
denominators a masking decision needs:

    C  = communication total per layer   (from ``Communication`` column)
    F  = the compute that could hide it  (sum of the attention-family kernels)
    c  = C / chunks, f = F / chunks      (per pipeline chunk)

and from those

    ideal bound  = 1 - 1/chunks                     (valid only when f >= c)
    real  bound  = 1 - (c + (chunks-1)*max(0, c-f) + drain) / C
    measured     = Overlapped / Communication
    achievement  = measured / real bound

The real bound is a pipeline model: one chunk of communication is exposed as
pipeline fill, each additional chunk exposes whatever its compute cannot cover,
and the last chunk's reverse communication is exposed as drain.

Why the distinction matters: when ``f < c`` the ``1 - 1/n`` figure is NOT
reachable and raising ``chunks`` does not help.  Only shrinking the
communication payload (or enlarging the hiding compute) does.  See
``references/comm-masking-method.md``.

Usage::

    python mask_bound_calc.py --profile-dir prof_dit8_ovl --layers 50 --chunks 7
    python mask_bound_calc.py --profile-dir . --fa-name EagleQuantBlockSparseAttention

Stdlib only; no NPU required (parses an already-collected capture).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

STEP_COLUMNS = (
    "Computing",
    "Communication(Not Overlapped)",
    "Overlapped",
    "Communication",
    "Free",
    "Stage",
)

# hcom_allGather_AicpuKernel_527_0_1 -> ("allGather", "527")
_COMM_RE = re.compile(r"hcom_([A-Za-z]+)(?:_AicpuKernel)?_(\d+)_")


def _find_one(root: Path, name: str) -> Path | None:
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def _read_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read a profiler CSV, dropping rows whose field count does not match.

    The captures contain a few rows with an embedded newline inside an
    unquoted field, which desynchronises ``csv.reader``; skipping them keeps
    the rest usable instead of failing the whole file.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    header = rows[0]
    return header, [r for r in rows[1:] if len(r) == len(header)]


def _median_stage_row(header: list[str], rows: list[list[str]]) -> list[str]:
    """Pick the row whose Stage is the median, so ramp/paused steps are dropped."""
    if "Stage" not in header:
        return rows[0]
    idx = header.index("Stage")
    scored = []
    for row in rows:
        try:
            scored.append((float(row[idx]), row))
        except (TypeError, ValueError):
            continue
    if not scored:
        return rows[0]
    scored.sort(key=lambda item: item[0])
    return scored[len(scored) // 2][1]


def _step_values(header: list[str], row: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in STEP_COLUMNS:
        if name not in header:
            continue
        try:
            out[name] = float(row[header.index(name)]) / 1000.0  # us -> ms
        except (TypeError, ValueError):
            out[name] = 0.0
    return out


def _select_c(
    step_comm_ms: float,
    families: dict[tuple[str, str], list[float]],
    mask_family: str,
    mask_gid: str | None = None,
) -> float:
    """Pick the communication total that the attention kernel is supposed to hide.

    Restricting C to the masked family is the faithful choice: the step-level
    ``Communication`` column also counts communicators that the attention kernel
    cannot hide (per-layer weight gathers of an offload, encoder/VAE traffic),
    and including them inflates C and therefore the computed bound.  When two
    communicators share a family name (a layerwise offload's all-gather sitting
    next to a sequence-parallel one), ``mask_gid`` picks the right one -- read
    the ids off the family table printed above.
    """
    if mask_family == "step":
        return step_comm_ms
    family_ms = sum(
        ms
        for (family, gid), (ms, _n) in families.items()
        if family == mask_family and (mask_gid is None or gid == mask_gid)
    )
    return family_ms if family_ms > 0 else step_comm_ms


def _kernel_totals(
    header: list[str], rows: list[list[str]], fa_name: str
) -> tuple[float, int, dict[tuple[str, str], list[float]]]:
    """Sum the attention kernels and bucket the collectives by (family, communicator id)."""
    name_i, dur_i = header.index("Name"), header.index("Duration(us)")
    fa_ms, fa_n = 0.0, 0
    families: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        name = row[name_i]
        try:
            dur = float(row[dur_i])
        except (TypeError, ValueError):
            continue
        if fa_name and fa_name in name:
            fa_ms += dur / 1000.0
            fa_n += 1
            continue
        match = _COMM_RE.match(name)
        if not match:
            continue
        key = (match.group(1), match.group(2))
        bucket = families.setdefault(key, [0.0, 0.0])
        bucket[0] += dur / 1000.0
        bucket[1] += 1.0
    return fa_ms, fa_n, families


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile-dir", required=True, help="dir containing the profiler capture")
    ap.add_argument("--layers", type=float, default=50.0, help="DiT layers per step")
    ap.add_argument("--chunks", type=int, default=7, help="head chunks used by the masking")
    ap.add_argument(
        "--fa-name",
        default="EagleQuantBlockSparseAttention",
        help="substring identifying the attention kernels that do the hiding",
    )
    ap.add_argument(
        "--drain-ratio",
        type=float,
        default=0.33,
        help="exposed reverse-comm of the last chunk, as a fraction of c",
    )
    ap.add_argument(
        "--mask-family",
        default="alltoall",
        help="collective family the masking hides; use 'step' to take C from the "
        "step_trace Communication column instead (counts every communicator, "
        "including ones the attention kernel cannot hide)",
    )
    ap.add_argument(
        "--mask-gid",
        default=None,
        help="communicator id of the masked family, needed when several communicators share "
        "the family name (e.g. an offload's all-gather next to a sequence-parallel one); "
        "read the ids off the family table printed above",
    )
    args = ap.parse_args()

    root = Path(args.profile_dir)
    step_csv = _find_one(root, "step_trace_time.csv")
    kern_csv = _find_one(root, "kernel_details.csv")
    if step_csv is None or kern_csv is None:
        print(
            f"missing capture under {root}: step_trace_time.csv={bool(step_csv)} kernel_details.csv={bool(kern_csv)}",
            file=sys.stderr,
        )
        return 2

    head, rows = _read_rows(step_csv)
    if not rows:
        print(f"no usable rows in {step_csv}", file=sys.stderr)
        return 2
    step = _step_values(head, _median_stage_row(head, rows))

    khead, krows = _read_rows(kern_csv)
    fa_ms, fa_n, families = _kernel_totals(khead, krows, args.fa_name)

    layers = max(args.layers, 1e-9)
    n = max(args.chunks, 2)

    comm_total = step.get("Communication", 0.0)
    exposed = step.get("Communication(Not Overlapped)", 0.0)
    overlapped = step.get("Overlapped", 0.0)
    stage = step.get("Stage", 0.0)
    computing = step.get("Computing", 0.0)
    free = step.get("Free", 0.0)

    c_total = _select_c(comm_total, families, args.mask_family, args.mask_gid) / layers
    f_total = fa_ms / layers
    c_chunk = c_total / n
    f_chunk = f_total / n
    ratio = c_chunk / f_chunk if f_chunk > 0 else float("inf")
    drain = args.drain_ratio * c_chunk
    steady = max(0.0, c_chunk - f_chunk) * (n - 1)
    exposed_bound = c_chunk + steady + drain
    real_bound = 1.0 - exposed_bound / c_total if c_total > 0 else 0.0
    ideal_bound = 1.0 - 1.0 / n
    measured = overlapped / comm_total if comm_total > 0 else 0.0
    achieved = measured / real_bound if real_bound > 0 else float("nan")

    print(f"profile      : {root}")
    print(
        f"step row     : rows={len(rows)}  Stage {stage:.1f} ms  Computing {computing:.1f} ms  "
        f"Comm(notOvl) {exposed:.1f} ms  Overlapped {overlapped:.1f} ms  "
        f"Comm {comm_total:.1f} ms  Free {free:.1f} ms"
    )
    print(f"attention    : {fa_ms:.1f} ms over {fa_n} kernels ({args.fa_name})")
    print("comm families:")
    for (family, gid), (ms, count) in sorted(families.items(), key=lambda kv: -kv[1][0]):
        print(f"  {family:<14} gid={gid:<6} n={int(count):<6} total={ms:9.1f} ms mean={ms / max(count, 1.0):7.2f} ms")
    print()
    step_note = "" if args.mask_family == "step" else f"; step column would be {comm_total:.1f}"
    gid_note = f" gid={args.mask_gid}" if args.mask_gid else ""
    print(f"C source     : family={args.mask_family}{gid_note}  ({c_total * layers:.1f} ms/step{step_note})")
    print(f"per layer    : C={c_total:.2f} ms  F={f_total:.2f} ms")
    print(f"per chunk    : c={c_chunk:.2f} ms  f={f_chunk:.2f} ms  ->  c/f={ratio:.2f}   (chunks={n})")
    print(
        f"ideal bound  : {100 * ideal_bound:.1f}% hiding (1-1/n)  <- reachable: "
        f"{'yes' if f_chunk >= c_chunk else 'NO (f < c)'}"
    )
    print(
        f"real bound   : {100 * real_bound:.1f}% hiding   (fill {c_chunk:.2f} + steady "
        f"{steady:.2f} + drain {drain:.2f} = {exposed_bound:.2f} ms exposed/layer)"
    )
    print(
        f"measured     : {100 * measured:.1f}% hidden (step-level Overlapped / Communication; "
        f"counts any other communicator that happens to overlap too)"
    )
    if real_bound > 0:
        print(f"achievement  : {100 * achieved:.0f}% of the real bound")
    if achieved > 1.05:
        print(
            "warning      : measured hiding exceeds the bound -> either the bound's inputs are "
            "off (C should be the masked family only; e.g. an offload's weight gather is not "
            "hideable) or the capture counts AI-core kernels issued on the side stream as "
            "communication (check the per-stream table of collective_attribution.py)"
        )
    if f_chunk < c_chunk:
        need = c_total - f_total
        print(
            f"action       : c/f>1 -> shrink per-layer communication by >= "
            f"{need:.2f} ms (payload quantization / fewer cross-island bytes / merged "
            f"collectives); extra chunks will not help"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
