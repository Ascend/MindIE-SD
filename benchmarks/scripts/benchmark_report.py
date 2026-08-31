#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""MindIE-SD benchmark report tool.

Single entry for baseline export (JSON), a timestamped report snapshot
(JSON with embedded per-series CSV content), on-demand HTML rendering, and
drift compare.

Report artifacts live under the report dir and are named
`benchmark-report_<YYYYmmdd-HHMMSS>.json` / `.html`. The baselines/*.json files
remain the single source of truth; the timestamped JSON is a run snapshot that
also carries the per-series CSV content. `cmd_baseline` additionally writes
per-op CSV data tables (`reports/<op>.csv`) that double as an editable data
source: peaks edited there are re-applied on the next `report` run.

Usage:
    # Export baselines from the latest benchmark reports; write a timestamped
    # report snapshot and render its HTML by default.
    python benchmark_report.py baseline --report_dir ../reports_seqlen_v2 \
        --baseline_dir ../baselines

    # Re-render HTML from an existing snapshot (default: latest under reports/).
    python benchmark_report.py render
    python benchmark_report.py render --report-json \
        ../reports/benchmark-report_20260824.json --html out.html

    # Compare latest reports against baseline; drift gate (exit 1 on violations).
    python benchmark_report.py compare --report_dir ../reports_seqlen_v2 \
        --baseline_dir ../baselines --threshold 0.03

Default invocation (no subcommand) == `baseline`.
"""

import argparse
import html
import json
import math
import pathlib
import sys
from datetime import datetime

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from common.metrics import util_metrics  # noqa: E402
from common.schema import (  # noqa: E402
    BASELINE_METRICS,
    COMPARE_METRICS,
    OP_SEQ_AXIS,
    OP_SERIES_KEY,
    OP_SLOT_ARGS,
    SLOT_OMIT_WHEN_DEFAULT,
)

DEFAULT_REPORT_DIR = str(BASE_DIR.joinpath("reports"))
DEFAULT_BASELINE_DIR = str(BASE_DIR.joinpath("baselines"))


# --- baseline helpers -------------------------------------------------------
def _slot_for(op_name, arguments):
    """Slot key for (op, arguments); defaults are omitted to keep keys stable.

    kv_len is dropped when unset or equal to q_len (the op-level default), so a
    workload that sets kv_len explicitly and one that omits it produce the same
    baseline slot.
    """
    keys = OP_SLOT_ARGS[op_name]
    omit = SLOT_OMIT_WHEN_DEFAULT.get(op_name, [])
    parts = []
    for key in keys:
        value = arguments.get(key)
        if value is None:
            continue
        if any(
            key == omit_key and value == arguments.get(default_key)
            for omit_key, default_key in omit
        ):
            continue
        parts.append(f"{key}={value}")
    return "|".join(parts)


def load_report_entries(report_dir):
    """Load all jsonl entries into a list of dicts.

    Files are read oldest-first (mtime) so a slot written by a newer run
    overwrites an older one; with the per-run report_dir convention this picks
    the latest run deterministically when a dir holds stale leftovers.
    """
    entries = []
    jsonl_files = sorted(
        pathlib.Path(report_dir).rglob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    for jsonl in jsonl_files:
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                # torn line from a run killed mid-write: skip, keep the rest
                continue
    return entries


def recompute_util(entry):
    """Return (MFU, MBU) recomputed from raw measured values.

    CUBE peaks come from the entry's own args (--config {"peak_flops": ...});
    MFU/MBU are None when the user did not provide them.
    """
    targets = entry["targets"]
    args = entry.get("arguments", {})
    return util_metrics(
        targets.get("calc_flops_power(tflops)"),
        targets.get("mem_bw(GB/s)"),
        args.get("peak_flops"),
        args.get("peak_bw"),
    )


def collect_baseline(entries):
    """Group by (op, slot) -> baseline metrics, MFU/MBU recomputed per entry.

    Entries without valid measurements (errored/crashed cases) are dropped so
    a failed case never shows up as a fake data row.
    """
    grouped = {}
    for e in entries:
        op = e.get("op_name")
        if op not in OP_SLOT_ARGS:
            continue
        targets = e.get("targets", {})
        if not targets or not any(
            k in targets for k in ("latency(us)", "calc_flops_power(tflops)", "MFU", "MBU")
        ):
            # errored/skipped case: xpu-perf summary is empty -> no data
            continue
        args = e.get("arguments", {})
        slot = _slot_for(op, args)
        mfu, mbu = recompute_util(e)
        slot_metrics = {}
        if mfu is not None:
            slot_metrics["MFU"] = mfu
        if mbu is not None:
            slot_metrics["MBU"] = mbu
        for metric in BASELINE_METRICS:
            if metric in targets and metric not in slot_metrics:
                slot_metrics[metric] = targets[metric]
        grouped.setdefault(op, {})[slot] = slot_metrics
    return grouped


def export_baseline_json(grouped, baseline_dir):
    baseline_dir = pathlib.Path(baseline_dir)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for op, cases in grouped.items():
        out_file = baseline_dir.joinpath(f"{op}.json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(cases, fh, indent=2, sort_keys=True)
    return baseline_dir


# --- compare ----------------------------------------------------------------
def relative_drift(current, baseline):
    if current is None or baseline is None:
        return None
    if baseline == 0:
        return 0.0 if current == baseline else float("inf")
    return abs(current - baseline) / abs(baseline)


def compare(current_grouped, baseline_dir, threshold):
    violations = []
    checked = 0
    baseline_dir = pathlib.Path(baseline_dir)
    for op, cases in current_grouped.items():
        baseline_file = baseline_dir.joinpath(f"{op}.json")
        if not baseline_file.exists():
            continue
        with open(baseline_file, encoding="utf-8") as fh:
            baselines = json.load(fh)
        for slot, targets in cases.items():
            if slot not in baselines:
                violations.append((op, slot, "missing_in_baseline", None))
                continue
            for metric in COMPARE_METRICS:
                cur = targets.get(metric)
                base = baselines[slot].get(metric)
                drift = relative_drift(cur, base)
                if drift is None:
                    continue
                checked += 1
                if drift > threshold:
                    violations.append(
                        (op, slot, metric, f"drift={drift:.4f} cur={cur} base={base}")
                    )
    return checked, violations


def cmd_compare(args):
    entries = load_report_entries(args.report_dir)
    current = collect_baseline(entries)
    checked, violations = compare(current, args.baseline_dir, args.threshold)
    print(f"Checked {checked} metric(s); violations: {len(violations)}")
    for op, slot, metric, detail in violations:
        print(f"  VIOLATION {op}[{slot}] {metric}: {detail}")
    if violations:
        sys.exit(1)


# --- report snapshot --------------------------------------------------------
def parse_slot(slot):
    """Parse slot 'k=v|k2=v2' into a dict."""
    out = {}
    for part in slot.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out


def _series_label(op_name, parsed, series_key):
    """Series label: dtype/quant + heads/dim + kernel function (when present)."""
    label = parsed.get(series_key, "default")
    if op_name in ("fa", "bsa"):
        parts = [label]
        if "num_heads" in parsed:
            parts.append(f"h{parsed['num_heads']}")
        if "head_dim" in parsed:
            parts.append(f"d{parsed['head_dim']}")
        if "func" in parsed:
            parts.append(f"fn={parsed['func']}")
        label = " ".join(parts)
    return label


def _aggregate_cases(op_name, cases):
    """Aggregate slots into series: label -> x -> metrics (BSA: label -> sparsity -> x).

    fa/bsa labels carry head count / dim / kernel function so reports can
    distinguish configs and show which Python function was tested.
    """
    series_key = OP_SERIES_KEY[op_name]
    seq_key = OP_SEQ_AXIS[op_name]
    agg = {}
    for slot, metrics in cases.items():
        parsed = parse_slot(slot)
        label = _series_label(op_name, parsed, series_key)
        x = parsed.get(seq_key)
        if x is None:
            continue
        if op_name == "bsa":
            sparsity = parsed.get("sparsity", "0.8")
            agg.setdefault(label, {}).setdefault(sparsity, {})[int(x)] = metrics
        else:
            agg.setdefault(label, {})[int(x)] = metrics
    return agg


def build_csv_section(grouped, peaks=None):
    """Per-series CSV content embedded in the report snapshot (no CSV files).

    Returns {op: {series_label: [row, ...]}}; BSA rows carry a sparsity column,
    other ops a seq_len column. Rows include MFU / MBU / latency_us plus the
    per-op peak values (peak_flops / peak_bw) so the CSV doubles as a data
    source: editing a peak there and re-running `report` updates the report.
    """
    peaks = peaks or {}
    csv_data = {}
    for op_name, cases in grouped.items():
        pk = peaks.get(op_name, {})
        agg = _aggregate_cases(op_name, cases)
        series = {}
        if op_name == "bsa":
            for label in sorted(agg):
                rows = []
                for sparsity in sorted(agg[label], key=float):
                    for x in sorted(agg[label][sparsity]):
                        m = agg[label][sparsity][x]
                        rows.append(
                            {
                                "sparsity": sparsity,
                                "seq_len": x,
                                "MFU": m.get("MFU"),
                                "MBU": m.get("MBU"),
                                "latency_us": m.get("latency(us)"),
                                "peak_flops": pk.get("peak_flops"),
                                "peak_bw": pk.get("peak_bw"),
                            }
                        )
                series[label] = rows
        else:
            for label in sorted(agg):
                rows = []
                for x in sorted(agg[label]):
                    m = agg[label][x]
                    rows.append(
                        {
                            "seq_len": x,
                            "MFU": m.get("MFU"),
                            "MBU": m.get("MBU"),
                            "latency_us": m.get("latency(us)"),
                            "peak_flops": pk.get("peak_flops"),
                            "peak_bw": pk.get("peak_bw"),
                        }
                    )
                series[label] = rows
        csv_data[op_name] = series
    return csv_data


def _load_run_command(report_dir):
    """Latest run_command.txt under the report dir (written by `mindie_bench run`)."""
    candidates = list(pathlib.Path(report_dir).rglob("run_command.txt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.read_text(encoding="utf-8").strip()


def build_report_data(
    grouped, peak_flops, peak_bw, info, command=None, generated_at=None, peaks=None
):
    info = info or {}
    return {
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": info.get("backend", {}),
        "runtime": info.get("runtime", {}),
        "env": {"peak_flops": peak_flops, "peak_bw": peak_bw},
        "command": command,
        "ops": grouped,
        "csv": build_csv_section(grouped, peaks=peaks),
    }


def write_report_json(report_dir, report, stamp=None):
    """Write report_dir/benchmark-report_<YYYYmmdd-HHMMSS>.json, return its path."""
    report_dir = pathlib.Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = report_dir.joinpath(f"benchmark-report_{stamp}.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return out_file


def find_info_json(report_dir):
    candidates = list(pathlib.Path(report_dir).rglob("info.json"))
    if not candidates:
        return {}
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def find_latest_report_json(report_dir):
    """Path of the newest benchmark-report_*.json (timestamp sorts lexically)."""
    candidates = list(pathlib.Path(report_dir).glob("benchmark-report_*.json"))
    return str(max(candidates, key=lambda p: p.name)) if candidates else None


def _op_peaks(entries):
    """First non-None peak per op (may differ per op after a CSV peak update)."""
    seen = set()
    peaks = {}
    for e in entries:
        op = e.get("op_name")
        if op in seen or op not in OP_SLOT_ARGS:
            continue
        args = e.get("arguments", {})
        flops = args.get("peak_flops")
        bw = args.get("peak_bw")
        if flops is not None or bw is not None:
            peaks[op] = {"peak_flops": flops, "peak_bw": bw}
            seen.add(op)
    return peaks


def read_peaks_from_csv(out_dir, op_names):
    """Read user-updated peak values from per-op CSV files (data source).

    Returns {op: {"peak_flops": x, "peak_bw": y}} from the first non-empty
    value found per column; ops without a peak column / valid value are
    omitted so the run's own peaks are kept.
    """
    import csv

    peaks = {}
    out_dir = pathlib.Path(out_dir)
    for op in op_names:
        path = out_dir / f"{op}.csv"
        if not path.exists():
            continue
        found = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for key in ("peak_flops", "peak_bw"):
                    if key in found:
                        continue
                    raw = (row.get(key) or "").strip()
                    if raw:
                        try:
                            found[key] = float(raw)
                        except ValueError:
                            continue
        if found:
            peaks[op] = found
    return peaks


def apply_csv_peak_updates(entries, csv_peaks):
    """Overwrite entry peak args with user-updated CSV values; return count."""
    updated = 0
    for e in entries:
        pk = csv_peaks.get(e.get("op_name"))
        if not pk:
            continue
        args = e.setdefault("arguments", {})
        for key in ("peak_flops", "peak_bw"):
            if pk.get(key) is not None:
                args[key] = pk[key]
                updated += 1
    return updated


def write_csv_files(grouped, out_dir, peaks=None):
    """Per-op CSV data tables (seq_len / latency / MFU / MBU / peaks...) under out_dir.

    The CSV files carry the same series data as the report snapshot and can be
    used to rebuild the HTML (data source).
    """
    import csv

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_data = build_csv_section(grouped, peaks=peaks)
    for op, series in csv_data.items():
        path = out_dir / f"{op}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for idx, (label, rows) in enumerate(series.items()):
                if idx == 0:
                    writer.writerow(["series"] + list(rows[0]))
                for row in rows:
                    writer.writerow([label] + [row.get(k) for k in rows[0]])
    return out_dir


def cmd_baseline(args):
    entries = load_report_entries(args.report_dir)
    info = find_info_json(args.report_dir)
    grouped = collect_baseline(entries)
    # CSV is a data source: peaks edited there override the run's own values,
    # and MFU/MBU are recomputed against the updated peaks.
    csv_peaks = read_peaks_from_csv(DEFAULT_REPORT_DIR, list(grouped))
    updated = apply_csv_peak_updates(entries, csv_peaks)
    if updated:
        grouped = collect_baseline(entries)
    export_baseline_json(grouped, args.baseline_dir)
    total = sum(len(c) for c in grouped.values())

    def first_peak(key):
        for e in entries:
            args_d = e.get("arguments", {})
            if args_d.get(key) is not None:
                return args_d.get(key)
        return None

    peak_flops = first_peak("peak_flops")
    peak_bw = first_peak("peak_bw")
    print(
        f"Exported baselines for {len(grouped)} ops, {total} cases -> {args.baseline_dir} "
        f"(peak_flops={peak_flops}, peak_bw={peak_bw})"
    )
    if updated:
        print(f"Applied peak update from CSV: {updated} value(s) {csv_peaks}")

    per_op_peaks = _op_peaks(entries)
    now = datetime.now()
    command = _load_run_command(args.report_dir)
    report = build_report_data(
        grouped,
        peak_flops,
        peak_bw,
        info,
        command=command,
        generated_at=now.astimezone().isoformat(timespec="seconds"),
        peaks=per_op_peaks,
    )
    out_dir = DEFAULT_REPORT_DIR
    report_path = write_report_json(out_dir, report, stamp=now.strftime("%Y%m%d-%H%M%S"))
    print(f"Wrote {report_path}")
    write_csv_files(grouped, out_dir, peaks=per_op_peaks)
    print(f"Wrote {out_dir}/<op>.csv")

    if not args.no_html:
        html_path = report_path.with_suffix(".html")
        render_html(report, html_path)
        print(f"Wrote {html_path}")


def cmd_render(args):
    report_file = args.report_json or find_latest_report_json(DEFAULT_REPORT_DIR)
    if report_file is None:
        sys.exit(f"No benchmark-report_*.json under {DEFAULT_REPORT_DIR}; run `baseline` first")
    report = json.loads(pathlib.Path(report_file).read_text(encoding="utf-8"))
    html_output = args.html or str(pathlib.Path(report_file).with_suffix(".html"))
    render_html(report, html_output)


# --- HTML render (from report snapshot data) --------------------------------
W = 920
H = 380
MARGIN = {"l": 70, "r": 20, "t": 24, "b": 46}
CHART_W = W - MARGIN["l"] - MARGIN["r"]
CHART_H = H - MARGIN["t"] - MARGIN["b"]
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _pct(v):
    """Utilization (MFU/MBU) rendered as a percentage in the HTML report."""
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def _human(v):
    v = float(v)
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return str(int(v))


def line_chart_svg(title, series, x_values, series_metric):
    """series: dict series_label -> dict {x: value}."""
    safe_title = html.escape(title)
    y_min, y_max = 0.0, 0.0
    for s in series.values():
        for v in s.values():
            if v is not None:
                y_max = max(y_max, v)
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.1

    def xs(v):
        if len(x_values) <= 1:
            return MARGIN["l"] + CHART_W / 2
        return MARGIN["l"] + CHART_W * (math.log2(v) - math.log2(x_values[0])) / (
            math.log2(x_values[-1]) - math.log2(x_values[0])
        )

    def ys(v):
        if v is None:
            return None
        return H - MARGIN["b"] - CHART_H * (v - y_min) / (y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">'
    ]
    parts.append(
        f'<text x="{MARGIN["l"]}" y="16" font-size="15" font-weight="600" fill="#222">'
        f"{safe_title}</text>"
    )
    for i in range(5):
        gy = y_min + (y_max - y_min) * i / 4
        yy = ys(gy)
        parts.append(
            f'<line x1="{MARGIN["l"]}" y1="{yy:.1f}" x2="{W - MARGIN["r"]}" y2="{yy:.1f}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{MARGIN["l"] - 6}" y="{yy + 4:.1f}" font-size="11" fill="#666" '
            f'text-anchor="end">{gy * 100:.0f}%</text>'
        )
    for v in x_values:
        xx = xs(v)
        parts.append(
            f'<line x1="{xx:.1f}" y1="{H - MARGIN["b"]}" x2="{xx:.1f}" '
            f'y2="{H - MARGIN["b"] + 4}" stroke="#bbb" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{xx:.1f}" y="{H - MARGIN["b"] + 18}" font-size="10" fill="#666" '
            f'text-anchor="middle">{_human(v)}</text>'
        )
    parts.append(
        f'<text x="{MARGIN["l"] + CHART_W / 2}" y="{H - 2}" font-size="12" fill="#333" '
        f'text-anchor="middle">seq len</text>'
    )

    for color_idx, (label, s) in enumerate(series.items()):
        color = PALETTE[color_idx % len(PALETTE)]
        pts = []
        for x in x_values:
            yv = ys(s.get(x))
            if yv is not None:
                pts.append(f"{xs(x):.1f},{yv:.1f}")
        if len(pts) >= 2:
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.4" '
                f'stroke-linejoin="round" points="{" ".join(pts)}"/>'
            )
            last = pts[-1].split(",")
            parts.append(
                f'<circle cx="{last[0]}" cy="{last[1]}" r="3.4" fill="{color}">'
                f'<title>{html.escape(label)} {html.escape(series_metric)}@{_human(x_values[-1])}'
                f'={_pct(s.get(x_values[-1]))}</title></circle>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _legend_html(series_labels, metrics=("MFU", "MBU")):
    """Color legend: metric swatches (squares) plus each plotted series (dots).

    Metrics use square swatches and series use round dots so a series colored
    like the first metric (e.g. gmm's first series vs MFU) is still readable.
    """
    parts = ['<div class="legend">']
    for idx, metric in enumerate(metrics):
        parts.append(f'<span class="sw" style="background:{PALETTE[idx]}"></span>{metric}')
    for idx, label in enumerate(series_labels):
        parts.append(
            f'<span class="dot" style="background:{PALETTE[idx % len(PALETTE)]}">'
            f"</span>{html.escape(label)}"
        )
    parts.append("</div>")
    return "".join(parts)


def build_op_html(op_name, cases):
    """Render one chart per displayed metric (all series combined) and the
    per-series performance tables below them.

    Displayed metrics come from schema.OP_DISPLAY_METRICS: fa/bsa/mm are
    compute-bound and show MFU only; gmm shows MFU+MBU.
    """
    from common.schema import OP_DISPLAY_METRICS

    axis = OP_SERIES_KEY[op_name]
    agg = _aggregate_cases(op_name, cases)
    display = OP_DISPLAY_METRICS[op_name]

    parts = [f'<h2>{html.escape(op_name.upper())}</h2>', f"<p>{len(cases)} baseline cases</p>"]

    if op_name == "bsa":
        all_x = sorted({x for label in agg.values() for sp in label.values() for x in sp})
        series = {metric: {} for metric in display}
        for label in sorted(agg):
            for sp in sorted(agg[label], key=float):
                key = f"{label} sp={sp}"
                for metric in display:
                    series[metric][key] = {x: agg[label][sp].get(x, {}).get(metric) for x in all_x}
    else:
        all_x = sorted({x for label in agg.values() for x in label})
        series = {}
        for metric in display:
            series[metric] = {
                label: {x: agg[label].get(x, {}).get(metric) for x in all_x}
                for label in sorted(agg)
            }

    if all_x:
        legend_labels = [label for metric in display for label in series[metric]]
        parts.append(_legend_html(legend_labels, metrics=display))
        for metric in display:
            title = f"{op_name.upper()} · {metric} ({axis})"
            parts.append(line_chart_svg(title, series[metric], all_x, metric))

    for label in sorted(agg):
        parts.append(f"<h3>Data — {html.escape(label)}</h3>")
        if op_name == "bsa":
            by_sparsity = agg[label]
            xs = sorted({x for m in by_sparsity.values() for x in m})
            if not xs:
                continue
            for metric in display:
                header = [f"{metric} / sparsity"] + [_human(x) for x in xs]
                rows = ""
                for sp in sorted(by_sparsity, key=float):
                    cells = [html.escape(sp)] + [
                        _pct(by_sparsity[sp].get(x, {}).get(metric)) for x in xs
                    ]
                    rows += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
                parts.append(
                    "<table><thead><tr>"
                    + "".join(f"<th>{h}</th>" for h in header)
                    + "</tr></thead><tbody>"
                    + rows
                    + "</tbody></table>"
                )
        else:
            metrics = agg[label]
            xs = sorted(metrics)
            if not xs:
                continue
            header = ["seq len", "latency(us)"] + list(display)
            rows = ""
            for x in xs:
                mm = metrics[x]
                cells = [f"<td>{_human(x)}</td>", f"<td>{_fmt(mm.get('latency(us)'))}</td>"]
                cells += [f"<td>{_pct(mm.get(m))}</td>" for m in display]
                rows += "<tr>" + "".join(cells) + "</tr>"
            parts.append(
                "<table><thead><tr>"
                + "".join(f"<th>{h}</th>" for h in header)
                + "</tr></thead><tbody>"
                + rows
                + "</tbody></table>"
            )
    return "\n".join(parts)


def render_html(data, html_output):
    out_path = pathlib.Path(html_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = data.get("backend", {})
    runtime = data.get("runtime", {})
    env = data.get("env", {})
    command = data.get("command")
    generated_at = html.escape(str(data.get("generated_at", "")))
    backend_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in backend.items()
    )
    env_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in env.items()
    )

    sections = []
    ops = data.get("ops", {})
    for op in sorted(ops):
        if op not in OP_SLOT_ARGS:
            continue
        sections.append(build_op_html(op, ops[op]))

    command_html = ""
    if command:
        command_html = (
            '<h2>Command</h2>'
            f'<pre style="background:#f4f6fa;border:1px solid #e2e8f0;border-radius:6px;'
            f'padding:12px;font-size:12px;overflow-x:auto;">{html.escape(command)}</pre>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MindIE-SD Benchmark Report</title>
<style>
  body {{ font-family: "Segoe UI", Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
         margin: 0; background: #f5f7fa; color: #222; }}
  .wrap {{ max-width: 1020px; margin: 24px auto; padding: 8px 24px 48px;
          background: #fff; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,.08); }}
  h1 {{ color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 8px; }}
  h2 {{ color: #0b3d91; margin-top: 36px; border-left: 4px solid #0b3d91; padding-left: 10px; }}
  h3 {{ color: #555; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0 26px; font-size: 13px; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: right; }}
  th {{ background: #eef3fb; color: #0b3d91; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .meta {{ color: #666; font-size: 13px; }}
  svg {{ display: block; margin: 8px 0 4px; background: #fafcff; border: 1px solid #e2e8f0;
        border-radius: 6px; }}
  .legend {{ font-size: 12px; color: #333; margin: 0 0 18px; }}
  .legend span {{ display: inline-block; margin-right: 18px; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%;
         margin-right: 4px; }}
  .sw {{ display: inline-block; width: 10px; height: 10px; margin-right: 4px; }}
</style>
</head>
<body><div class="wrap">
<h1>MindIE-SD Core Ops Benchmark Report</h1>
<p class="meta">Generated: {generated_at} · Device:
{html.escape(str(backend.get('device_name', 'n/a')))} ·
PyTorch {html.escape(str(backend.get('torch_version', 'n/a')))} · torch_npu
{html.escape(str(backend.get('torch_npu_version', 'n/a')))} ·
device_ids {html.escape(str(runtime.get('device_ids', 'n/a')))} ·
peak_flops {env.get('peak_flops', 'n/a')} TFLOPS · peak_bw {env.get('peak_bw', 'n/a')} GB/s</p>
<h2>Environment</h2>
<table><thead><tr><th>attr</th><th>value</th></tr></thead><tbody>{backend_html}</tbody></table>
<h3>Peak config (CUBE flops / bandwidth)</h3>
<table><thead><tr><th>key</th><th>value</th></tr></thead><tbody>{env_html}</tbody></table>
{command_html}
{''.join(sections)}
</div></body></html>"""

    out_path.write_text(doc, encoding="utf-8")


# --- CLI --------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="MindIE-SD benchmark report tool "
        "(baseline export + report snapshot + HTML + compare)."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_b = sub.add_parser(
        "baseline",
        help="Export baseline JSON + timestamped report snapshot; render HTML by default.",
    )
    p_b.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Latest xpu-perf report dir")
    p_b.add_argument(
        "--baseline_dir", default=DEFAULT_BASELINE_DIR, help="Baseline JSON output dir"
    )
    p_b.add_argument("--no-html", action="store_true", help="Skip HTML rendering")
    p_b.set_defaults(func=cmd_baseline)

    p_r = sub.add_parser("render", help="Render HTML from a report snapshot (default: latest).")
    p_r.add_argument(
        "--report-json",
        default=None,
        help="Snapshot JSON path (default: latest benchmark-report_*.json under reports/)",
    )
    p_r.add_argument(
        "--html", default=None, help="HTML output path (default: next to the snapshot)"
    )
    p_r.set_defaults(func=cmd_render)

    p_c = sub.add_parser("compare", help="Compare latest reports against baseline; drift gate.")
    p_c.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Latest xpu-perf report dir")
    p_c.add_argument("--baseline_dir", default=DEFAULT_BASELINE_DIR, help="Baseline JSON dir")
    p_c.add_argument("--threshold", type=float, default=0.03, help="Relative drift threshold")
    p_c.set_defaults(func=cmd_compare)

    return parser


def main():
    known = {"baseline", "render", "compare"}
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help") or sys.argv[1] not in known:
        sys.argv.insert(1, "baseline")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
