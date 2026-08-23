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
also carries the per-series CSV content (no CSV files are written).

Usage:
    # Export baselines from the latest benchmark reports; write a timestamped
    # report snapshot and render its HTML by default.
    python benchmark_report.py baseline --report_dir ../reports_seqlen_v2 \
        --baseline_dir ../baselines --env ../xpu-perf-plugin/vendor_ops/NPU/env.json

    # Re-render HTML from an existing snapshot (default: latest snapshot).
    python benchmark_report.py render --report_dir ../reports_seqlen_v2

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
from common.env_util import load_peaks  # noqa: E402
from common.metrics import util_metrics  # noqa: E402
from common.schema import (  # noqa: E402
    BASELINE_METRICS,
    COMPARE_METRICS,
    OP_SERIES_KEY,
    OP_SLOT_ARGS,
    OP_SEQ_AXIS,
    SLOT_OMIT_WHEN_DEFAULT,
)

DEFAULT_REPORT_DIR = str(BASE_DIR.joinpath("reports"))
DEFAULT_BASELINE_DIR = str(BASE_DIR.joinpath("baselines"))
DEFAULT_ENV = str(BASE_DIR.joinpath("xpu-perf-plugin", "vendor_ops", "NPU", "env.json"))


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
        if any(key == omit_key and value == arguments.get(default_key) for omit_key, default_key in omit):
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
            if line:
                entries.append(json.loads(line))
    return entries


def recompute_util(entry, peak_flops, peak_bw):
    """Return (MFU, MBU) recomputed from raw measured values with env peaks."""
    targets = entry["targets"]
    return util_metrics(
        targets.get("calc_flops_power(tflops)"),
        targets.get("mem_bw(GB/s)"),
        peak_flops,
        peak_bw,
    )


def collect_baseline(entries, peak_flops, peak_bw):
    """Group by (op, slot) -> baseline metrics, MFU/MBU recomputed with peaks."""
    grouped = {}
    for e in entries:
        op = e.get("op_name")
        if op not in OP_SLOT_ARGS:
            continue
        args = e.get("arguments", {})
        targets = e.get("targets", {})
        slot = _slot_for(op, args)
        mfu, mbu = recompute_util(e, peak_flops, peak_bw)
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
                    violations.append((op, slot, metric, f"drift={drift:.4f} cur={cur} base={base}"))
    return checked, violations


def cmd_compare(args):
    entries = load_report_entries(args.report_dir)
    peak_flops, peak_bw = load_peaks(args.env)
    current = collect_baseline(entries, peak_flops, peak_bw)
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


def _aggregate_cases(op_name, cases):
    """Aggregate slots into series: label -> x -> metrics (BSA: label -> sparsity -> x)."""
    series_key = OP_SERIES_KEY[op_name]
    seq_key = OP_SEQ_AXIS[op_name]
    agg = {}
    for slot, metrics in cases.items():
        parsed = parse_slot(slot)
        label = parsed.get(series_key, "default")
        x = parsed.get(seq_key)
        if x is None:
            continue
        if op_name == "bsa":
            sparsity = parsed.get("sparsity", "0.8")
            agg.setdefault(label, {}).setdefault(sparsity, {})[int(x)] = metrics
        else:
            agg.setdefault(label, {})[int(x)] = metrics
    return agg


def build_csv_section(grouped):
    """Per-series CSV content embedded in the report snapshot (no CSV files).

    Returns {op: {series_label: [row, ...]}}; BSA rows carry a sparsity column,
    other ops a seq_len column. Rows include MFU / MBU / latency_us.
    """
    csv_data = {}
    for op_name, cases in grouped.items():
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
                        }
                    )
                series[label] = rows
        csv_data[op_name] = series
    return csv_data


def build_report_data(grouped, peak_flops, peak_bw, info, generated_at=None):
    info = info or {}
    return {
        "generated_at": generated_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": info.get("backend", {}),
        "runtime": info.get("runtime", {}),
        "env": {"peak_flops": peak_flops, "peak_bw": peak_bw},
        "ops": grouped,
        "csv": build_csv_section(grouped),
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


def cmd_baseline(args):
    entries = load_report_entries(args.report_dir)
    peak_flops, peak_bw = load_peaks(args.env)
    grouped = collect_baseline(entries, peak_flops, peak_bw)
    export_baseline_json(grouped, args.baseline_dir)
    total = sum(len(c) for c in grouped.values())
    print(
        f"Exported baselines for {len(grouped)} ops, {total} cases -> {args.baseline_dir} "
        f"(peak_flops={peak_flops}, peak_bw={peak_bw})"
    )

    now = datetime.now()
    info = find_info_json(args.report_dir)
    report = build_report_data(
        grouped,
        peak_flops,
        peak_bw,
        info,
        generated_at=now.astimezone().isoformat(timespec="seconds"),
    )
    report_path = write_report_json(args.report_dir, report, stamp=now.strftime("%Y%m%d-%H%M%S"))
    print(f"Wrote {report_path}")

    if not args.no_html:
        html_path = report_path.with_suffix(".html")
        render_html(report, html_path)
        print(f"Wrote {html_path}")


def cmd_render(args):
    report_file = args.report_json or find_latest_report_json(args.report_dir)
    if report_file is None:
        sys.exit(f"No benchmark-report_*.json under {args.report_dir}; run `baseline` first")
    report = json.loads(pathlib.Path(report_file).read_text(encoding="utf-8"))
    html_output = args.html_output or str(pathlib.Path(report_file).with_suffix(".html"))
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
    parts.append(f'<text x="{MARGIN["l"]}" y="16" font-size="15" font-weight="600" fill="#222">{safe_title}</text>')
    for i in range(5):
        gy = y_min + (y_max - y_min) * i / 4
        yy = ys(gy)
        parts.append(
            f'<line x1="{MARGIN["l"]}" y1="{yy:.1f}" x2="{W - MARGIN["r"]}" y2="{yy:.1f}" '
            f'stroke="#e8e8e8" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{MARGIN["l"] - 6}" y="{yy + 4:.1f}" font-size="11" fill="#666" text-anchor="end">{gy:.2f}</text>'
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

    color_idx = 0
    for label, s in series.items():
        color = PALETTE[color_idx % len(PALETTE)]
        color_idx += 1
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
                f'={_fmt(s.get(x_values[-1]))}</title></circle>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _legend_html(series_labels):
    """Color legend for the two metric charts plus each plotted series line."""
    parts = ['<div class="legend">']
    parts.append(f'<span class="dot" style="background:{PALETTE[0]}"></span>MFU')
    parts.append(f'<span class="dot" style="background:{PALETTE[1]}"></span>MBU')
    for idx, label in enumerate(series_labels):
        parts.append(f'<span class="dot" style="background:{PALETTE[idx % len(PALETTE)]}"></span>{html.escape(label)}')
    parts.append("</div>")
    return "".join(parts)


def build_op_html(op_name, cases):
    """Render charts + tables for one op from report snapshot data."""
    axis = OP_SERIES_KEY[op_name]
    agg = _aggregate_cases(op_name, cases)

    parts = [f'<h2>{html.escape(op_name.upper())}</h2>', f"<p>{len(cases)} baseline cases</p>"]
    for label in sorted(agg):
        if op_name == "bsa":
            by_sparsity = agg[label]
            xs = sorted({x for m in by_sparsity.values() for x in m})
            if not xs:
                continue
            series_mfu = {
                f"sp={sp}": {x: by_sparsity[sp].get(x, {}).get("MFU") for x in xs}
                for sp in sorted(by_sparsity, key=float)
            }
            series_mbu = {
                f"sp={sp}": {x: by_sparsity[sp].get(x, {}).get("MBU") for x in xs}
                for sp in sorted(by_sparsity, key=float)
            }
        else:
            metrics = agg[label]
            xs = sorted(metrics)
            if not xs:
                continue
            series_mfu = {label: {x: metrics[x].get("MFU") for x in xs}}
            series_mbu = {label: {x: metrics[x].get("MBU") for x in xs}}

        title = f"{op_name.upper()} — {axis}={label} (seq len sweep)"
        parts.append(_legend_html(list(series_mfu)))
        parts.append(line_chart_svg(f"{title} · MFU", series_mfu, xs, "MFU"))
        parts.append(line_chart_svg(f"{title} · MBU", series_mbu, xs, "MBU"))

        parts.append(f"<h3>Data — {html.escape(axis)}={html.escape(label)}</h3>")
        if op_name == "bsa":
            header = ["sparsity"] + [_human(x) for x in xs]
            rows = ""
            for sp in sorted(by_sparsity, key=float):
                cells = [html.escape(sp)] + [_fmt(by_sparsity[sp].get(x, {}).get("MFU")) for x in xs]
                rows += "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"
            parts.append(
                "<table><thead><tr>"
                + "".join(f"<th>{h}</th>" for h in header)
                + "</tr></thead><tbody>"
                + rows
                + "</tbody></table>"
            )
        else:
            rows = ""
            for x in xs:
                mm = metrics[x]
                rows += (
                    f"<tr><td>{_human(x)}</td><td>{_fmt(mm.get('latency(us)'))}</td>"
                    f"<td>{_fmt(mm.get('MFU'))}</td><td>{_fmt(mm.get('MBU'))}</td></tr>"
                )
            parts.append(
                "<table><thead><tr><th>seq len</th><th>latency(us)</th><th>MFU</th>"
                "<th>MBU</th></tr></thead><tbody>" + rows + "</tbody></table>"
            )
    return "\n".join(parts)


def render_html(data, html_output):
    out_path = pathlib.Path(html_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    backend = data.get("backend", {})
    runtime = data.get("runtime", {})
    env = data.get("env", {})
    generated_at = html.escape(str(data.get("generated_at", "")))
    backend_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in backend.items()
    )

    sections = []
    ops = data.get("ops", {})
    for op in sorted(ops):
        if op not in OP_SLOT_ARGS:
            continue
        sections.append(build_op_html(op, ops[op]))

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
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; }}
</style>
</head>
<body><div class="wrap">
<h1>MindIE-SD Core Ops Benchmark Report</h1>
<p class="meta">Generated: {generated_at} · Device: {html.escape(str(backend.get('device_name', 'n/a')))} ·
PyTorch {html.escape(str(backend.get('torch_version', 'n/a')))} · torch_npu {html.escape(str(backend.get('torch_npu_version', 'n/a')))} ·
device_ids {html.escape(str(runtime.get('device_ids', 'n/a')))} ·
peak_flops {env.get('peak_flops', 'n/a')} TFLOPS · peak_bw {env.get('peak_bw', 'n/a')} GB/s</p>
<h2>Environment</h2>
<table><thead><tr><th>attr</th><th>value</th></tr></thead><tbody>{backend_html}</tbody></table>
{''.join(sections)}
</div></body></html>"""

    out_path.write_text(doc, encoding="utf-8")


# --- CLI --------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="MindIE-SD benchmark report tool (baseline export + report snapshot + HTML + compare)."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_b = sub.add_parser("baseline", help="Export baseline JSON + timestamped report snapshot; render HTML by default.")
    p_b.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Latest xpu-perf report dir")
    p_b.add_argument("--baseline_dir", default=DEFAULT_BASELINE_DIR, help="Baseline JSON output dir")
    p_b.add_argument("--env", default=DEFAULT_ENV, help="env.json with peak_flops/peak_bw")
    p_b.add_argument("--no-html", action="store_true", help="Skip HTML rendering")
    p_b.set_defaults(func=cmd_baseline)

    p_r = sub.add_parser("render", help="Render HTML from a report snapshot (default: latest).")
    p_r.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Report dir holding benchmark-report_*.json")
    p_r.add_argument("--report-json", default=None, help="Explicit snapshot JSON path")
    p_r.add_argument("--html-output", default=None, help="HTML output path (default: next to the snapshot)")
    p_r.set_defaults(func=cmd_render)

    p_c = sub.add_parser("compare", help="Compare latest reports against baseline; drift gate.")
    p_c.add_argument("--report_dir", default=DEFAULT_REPORT_DIR, help="Latest xpu-perf report dir")
    p_c.add_argument("--baseline_dir", default=DEFAULT_BASELINE_DIR, help="Baseline JSON dir")
    p_c.add_argument("--env", default=DEFAULT_ENV, help="env.json with peak_flops/peak_bw")
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
