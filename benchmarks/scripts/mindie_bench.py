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

"""MindIE-SD benchmark CLI (vLLM-style).

Unified entry with subcommands and kebab-case options:

    # all ops, built-in single-slot defaults (smoke test)
    python mindie_bench.py run

    # per-op structure params + shared scan matrix
    python mindie_bench.py run \\
        --op {fa: {num_heads: 32}, mm: {K: 5120, N: 13824}} \\
        --config {seqlen: [1024, 2048], dtype: [bf16, fp8]}

    # pick one op with a specific dtype/quant tier (kernel auto-selected)
    python mindie_bench.py run --op {bsa: {}} \\
        --config {seqlen: [1024, 2048], sparse: [0.6, 0.8], timeout: 300, peak_flops: 377.78}

    # report / compare (report merges every run under the report dir into one
    # HTML; each op gets one chart per displayed metric — fa/bsa/mm: MFU only,
    # gmm: MFU+MBU — with per-series tables)
    python mindie_bench.py report --report-dir reports --baseline-dir baselines
    python mindie_bench.py compare --report-dir reports --baseline-dir baselines

Responsibility split (strict):
    --op     op selection + STRUCTURE params only (num_heads/head_dim/K/N/
             experts/top_k/... and the reserved "func" = kernel-source label
             shown in reports; the kernel is chosen by dtype/quant_algo)
    --config scan matrix + peaks: seqlen (-> per-op seq axis), dtype, sparse
             (-> sparsity), quant_algo (quantization tier, same class as
             dtype), seed (RNG reproducibility), timeout (per-case seconds),
             peak_flops / peak_bw (device peaks, MFU/MBU denominators). List
             values expand to a cartesian product, scalars are fixed.

--op accepts JSON or a lenient bare form without quotes:
    {"fa": {}}  ==  {fa: {}}
Value is either {} (op defaults) or an object of params.
Scan keys (seqlen/q_len/dtype/sparse/...) in --op and structure keys or func
in --config are rejected. Every run writes into its own timestamped
report-dir (reports/run_<ts>) unless --report-dir is given explicitly, so
multiple runs coexist under reports/ and `report` merges them.
"""

import argparse
import datetime
import itertools
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import types

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
PLUGIN_DIR = BASE_DIR / "xpu-perf-plugin"

# Single source for the per-op seq-scan axis (schema table, not a copy).
sys.path.insert(0, str(BASE_DIR))
from common.schema import OP_SEQ_AXIS as SEQ_AXIS  # noqa: E402

VALID_OPS = ("fa", "bsa", "gmm", "mm")

# --op must not carry scan keys; --config only carries these keys.
# Keys are compared after _normalize_key() lowercases them, so use lowercase.
OP_BLOCKED_KEYS = {
    "seqlen", "seq_len", "q_len", "num_tokens", "m", "dtype", "sparse",
    "sparsity", "quant_algo", "quant",
}
CONFIG_ALLOWED_KEYS = {
    "seqlen", "seq_len", "dtype", "sparse", "sparsity", "quant_algo", "seed",
    "timeout", "peak_flops", "peak_bw",
}

# built-in fixed-param defaults (aligned with the example 默认规格);
# used for params not given in --op/--config
OP_DEFAULTS = {
    "fa": {
        "arg_type": "default",
        "batch_size": 1,
        "num_heads": 32,
        "head_dim": 128,
        "causal": False,
        "dtype": "bf16",
        "block_size": 32,
        "scale_alg": 2,
    },
    "bsa": {
        "arg_type": "default",
        "batch_size": 1,
        "num_heads": 32,
        "head_dim": 128,
        "causal": False,
        "dtype": "bf16",
        "sparsity": 0.8,
        "mask_type": "rf_v3",
    },
    "gmm": {
        "arg_type": "default",
        "hidden_size": 1536,
        "moe_inter": 3200,
        "experts": 128,
        "top_k": 16,
        "quant_algo": "NO_QUANT",
    },
    "mm": {
        "arg_type": "default",
        "K": 5120,
        "N": 13824,
        "quant_algo": "NO_QUANT",
        "group_size": 32,
        "scale_alg": 2,
    },
}


# --- lenient dict parsing ---------------------------------------------------
_LENIENT_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
# lookahead separators: the delimiter is NOT consumed so adjacent bare values
# like "W8A8, W8A8_MXFP8" each keep their leading comma
_LENIENT_VAL_RE = re.compile(r"([\[:]\s*)([A-Za-z_][A-Za-z0-9_.]*)(?=\s*[,}\]])")
_LENIENT_SEQ_VAL_RE = re.compile(r"(,\s*)([A-Za-z_][A-Za-z0-9_.]*)(?=\s*[,}\]])")
# Python-style literals -> JSON literals (case-insensitive); None -> null.
_LITERAL_TO_JSON = {
    "true": "true",
    "false": "false",
    "null": "null",
    "none": "null",
}


def _quote_key(match):
    return match.group(1) + '"' + match.group(2) + '"' + match.group(3)


def _quote_val(match):
    token = match.group(2)
    if token.lower() in _LITERAL_TO_JSON:
        # Normalize case and Python spelling (True/False/None -> true/false/
        # null) so json.loads succeeds instead of raising a misleading
        # "expecting value" error.
        return match.group(1) + _LITERAL_TO_JSON[token.lower()]
    return match.group(1) + '"' + token + '"'


def parse_dict(value, name):
    """Parse JSON or a lenient bare-form dict (bare keys/values get quoted).

    {fa: {}} is the defaults form; list elements like
    [bf16, fp8] are quoted as well.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        pass
    cleaned = _LENIENT_KEY_RE.sub(_quote_key, value)
    cleaned = _LENIENT_VAL_RE.sub(_quote_val, cleaned)
    cleaned = _LENIENT_SEQ_VAL_RE.sub(_quote_val, cleaned)
    try:
        return json.loads(cleaned)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {name}: {exc}") from exc


# --- --op / --config parsing ------------------------------------------------
def _normalize_key(key):
    name = key.strip().lower()
    if name in ("seqlen", "seq_len"):
        return "seqlen"
    if name == "sparse":
        return "sparsity"
    if name == "quant":
        return "quant_algo"
    return name


def parse_op_spec(op_str):
    """--op -> {op: structure params}. Omitted -> all ops with defaults."""
    if not op_str or not op_str.strip():
        return {op: {} for op in VALID_OPS}
    data = parse_dict(op_str, "--op")
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError('--op must be an object like {"fa": {}}')
    spec = {}
    for op, value in data.items():
        op = op.strip().lower()
        if op not in VALID_OPS:
            raise argparse.ArgumentTypeError(
                f"unknown op '{op}'; choose from {', '.join(VALID_OPS)}"
            )
        if isinstance(value, str):
            if value != "default":
                raise argparse.ArgumentTypeError(
                    f"--op value for '{op}' must be {{}} or an object"
                )
            params = {}
        elif isinstance(value, dict):
            params = value
        else:
            raise argparse.ArgumentTypeError(
                f"--op value for '{op}' must be {{}} or an object"
            )
        for key, val in params.items():
            norm = _normalize_key(key)
            if norm in OP_BLOCKED_KEYS:
                raise argparse.ArgumentTypeError(
                    f"param '{key}' of op '{op}' is a scan key; "
                    "put seqlen/dtype/sparse in --config"
                )
            if isinstance(val, list):
                raise argparse.ArgumentTypeError(
                    f"param '{key}' of op '{op}' must be a scalar; scanning goes to --config"
                )
        spec[op] = params
    return spec


def parse_config(config_str):
    """--config -> scan matrix + peaks; only whitelisted keys allowed."""
    if config_str is None:
        return None
    data = parse_dict(config_str, "--config")
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("--config must be an object")
    out = {}
    for key, value in data.items():
        norm = _normalize_key(key)
        if norm not in CONFIG_ALLOWED_KEYS:
            raise argparse.ArgumentTypeError(
                f"key '{key}' is not allowed in --config "
                "(only seqlen/dtype/sparse/quant_algo/seed/timeout/"
                "peak_flops/peak_bw)"
            )
        if norm == "dtype":
            if isinstance(value, list):
                value = [v.lower() if isinstance(v, str) else v for v in value]
            elif isinstance(value, str):
                value = value.lower()
        out[norm] = value
    return out


# --- device auto-detection --------------------------------------------------
def _detect_free_torch(threshold=0.5):
    """Idle device ids via torch.npu mem_get_info (in-process, container-safe).

    Returns a list of device ids whose free HBM fraction exceeds ``threshold``,
    or None when torch.npu is unavailable. Devices that fail to query are
    treated as free (cannot tell -> do not block them).
    """
    try:
        import torch

        npu = getattr(torch, "npu", None)
        if npu is None or not npu.is_available():
            return None
        free = []
        for i in range(npu.device_count()):
            try:
                free_bytes, _ = npu.mem_get_info(i)
                total_bytes = npu.get_device_properties(i).total_memory
                if total_bytes > 0 and free_bytes / total_bytes >= threshold:
                    free.append(i)
            except Exception:  # noqa: BLE001 - query failure -> assume free
                free.append(i)
        return free
    except Exception:  # noqa: BLE001 - torch import/init failure
        return None


def _parse_npu_smi_usage(output, threshold):
    """Parse `npu-smi info -t usages` output: device id + HBM usage %.

    Handles both the plain "0 12.5" line form and the bordered table form
    (rows start with "|"; the HBM-Usage column is the second field, trailing
    columns such as AICore-Usage are ignored). Returns device ids whose HBM
    usage stays below the threshold; empty when unparseable.
    """
    free = []
    for line in output.splitlines():
        parts = line.replace("|", " ").split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                usage = float(parts[1].rstrip("%"))
            except ValueError:
                continue
            if usage < threshold:
                free.append(int(parts[0]))
    return free


def _detect_free_npu_smi(threshold=50.0):
    """Fallback device detection by parsing `npu-smi info -t usages`."""
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        return None
    try:
        proc = subprocess.run(
            [npu_smi, "info", "-t", "usages"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    free = _parse_npu_smi_usage(proc.stdout, threshold)
    return free or None


def detect_free_devices(threshold=50.0):
    """Detect idle NPU device ids (torch.npu first, npu-smi fallback).

    Returns a list of idle device ids, or None when nothing can be detected
    (caller then falls back to the --devices default).
    """
    free = _detect_free_torch()
    if free is not None:
        return free or None
    return _detect_free_npu_smi(threshold)


def _resolve_devices(ns):
    """--devices value: explicit list, or 'auto' -> detect idle devices."""
    if ns.devices not in (None, "", "auto"):
        return ns.devices
    free = detect_free_devices()
    if free is None:
        print("WARNING: cannot detect free devices; falling back to device 0")
        return "0"
    print(f"Detected {len(free)} free device(s): {free} -> parallel runs")
    return ",".join(str(dev) for dev in free)


# --- case generation --------------------------------------------------------
def expand_cases(template):
    """Cartesian product over list-valued params -> list of scalar case dicts."""
    list_keys = [k for k, v in template.items() if isinstance(v, list)]
    if not list_keys:
        return [dict(template)]
    cases = []
    for combo in itertools.product(*(template[k] for k in list_keys)):
        case = dict(template)
        for key, value in zip(list_keys, combo):
            case[key] = value
        cases.append(case)
    return cases


def build_inline_cases(op_spec, config):
    """Generate {op: [cases]} from --op + --config (no workload file)."""
    test_cases = {}
    for op, op_params in op_spec.items():
        template = _op_template(op, op_params, config)
        test_cases[op] = expand_cases(template)
    return test_cases


def _op_template(op, op_params, config):
    """Merged param template for one op (defaults < config < --op params)."""
    template = dict(OP_DEFAULTS[op])
    if config:
        cfg = dict(config)
        if "seqlen" in cfg:
            cfg[SEQ_AXIS[op]] = cfg.pop("seqlen")
        template.update(cfg)
    template.update(op_params)
    return template


# --- run: bridge into npu_launch -------------------------------------------
def _resolve_report_dir(ns):
    if ns.report_dir:
        return ns.report_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return str(BASE_DIR / "reports" / f"run_{timestamp}")


def build_launch_args(ns, report_dir, op_names, devices=None):
    """Namespace compatible with npu_launch.parse_args() output.

    ``devices`` is the resolved value from _resolve_devices(): an explicit id
    list, or None for "all devices" (auto-detection failed / not requested).
    """
    import npu_launch  # noqa: PLC0415  # lazy: needs xpu_perf on the NPU box

    return types.SimpleNamespace(
        backend="NPU",
        op_defs=npu_launch.OP_DEFS_DIR,
        vendor_ops=[npu_launch.VENDOR_NPU_DIR],
        env=None,
        numa="-1",
        device=devices,
        node_world_size=1,
        node_rank=0,
        master_addr="localhost",
        server_port=49371,
        host_port=49372,
        device_port=49373,
        task_dir=str(BASE_DIR / "example"),
        task=",".join(op_names),
        workload=None,
        report_dir=report_dir,
        script_dir=npu_launch.FILE_DIR,
        backend_name_list=["NPU"],
        backend_mod_list={"NPU": npu_launch._npu_backend_module()},
    )


def _join_nargs(value):
    """Re-join shell-split tokens of a bare-form dict (--op {fa: {}})."""
    if isinstance(value, list):
        return " ".join(value)
    return value


def _print_run_config(ns, op_spec, config, devices, report_dir, test_cases):
    """Print the full effective configuration (all params incl. defaults)."""
    print("=== benchmark configuration ===")
    print(f"command: {' '.join(sys.argv)}")
    print(f"ops: {', '.join(op_spec)}")
    for op, op_params in op_spec.items():
        template = _op_template(op, op_params, config)
        print(f"  {op}: {json.dumps(template, sort_keys=True)}")
        print(f"      cases: {len(test_cases.get(op, []))}")
    print(f"config (scan matrix): {json.dumps(config) if config else '(none, single-slot)'}")
    print(f"devices: {devices or '(all)'}  (default: 0)")
    print(f"report-dir: {report_dir}  (default: reports/run_<timestamp>)")
    if config:
        print(f"CUBE peaks: peak_flops={config.get('peak_flops')} TFLOPS, "
              f"peak_bw={config.get('peak_bw')} GB/s")
    else:
        print("CUBE peaks: not provided — add --config {peak_flops: <TFLOPS>, peak_bw: <GB/s>}")


def _save_run_command(ns, report_dir, op_spec, config, devices):
    """Record the executed command + config under the report dir so the HTML
    report can show exactly how the run was produced."""
    os.makedirs(report_dir, exist_ok=True)
    lines = [
        " ".join(sys.argv),
        "",
        f"ops: {json.dumps(op_spec)}",
        f"config: {json.dumps(config)}",
        f"devices: {devices}",
    ]
    path = os.path.join(report_dir, "run_command.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def cmd_run(ns):
    op_str = _join_nargs(ns.op)
    config_str = _join_nargs(ns.config)
    op_spec = parse_op_spec(op_str)
    config = parse_config(config_str)
    sys.path.insert(0, str(PLUGIN_DIR))
    import npu_launch  # noqa: PLC0415

    report_dir = _resolve_report_dir(ns)
    devices = _resolve_devices(ns)
    launch_args = build_launch_args(ns, report_dir, list(op_spec), devices=devices)
    test_cases = build_inline_cases(op_spec, config)

    total = sum(len(c) for c in test_cases.values())
    if total == 0:
        raise SystemExit("No test cases; nothing to benchmark")

    _print_run_config(ns, op_spec, config, devices, report_dir, test_cases)
    _save_run_command(ns, report_dir, op_spec, config, devices)

    counts = ", ".join(f"{op}={len(c)}" for op, c in test_cases.items())
    print(f"Benchmarking {total} case(s): {counts} -> {report_dir}")
    _run_bench_inline(npu_launch, launch_args, test_cases)


def _run_bench_inline(npu_launch, launch_args, test_cases):
    """Run npu_launch.run_bench with injected cases when the signature allows.

    dev's npu_launch.run_bench(args) parses workloads from --task_dir and does
    not accept test_cases; the runtime benchmark branch adds the injection.
    Detect the signature so the CLI fails with a clear hint instead of a
    TypeError when only this branch is merged.
    """
    import inspect

    params = inspect.signature(npu_launch.run_bench).parameters
    if "test_cases" in params:
        npu_launch.run_bench(launch_args, test_cases=test_cases)
    else:
        raise SystemExit(
            "npu_launch.run_bench predates inline test_cases injection; "
            "merge the runtime benchmark branch (dev-benchmark-runtime) first"
        )


# --- report / compare: bridge into benchmark_report -------------------------
def cmd_report(ns):
    sys.path.insert(0, str(SCRIPTS_DIR))
    import benchmark_report  # noqa: PLC0415

    if ns.render:
        args = types.SimpleNamespace(report_json=ns.report_json, html=ns.html)
        benchmark_report.cmd_render(args)
        return
    # env=None: peaks come from --config per case (new benchmark_report); a
    # None env also keeps dev's env-file-based cmd_baseline from crashing.
    args = types.SimpleNamespace(
        report_dir=ns.report_dir,
        baseline_dir=ns.baseline_dir,
        no_html=ns.no_html,
        env=None,
    )
    benchmark_report.cmd_baseline(args)


def cmd_compare(ns):
    sys.path.insert(0, str(SCRIPTS_DIR))
    import benchmark_report  # noqa: PLC0415

    args = types.SimpleNamespace(
        report_dir=ns.report_dir,
        baseline_dir=ns.baseline_dir,
        threshold=ns.threshold,
        env=None,
    )
    benchmark_report.cmd_compare(args)


# --- CLI --------------------------------------------------------------------
_DEFAULT_REPORT_DIR = str(BASE_DIR / "reports")
_DEFAULT_BASELINE_DIR = str(BASE_DIR / "baselines")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="mindie_bench",
        description="MindIE-SD core-op benchmark CLI (FA/BSA/GMM/MM), vLLM-style.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run core-op benchmarks")
    grp = p_run.add_argument_group("workload")
    grp.add_argument(
        "--op",
        default=None,
        nargs="+",
        metavar="JSON",
        help='op selection with structure params, e.g. {fa: {}, '
        'mm: {K: 5120, N: 13824}}; value {} (defaults) or an object; reserved key '
        '"func" = Python kernel function name; scan keys (seqlen/dtype/sparse/'
        "quant_algo) are rejected here (use --config); omitted = all ops; "
        "bare form may be written without outer quotes",
    )
    grp.add_argument(
        "--config",
        default=None,
        nargs="+",
        metavar="JSON",
        help="scan matrix + peaks: seqlen/dtype/sparse(sparsity)/quant_algo/"
        "seed/timeout/peak_flops/peak_bw; list values -> cartesian product, "
        "scalars fixed, e.g. {seqlen: [1024, 2048], dtype: [bf16, fp8], "
        "timeout: 300, peak_flops: 377.78}; bare form may be written without "
        "outer quotes",
    )
    grp2 = p_run.add_argument_group("runtime")
    grp2.add_argument("--devices", default="0", metavar="IDS|auto",
                      help="NPU logical device ids: comma list (default 0), or "
                      "'auto' to detect idle devices via npu-smi; multiple "
                      "devices run different cases in parallel")
    grp2.add_argument("--report-dir", default=None, metavar="DIR",
                      help="report output dir (default: reports/run_<timestamp>, "
                      "so repeated runs never overwrite each other)")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="Export baselines + report snapshot + HTML")
    p_rep.add_argument("--report-dir", default=_DEFAULT_REPORT_DIR, metavar="DIR",
                       help="dir holding one or more runs; all jsonl under it are merged "
                       "into a single HTML")
    p_rep.add_argument("--baseline-dir", default=_DEFAULT_BASELINE_DIR, metavar="DIR")
    p_rep.add_argument("--no-html", action="store_true", help="skip HTML rendering")
    p_rep.add_argument("--render", action="store_true",
                       help="re-render HTML from a snapshot (no recompute)")
    p_rep.add_argument("--report-json", default=None, metavar="FILE",
                       help="snapshot JSON for --render")
    p_rep.add_argument("--html", default=None, metavar="FILE",
                       help="HTML output path for --render")
    p_rep.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="Compare latest reports against baseline (drift gate)")
    p_cmp.add_argument("--report-dir", default=_DEFAULT_REPORT_DIR, metavar="DIR")
    p_cmp.add_argument("--baseline-dir", default=_DEFAULT_BASELINE_DIR, metavar="DIR")
    p_cmp.add_argument("--threshold", type=float, default=0.03, metavar="RATIO")
    p_cmp.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
