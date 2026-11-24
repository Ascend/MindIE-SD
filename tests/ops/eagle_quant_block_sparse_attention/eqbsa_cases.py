# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Shared, device-free scene selection for accuracy and performance."""

import argparse
import json
from pathlib import Path
import sys

from tests.ops.eagle_quant_block_sparse_attention.eagle_quant_block_sparse_attention_accuracy_common import (
    PRODUCTION,
    name,
    preview,
    scenario,
    validate,
)
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_state import read

KERNELS = {"cann_fp8_bsa": "BlockSparseAttention", "eqbsa": "EagleQuantBlockSparseAttention"}
CANN_BLOCK_Q, CANN_BLOCK_KV = 128, 256
DEFAULT_TEST_COUNT = 3  # Independent captures per scenario on Ascend 950.


def cann_supported(plan):
    return plan["batch_size"] == 1 and plan["layout"] == "BNSD" and plan["output_dtype"] == "bf16"


def impl_list(value):
    impls = [v.strip() for v in value.split(",")]
    if len(set(impls)) != len(impls) or any(impl not in KERNELS for impl in impls):
        raise argparse.ArgumentTypeError("expected a comma-separated list of distinct cann_fp8_bsa and/or eqbsa")
    return impls


def scene_name(s):
    label = name(s)
    return "cann_fp8_bsa_" + label if s.get("impl") == "cann_fp8_bsa" else label


def scene_preview(s):
    detail = preview(s)
    if s.get("impl") == "cann_fp8_bsa":
        for key, block in (
            ("query", CANN_BLOCK_Q),
            ("key", CANN_BLOCK_KV),
            ("value", CANN_BLOCK_KV),
        ):
            b, n, seq, d = detail[key]["shape"]
            detail[key]["dtype"] = "float8_e4m3fn"
            detail[key + "_scale"]["shape"] = [b, n, (seq + block - 1) // block, (d + 127) // 128]
        detail["block_sparse_mask"]["shape"][-1] = (s["kv_seq_len"] + CANN_BLOCK_KV - 1) // CANN_BLOCK_KV
        detail["block_shape"] = [CANN_BLOCK_Q, CANN_BLOCK_KV]
        detail["softmax_lse"] = None  # The CANN FP8 workload returns attention_out only.
    return detail


def selections(plan, perf=False):
    base = {k: plan[k] for k in PRODUCTION}
    if perf:
        eq_flags = [0, 1] if plan["softmax_lse_flag"] is None else [plan["softmax_lse_flag"]]
        return [
            scenario(**base, impl=impl, softmax_lse_flag=flag, distribution="normal")
            for impl in plan.get("impls", ["eqbsa"])  # Older saved plans only profiled EQBSA.
            for flag in (eq_flags if impl == "eqbsa" else [0])
        ]
    return [
        scenario(**base, softmax_lse_flag=flag, distribution=dist)
        for flag, dist in [(0, "uniform"), (1, "normal"), (0, "normal"), (1, "uniform")]
        if (plan.get("softmax_lse_flag") is None or flag == plan["softmax_lse_flag"])
        and (plan.get("distribution") is None or dist == plan["distribution"])
    ]


def warmup(s):
    length = max(s["query_seq_len"], s["kv_seq_len"])
    return 20 if length < 30000 else 2 if length < 50000 else 1


def parser(kind):
    defaults = dict(PRODUCTION, softmax_lse_flag=None, distribution="normal" if kind != "accuracy" else None)
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description=f"EQBSA {kind}: INT8 Q/K, FP8 E4M3 V, FP32 scales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for key, default in PRODUCTION.items():
        opts = (
            {"type": int}
            if isinstance(default, int)
            else {"choices": ("BNSD", "TND") if key == "layout" else ("bf16", "fp16")}
        )
        p.add_argument("--" + key.replace("_", "-"), help=f"default: {default}", **opts)
    p.add_argument(
        "--softmax-lse-flag",
        type=int,
        choices=(0, 1),
        help="EQBSA LSE mode; default: 0 for one forward, both 0 and 1 in a batch",
    )
    if kind == "accuracy":
        p.add_argument(
            "--distribution", choices=("normal", "uniform"), help="default: both distributions, four scenes total"
        )
        p.add_argument("--device-id", type=int, nargs="+", help="distinct physical cards; default: auto-select one")
        p.add_argument("--mode", choices=("all", "accuracy", "stress"), help="default: all accuracy, then all stress")
        p.add_argument("--stress-runs", type=int, help="default: 1000, baseline excluded")
        defaults.update(mode="all", stress_runs=1000)
    else:
        p.add_argument("--device-id", type=int, help="one physical card only")
        if kind == "perf":
            p.add_argument(
                "--impls",
                type=impl_list,
                help="ordered CSV; default: cann_fp8_bsa,eqbsa for B1/BNSD/BF16, otherwise eqbsa",
            )
            p.add_argument(
                "--test-count",
                type=int,
                help=f"default: {DEFAULT_TEST_COUNT} independent captures per scene on Ascend 950",
            )
            p.add_argument(
                "--warm-up", type=int, help="default by max(Q,KV): <30000:20, <50000:2, otherwise:1; permits 0"
            )
            defaults.update(test_count=DEFAULT_TEST_COUNT, warm_up=None, impls=list(KERNELS))
        else:
            p.add_argument("--impl", choices=tuple(KERNELS), help="single target workload; default: eqbsa")
            defaults.update(impls=["eqbsa"], softmax_lse_flag=0)
            p.add_argument("--msprof-mode", action="store_true", help="one target call, also the default behavior")
    if kind != "profile":
        p.add_argument("--resume", help="restore immutable batch and committed results; omitted values use old plan")
        p.add_argument("--output-dir", help="new, previously unused batch directory")
    p.add_argument("--dry-run", "--list-cases", action="store_true", help="preview without NPU access or result files")
    lines = ["Default complete scene tuples (all selected unless filtered):"]
    for s in selections(defaults, perf=kind != "accuracy"):
        detail = scene_preview(s)
        selector = " ".join(f"--{k.replace('_', '-')} {s[k]}" for k in PRODUCTION)
        selector += f" --softmax-lse-flag {s['softmax_lse_flag']}"
        if kind == "accuracy":
            selector += f" --distribution {s['distribution']}"
        else:
            selector += f" --{'impls' if kind == 'perf' else 'impl'} {s['impl']}"
        lines += [
            f"  {scene_name(s)}: Q={detail['query']['shape']}, K/V={detail['key']['shape']}, O={detail['attention_out']}, LSE={detail['softmax_lse']}",
            "    " + selector,
        ]
    lines += [
        "Normal: independently sample mean U[-5,5], std U[0.1,2] for Q/K/V; BF16 sources; no clipping.",
        "Progress refreshes after checked/committed rounds (first, every 10, last, or >=30s at round end).",
        "Stage logs do not provide an intra-kernel heartbeat. Stress timings are host time, not kernel performance.",
        f"Ascend 950 performance default: {DEFAULT_TEST_COUNT} captures per scene, "
        f"{warmup(PRODUCTION)} warmups for Q={PRODUCTION['query_seq_len']}/KV={PRODUCTION['kv_seq_len']}.",
        "Spread >10%: consider more warmups and a fresh complete batch; retain any remaining variation.",
    ]
    p.epilog = "\n".join(lines)
    return p, defaults


def parse(kind, argv=None):
    p, defaults = parser(kind)
    argv = sys.argv[1:] if argv is None else argv
    flags = [a.split("=", 1)[0] for a in argv if a.startswith("--")]
    if len(flags) != len(set(flags)):
        p.error("duplicate option")
    args = p.parse_args(argv)
    resume = getattr(args, "resume", None)
    saved = read(Path(resume) / "plan.json")["parameters"] if resume else None
    plan = dict(saved or defaults)
    for key in defaults:
        value = getattr(args, key, None)
        if value is not None:
            if saved and value != plan.get(key):
                p.error(f"resume conflicts with {key}")
            plan[key] = value
    if resume and args.output_dir:
        p.error("resume and output-dir are mutually exclusive")
    devices = args.device_id if kind == "accuracy" else [args.device_id] if args.device_id is not None else None
    if devices is not None and (min(devices) < 0 or len(set(devices)) != len(devices)):
        p.error("device-id requires distinct nonnegative physical IDs")
    if (
        plan.get("stress_runs", 1) <= 0
        or plan.get("test_count", 1) <= 0
        or (plan.get("warm_up") is not None and plan["warm_up"] < 0)
    ):
        p.error("stress-runs/test-count must be positive; warm-up must be nonnegative")
    if kind == "profile" and args.impl:
        plan["impls"] = [args.impl]
    if kind == "perf" and not resume and args.impls is None and not cann_supported(plan):
        plan["impls"] = ["eqbsa"]
    if "cann_fp8_bsa" in plan.get("impls", []):
        if not cann_supported(plan):
            p.error("cann_fp8_bsa requires batch-size 1, BNSD layout and bf16 output")
        if plan["impls"] == ["cann_fp8_bsa"] and plan["softmax_lse_flag"] == 1:
            p.error("cann_fp8_bsa has LSE disabled; select eqbsa for --softmax-lse-flag 1")
    chosen = selections(plan, kind != "accuracy")
    for s in chosen:
        validate(s)
    if kind == "profile" and args.device_id is None and not args.dry_run:
        p.error("single-forward profile requires --device-id")
    return args, plan


def print_preview(args, plan, perf=False):
    cards = args.device_id
    if getattr(args, "resume", None) and cards is None:
        segments = sorted(Path(args.resume).glob("segment_*.json"), key=lambda p: read(p)["started"])
        cards = read(segments[-1])["devices"] if segments else None
    cards = cards if isinstance(cards, list) else [cards] if cards is not None else ["auto"]
    selected = selections(plan, perf)
    print(
        json.dumps(
            {
                "plan": plan,
                "scenes": [scene_preview(s) for s in selected],
                "assignment": {scene_name(s): cards[i % len(cards)] for i, s in enumerate(selected)},
                "warm_up": {
                    scene_name(s): plan.get("warm_up") if plan.get("warm_up") is not None else warmup(s)
                    for s in selected
                }
                if perf
                else None,
            },
            indent=2,
        )
    )
