"""Independent msprof captures, strict extraction, and resumable summaries."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import traceback

from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_state import Batch, ROOT, digest, read, save
from tests.ops.eagle_quant_batch_matmul_swiglu.qbmm_common import preview, scenario

MODULE = "tests.ops.eagle_quant_batch_matmul_swiglu.profile_eagle_quant_batch_matmul_swiglu_prod"
KERNEL = "EagleQuantBatchMatmulSwiglu"
LAYER_MODULE = "tests.ops.eagle_quant_batch_matmul_swiglu.profile_qbmm_swiglu_layer"


def units(plan):
    return [
        (path, layout, run)
        for path in plan["paths"]
        for layout in plan["formats"]
        for run in range(1, plan["runs"] + 1)
    ]


def layer_artifacts(folder, plan, layout, device):
    timing = folder / ("timing.json" if plan["no_msprof"] else "profiled_timing.json")
    data = read(timing)
    expected = dict(
        path="torch_npu",
        weight_format=layout,
        device=device,
        warmup=5,
        iterations=plan["iterations"],
        samples=plan["iterations"],
        profiled=not plan["no_msprof"],
        intermediate_dtype="BF16",
        output_dtype="torch.bfloat16",
    )
    if any(data.get(key) != value for key, value in expected.items()):
        raise RuntimeError("layer timing metadata differs from capture plan")
    if data.get("scenario") != scenario(**{key: plan[key] for key in ("m", "k", "n", "seed")}):
        raise RuntimeError("layer scenario differs from capture plan")
    samples = data["event_samples_us"]
    if len(samples) != plan["iterations"] or not all(math.isfinite(v) and v > 0 for v in samples):
        raise RuntimeError("invalid layer Event samples")
    if data["event_us"] != min(samples):
        raise RuntimeError("layer minimum differs from Event samples")
    artifacts = [timing, folder / "event_samples.csv"]
    profile = None
    if not plan["no_msprof"]:
        profiles = [p for p in folder.rglob("PROF_*") if p.is_dir()]
        if len(profiles) != 1:
            raise RuntimeError("expected exactly one PROF directory")
        profile = profiles[0]
        for pattern in ("op_summary*.csv", "msprof_*.json"):
            files = [p for p in profile.rglob(pattern) if p.stat().st_size > 0]
            if not files:
                raise RuntimeError(f"missing profiler artifact {pattern}")
            artifacts.extend(files)
    return data["event_us"], artifacts, profile, timing, samples


def duration(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(encoding="utf-8", errors="replace"))
    if text.count("Operator Basic Information") != 1:
        raise ValueError("expected exactly one Operator Basic Information section")
    section = text.split("Operator Basic Information", 1)[1]
    labels = re.findall(r"Task\s+Duration\s*\(us\)\s*:\s*([^\r\n]*)", section)
    if len(labels) != 1 or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", labels[0].strip()):
        raise ValueError(f"expected one numeric Task Duration(us), found {labels}")
    value = float(labels[0])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Task Duration must be finite and positive")
    return value


def identity(device):
    text = subprocess.check_output(
        ["npu-smi", "info", "-t", "board", "-i", str(device)], text=True, stderr=subprocess.STDOUT
    )
    fields = {}
    for line in text.splitlines():
        match = re.match(r"\s*([^:]+):\s*(.*?)\s*$", line)
        if match and re.search(r"serial|\bSN\b|board id|chip id", match[1], re.I):
            fields[match[1].strip()] = match[2]
    if not fields or not any(re.search(r"serial|\bSN\b", key, re.I) for key in fields):
        raise RuntimeError("cannot establish stable physical identity from npu-smi board serial")
    return dict(device=device, board=fields, host=__import__("platform").node())


def raw_artifacts(folder):
    # msOpProf documents OpBasicInfo.csv and dump/ as the basic and raw data.
    basic = list(folder.rglob("OpBasicInfo*.csv"))
    if len(basic) != 1:
        raise RuntimeError("expected one OpBasicInfo CSV for the selected kernel")
    content = basic[0].read_text(encoding="utf-8-sig", errors="replace")
    if re.sub(r"[^a-z0-9]", "", KERNEL.lower()) not in re.sub(r"[^a-z0-9]", "", content.lower()):
        raise RuntimeError("OpBasicInfo does not identify the target QBMM kernel")
    raw = [
        p for p in folder.rglob("*") if p.is_file() and p.stat().st_size > 0 and "dump" in p.relative_to(folder).parts
    ]
    if not raw:
        raise RuntimeError("missing nonempty raw dump data")
    return [basic[0], *raw]


def table(rows):
    widths = [((max(len(str(row[c])) for row in rows) // 8) + 1) * 8 for c in range(len(rows[0]))]
    return ["".join(str(v).ljust(w - 1) + "\t" for v, w in zip(row[:-1], widths[:-1])) + str(row[-1]) for row in rows]


def summary(batch, reason):
    plan = batch.plan["parameters"]
    rows = [("Scenario", "Test", "Metric", "Duration(us)", "DataDir")]
    minima = [("Scenario", "Metric", "Min(us)")]
    groups = {}
    progress = [("Path", "Weight", "Completed", "Status")]
    for path in plan["paths"]:
        for layout in plan["formats"]:
            completed = sum(
                bool(record and record["status"] == "PASS")
                for record in (batch.get(f"{path}/{layout}/{i}") for i in range(1, plan["runs"] + 1))
            )
            progress.append(
                (
                    path,
                    layout,
                    f"{completed}/{plan['runs']}",
                    "COMPLETE" if completed == plan["runs"] else "PENDING" if not completed else "PARTIAL",
                )
            )
    details = []
    count = 0
    for path, layout, i in units(plan):
        operation = "FusedQBMM-FP32SwiGLU-BF16out" if path == "qbmm" else "QBMM-BF16out+SwiGLU-BF16out"
        name = f"{operation}_{layout}_M{plan['m']}_K{plan['k']}_N{plan['n']}_int8_normal"
        record = batch.get(f"{path}/{layout}/{i}")
        if record and record["status"] == "PASS":
            count += 1
            metric = "Task Duration" if path == "qbmm" else "Layer Duration"
            groups.setdefault((name, metric), []).append(record["duration"])
            rows.append((name, i, metric, f"{record['duration']:.9g}", str(batch.path / record["directory"])))
            details.append(
                dict(
                    scenario=name,
                    run=i,
                    metric=metric,
                    duration_us=record["duration"],
                    data_dir=str(batch.path / record["directory"]),
                    profile_dir=str(batch.path / record["profile_dir"]) if record["profile_dir"] else None,
                    timing_file=str(batch.path / record["timing_file"]) if record["timing_file"] else None,
                    event_samples_us=record["event_samples_us"],
                )
            )
    for (name, metric), values in groups.items():
        minima.append((name, metric, f"{min(values):.9g}"))
    lines = (
        [f"{reason}; completed={count}/{len(units(plan))}"]
        + table(progress)
        + [""]
        + table(rows)
        + [""]
        + table(minima)
    )
    for (name, metric), values in groups.items():
        if metric != "Task Duration":
            continue
        low, high = min(values), max(values)
        spread = (high - low) / low * 100
        lines.append(f"{name}: max={high:.9g}us; spread={spread:.2f}%; warm-up={plan['warm_up']}; launch-count=1")
        if spread > 10:
            lines.append("WARNING spread >10%; increase warm-up and collect a fresh complete 20-capture batch.")
    lines += [
        "Layer Duration: fixed warmup=5; per-run minimum of " + str(plan["iterations"]) + " Event samples.",
        "Task Duration and Layer Duration use different measurement scopes; no mixed-metric speedup is reported.",
        "Note: Some hardware models may show increased latency across runs. The minimum across all groups for each scenario is reported as the best performance observed under the current test conditions; this does not mean the variation has been eliminated.",
    ]
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    (batch.path / "final_summary.txt").write_text(text, encoding="utf-8")
    (batch.path / "task_duration_summary.txt").write_text(text, encoding="utf-8")
    best = [
        min((row for row in details if row["scenario"] == name), key=lambda row: row["duration_us"])
        for name, metric in groups
    ]
    save(
        batch.path / "summary.json",
        dict(reason=reason, completed=count, total=len(units(plan)), details=details, minima=best, plan=plan),
    )


def main():
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Fused QBMM/SwiGLU and native two-op baseline: ND/NZ weights, INT8 inputs, FP32 scales, BF16 output",
        epilog="Defaults: M=5672 K=5376 N=28672; x1=[M,K], x2=[K,N], scale=[N], token=[M], y=[M,N/2]. "
        "Positive M/K/N, N must be even (NZ: divisible by 64). No transpose/bias; seed=20261030. "
        "Default paths=qbmm,torch_npu; formats=ND,NZ; runs=20; layer iterations=10, fixed warmup=5. "
        "New runs auto-select one card; resume keeps its physical identity. "
        "Output: repo/log/eagle_quant_batch_matmul_swiglu/perf_TIMESTAMP/final_summary.txt. "
        "Fused --warm-up defaults to 20 and permits 0; NZ packing is outside timing.",
    )
    for name in ("m", "k", "n", "seed", "runs", "iterations", "warm-up", "device-id"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--test-count", type=int, dest="runs", help=argparse.SUPPRESS)
    p.add_argument("--path", choices=["qbmm", "torch_npu"], help="default: both paths")
    p.add_argument("--weight-format", choices=["ND", "NZ"], help="default: both layouts")
    p.add_argument("--no-msprof", action="store_true", default=None, help="direct Events; requires --path torch_npu")
    p.add_argument("--resume")
    p.add_argument("--output-dir")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    # Reject repeated scalar selectors instead of silently using the last value.
    flags = [arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")]
    if len(flags) != len(set(flags)) or ("--runs" in flags and "--test-count" in flags):
        p.error("duplicate option")
    plan = (
        read(Path(args.resume) / "plan.json")["parameters"]
        if args.resume
        else dict(
            m=5672,
            k=5376,
            n=28672,
            seed=20261030,
            runs=20,
            warm_up=20,
            iterations=10,
            paths=["qbmm", "torch_npu"],
            formats=["ND", "NZ"],
            no_msprof=False,
        )
    )
    args.paths = [args.path] if args.path else None
    args.formats = [args.weight_format] if args.weight_format else None
    for name in tuple(plan):
        value = getattr(args, name)
        if value is not None:
            if args.resume and value != plan[name]:
                p.error(f"resume conflicts with {name}")
            plan[name] = value
    if min(plan[k] for k in ("m", "k", "n", "runs", "iterations")) <= 0 or plan["warm_up"] < 0:
        p.error("dimensions, runs and iterations must be positive; warm-up nonnegative")
    if plan["n"] % 2:
        p.error("N must be even")
    if "NZ" in plan["formats"] and plan["n"] % 64:
        p.error("NZ requires N/2 aligned to 32")
    if plan["no_msprof"] and plan["paths"] != ["torch_npu"]:
        p.error("--no-msprof requires only --path torch_npu")
    if args.device_id is not None and args.device_id < 0:
        p.error("physical device-id must be nonnegative")
    if args.resume and args.output_dir:
        p.error("resume and output-dir are mutually exclusive")
    captures = units(plan)
    selections = [
        dict(
            path=path,
            weight_format=layout,
            runs=plan["runs"],
            tool="msprof op" if path == "qbmm" else "Events" if plan["no_msprof"] else "msprof + Events",
        )
        for path in plan["paths"]
        for layout in plan["formats"]
    ]
    if args.dry_run:
        selected = scenario(m=plan["m"], k=plan["k"], n=plan["n"], seed=plan["seed"])
        inputs = [
            dict(preview(selected), layout=dict(x1="ND", x2=layout, scale="ND", pertoken_scale="ND"))
            for layout in plan["formats"]
        ]
        print(
            json.dumps(
                dict(
                    plan=plan,
                    capture_groups=selections,
                    total_captures=len(captures),
                    inputs=inputs,
                    kernel=KERNEL,
                    distribution="normal",
                    device=args.device_id,
                    launch_count=1,
                    output_root=str(ROOT / "log"),
                ),
                indent=2,
            )
        )
        return 0
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        p.error("unset ASCEND_RT_VISIBLE_DEVICES")
    print(f"PLAN {'resume' if args.resume else 'new'} groups={len(selections)} captures={len(captures)}", flush=True)
    for index, selection in enumerate(selections, 1):
        print(
            f"PLAN {index}/{len(selections)} {selection['path']}/{selection['weight_format']} "
            f"runs={selection['runs']} tool={selection['tool']}",
            flush=True,
        )
    print(
        "PLAN groups execute sequentially; only --path selects a single path; resume preserves its saved selection.",
        flush=True,
    )
    batch = Batch("perf", plan, args.resume, args.output_dir)
    reason, code = "PARTIAL", 1
    try:
        batch.ensure_no_failures()
        if all(batch.get(f"{path}/{layout}/{i}") for path, layout, i in units(plan)):
            reason, code = "COMPLETE (committed results; no NPU access)", 0
            return code
        device_file = batch.path / "device.json"
        previous = read(device_file) if device_file.exists() else None
        device = args.device_id
        if previous:
            if device is not None and device != previous["identity"]["device"]:
                raise ValueError("performance resume cannot change physical device")
            device = previous["identity"]["device"]
        if device is None:
            report = subprocess.check_output(
                [sys.executable, str(ROOT / "tests/tools/select_npu_device.py"), "--format=report"], text=True
            )
            (batch.path / "device_selection.log").write_text(report, encoding="utf-8")
            print(report, flush=True)
            matches = re.findall(r"Selected NPU:\s*(\d+)", report)
            if len(matches) != 1:
                raise RuntimeError("cannot parse selected NPU from report")
            device = int(matches[0])
        profiler = None if plan["no_msprof"] else shutil.which("msprof")
        if not profiler and not plan["no_msprof"]:
            raise RuntimeError("msprof not found; source CANN set_env.sh")
        current = dict(
            identity=identity(device), profiler_path=profiler, profiler_sha256=digest(profiler) if profiler else None
        )
        if previous and previous != current:
            raise RuntimeError("physical device or profiler changed; start a new batch")
        if not previous:
            save(device_file, current)
        batch.history([device])

        def stop(sig, frame):
            raise SystemExit(128 + sig)

        signal.signal(signal.SIGTERM, stop)
        for path, layout, i in captures:
            key = f"{path}/{layout}/{i}"
            if batch.get(key):
                continue
            folder = batch.attempt(key)
            workload = [
                sys.executable,
                "-m",
                MODULE if path == "qbmm" else LAYER_MODULE,
                "--device-id",
                str(device),
                "--weight-format",
                layout,
            ]
            for name in ("m", "k", "n", "seed"):
                workload += ["--" + name, str(plan[name])]
            if path == "qbmm":
                command = [
                    profiler,
                    "op",
                    f"--kernel-name={KERNEL}",
                    f"--warm-up={plan['warm_up']}",
                    "--launch-count=1",
                    "--kill=off",
                    f"--output={folder}",
                ] + workload
            else:
                timing = folder / ("timing.json" if plan["no_msprof"] else "profiled_timing.json")
                workload += ["--iterations", str(plan["iterations"]), "--output", str(timing)]
                if plan["no_msprof"]:
                    command = workload
                else:
                    workload += ["--msprof-mode"]
                    # Pass argv directly, matching the other composed profiling workloads.
                    # Collection automatically exports results; --export is an offline mode.
                    command = [
                        profiler,
                        f"--output={folder}",
                        "--task-time=on",
                        "--ascendcl=on",
                        "--runtime-api=on",
                    ] + workload
            save(folder / "command.json", dict(argv=command))
            print(f"START {key}/{plan['runs']} device={device} dir={folder}", flush=True)
            log_path = folder / ("timing.log" if plan["no_msprof"] else "msprof.log")
            with log_path.open("w", encoding="utf-8") as log:
                with subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                    pass_fds=(batch.lock.fileno(),),
                ) as child:
                    try:
                        for line in child.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        result = child.wait()
                    finally:
                        if child.poll() is None:
                            # start_new_session makes the child its POSIX process group leader.
                            os.kill(-child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.kill(-child.pid, signal.Signals["SIGKILL"])
                                child.wait()
            if result:
                raise RuntimeError(f"capture exit={result}; see {log_path}")
            profile, timing, samples = None, None, None
            if path == "qbmm":
                value, raw = duration(folder / "msprof.log"), raw_artifacts(folder)
            else:
                value, raw, profile, timing, samples = layer_artifacts(folder, plan, layout, device)
            batch.commit(
                key,
                duration=value,
                directory=str((profile or folder).relative_to(batch.path)),
                device=device,
                profile_dir=str(profile.relative_to(batch.path)) if profile else None,
                timing_file=str(timing.relative_to(batch.path)) if timing else None,
                event_samples_us=samples,
                artifacts=[log_path, folder / "command.json", device_file, *raw],
            )
            print(f"PASS {key} {'Task' if path == 'qbmm' else 'Layer'} Duration(us)={value}", flush=True)
        reason, code = "COMPLETE", 0
    except KeyboardInterrupt:
        reason, code = "PARTIAL SIGINT", 130
    except SystemExit as exc:
        reason, code = "PARTIAL signal", int(exc.code)
    except Exception:
        traceback.print_exc()
        reason, code = "PARTIAL error", 1
    finally:
        summary(batch, reason)
        batch.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
