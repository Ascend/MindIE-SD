# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Collect the paired 950 gate/quant workloads and retain independent runs."""

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

from tests.ops.sigmoid_mul_dynamic_mx_quant.batch import Batch, prepare_vendor
from tests.ops.sigmoid_mul_dynamic_mx_quant.summarize import PATHS, read_run, summarize


def run_unit(folder, path, args):
    app = [
        sys.executable,
        "-m",
        "tests.ops.sigmoid_mul_dynamic_mx_quant.profile_workload",
        "--device-id",
        str(args.device_id),
        "--path",
        path,
        "--scale-alg",
        str(args.scale_alg),
    ]
    if path == "kernel":
        command = [
            "msprof",
            "op",
            "--kernel-name=SigmoidMulDynamicMxQuant",
            "--launch-count=1",
            "--kill=off",
            f"--warm-up={args.warmup}",
            f"--output={folder}",
            *app,
        ]
    else:
        name = "timing.json" if args.no_msprof else "profiled_timing.json"
        app.extend(["--iterations", str(args.iterations), "--output", str(folder / name)])
        if not args.no_msprof:
            app.append("--msprof-mode")
        command = (
            app
            if args.no_msprof
            else ["msprof", f"--output={folder}", "--task-time=on", "--ascendcl=on", "--runtime-api=on", *app]
        )
    folder.mkdir()
    print(shlex.join(command), flush=True)
    log = folder / ("timing.log" if args.no_msprof else "msprof.log")
    with (
        log.open("w", encoding="utf-8") as stream,
        subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
        ) as process,
    ):
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
            if process.wait():
                raise RuntimeError(f"capture failed; see {log}")
        except (KeyboardInterrupt, SystemExit):
            if process.poll() is None:
                os.kill(-process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Ascend profiling runs on Linux, where negative PIDs address process groups.
                    os.kill(-process.pid, signal.SIGKILL)  # pylint: disable=no-member
            raise
    read_run(folder, path, vars(args))
    return [p for p in folder.rglob("*") if p.is_file()]


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--scale-alg", type=int, choices=(0, 1), help="MX scale algorithm (default: 0)")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--path", choices=PATHS, action="append", help="select one path (default: kernel and native)")
    parser.add_argument("--no-msprof", action="store_true", default=None, help="measure --path native without msprof")
    parser.add_argument("--dry-run", action="store_true", help="show the 950 capture plan without using a device")
    parser.epilog = "Per-run measurements and scenario minima are saved in final_summary.txt."
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--output", type=Path)
    args = parser.parse_args()
    defaults = {
        "runs": 3,
        "scale_alg": 0,
        "warmup": int(os.environ.get("MSPROF_WARMUP", 20)),
        "iterations": 10,
        "path": None,
        "no_msprof": False,
        "device_id": None,
    }
    if args.path:
        if len(args.path) != 1:
            parser.error("specify --path once")
        args.path = args.path[0]
    if args.resume:
        previous = json.loads((args.resume / "plan.json").read_text(encoding="utf-8"))["parameters"]
        defaults.update({k: previous[k] for k in defaults if k in previous})
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if min(args.runs, args.iterations) < 1 or args.warmup < 0 or (args.device_id is not None and args.device_id < 0):
        parser.error("invalid device, run count, iteration count or warmup")
    if args.path is not None and args.path not in PATHS:
        parser.error("saved plan is incompatible with the BF16 performance baseline")
    if args.no_msprof and args.path != "native":
        parser.error("--no-msprof requires --path native")
    paths = [args.path] if args.path else PATHS
    print(
        f"950 BF16 mode=1 scale_alg={args.scale_alg} paths={','.join(paths)} runs={args.runs} captures={args.runs * len(paths)}"
    )
    print(f"Kernel warmup={args.warmup}, Event warmup=5, iterations={args.iterations}")
    if args.dry_run:
        return
    if args.device_id is None:
        device_report = subprocess.check_output(
            [sys.executable, "-m", "tests.tools.select_npu_device", "--format", "report"], text=True
        )
        print(device_report, end="")
        args.device_id = int(next(s.split(":")[1] for s in device_report.splitlines() if s.startswith("Selected NPU:")))
    prepare_vendor()
    board = subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(args.device_id)], text=True)
    identity = [line.strip() for line in board.splitlines() if "Serial Number" in line or "Chip Name" in line]
    if not any("950" in line for line in identity):
        parser.error("this operator requires Ascend 950")
    plan = {k: v for k, v in vars(args).items() if k not in ("output", "resume", "dry_run")}
    plan["precision_mode"] = 1
    plan["board"] = identity
    batch = Batch("performance", plan, args.output, args.resume)
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        for path in paths:
            for run in range(1, args.runs + 1):
                key = f"mode1_{path}_{run}"
                if batch.complete(key):
                    continue
                folder = batch.path / key
                if folder.exists():
                    folder.rename(batch.path / f"partial_{key}_{len(list(batch.path.glob(f'partial_{key}_*')))}")
                artifacts = run_unit(folder, path, args)
                batch.save(key, "PASS", artifacts)
        summarize(batch.path)
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        summarize(batch.path, partial_reason=f"{type(error).__name__} {error}")
        if isinstance(error, KeyboardInterrupt):
            raise SystemExit(130) from error
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        batch.close()


if __name__ == "__main__":
    main()
