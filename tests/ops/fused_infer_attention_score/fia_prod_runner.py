# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Select and resume large FIA accuracy/stress or msprof batches."""

import argparse
import concurrent.futures
import csv
import datetime
import importlib.metadata
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from tests.ops.fused_infer_attention_score.fia_prod_state import (
    atomic_json,
    file_hash,
    load_record,
    plan_fingerprint,
    read_json,
    record_path,
    summarize,
)
from tests.ops.fused_infer_attention_score.fia_quant_common import (
    FIA_QUANT_CASE_BY_NAME,
    add_shape_arguments,
    select_shapes,
    selected_paths,
)
from tests.ops.fused_infer_attention_score.profile_fia_dit_quant import operator_package_dir

REPO_ROOT = Path(__file__).resolve().parents[3]
STOP = threading.Event()


def parse_args(kind):
    parser = argparse.ArgumentParser(
        description=f"FIA {kind}: default all 8 path/shape combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--path", choices=FIA_QUANT_CASE_BY_NAME)
    parser.add_argument(
        "--device-id", type=int, nargs="+", help="Accuracy: one or more physical IDs; performance: one."
    )
    add_shape_arguments(parser)
    parser.add_argument("--dry-run", action="store_true", help="Preview selection without accessing devices.")
    parser.add_argument("--output-dir", type=Path, help="Batch directory containing final_summary.txt.")
    parser.add_argument("--resume", type=Path, help="Explicit existing batch; no automatic resume.")
    parser.add_argument("--seed", type=int)
    if kind == "accuracy":
        parser.add_argument("--mode", choices=("all", "accuracy", "stress"), help="Default all: accuracy then stress.")
        parser.add_argument("--stress-runs", type=int, help="Default 1000 repeats after independent baseline.")
    else:
        parser.add_argument("--runs", type=int, help="Default: Ascend 950=3, 910B=20 independent captures.")
        parser.add_argument("--warmup", type=int, help="Default by max(Q,KV): <30000=20, <50000=2, otherwise=1.")
        parser.add_argument(
            "--soc-version", choices=("950", "910B"), help="Optional explicit architecture for preview."
        )
    args = parser.parse_args()
    # Single-valued flags must not silently accept the last duplicate.
    flags = [value.split("=", 1)[0] for value in sys.argv[1:] if value.startswith("--")]
    if len(flags) != len(set(flags)):
        parser.error("Duplicate options are not supported")
    if args.resume and args.output_dir:
        parser.error("--resume and --output-dir are mutually exclusive")
    if args.device_id is not None and (
        not args.device_id or min(args.device_id) < 0 or len(set(args.device_id)) != len(args.device_id)
    ):
        parser.error("Device IDs must be nonnegative and distinct")
    if kind == "performance" and args.device_id and len(args.device_id) != 1:
        parser.error("Performance requires exactly one physical device")
    for name in ("runs", "stress_runs"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "warmup", None) is not None and args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    return args


def scenarios_for(args, kind):
    scenarios = []
    for shape in select_shapes(args):
        for path in selected_paths(args.path):
            index = list(FIA_QUANT_CASE_BY_NAME).index(path)
            distribution = "normal" if kind == "performance" or index % 2 == 0 else "uniform"
            label = f"Q={shape.query_shape};KV={shape.kv_shape};{path};BNSD;FP8->BF16;{distribution}"
            scenarios.append(
                {
                    "id": f"s{len(scenarios):02d}",
                    "label": label,
                    "shape": vars(shape),
                    "path": path,
                    "distribution": distribution,
                }
            )
    return scenarios


def environment_identity():
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torch_npu", "mindiesd")}
    sources = {str(path.relative_to(REPO_ROOT)): file_hash(path) for path in Path(__file__).parent.glob("*.py")}
    package = operator_package_dir()
    artifacts = {
        str(item): file_hash(item)
        for item in package.rglob("*")
        if item.is_file() and item.suffix in (".so", ".o", ".json")
    }
    if not artifacts:
        raise ValueError("No installed MindIE-SD operator artifacts found")
    return {"versions": versions, "sources": sources, "artifacts": artifacts}


def selected_devices(args, old=None):
    if args.device_id:
        return args.device_id
    if old:
        return old["devices"]
    output = subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "tests/tools/select_npu_device.py"), "--format=id"], text=True
    )
    return [int(output.strip())]


def run_process(command, log_path, lock_fd):
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONUNBUFFERED="1")
    with Path(log_path).open("w", encoding="utf-8") as stream:
        # Explicit wait below keeps the tee and cancellation watcher active until exit.
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            pass_fds=(lock_fd,),
        )

        output_closed = threading.Event()

        def monitor():
            while not output_closed.wait(0.25):
                if STOP.is_set():
                    try:
                        os.killpg(process.pid, signal.SIGTERM)  # pylint: disable=no-member
                        time.sleep(10)
                        os.killpg(process.pid, signal.SIGKILL)  # pylint: disable=no-member
                    except ProcessLookupError:
                        pass
                    return

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
        output_closed.set()
        watcher.join()
    if code:
        STOP.set()
        raise RuntimeError(f"Process failed ({code}); log={log_path}")


def run_device(batch, plan, scenarios, device, stage, lock_fd):
    try:
        _run_device(batch, plan, scenarios, device, stage, lock_fd)
    except Exception:
        STOP.set()
        raise


def _run_device(batch, plan, scenarios, device, stage, lock_fd):
    fingerprint = plan_fingerprint(plan)
    for scenario in scenarios:
        if STOP.is_set():
            return
        count = plan["stages"][stage]
        paths = [record_path(batch, scenario["id"], stage, index) for index in range(1, count + 1)]
        if all(path.exists() and load_record(path, fingerprint)["status"] == "PASS" for path in paths):
            continue
        if stage != "performance":
            log = batch / f"{scenario['id']}_{stage}_{uuid.uuid4().hex[:8]}.log"
            command = [
                sys.executable,
                "-u",
                "-m",
                "tests.ops.fused_infer_attention_score.fia_prod_worker",
                "--batch",
                str(batch),
                "--scenario",
                scenario["id"],
                "--stage",
                stage,
                "--device-id",
                str(device),
                "--worker-log",
                str(log),
            ]
            run_process(command, log, lock_fd)
            if not all(path.exists() and load_record(path, fingerprint)["status"] == "PASS" for path in paths):
                raise RuntimeError(f"Missing or failed accuracy records: {scenario['label']}")
            continue
        for iteration, path in enumerate(paths, 1):
            if path.exists():
                if load_record(path, fingerprint)["status"] != "PASS":
                    raise ValueError(f"Previously committed failure: {path}")
                continue
            directory = batch / scenario["id"] / f"run_{iteration:02d}_{uuid.uuid4().hex[:8]}"
            directory.mkdir(parents=True)
            log = directory / "msprof.log"
            command = [
                "msprof",
                "op",
                "--kernel-name=EagleFusedInferAttentionScore",
                f"--warm-up={scenario['warmup']}",
                "--launch-count=1",
                "--kill=off",
                f"--output={directory}",
                sys.executable,
                "-u",
                str(Path(__file__).with_name("profile_fia_dit_quant.py")),
                "--device-id",
                str(device),
                "--path",
                scenario["path"],
                "--seed",
                str(plan["seed"]),
            ]
            for key, value in scenario["shape"].items():
                command.extend(("--" + key.replace("_", "-"), str(value)))
            run_process(command, log, lock_fd)
            content = log.read_text(encoding="utf-8")
            durations = re.findall(r"Task Duration\s*\(us\)\s*[:=|]?\s*([0-9]+(?:\.[0-9]+)?)", content)
            if len(durations) != 1 or float(durations[0]) <= 0:
                raise ValueError(f"Expected one positive Task Duration(us): {log}")
            csv_files = [item for item in directory.rglob("*.csv") if item.stat().st_size > 0]
            if not csv_files:
                raise ValueError(f"Missing profiler CSV artifacts: {directory}")
            target_rows = []
            for csv_file in csv_files:
                with csv_file.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.reader(stream)
                    next(reader, None)
                    target_rows.extend(
                        row
                        for row in reader
                        if any("eaglefusedinferattentionscore" in cell.replace("_", "").lower() for cell in row)
                    )
            if not target_rows:
                raise ValueError(f"No target kernel data rows in profiler CSV: {directory}")
            atomic_json(
                path,
                {
                    "version": 1,
                    "unit": path.stem,
                    "plan_fingerprint": fingerprint,
                    "status": "PASS",
                    "duration_us": float(durations[0]),
                    "device_id": device,
                    "data_dir": str(directory),
                    "artifacts": {str(item): file_hash(item) for item in [log, *csv_files]},
                },
            )


def main(kind="accuracy"):
    args = parse_args(kind)
    old = read_json(args.resume / "plan.json") if args.resume else None
    if old:
        if old["kind"] != kind:
            raise ValueError("Resume batch type differs")
        # Restore omitted fields; explicit conflicts are rejected before device initialization.
        for key, value in old["options"].items():
            requested = getattr(args, key, None)
            if requested is not None and requested != value:
                raise ValueError(f"Resume option conflict: {key}")
            setattr(args, key, value)
    scenarios = scenarios_for(args, kind)
    for item in scenarios:
        print(item["label"], flush=True)
    print(
        f"scenarios={len(scenarios)}; performance defaults: 950=3 runs, 910B=20 runs; "
        "attention warmup by max(Q,KV): <30000=20, <50000=2, otherwise=1",
        flush=True,
    )
    if args.dry_run:
        return
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("Unset ASCEND_RT_VISIBLE_DEVICES")
    batch = (
        args.resume
        or args.output_dir
        or REPO_ROOT / "log/fused_infer_attention_score" / f"{kind}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    if not old:
        batch.mkdir(parents=True, exist_ok=False)
    import fcntl

    with (batch / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if old and summarize(batch, old, "saved progress"):
            return
        identity = environment_identity()
        devices = selected_devices(args, old)
        if old and (old["identity"] != identity or old["hostname"] != socket.gethostname()):
            raise ValueError("Code, operator artifacts, software or host changed")
        if old and kind == "performance" and devices != old["devices"]:
            raise ValueError("Performance resume must use the original device")
        board = {
            str(device): subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(device)], text=True)
            for device in devices
        }
        if old and kind == "performance" and board != old["boards"]:
            raise ValueError("Performance device identity changed")
        if old:
            plan = old
            plan["history"].append({"devices": devices, "boards": board})
            plan["devices"] = devices
        else:
            if kind == "accuracy":
                mode = args.mode or "all"
                stages = {"accuracy": 1} if mode in ("all", "accuracy") else {}
                if mode in ("all", "stress"):
                    stages["stress"] = args.stress_runs or 1000
            else:
                detected = "950" if "950" in next(iter(board.values())) else "910B"
                if args.soc_version and args.soc_version != detected:
                    raise ValueError("Explicit architecture differs from selected device")
                stages = {"performance": args.runs or (3 if detected == "950" else 20)}
                for item in scenarios:
                    length = max(item["shape"]["query_seq_len"], item["shape"]["kv_seq_len"])
                    item["warmup"] = (
                        args.warmup if args.warmup is not None else 20 if length < 30000 else 2 if length < 50000 else 1
                    )
            options = {
                key: value
                for key, value in vars(args).items()
                if key not in ("resume", "output_dir", "device_id", "dry_run")
            }
            plan = {
                "version": 1,
                "kind": kind,
                "options": options,
                "scenarios": scenarios,
                "stages": stages,
                "seed": args.seed if args.seed is not None else 20260811,
                "hostname": socket.gethostname(),
                "identity": identity,
                "devices": devices,
                "boards": board,
                "history": [{"devices": devices, "boards": board}],
            }
        atomic_json(batch / "plan.json", plan)
        exit_code, reason = 0, "completed"

        def interrupted(signum, _frame):
            STOP.set()
            raise InterruptedError(signum)

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            for stage in plan["stages"]:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
                    futures = [
                        executor.submit(
                            run_device,
                            batch,
                            plan,
                            plan["scenarios"][index :: len(devices)],
                            device,
                            stage,
                            lock.fileno(),
                        )
                        for index, device in enumerate(devices)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
        except InterruptedError as exc:
            exit_code, reason = 128 + exc.args[0], str(exc)
        except Exception as exc:
            STOP.set()
            exit_code, reason = 1, str(exc)
        finally:
            success = summarize(batch, plan, reason)
        if exit_code or not success:
            raise SystemExit(exit_code or 1)


if __name__ == "__main__":
    main()
