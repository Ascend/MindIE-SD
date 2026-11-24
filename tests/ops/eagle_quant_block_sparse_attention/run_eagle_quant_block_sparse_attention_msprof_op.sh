#!/usr/bin/env bash
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# Source an explicitly requested environment; otherwise use the caller's CANN environment.
if [[ -n "${ASCEND_ENV_SH:-}" ]]; then
    source "${ASCEND_ENV_SH}"
fi
unset ASCEND_RT_VISIBLE_DEVICES
cd "${REPO_ROOT}"
# Keep the batch process in charge of the lock, signals and per-capture commits.
# msprof runs the standalone profile script once per capture.
exec python3 -u - "${BASH_SOURCE[0]##*/}" "$@" <<'PYTHON'
import os
import re
import shutil
import signal
import subprocess
import sys
import traceback

from tests.ops.eagle_quant_block_sparse_attention.eqbsa_cases import (
    KERNELS,
    parse,
    print_preview,
    scene_name,
    selections,
    warmup,
)
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_state import (
    Batch,
    HERE,
    ROOT,
    digest,
    duration,
    identity,
    read,
    save,
    select_device,
    table,
    tee_log,
)
from tests.ops.eagle_quant_block_sparse_attention.eagle_quant_block_sparse_attention_accuracy_common import (
    PRODUCTION,
)


def units(plan):
    return [(s, i) for s in selections(plan, perf=True) for i in range(1, plan["test_count"] + 1)]


def raw_artifacts(folder, kernel):
    basic = list(folder.rglob("OpBasicInfo*.csv"))
    if len(basic) != 1:
        raise RuntimeError("expected one target OpBasicInfo CSV")
    text = basic[0].read_text(encoding="utf-8-sig", errors="replace")
    # The CANN kernel name is also a suffix of the Eagle and EagleQuant names.
    matches = re.findall(r"(?:eagle[_ -]?(?:quant[_ -]?)?)?block[_ -]?sparse[_ -]?attention", text, re.IGNORECASE)
    if {re.sub(r"[_ -]", "", match).lower() for match in matches} != {kernel.lower()}:
        raise RuntimeError(f"OpBasicInfo does not uniquely identify {kernel}")
    raw = [p for p in folder.rglob("*") if p.is_file() and p.stat().st_size and "dump" in p.relative_to(folder).parts]
    if not raw:
        raise RuntimeError("missing nonempty raw profiler dump")
    return [basic[0], *raw]


def summary(batch, reason):
    plan = batch.plan["parameters"]
    rows = [("Scenario", "Test", "Metric", "Duration(us)", "DataDir")]
    minima = [("Scenario", "Metric", "Min(us)", "Samples")]
    progress = [("Scenario", "Completed/Planned", "Status")]
    groups = {}
    completed = {}
    for s in selections(plan, perf=True):
        label = scene_name(s)
        values = []
        for i in range(1, plan["test_count"] + 1):
            r = batch.get(f"{label}/{i}")
            if r and r["status"] == "PASS":
                values.append(r["duration_us"])
                rows.append((label, i, "Task Duration", f"{r['duration_us']:.9g}", str(batch.path / r["directory"])))
        count = len(values)
        progress.append(
            (label, f"{count}/{plan['test_count']}", "COMPLETE" if count == plan["test_count"] else "PARTIAL")
        )
        if values:
            groups[label] = values
            minima.append((label, "Task Duration", f"{min(values):.9g}", count))
            if count == plan["test_count"]:
                completed[s["impl"], s["softmax_lse_flag"]] = min(values)
    # One uninterrupted round table immediately followed by the single minimum table.
    lines = [reason] + table(progress) + [""] + table(rows) + [""] + table(minima)
    if len(rows) == 1:
        lines.append("No completed results")
    for label, values in groups.items():
        low, high = min(values), max(values)
        spread = (high - low) / low * 100
        lines.append(
            f"{label}: warm-up={plan['warm_up']}; max={high:.9g}us; spread={spread:.2f}%; samples={len(values)}"
        )
        if len(values) < 2:
            lines.append("Insufficient samples to assess variation")
        elif spread > 10:
            lines.append(
                "WARNING spread >10%; consider more warmups on the same card, then a fresh batch "
                f"of {plan['test_count']} captures per scene."
            )
    eq0, eq1 = completed.get(("eqbsa", 0)), completed.get(("eqbsa", 1))
    cann = completed.get(("cann_fp8_bsa", 0))
    if cann is not None and eq0 is not None:
        lines.append(
            f"Implementation comparison (LSE off): CANN FP8 BSA / EQBSA = {cann:.9g} / {eq0:.9g} = {cann / eq0:.4f}x"
        )
    if eq0 is not None and eq1 is not None:
        lines.append(
            f"EQBSA LSE overhead (LSE on vs off): {eq1:.9g} - {eq0:.9g} = {eq1 - eq0:+.9g} us "
            f"({(eq1 / eq0 - 1) * 100:+.2f}%)"
        )
    lines.append(
        "Note: Some hardware models may show increased latency across runs. The minimum across all groups for each scenario is reported as the best performance observed under the current test conditions; this does not mean the variation has been eliminated."
    )
    text = "\n".join(lines) + "\n"
    print(text, flush=True)
    for filename in ("final_summary.txt", "task_duration_summary.txt"):
        (batch.path / filename).write_text(text, encoding="utf-8")


def capture(batch, folder, command):
    save(folder / "command.json", {"argv": command})
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    active_path = batch.path / "active_capture.json"
    with (
        (folder / "msprof.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
            pass_fds=(batch.lock.fileno(),),
        ) as child,
    ):
        try:
            save(active_path, {"pgid": child.pid})
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        finally:
            # Kill the entire owned group, including tool descendants, before releasing its lease.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            if child.poll() is None:
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            active_path.unlink(missing_ok=True)
    if code:
        raise RuntimeError(f"msprof exit={code}; see {folder / 'msprof.log'}")


def run_batch(batch, args, plan):
    reason, code = "PARTIAL", 1

    def stop(sig, frame):
        raise SystemExit(128 + sig)

    previous_handler = signal.signal(signal.SIGTERM, stop)
    try:
        batch.ensure_no_failures()
        device_file = batch.path / "device.json"
        previous = read(device_file) if device_file.exists() else None
        if previous and args.device_id is not None and args.device_id != previous["identity"]["device"]:
            raise ValueError("performance resume must keep the original physical card")
        for s in selections(plan, perf=True):
            complete = sum(batch.get(f"{scene_name(s)}/{i}") is not None for i in range(1, plan["test_count"] + 1))
            print(f"SAVED {scene_name(s)} captures={complete}/{plan['test_count']}", flush=True)
        if all(batch.get(f"{scene_name(s)}/{i}") for s, i in units(plan)):
            reason, code = "COMPLETE (saved results; no device access)", 0
            return code
        device = previous["identity"]["device"] if previous else args.device_id
        if device is None:
            device = select_device()
        profiler = shutil.which("msprof")
        if not profiler:
            raise RuntimeError("msprof not found; source CANN set_env.sh")
        current = {"identity": identity(device), "profiler_path": profiler, "profiler_sha256": digest(profiler)}
        if previous and previous != current:
            raise RuntimeError("original physical device or profiler changed")
        if not previous:
            save(device_file, current)
        batch.history([device])
        print(
            f"PLAN impls={plan['impls']} physical_device={device} warm-up={plan['warm_up']} "
            f"independent_captures={plan['test_count']} launch-count=1; report each sample and minimum",
            flush=True,
        )
        for s, i in units(plan):
            key = f"{scene_name(s)}/{i}"
            if batch.get(key):
                continue
            folder = batch.attempt(key)
            kernel = KERNELS[s["impl"]]
            command = [
                profiler,
                "op",
                f"--kernel-name={kernel}",
                f"--warm-up={plan['warm_up']}",
                "--launch-count=1",
                "--kill=off",
                f"--output={folder}",
                sys.executable,
                "-u",
                str(HERE / "profile_eagle_quant_block_sparse_attention_prod.py"),
                "--impl",
                s["impl"],
                "--device-id",
                str(device),
                "--softmax-lse-flag",
                str(s["softmax_lse_flag"]),
                "--msprof-mode",
            ]
            for key_name in PRODUCTION:
                command += ["--" + key_name.replace("_", "-"), str(s[key_name])]
            print(f"START {key} DataDir={folder}", flush=True)
            capture(batch, folder, command)
            value = duration(folder / "msprof.log")
            artifacts = raw_artifacts(folder, kernel)
            batch.commit(
                key,
                device=device,
                impl=s["impl"],
                kernel=kernel,
                identity=current["identity"],
                duration_us=value,
                directory=str(folder.relative_to(batch.path)),
                artifacts=[device_file, folder / "command.json", folder / "msprof.log", *artifacts],
            )
            print(f"COMMITTED {key} Task Duration(us)={value}", flush=True)
        reason, code = "COMPLETE", 0
    except KeyboardInterrupt:
        reason, code = "PARTIAL SIGINT", 130
    except SystemExit as error:
        reason, code = "PARTIAL signal", int(error.code)
    except Exception:  # noqa: BLE001 - preserve every committed result on scheduler/tool errors.
        traceback.print_exc()
        reason, code = "PARTIAL error", 1
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        summary(batch, reason)
    return code


def main(argv=None):
    args, plan = parse("perf", argv)
    if plan["warm_up"] is None:
        plan["warm_up"] = warmup(selections(plan, perf=True)[0])
    if args.dry_run:
        print_preview(args, plan, perf=True)
        return 0
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("unset ASCEND_RT_VISIBLE_DEVICES")
    batch = Batch("perf", plan, args.resume, args.output_dir)
    try:
        with tee_log(batch.path / "performance.log", stderr=True):
            return run_batch(batch, args, plan)
    finally:
        batch.close()


sys.argv = sys.argv[1:]  # Show the shell entry point in argparse help and errors.
sys.exit(main())
PYTHON
