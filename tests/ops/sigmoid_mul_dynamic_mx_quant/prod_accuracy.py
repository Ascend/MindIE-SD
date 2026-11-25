# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Production-shape accuracy and resumable fixed-input stress on Ascend 950."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import torch

from tests.ops.sigmoid_mul_dynamic_mx_quant.batch import Batch, digest, prepare_vendor, save_record, write_json
from tests.ops.sigmoid_mul_dynamic_mx_quant.workload import SHAPE, compare, fused, make_inputs, metadata, reference


def snapshot(path, create):
    manifest = path.with_suffix(".json")
    if not manifest.exists():
        tensors = create()
        pending = path.with_suffix(".pending")
        torch.save(tensors, pending)
        pending.replace(path)
        write_json(manifest, {"sha256": digest(path)})
        return tensors
    if digest(path) != json.loads(manifest.read_text(encoding="utf-8"))["sha256"]:
        raise ValueError(f"snapshot checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


def run_case(root, phase, mode, scale_alg, distribution, device, repeats):
    print(
        f"device={device} phase={phase} precision_mode={mode} scale_alg={scale_alg} distribution={distribution} preparing inputs",
        flush=True,
    )
    cpu = snapshot(root / f"inputs_{mode}.pt", lambda: make_inputs(SHAPE, distribution))
    x, gate = (t.to(device) for t in cpu)
    if phase == "accuracy":
        actual = fused(x, gate, mode, scale_alg)
        metadata(actual, SHAPE, device)
        print("Computing NPU golden and comparing both outputs", flush=True)
        compare(actual, reference(x, gate, mode, scale_alg), small=False)
        return

    def baseline():
        out = fused(x, gate, mode, scale_alg)
        metadata(out, SHAPE, device)
        return tuple(t.view(torch.uint8).cpu() for t in out)

    expected = tuple(t.to(device) for t in snapshot(root / f"baseline_{mode}_alg{scale_alg}.pt", baseline))
    progress = root / f"stress_{mode}_alg{scale_alg}.json"
    done = json.loads(progress.read_text(encoding="utf-8"))["round"] if progress.exists() else 0
    if not isinstance(done, int) or not 0 <= done <= repeats:
        raise ValueError(f"invalid stress checkpoint {progress}")
    print(f"Resuming at {done}/{repeats}", flush=True)
    for repeat in range(done + 1, repeats + 1):
        out = fused(x, gate, mode, scale_alg)
        metadata(out, SHAPE, device)
        for name, actual, target in zip(("E4M3FN", "E8M0"), out, expected):
            bits = actual.view(torch.uint8)
            if not torch.equal(bits, target):
                location = torch.nonzero(bits != target)[0].tolist()
                raise AssertionError(f"repeat={repeat} output={name} first difference={location}")
        write_json(progress, {"round": repeat, "device": str(device)})
        if repeat % 100 == 0:
            print(f"device={device} mode={mode} round={repeat}/{repeats}", flush=True)
    if not all(torch.equal(actual.cpu(), original) for actual, original in zip((x, gate), cpu)):
        raise AssertionError("input changed during fixed-input stress")


def run_worker(root, phase, cases, device_id, repeats, segment):
    prepare_vendor()
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    import mindiesd  # noqa: F401

    device = torch.device(f"npu:{device_id}")
    for mode, scale_alg, distribution in cases:
        key = f"{phase}_{mode}_alg{scale_alg}"
        log = root / f"{key}_device{device_id}_{segment}.log"
        try:
            with log.open("w", encoding="utf-8") as stream, redirect_stdout(stream), redirect_stderr(stream):
                run_case(root, phase, mode, scale_alg, distribution, device, repeats)
        except AssertionError as error:
            save_record(root, key, "FAIL", [log], detail=str(error))
            raise
        artifacts = [log, root / f"inputs_{mode}.pt", root / f"inputs_{mode}.json"]
        if phase == "stress":
            artifacts.extend(
                [
                    root / f"baseline_{mode}_alg{scale_alg}.pt",
                    root / f"baseline_{mode}_alg{scale_alg}.json",
                    root / f"stress_{mode}_alg{scale_alg}.json",
                ]
            )
        save_record(root, key, "PASS", artifacts, detail=f"device={device_id}")
        print(log.read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, nargs="+")
    parser.add_argument("--mode", choices=("all", "accuracy", "stress"))
    parser.add_argument("--stress-runs", type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=Path)
    group.add_argument("--resume", type=Path)
    args = parser.parse_args()
    previous = {}
    if args.resume:
        previous = json.loads((args.resume / "plan.json").read_text(encoding="utf-8"))["parameters"]
    repeats = args.stress_runs if args.stress_runs is not None else previous.get("stress_runs", 1000)
    mode_choice = args.mode or previous.get("mode", "all")
    if repeats < 1 or (args.device_id and (min(args.device_id) < 0 or len(set(args.device_id)) != len(args.device_id))):
        parser.error("expected distinct nonnegative devices and positive stress runs")
    devices = args.device_id
    if devices is None and args.resume:
        segments = sorted(args.resume.glob("segment_*.json"))
        if segments:
            devices = json.loads(segments[-1].read_text(encoding="utf-8"))["devices"]
    if devices is None:
        devices = [
            int(
                subprocess.check_output(
                    [sys.executable, "-m", "tests.tools.select_npu_device", "--format", "id"], text=True
                )
            )
        ]
    identities = {}
    for device in devices:
        board = subprocess.check_output(["npu-smi", "info", "-t", "board", "-i", str(device)], text=True)
        identities[str(device)] = [
            line.strip() for line in board.splitlines() if "Serial Number" in line or "Chip Name" in line
        ]
        if not any("950" in line for line in identities[str(device)]):
            parser.error(f"device {device} must be Ascend 950")
    plan = {"shape": list(SHAPE), "mode": mode_choice, "scale_algorithms": [0, 1], "stress_runs": repeats}
    batch = Batch("accuracy", plan, args.output, args.resume)
    segment = time.time_ns()
    phases = ("accuracy", "stress") if mode_choice == "all" else (mode_choice,)
    allocation = {mode: devices[mode % len(devices)] for mode in (0, 1)}
    write_json(
        batch.path / f"segment_{segment}.json", {"devices": devices, "identity": identities, "mapping": allocation}
    )
    keys = [f"{phase}_{mode}_alg{alg}" for phase in phases for mode in (0, 1) for alg in (0, 1)]
    try:
        for phase in phases:
            with ProcessPoolExecutor(
                max_workers=min(len(devices), 2), mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                workers = {}
                for device in devices:
                    cases = [
                        (m, alg, dist)
                        for m, dist in enumerate(("normal", "uniform"))
                        for alg in (0, 1)
                        if allocation[m] == device and not batch.complete(f"{phase}_{m}_alg{alg}")
                    ]
                    if cases:
                        future = pool.submit(run_worker, batch.path, phase, cases, device, repeats, segment)
                        workers[future] = device
                for future in workers:
                    future.result()
    finally:
        for key in keys:
            record = batch.record(key)
            status = json.loads(record.read_text(encoding="utf-8"))["status"] if record.exists() else "PARTIAL"
            print(f"{status} {key} {SHAPE}", flush=True)
        batch.close()


if __name__ == "__main__":
    main()
