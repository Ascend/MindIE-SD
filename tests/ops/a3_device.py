# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Resolve an allocated A3 die without modifying runtime visibility."""

import ctypes
import json
import os
import re
import subprocess
import sys


def visible_ids():
    mask = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")
    if mask is None:
        return None
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", mask):
        raise ValueError(f"Invalid ASCEND_RT_VISIBLE_DEVICES: {mask!r}")
    ids = [int(item) for item in mask.split(",")]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate logical IDs in ASCEND_RT_VISIBLE_DEVICES: {mask!r}")
    return ids


def query_smi(*args):
    command = ["npu-smi", "info", *args]
    try:
        return subprocess.run(command, capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"{' '.join(command)} failed: {getattr(error, 'stderr', '') or error}") from error


def find_die(table, logic_id):
    table = table.replace("|", " ")
    if not re.search(r"NPU\s+ID\s+Chip\s+ID\s+Chip\s+Logic\s+ID", table):
        raise RuntimeError("Missing npu-smi NPU ID / Chip ID / Chip Logic ID columns")
    mapping = {}
    dies = set()
    for row in table.splitlines():
        match = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)(?:\s|$)", row)
        if match is None:  # Includes MCU rows with '-' in the Chip Logic ID column.
            continue
        card, chip, logic = map(int, match.groups())
        if logic in mapping or (card, chip) in dies:
            raise RuntimeError(f"Duplicate npu-smi mapping: {row.strip()}")
        mapping[logic] = (card, chip)
        dies.add((card, chip))
    if logic_id not in mapping:
        raise RuntimeError(f"Chip Logic ID {logic_id} is absent from npu-smi info -m")
    return mapping[logic_id]


def physical_device_id(ordinal):
    """Query the current process's CANN user-to-physical mapping without selecting a device."""
    try:
        translate = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
    except (OSError, AttributeError):
        translate = ctypes.CDLL("libacl_rt.so").aclrtGetPhyDevIdByUserDevId
    translate.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    translate.restype = ctypes.c_int
    physical = ctypes.c_int32(-1)
    status = translate(ordinal, ctypes.byref(physical))
    if status != 0 or physical.value < 0:
        raise RuntimeError(f"CANN physical device lookup for ordinal {ordinal} returned {status}, ID {physical.value}")
    return physical.value


def unmasked_logic_id(ordinal, table):
    try:
        physical = physical_device_id(ordinal)
        remap = query_smi("-t", "phyid-remap", "-p", str(physical))
        fields = re.findall(r"^\s*(Chip Physical ID|Chip Logic ID|NPU ID|Chip ID)\s*:\s*([0-9]+)\s*$", remap, re.M)
        ids = {name: int(value) for name, value in fields}
        if len(fields) != 4 or len(ids) != 4 or ids["Chip Physical ID"] != physical:
            raise RuntimeError(f"Invalid npu-smi phyid-remap for physical ID {physical}")
        logic = ids["Chip Logic ID"]
        if find_die(table, logic) != (ids["NPU ID"], ids["Chip ID"]):
            raise RuntimeError(f"Conflicting npu-smi mappings for physical ID {physical}, Chip Logic ID {logic}")
        return logic
    except (AttributeError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Cannot verify unmasked A3 ordinal {ordinal}: {error}. Check the CANN/driver mapping or use a verified "
            "ASCEND_RT_VISIBLE_DEVICES allocation for non-profiling runs."
        ) from error


def verify_selected_die(ordinal):
    ids = visible_ids()
    if ordinal < 0 or (ids is not None and ordinal >= len(ids)):
        raise ValueError(f"Invalid visible ordinal {ordinal} for ASCEND_RT_VISIBLE_DEVICES={ids}")
    table = query_smi("-m")
    logic = unmasked_logic_id(ordinal, table) if ids is None else ids[ordinal]
    card, chip = find_die(table, logic)
    board = query_smi("-t", "board", "-i", str(card), "-c", str(chip))
    fields = {}
    for line in board.splitlines():
        key, separator, value = line.partition(":")
        if separator and value.strip():
            fields[key.strip().lower()] = value.strip()
    if not fields.get("chip name"):
        raise RuntimeError(f"Missing Chip Name in board information for NPU {card}, chip {chip}")
    identity = {
        key: value
        for key, value in fields.items()
        if key in ("chip name", "npu name", "board id", "pcie bus info")
        or re.search(r"\b(sn|uuid|serial number|bus-id|bus id)\b", key)
    }
    record = dict(runtime_ordinal=ordinal, chip_logic_id=logic, npu_id=card, chip_id=chip, identity=identity)
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
    return record


def production_a3_ordinal(requested, count):
    ids = visible_ids()
    if ids is None:
        table = query_smi("-m")
        ids = [unmasked_logic_id(ordinal, table) for ordinal in range(count)]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Ambiguous A3 die mapping; check the allocation and ASCEND_RT_VISIBLE_DEVICES")
    elif requested not in ids:
        raise ValueError(f"Chip Logic ID {requested} is outside ASCEND_RT_VISIBLE_DEVICES={ids}")
    if requested not in ids:
        raise ValueError(f"Chip Logic ID {requested} is not visible to the A3 runtime")
    return ids.index(requested)


def bind_production_device(requested):
    """A3 CLI accepts driver logic IDs. A2/A5 retain their visible-ordinal CLI."""
    import torch_npu

    if requested < 0:
        raise ValueError("Device ID must be nonnegative")
    # The public get_soc_version() initializes a context before we know the allocated ordinal.
    soc = torch_npu._C._npu_get_soc_version()
    if 250 <= soc <= 256:
        target = "ascend910_93"
    elif 220 <= soc <= 225:
        target = "ascend910b"
    elif soc == 260:
        target = "ascend950"
    else:
        raise RuntimeError(f"EagleSplitNormRopeConcat is not registered for SoC {soc}")
    configured = os.environ.get("ASCEND_COMPUTE_UNIT")
    if configured in ("ascend910b", "ascend910_93", "ascend950") and configured != target:
        raise RuntimeError(f"ASCEND_COMPUTE_UNIT={configured} conflicts with detected {target} (SoC {soc})")
    npu = torch_npu.npu
    count = npu.device_count()
    ordinal = production_a3_ordinal(requested, count) if target == "ascend910_93" else requested
    if ordinal >= count:
        raise ValueError(f"Requested device {requested} maps outside the visible devices (ordinal {ordinal})")
    npu.set_device(ordinal)
    if target == "ascend910_93":
        record = verify_selected_die(ordinal)
        if record["chip_logic_id"] != requested:
            raise RuntimeError("A3 device ID must identify the allocated Chip Logic ID")
    else:
        record = dict(runtime_ordinal=ordinal)
    record["soc_version"] = soc
    pin = os.environ.get("EAGLE_TEST_DEVICE_RECORD")
    if pin is not None and record != json.loads(pin):
        raise RuntimeError(f"Selected device identity changed since preflight: {record}")
    return record


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, required=True)
    options = parser.parse_args()
    print(json.dumps(bind_production_device(options.device_id), sort_keys=True))
