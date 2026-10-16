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

"""Shared NPU picker for MindIE-SD tests (tests/tools/select_npu_device.py).

Parse `npu-smi info` and print a physical NPU ID for msprof / msprof op.
Preference: cards with no running process; if none, cards with NPU Util 0%
(lowest HBM if several). Callers pass that id to Python as --device-id and
must NOT export ASCEND_RT_VISIBLE_DEVICES (incompatible with Profiling).

Supports both the pre-25.x packed table (NPU+Name in one cell, AICore(%)) and
npu-smi 25.7+ (separate NPU ID / Name / Health columns, NPU Util(%)).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys

_HEALTHS = ("OK", "Warning", "Alarm", "Critical", "UNKNOWN")
_BUS_RE = re.compile(r"^(?:[0-9A-Fa-f]{4}:[0-9A-Fa-f:.]+|NA)$")
_OLD_NAME_RE = re.compile(r"^(\d+)\s+(\S.*)$")
_OLD_CHIP_RE = re.compile(r"^\d+(?:\s+\d+)?$")


def _run_npu_smi(args):
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        raise RuntimeError("npu-smi not found in PATH")
    proc = subprocess.run(
        [npu_smi, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and not text.strip():
        raise RuntimeError(f"failed to run npu-smi {' '.join(args)}: {proc.returncode}")
    return text


def _split_cells(line):
    if not line.startswith("|"):
        return []
    cells = [part.strip() for part in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _parse_util_hbm(blob):
    util = 0.0
    util_m = re.search(r"([\d.]+)", blob)
    if util_m:
        util = float(util_m.group(1))
    pairs = [(int(used), int(total)) for used, total in re.findall(r"(\d+)\s*/\s*(\d+)", blob)]
    if not pairs:
        return util, 0, 0
    hbm_used, hbm_total = pairs[-1]
    return util, hbm_used, hbm_total


def _match_name_row(cells):
    """Return (npu_id, health) or None."""
    if len(cells) < 3:
        return None
    health_v25 = cells[2].split()[0] if cells[2] else ""
    # 25.7+: | 0 | Ascend950PR | OK | 206.3 ... |
    if (
        re.fullmatch(r"\d+", cells[0])
        and cells[1]
        and not _BUS_RE.fullmatch(cells[1])
        and health_v25 in _HEALTHS
    ):
        return int(cells[0]), health_v25
    # Older: | 0     910B3 | OK | ... |
    old = _OLD_NAME_RE.match(cells[0])
    health_old = cells[1].split()[0] if cells[1] else ""
    if old and health_old in _HEALTHS:
        return int(old.group(1)), health_old
    return None


def _match_metric_row(cells):
    """Return (util, hbm_used, hbm_total) or None."""
    if len(cells) < 3:
        return None
    # 25.7+: |  |  | 0000:01:00.0 | 0   0 / 0   32981 / 114688 |
    if len(cells) >= 4 and not cells[0] and not cells[1] and _BUS_RE.fullmatch(cells[2]):
        return _parse_util_hbm(cells[3])
    # Older: | 0 | 0000:C1:00.0 | 0   0 / 0   3379 / 65536 |
    if _OLD_CHIP_RE.fullmatch(cells[0]) and _BUS_RE.fullmatch(cells[1]):
        return _parse_util_hbm(cells[2])
    return None


def _parse_npu_smi_info(text):
    """Parse `npu-smi info` table into per-card stats."""
    cards = {}
    proc_section = False
    pending_npu = None
    pending_health = "OK"

    for line in text.splitlines():
        if "Process id" in line and "NPU" in line:
            proc_section = True
            pending_npu = None
            continue
        if "Name" in line and "Health" in line and "Process" not in line:
            proc_section = False
            pending_npu = None
            continue
        if not line.startswith("|"):
            continue

        if "No running processes found in NPU" in line:
            found = re.search(r"NPU\s+(\d+)", line)
            if found:
                npu_id = int(found.group(1))
                if npu_id in cards:
                    cards[npu_id]["processes"] = 0
            continue

        cells = _split_cells(line)
        if not cells:
            continue

        if proc_section:
            npu_id = None
            if cells[0].isdigit() and len(cells) >= 2 and cells[1].isdigit():
                npu_id = int(cells[0])
            else:
                packed = re.match(r"^(\d+)\s+\d+$", cells[0])
                if packed and len(cells) >= 2 and cells[1].isdigit():
                    npu_id = int(packed.group(1))
            if npu_id is not None and npu_id in cards:
                cards[npu_id]["processes"] += 1
            continue

        named = _match_name_row(cells)
        if named is not None:
            pending_npu, pending_health = named
            continue

        if pending_npu is not None:
            metrics = _match_metric_row(cells)
            if metrics is None:
                continue
            util, hbm_used, hbm_total = metrics
            cards[pending_npu] = {
                "aicore": util,
                "hbm_used": hbm_used,
                "hbm_total": hbm_total,
                "processes": 0,
                "health": pending_health,
            }
            pending_npu = None

    if not cards:
        preview = "\n".join(text.splitlines()[:24])
        raise RuntimeError("failed to parse npu-smi info output. First lines:\n" + preview)
    return cards


def select_npu(cards):
    """Return (npu_id, reason, fallback_notice).

    1. Cards with no running process (skip Critical/UNKNOWN if others exist).
       Several: lowest HBM, then lowest util.
    2. Else cards with NPU Util == 0 (same health skip, then lowest HBM).
    3. Else lowest HBM among the rest.
    fallback_notice is a terminal line when step 1 found nothing, else None.
    """

    def unhealthy(info):
        return info.get("health", "OK") in ("Critical", "UNKNOWN")

    def pick_min_hbm(candidates):
        return min(candidates, key=lambda item: (item[1]["hbm_used"], item[1]["aicore"], item[0]))

    usable = [(npu_id, info) for npu_id, info in cards.items() if not unhealthy(info)]
    pool = usable if usable else list(cards.items())

    no_task = [(npu_id, info) for npu_id, info in pool if info["processes"] == 0]
    if no_task:
        npu_id, info = pick_min_hbm(no_task)
        reason = (
            f'no-task card: util={info["aicore"]:.0f}%, '
            f'HBM={info["hbm_used"]}/{info["hbm_total"]} MB, '
            f'health={info.get("health", "OK")}, processes=0'
        )
        return npu_id, reason, None

    zero_util = [(npu_id, info) for npu_id, info in pool if info["aicore"] <= 0.0]
    if zero_util:
        npu_id, info = pick_min_hbm(zero_util)
    else:
        npu_id, info = pick_min_hbm(pool)
    reason = (
        f'no no-task card, picked NPU {npu_id}: util={info["aicore"]:.0f}%, '
        f'HBM={info["hbm_used"]}/{info["hbm_total"]} MB, '
        f'health={info.get("health", "OK")}, processes={info["processes"]}'
    )
    notice = f"没有选到没有任务的卡，选了 NPU {npu_id}"
    return npu_id, reason, notice


def main():
    parser = argparse.ArgumentParser(
        description="Select physical NPU id for benchmark/profiling (npu-smi ID)."
    )
    parser.add_argument("--format", choices=["id", "report"], default="report")
    args = parser.parse_args()

    cards = _parse_npu_smi_info(_run_npu_smi(["info"]))
    npu_id, reason, notice = select_npu(cards)

    if notice:
        # stderr: 终端打屏；--format=id 的 stdout 仍只有卡号，供 sh 捕获
        print(notice, file=sys.stderr)

    if args.format == "id":
        print(npu_id)
        return 0

    print(f"Selected NPU: {npu_id}")
    print(f"Reason: {reason}")
    if notice:
        print(notice)
    print("Cards:")
    for cid in sorted(cards):
        info = cards[cid]
        print(
            f'  NPU {cid}: util={info["aicore"]:.0f}%, '
            f'HBM={info["hbm_used"]}/{info["hbm_total"]} MB, '
            f'health={info.get("health", "OK")}, '
            f'processes={info["processes"]}'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
