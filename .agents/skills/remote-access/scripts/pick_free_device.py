#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""Detect the Ascend NPU card with the lowest HBM usage.

Usage as standalone:
    python pick_free_device.py --host <远端IP> --user <用户名> --password <密码>

Usage as import:
    from pick_free_device import pick_free_device
    ssh = paramiko.SSHClient()
    ssh.connect(...)
    dev, usage = pick_free_device(ssh, container="<容器名>")
"""

import argparse
import sys
import paramiko

DEFAULT_HOST = ""
DEFAULT_USER = ""
DEFAULT_CONTAINER = ""


def pick_free_device(ssh_client, container="", num_cards=8):
    """Return (device_id, hbm_usage_pct) with the lowest HBM usage.

    Args:
        ssh_client: Connected paramiko SSHClient.
        container: Docker container name.
        num_cards: Number of NPU cards to scan.

    Returns:
        Tuple (device_id, hbm_usage_pct).
    """
    cmd = f"docker exec {container} bash -lc 'npu-smi info -t usages -i 0-{num_cards - 1}'"
    _stdin, stdout, stderr = ssh_client.exec_command(cmd, timeout=15)
    output = stdout.read().decode("utf-8", errors="replace")

    best_dev, best_usage = 0, 100
    for line in output.split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            dev_id = int(parts[0])
            try:
                usage_val = float(parts[-1])
                if usage_val < best_usage:
                    best_dev, best_usage = dev_id, usage_val
            except ValueError:
                continue

    return best_dev, best_usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect the NPU card with the lowest HBM usage")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", required=True)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--num-cards", type=int, default=8)
    args = parser.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.host, username=args.user, password=args.password, timeout=30)

    dev, usage = pick_free_device(ssh, container=args.container, num_cards=args.num_cards)
    ssh.close()

    sys.stdout.write(f"{dev} {usage:.0f}\n")
