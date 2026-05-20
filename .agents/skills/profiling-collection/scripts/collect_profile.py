#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
在远端昇腾设备上运行 profiling，压缩并回传结果。

部署由 ascend-deploy/scripts/deploy_to_remote.py 单独完成。
本脚本仅负责: SSH连接 → 执行 profiling → 压缩 → 下载。

用法:
    python collect_profile.py \
        --host <IP> --user <用户> --password <密码> \
        --container <容器名> --workspace /home/... \
        --script wan_infer.py --device-id 0
"""
# pylint: disable=duplicate-code

import argparse
import logging
import os
import sys
import time

import paramiko

logger = logging.getLogger(__name__)


DEFAULT_HOST = ""
DEFAULT_USER = ""
DEFAULT_CONTAINER = ""
DEFAULT_WORKSPACE = ""
DEFAULT_PROJECT = "MindIE-SD"
DEFAULT_DEVICE_ID = 0
DEFAULT_PROFILE_DIR = "profile_l1"
DEFAULT_SCRIPT_SUBDIR = "examples/dummy_run"


def _parse_args():
    parser = argparse.ArgumentParser(description="Run profiling on remote Ascend, collect results")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", required=True)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--script-subdir", default=DEFAULT_SCRIPT_SUBDIR, help="Remote subdirectory containing the inference script"
    )
    parser.add_argument("--script", required=True, help="Inference script name on remote")
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR, help="Profile output directory name on remote")
    parser.add_argument("--output-dir", default=None, help="Local directory to save downloaded results (default: cwd)")
    parser.add_argument("--compile", action="store_true", help="Enable MindieSDBackend compilation")
    parser.add_argument(
        "--warmup-steps", type=int, default=5, help="Number of warmup steps before profiling (default: 5)"
    )
    parser.add_argument(
        "--skip-profiling", action="store_true", help="Skip profiling run (compress+download existing results)"
    )
    return parser.parse_args()


def _exec(ssh, cmd, timeout=30):
    _stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


def _log_safe(text, label=""):
    try:
        logger.info("%s%s", label, text)
    except UnicodeEncodeError:
        safe = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        logger.info("%s%s", label, safe[:2000])


def _run_remote_profiling(ssh, args):
    compile_flag = " --compile" if args.compile else ""
    script_path = f"{args.script_subdir}/{args.script}"
    cmd = (
        f'docker exec {args.container} bash -lc "'
        f'source /usr/local/Ascend/ascend-toolkit/set_env.sh && '
        f'cd {args.workspace}/{args.project} && '
        f'python {script_path} '
        f'--device_id {args.device_id} --profile{compile_flag}'
        f'"'
    )
    logger.info("[profiling] Running: %s", cmd)
    exit_code, out, err = _exec(ssh, cmd, timeout=1800)
    _log_safe(out)
    if err:
        safe_err = err.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        logger.warning("[profiling stderr] %s", safe_err[:2000])
    return exit_code, out, err


def _compress_remote(ssh, args, remote_project_dir):
    cmd = (
        f'docker exec {args.container} bash -lc "'
        f'cd {remote_project_dir} && '
        f'if [ -d {args.profile_dir} ]; then '
        f'tar czf {args.profile_dir}.tar.gz {args.profile_dir}/ && '
        f'echo COMPRESS_OK; '
        f'else echo COMPRESS_FAIL_DIR_NOT_FOUND; fi'
        f'"'
    )
    logger.info("[compress] Running: cd %s && tar czf ...", remote_project_dir)
    exit_code, out, err = _exec(ssh, cmd, timeout=60)
    _log_safe(out)
    if err:
        _log_safe(err, "[compress stderr] ")
    if "COMPRESS_FAIL" in out:
        logger.error("[compress] profile directory not found")
        return None
    return f"{remote_project_dir}/{args.profile_dir}.tar.gz"


def _download_result(sftp, remote_tar_path, local_output_dir):
    local_tar_path = os.path.join(local_output_dir, os.path.basename(remote_tar_path))
    logger.info("[download] %s -> %s", remote_tar_path, local_tar_path)
    sftp.get(remote_tar_path, local_tar_path)
    fsize = os.path.getsize(local_tar_path)
    logger.info("[download] OK, %d bytes", fsize)
    return local_tar_path


def main(ssh=None):
    args = _parse_args()
    output_dir = args.output_dir if args.output_dir else os.getcwd()

    logger.info("Remote:       %s (user=%s)", args.host, args.user)
    logger.info("Container:    %s", args.container)
    logger.info("Workspace:    %s/%s", args.workspace, args.project)
    logger.info("Script:       %s/%s", args.script_subdir, args.script)
    logger.info("Output dir:   %s", output_dir)
    logger.info()

    os.makedirs(output_dir, exist_ok=True)

    remote_project_dir = f"{args.workspace}/{args.project}"

    _own_connection = ssh is None
    if _own_connection:
        logger.info("=" * 60)
        logger.info("Step 1/3: Connecting SSH...")
        logger.info("=" * 60)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(args.host, username=args.user, password=args.password, timeout=30)
        logger.info("SSH connected.")
    else:
        logger.info("Using existing SSH connection.")

    sftp = ssh.open_sftp()

    try:
        if not args.skip_profiling:
            logger.info()
            logger.info("=" * 60)
            logger.info("Step 2/3: Running profiling on remote...")
            logger.info("=" * 60)
            t_start = time.time()
            exit_code, out, err = _run_remote_profiling(ssh, args)
            elapsed = time.time() - t_start
            logger.info("Profiling finished in %.0fs (exit_code=%d)", elapsed, exit_code)
            if exit_code != 0:
                logger.warning("[profiling] non-zero exit code, continuing to collect results")
        else:
            logger.info()
            logger.info("Step 2/3: Skipping profiling (--skip-profiling)")
            logger.info("Compressing and downloading existing profile data...")

        logger.info()
        logger.info("=" * 60)
        logger.info("Step 3/3: Compressing and downloading results...")
        logger.info("=" * 60)
        tar_remote = _compress_remote(ssh, args, remote_project_dir)
        if tar_remote is None:
            logger.error("[compress] FAILED: profile dir not found on remote")
            sys.exit(1)

        local_tar = _download_result(sftp, tar_remote, output_dir)
        logger.info()
        logger.info("Done. Profile archive saved to: %s", local_tar)
        logger.info("Unpack with: tar xzf %s", os.path.basename(local_tar))

    finally:
        sftp.close()
        if _own_connection:
            ssh.close()


if __name__ == "__main__":
    main()
