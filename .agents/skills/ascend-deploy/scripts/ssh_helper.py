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

"""Reusable SSH runner for remote MindIE-SD operations (single-connection reuse).

与 ascend-deploy §1 连接复用原则一致：一个 SSH 连接执行多次命令，
避免远端 MaxStartups 限制导致的拒绝访问。

Usage:
    python ssh_helper.py --host <IP> --user <用户名> --container <容器名> \
        --cmd "npu-smi info -l"

密码来源（优先级）: 环境变量 MINDIE_SSH_PASSWORD > 交互式输入（不回显）> --password。
优先使用环境变量或交互输入，避免明文密码出现在进程列表 / shell history。
"""

import argparse
import getpass
import os
import shlex
import sys
import threading

import paramiko


def make_ssh(host, user, password, timeout=30):
    """Create a single SSH connection (reuse across commands).

    不设置 AutoAddPolicy：paramiko 默认 MissingHostKeyPolicy 为 RejectPolicy，
    未知主机密钥直接拒绝（防中间人）；已知主机（SSHClient 自动加载系统
    known_hosts）正常连接。
    """
    ssh = paramiko.SSHClient()
    ssh.connect(host, username=user, password=password, timeout=timeout)
    return ssh


def run(ssh, cmd, timeout=600):
    """Run one command on the reused connection; return (exit_code, stdout, stderr).

    stdout / stderr 用两个线程并发读取，避免 paramiko 通道 64KB 管道缓冲填满时
    串行读取造成死锁（远端大量 stderr 输出的编译/安装场景）。
    """
    _stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    chunks = {"out": [], "err": []}

    def _pump(src, key):
        try:
            while True:
                chunk = src.read(4096)
                if not chunk:
                    break
                chunks[key].append(chunk)
        except Exception:  # noqa: BLE001 - channel EOF/超时边缘，忽略
            pass

    t_out = threading.Thread(target=_pump, args=(stdout, "out"))
    t_err = threading.Thread(target=_pump, args=(stderr, "err"))
    t_out.start()
    t_err.start()
    code = stdout.channel.recv_exit_status()
    t_out.join()
    t_err.join()
    return (
        code,
        b"".join(chunks["out"]).decode("utf-8", "replace"),
        b"".join(chunks["err"]).decode("utf-8", "replace"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run commands on remote Ascend host with single-connection reuse"
    )
    parser.add_argument("--host", required=True, help="remote host IP")
    parser.add_argument("--user", required=True, help="SSH username")
    parser.add_argument(
        "--password",
        default=None,
        help="SSH password（可选；优先用环境变量 MINDIE_SSH_PASSWORD 或交互输入，"
             "避免明文进进程列表/history）",
    )
    parser.add_argument("--container", default=None, help="docker container name (run inside it)")
    parser.add_argument("--cmd", required=True, help="command to run on remote")
    parser.add_argument("--timeout", type=int, default=600, help="command timeout (s)")
    args = parser.parse_args()

    password = args.password or os.environ.get("MINDIE_SSH_PASSWORD")
    if password is None:
        password = getpass.getpass("SSH password: ")

    ssh = make_ssh(args.host, args.user, password)
    try:
        if args.container:
            # shlex.quote 分别转义容器名与命令：远端 shell 与内层 bash 各只解一次
            # 引号，命令原样到达（含双引号/$/反引号的安全传递）
            cmd = f"docker exec {shlex.quote(args.container)} bash -lc {shlex.quote(args.cmd)}"
        else:
            cmd = args.cmd
        code, out, err = run(ssh, cmd, timeout=args.timeout)
        if out:
            print(out)
        if err:
            print(f"[stderr]\n{err}", file=sys.stderr)
        print(f"[exit code: {code}]")
        return code
    finally:
        ssh.close()


if __name__ == "__main__":
    sys.exit(main())
