#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""MindIE-SD incremental deploy to remote Ascend device."""
# pylint: disable=redefined-outer-name

import argparse
import logging
import os
from pathlib import Path

import paramiko

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy MindIE-SD to remote Ascend device.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--workspace", required=True, help="远端工作目录")
    parser.add_argument("--container", required=True, help="远端容器名")
    parser.add_argument("--local-root", required=True, type=Path, help="本地源码根目录")
    return parser.parse_args()


EXCLUDE_DIRS = {'.git', '__pycache__', 'dist', 'mindiesd.egg-info', '_build', '.pytest_cache', '.coverage'}
IGNORE_PATTERNS = [
    'build/build/',
    'build/vendors/',
    'build/output/',
    'build/custom_project_tik/',
    'mindiesd/ops/',
    'mindiesd/plugin/',
    'docs/_build/',
]


def _is_text_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    return ext in (".py", ".sh", ".json", ".txt", ".md", ".yaml", ".yml", ".cfg")


def should_skip(rel_path):
    parts = rel_path.replace('\\', '/').split('/')
    for part in parts:
        if part in EXCLUDE_DIRS:
            return True
    for pat in IGNORE_PATTERNS:
        if rel_path.replace('\\', '/').startswith(pat):
            return True
    return False


def collect_local_files(local_root):
    files = {}
    for root, dirs, filenames in os.walk(local_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        for fn in filenames:
            fpath = Path(root) / fn
            rel = str(fpath.relative_to(local_root))
            if should_skip(rel):
                continue
            files[rel] = {'path': str(fpath), 'size': fpath.stat().st_size}
    return files


class DeployConfig:
    """Configuration for remote deployment."""

    def __init__(self, host, user, password, workspace, container, local_root, ssh=None, sftp=None):
        self.host = host
        self.user = user
        self.password = password
        self.workspace = workspace
        self.container = container
        self.local_root = local_root
        self.ssh = ssh
        self.sftp = sftp


def deploy(cfg):
    _own_connection = cfg.ssh is None
    if _own_connection:
        logger.info('Connecting SSH...')
        cfg.ssh = paramiko.SSHClient()
        cfg.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        cfg.ssh.connect(cfg.host, username=cfg.user, password=cfg.password, timeout=30)

    if cfg.sftp is None:
        cfg.sftp = cfg.ssh.open_sftp()
        _own_sftp = True
    else:
        _own_sftp = False

    logger.info('Collecting local files...')
    local_files = collect_local_files(cfg.local_root)
    logger.info('  %d local files', len(local_files))

    remote_project = f'{cfg.workspace}/{cfg.local_root.name}'

    # Ensure remote directory exists
    try:
        cfg.sftp.stat(remote_project)
    except FileNotFoundError:
        _exec(cfg.ssh, f'mkdir -p {remote_project}')

    # Incremental transfer: only upload changed files
    logger.info('Transferring changed/new files...')
    uploaded = 0
    skipped = 0
    for rel_path, info in sorted(local_files.items()):
        remote_path = f'{remote_project}/{rel_path}'.replace('\\', '/')
        try:
            rstat = cfg.sftp.stat(remote_path)
            if rstat.st_size == info['size']:
                skipped += 1
                continue
        except FileNotFoundError:
            pass

        # Ensure parent dir exists on remote
        remote_dir = os.path.dirname(remote_path)
        try:
            cfg.sftp.stat(remote_dir)
        except FileNotFoundError:
            _exec(cfg.ssh, f'mkdir -p {remote_dir}')

        if _is_text_file(rel_path):
            with open(info['path'], 'rb') as fh:
                data = fh.read()
            data = data.replace(b'\r\n', b'\n')
            from io import BytesIO

            cfg.sftp.putfo(BytesIO(data), remote_path)
        else:
            cfg.sftp.put(info['path'], remote_path)
        uploaded += 1
        if uploaded % 50 == 0:
            logger.info('  uploaded %d files...', uploaded)

    cfg.sftp.close()
    logger.info('  %d uploaded, %d unchanged', uploaded, skipped)

    logger.info("")
    logger.info('Building inside container...')
    build_cmd = (
        f'cd {cfg.workspace} && '
        f'source /usr/local/Ascend/ascend-toolkit/set_env.sh && '
        f'cd MindIE-SD && '
        f'pip install build wheel -q && '
        f'python setup.py build_py && '
        f'pip install -e . && '
        f'echo DEPLOY_SUCCESS'
    )
    cmd = f'docker exec {cfg.container} bash -lc "{build_cmd}"'
    stdin, stdout, stderr = cfg.ssh.exec_command(cmd, timeout=1800)
    for line in iter(stdout.readline, ''):
        if line:
            logger.info('  %s', line.rstrip())
    for line in iter(stderr.readline, ''):
        if line:
            logger.warning('  [err] %s', line.rstrip())

    if _own_sftp:
        cfg.sftp.close()
    if _own_connection:
        cfg.ssh.close()
    logger.info('Done. Check for DEPLOY_SUCCESS above.')


def _exec(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    stdout.channel.recv_exit_status()


if __name__ == '__main__':
    args = parse_args()
    cfg = DeployConfig(
        host=args.host,
        user=args.user,
        password=args.password,
        workspace=args.workspace,
        container=args.container,
        local_root=args.local_root,
    )
    deploy(cfg)
