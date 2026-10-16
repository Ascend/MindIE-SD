#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""MindIE-SD incremental deploy to remote Ascend device.

SSH 通道**复用 remote-access 技能的统一实现**（`remote-access/scripts/ssh_helper.py`）：
同一凭据来源优先级与同一主机密钥纪律，避免两处各写一套而口径不一致。

凭据来源（优先级）：环境变量 `MINDIE_SSH_PASSWORD` > 交互式输入（不回显）> `--password`。
优先用前两者，**避免明文密码进入进程列表 / shell history**。
"""
# pylint: disable=redefined-outer-name

import argparse
import getpass
import logging
import os
import sys
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy MindIE-SD to remote Ascend device.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--password",
        default=None,
        help="SSH password（可选；优先用环境变量 MINDIE_SD_SSH_PASSWORD / MINDIE_SSH_PASSWORD 或交互输入）",
    )
    parser.add_argument("--workspace", required=True, help="远端工作目录")
    parser.add_argument("--container", required=True, help="远端容器名")
    parser.add_argument("--local-root", required=True, type=Path, help="本地源码根目录")
    parser.add_argument(
        "--allow-unknown-host",
        action="store_true",
        help="首次接入陌生主机时放开主机密钥校验（默认 RejectPolicy，未知主机直接拒绝）",
    )
    return parser.parse_args()


def load_ssh_helper():
    """按需导入 remote-access 技能的统一 SSH 通道实现。"""
    helper_dir = Path(__file__).resolve().parents[2] / "remote-access" / "scripts"
    if not (helper_dir / "ssh_helper.py").exists():
        raise SystemExit(
            "缺少 remote-access/scripts/ssh_helper.py —— 本脚本复用该技能的统一 SSH 通道；"
            "请确保 .agents/skills/ 下 env-install 与 remote-access 两个技能同时存在。"
        )
    sys.path.insert(0, str(helper_dir))
    import ssh_helper  # 跨技能复用，按需导入（非顶层导入，避免硬依赖 remote-access 的安装顺序）

    return ssh_helper


def resolve_password(cli_password):
    """凭据优先级：环境变量 > 交互输入 > --password（并提示风险）。"""
    for env_name in ("MINDIE_SD_SSH_PASSWORD", "MINDIE_SSH_PASSWORD"):
        value = os.environ.get(env_name)
        if value:
            return value
    if cli_password:
        logger.warning(
            "使用 --password 传参会把明文密码留在进程列表/shell history；"
            "建议改用环境变量 MINDIE_SSH_PASSWORD 或去掉该参数走交互输入。"
        )
        return cli_password
    return getpass.getpass("SSH password: ")


EXCLUDE_DIRS = {
    '.git',
    '__pycache__',
    'dist',
    'mindiesd.egg-info',
    '_build',
    '.pytest_cache',
    '.coverage',
}
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
    return any(rel_path.replace('\\', '/').startswith(pat) for pat in IGNORE_PATTERNS)


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

    def __init__(
        self,
        host,
        user,
        password,
        workspace,
        container,
        local_root,
        helper=None,
        allow_unknown_host=False,
        ssh=None,
        sftp=None,
    ):
        self.host = host
        self.user = user
        self.password = password
        self.workspace = workspace
        self.container = container
        self.local_root = local_root
        self.helper = helper
        self.allow_unknown_host = allow_unknown_host
        self.ssh = ssh
        self.sftp = sftp


def deploy(cfg):
    helper = cfg.helper or load_ssh_helper()
    cfg.helper = helper
    _own_connection = cfg.ssh is None
    if _own_connection:
        logger.info('Connecting SSH...')
        cfg.ssh = helper.make_ssh(cfg.host, cfg.user, cfg.password, allow_unknown_host=cfg.allow_unknown_host)

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
    # 用统一通道执行：stdout/stderr 并发读取，避免构建期大量 stderr 把 64KB 管道填满导致死锁
    code, out, err = helper.run(cfg.ssh, cmd, timeout=1800)
    for line in out.splitlines():
        if line:
            logger.info('  %s', line.rstrip())
    for line in err.splitlines():
        if line:
            logger.warning('  [err] %s', line.rstrip())
    if code != 0:
        logger.warning('  远端构建退出码 = %s（检查上面的 [err] 输出）', code)

    if _own_sftp:
        cfg.sftp.close()
    if _own_connection:
        cfg.ssh.close()
    logger.info('Done. Check for DEPLOY_SUCCESS above.')


def _exec(ssh, cmd, helper=None):
    """执行一条短命令并等待退出（复用统一通道的并发读取，防管道填满死锁）。"""
    helper = helper or load_ssh_helper()
    helper.run(ssh, cmd, timeout=30)


if __name__ == '__main__':
    args = parse_args()
    cfg = DeployConfig(
        host=args.host,
        user=args.user,
        password=resolve_password(args.password),
        workspace=args.workspace,
        container=args.container,
        local_root=args.local_root,
        helper=load_ssh_helper(),
        allow_unknown_host=args.allow_unknown_host,
    )
    deploy(cfg)
