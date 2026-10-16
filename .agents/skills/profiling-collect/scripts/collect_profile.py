#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
"""
在远端昇腾设备上运行 profiling，压缩并回传结果。

部署由 env-install/scripts/deploy_to_remote.py 单独完成。
本脚本仅负责: 落开始标记 → SSH连接 → 执行 profiling → 压缩 → 下载 → 产物门禁。

用法:
    python collect_profile.py \
        --host <IP> --user <用户> --password <密码> \
        --container <容器名> --workspace <远端工作目录> \
        --script wan_infer.py --device-id 0

产物门禁（默认开）：采集**之前**在本地输出目录落一个开始标记（`.start_epoch`），
采集结束后解包产物并调用 `check_output.py --dir <解包目录> --start-marker <标记>`，
**显式三态**：`0` 通过 / `1` 有检查判失败 / `3` 无法判定（含"没给标记 / 标记缺失"）。
`3` **不得被当成通过**，也不会被静默吞掉（脚本会把三态分别打印并把退出码透传）。
`--skip-profiling` 时本脚本**不冒充**新鲜度判据（不传标记）⇒ 门禁给出 `3` 并写明原因。

退出码: `0` 采集成功且门禁通过；`1` 压缩/下载失败或门禁判失败；`3` 门禁无法判定
（含 `--no-check-output` 的显式跳过：跳过检查 ≠ 通过）；`2` 参数前置缺失。
"""
# pylint: disable=duplicate-code

import argparse
import logging
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

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

START_MARKER_NAME = ".start_epoch"
CHECK_OUTPUT_SCRIPT = Path(__file__).resolve().parent / "check_output.py"

# 门禁三态（与 check_output.py 的退出码一一对应；3 与 0 必须区别对待）
GATE_PASS, GATE_FAIL, GATE_UNKNOWN = 0, 1, 3


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run profiling on remote Ascend, collect results")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=None)
    parser.add_argument("--container", default=DEFAULT_CONTAINER)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--script-subdir",
        default=DEFAULT_SCRIPT_SUBDIR,
        help="Remote subdirectory containing the inference script",
    )
    parser.add_argument("--script", default=None, help="Inference script name on remote")
    parser.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID)
    parser.add_argument("--profile-dir", default=DEFAULT_PROFILE_DIR, help="Profile output directory name on remote")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory to save downloaded results (default: cwd)",
    )
    parser.add_argument("--compile", action="store_true", help="Enable MindieSDBackend compilation")
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip profiling run (compress+download existing results)",
    )
    parser.add_argument(
        "--start-marker",
        default=None,
        help=f"开始标记文件路径（默认 <output-dir>/{START_MARKER_NAME}）",
    )
    parser.add_argument(
        "--check-output",
        dest="check_output",
        action="store_true",
        default=True,
        help="采集后解包并跑产物门禁（默认开）",
    )
    parser.add_argument(
        "--no-check-output",
        dest="check_output",
        action="store_false",
        help="跳过产物门禁（跳过检查 ≠ 通过：退出码 3，结论记『未验证』）",
    )
    parser.add_argument("--no-extract", action="store_true", help="不自动解包（只下载 tar.gz）")
    parser.add_argument("--selftest", action="store_true", help="门禁接线自测（不需要 SSH / 设备）")
    args = parser.parse_args(argv)
    # --password / --script 对真实采集仍是必需；只有 --selftest 例外（零 SSH 自证）
    if not args.selftest and (not args.password or not args.script):
        parser.error("--password 与 --script 为必需参数（--selftest 除外）")
    return args


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
    _exit_code, out, err = _exec(ssh, cmd, timeout=60)
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


def _write_start_marker(output_dir, marker_path=None):
    """采集**之前**落开始标记（必须早于产物，否则"新鲜度"检查形同虚设）。"""
    path = Path(marker_path) if marker_path else Path(output_dir) / START_MARKER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{time.time():.6f}\n", encoding="utf-8")
    logger.info("[gate] 开始标记已写入（先于采集）: %s", path)
    return path


def _extract_archive(local_tar, output_dir):
    """解包 tar.gz 到 output_dir；返回解包根目录（供门禁按 rglob 找 CSV）。"""
    with tarfile.open(local_tar, "r:gz") as tf:
        tf.extractall(output_dir)
    return Path(output_dir)


def _run_output_gate(profile_dir, marker_path=None):
    """调 check_output.py；返回 (退出码, 输出文本)。退出码 0/1/3（2 视为无法判定的前置缺失）。"""
    cmd = [sys.executable, str(CHECK_OUTPUT_SCRIPT), "--dir", str(profile_dir)]
    if marker_path is not None:
        cmd += ["--start-marker", str(marker_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return proc.returncode, f"{proc.stdout or ''}{proc.stderr or ''}"


def _report_gate(rc, text):
    """三态显式落地：0 通过 / 1 判失败 / 3 无法判定（3 不得被当成通过）。"""
    for line in text.splitlines():
        _log_safe(line, "[gate] ")
    if rc == GATE_PASS:
        logger.info("[gate] 退出码 0 = 通过（全部检查通过且无不可判定项）")
        return GATE_PASS
    if rc == GATE_FAIL:
        logger.error("[gate] 退出码 1 = 有检查判失败 —— 产物不得交下游，请按上文提示重采")
        return GATE_FAIL
    logger.error("[gate] 退出码 %d = 无法判定（**不得当作通过**）—— 原因见上文", rc)
    return GATE_UNKNOWN


def _selftest(base=None):
    """门禁接线自测（零 SSH / 零设备）：三类夹具逐条核对三态与退出码语义。"""
    root = Path(base or Path(__file__).resolve().parents[1]) / ".collect_profile_selftest"
    if root.exists():
        for item in sorted(root.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
    root.mkdir(parents=True, exist_ok=True)

    good = "Name,Duration(us),Task Type\nk1,10.0,AI_CORE\n"
    util = "Op Name,aic_vec_ratio,aic_mac_ratio,aic_mte2_ratio,aic_mte3_ratio\nk1,0.1,0.2,0.3,0.4\n"
    results = []
    try:
        # ① 标记先写 + 产物后写 ⇒ 通过（0）
        case = root / "ok"
        case.mkdir()
        marker = _write_start_marker(case)
        (case / "kernel_details.csv").write_text(good, encoding="utf-8")
        (case / "op_summary_0.csv").write_text(util, encoding="utf-8")
        rc, text = _run_output_gate(case, marker)
        results.append(("合格：标记先于产物 + 有数据行", GATE_PASS, rc, text))

        # ② 产物时间戳早于标记 ⇒ 判失败（1）
        case = root / "stale"
        case.mkdir()
        marker = _write_start_marker(case)
        for name, body in (("kernel_details.csv", good), ("op_summary_0.csv", util)):
            target = case / name
            target.write_text(body, encoding="utf-8")
            old = time.time() - 3600
            os.utime(target, (old, old))
        rc, text = _run_output_gate(case, marker)
        results.append(("陈旧产物（早于标记）", GATE_FAIL, rc, text))

        # ③ 标记缺失（不传标记）⇒ 无法判定（3），且报告必须写明原因
        case = root / "no_marker"
        case.mkdir()
        (case / "kernel_details.csv").write_text(good, encoding="utf-8")
        (case / "op_summary_0.csv").write_text(util, encoding="utf-8")
        rc, text = _run_output_gate(case, None)
        results.append(("标记缺失（未传 --start-marker）", GATE_UNKNOWN, rc, text))

        failures = []
        for label, want, got, text in results:
            ok = got == want
            note = ""
            if label.startswith("标记缺失"):
                has_reason = "无法判定" in text and "开始标记" in text
                ok = ok and has_reason
                note = f"；报告含原因={has_reason}"
            print(f"  [{'✓' if ok else '✗'}] {label}: 期望退出码 {want}，实得 {got}{note}")
            if not ok:
                failures.append(label)
        if failures:
            print("collect_profile --selftest: FAIL")
            for item in failures:
                print(f"  - {item}")
            return 1
        print(
            "collect_profile --selftest: PASS（3 类夹具逐条命中：通过 0 / 判失败 1 / 无法判定 3；"
            "且『无法判定』报告写明原因）"
        )
        return 0
    finally:
        for item in sorted(root.rglob("*"), reverse=True):
            item.unlink() if item.is_file() else item.rmdir()
        root.rmdir()


def main(ssh=None):
    args = _parse_args()
    if args.selftest:
        return _selftest()
    output_dir = args.output_dir if args.output_dir else os.getcwd()

    logger.info("Remote:       %s (user=%s)", args.host, args.user)
    logger.info("Container:    %s", args.container)
    logger.info("Workspace:    %s/%s", args.workspace, args.project)
    logger.info("Script:       %s/%s", args.script_subdir, args.script)
    logger.info("Output dir:   %s", output_dir)
    logger.info()

    os.makedirs(output_dir, exist_ok=True)

    # 开始标记必须在**采集之前**落下（否则"新鲜度"检查形同虚设）
    marker = None if args.skip_profiling else _write_start_marker(output_dir, args.start_marker)

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
            exit_code, _out, _err = _run_remote_profiling(ssh, args)
            elapsed = time.time() - t_start
            logger.info("Profiling finished in %.0fs (exit_code=%d)", elapsed, exit_code)
            if exit_code != 0:
                logger.warning("[profiling] non-zero exit code, continuing to collect results")
        else:
            logger.info()
            logger.info("Step 2/3: Skipping profiling (--skip-profiling)")
            logger.info("Compressing and downloading existing profile data...")
            logger.info("[gate] 本次未采集 ⇒ 不传开始标记：新鲜度将报『无法判定』（不是通过）")

        logger.info()
        logger.info("=" * 60)
        logger.info("Step 3/3: Compressing and downloading results...")
        logger.info("=" * 60)
        tar_remote = _compress_remote(ssh, args, remote_project_dir)
        if tar_remote is None:
            logger.error("[compress] FAILED: profile dir not found on remote")
            return 1

        local_tar = _download_result(sftp, tar_remote, output_dir)
        logger.info()
        logger.info("Done. Profile archive saved to: %s", local_tar)
        logger.info("Unpack with: tar xzf %s", os.path.basename(local_tar))

    finally:
        sftp.close()
        if _own_connection:
            ssh.close()

    # ---- 产物门禁（三态显式落地；跳过检查 ≠ 通过）
    if not args.check_output:
        logger.error(
            "[gate] 已显式跳过产物门禁（--no-check-output）⇒ 结论属『未验证』，**不得当作通过**"
            "（退出码 3）；如需下载成功与否，请自行确认 tar 已落地"
        )
        return GATE_UNKNOWN

    if args.no_extract:
        logger.error("[gate] --no-extract ⇒ 无解包目录，无法判定产物是否新鲜/完整（退出码 3）")
        return GATE_UNKNOWN

    logger.info()
    logger.info("=" * 60)
    logger.info("Step 4/4: 产物门禁（check_output.py）...")
    logger.info("=" * 60)
    try:
        check_dir = _extract_archive(local_tar, output_dir)
    except (tarfile.TarError, OSError) as exc:
        logger.error("[gate] 解包失败（%s）⇒ 无法判定（退出码 3）", exc)
        return GATE_UNKNOWN

    rc, text = _run_output_gate(check_dir, marker)
    return _report_gate(rc, text)


if __name__ == "__main__":
    raise SystemExit(main())
