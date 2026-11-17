# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

"""Run CPU units and real operator tests in independent Python processes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests"


def incremental_selection(args):
    """Resolve the test files affected by the current git changes."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tests.incremental_test_finder import IncrementalTestFinder  # pylint: disable=import-outside-toplevel

    finder = IncrementalTestFinder(repo_root=str(ROOT), base_branch=args.base_branch)
    tests, change_info, changes = finder.get_incremental_tests(
        since_ref=args.base_branch,
        include_staged=not args.no_staged,
        include_unstaged=not args.no_unstaged,
    )
    finder.print_test_plan(tests, change_info, changes)
    selected = [(ROOT / name).resolve() for name in sorted(tests)]
    return [path for path in selected if path.is_file() and path.is_relative_to(TESTS)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cpu_only", action="store_true")
    group.add_argument("--npu_only", action="store_true")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--base-branch", default="dev")
    parser.add_argument("--no-staged", action="store_true")
    parser.add_argument("--no-unstaged", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("test_files", nargs="*")
    args = parser.parse_args(argv)
    if args.dry_run and not args.incremental:
        parser.error("--dry-run requires --incremental")
    if args.incremental and args.test_files:
        parser.error("--incremental cannot be combined with explicit test selection")
    modes = ["CPU"] if args.cpu_only else ["NPU"] if args.npu_only else ["CPU", "NPU"]
    selected = []
    if args.test_files:
        for name in args.test_files:
            path = Path(name)
            if not path.exists():
                path = TESTS / name
            path = path.resolve()
            if not path.is_relative_to(TESTS) or not path.exists():
                parser.error(f"Test path must exist inside {TESTS}: {name}")
            selected.append(path)
    elif args.incremental:
        selected = incremental_selection(args)
        if args.dry_run:
            return 0
        if not selected:
            print("No tests affected by the current changes.", flush=True)
            return 0
    selected = sorted({file for path in selected for file in (path.rglob("test_*.py") if path.is_dir() else [path])})
    if args.test_files and not selected:
        parser.error("No tests found in the selected paths")
    expected_cpu = {path for path in TESTS.rglob("test_*.py") if not path.is_relative_to(TESTS / "ops")}
    full_cpu_selection = not selected or expected_cpu.issubset(set(selected))
    output_base = Path(
        os.environ.get("MINDIE_TEST_RESULTS", str(Path(tempfile.gettempdir()) / "mindiesd-test-results"))
    )
    output_base.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="run-", dir=output_base))
    print(f"Test artifacts: {output}", flush=True)
    coverage_files = []
    result = 0
    for mode in modes:
        paths = [path for path in selected if path.is_relative_to(TESTS / "ops") == (mode == "NPU")]
        if selected and not paths:
            if len(modes) == 1:
                parser.error(f"Selected files do not belong to the {mode} suite")
            continue
        paths = paths or [TESTS / "ops" if mode == "NPU" else TESTS]
        env = dict(os.environ, MINDIE_TEST_MODE=mode)
        if mode == "CPU":
            env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        else:
            preflight = [
                sys.executable,
                "-c",
                "import torch, torch_npu; "
                "assert torch.npu.is_available(), 'NPU suite requires a working device backend'",
            ]
            rc = subprocess.call(preflight, cwd=ROOT, env=env)
            if rc:
                result = result or rc
                continue
        data = output / f"{mode.lower()}.coverage"
        env["COVERAGE_FILE"] = str(data)
        command = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile",
            str(ROOT / ".coveragerc"),
            "-m",
            "pytest",
            f"--test-mode={mode}",
            "-q",
            f"--junitxml={output / (mode.lower() + '.xml')}",
            *map(str, paths),
        ]
        rc = subprocess.call(command, cwd=ROOT, env=env)
        result = result or rc
        coverage_files.append(data)
    if coverage_files:
        env = dict(os.environ, COVERAGE_FILE=str(output / "combined.coverage"))
        rc = subprocess.call(
            [sys.executable, "-m", "coverage", "combine", "--keep", *map(str, coverage_files)], cwd=ROOT, env=env
        )
        result = result or rc
        report = output / "coverage.json"
        rc = subprocess.call([sys.executable, "-m", "coverage", "json", "-o", str(report)], cwd=ROOT, env=env)
        result = result or rc
        if full_cpu_selection and "CPU" in modes and result == 0:
            rc = subprocess.call(
                [
                    sys.executable,
                    str(TESTS / "scripts/check_file_coverage.py"),
                    str(report),
                    "--output",
                    str(output / "file-coverage.json"),
                ],
                cwd=ROOT,
            )
            result = result or rc
    return result


if __name__ == "__main__":
    raise SystemExit(main())
