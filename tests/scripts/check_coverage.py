#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Enforce coverage thresholds for newly added source files in CI."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageTarget:
    name: str
    path: str
    line_threshold: float
    branch_threshold: float


LINE_THRESHOLD = 60.0
BRANCH_THRESHOLD = 30.0

CONDITION_RE = re.compile(r"\((\d+)/(\d+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check added-file coverage thresholds.")
    parser.add_argument("--xml", required=True, help="Path to a coverage.xml file.")
    parser.add_argument(
        "--changed-files",
        required=True,
        help="Path to a file containing changed files, one per line.",
    )
    parser.add_argument(
        "--target-branch",
        required=True,
        help="Target branch name used to decide whether a changed file is newly added.",
    )
    return parser.parse_args()


def normalise_path(filename: str) -> str:
    parts = Path(filename.replace("\\", "/")).parts
    if "mindiesd" in parts:
        return "/".join(parts[parts.index("mindiesd") :])
    return filename.replace("\\", "/").lstrip("./")


def parse_branch_coverage(line_elem: ET.Element) -> tuple[int, int]:
    condition = line_elem.attrib.get("condition-coverage", "")
    match = CONDITION_RE.search(condition)
    if match is None:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def collect_file_metrics(root: ET.Element, target: CoverageTarget) -> tuple[int, int, int, int]:
    covered_lines = 0
    total_lines = 0
    covered_branches = 0
    total_branches = 0

    for class_elem in root.findall(".//class"):
        filename = normalise_path(class_elem.attrib.get("filename", ""))
        if filename != target.path:
            continue

        for line_elem in class_elem.findall("./lines/line"):
            total_lines += 1
            if int(line_elem.attrib.get("hits", "0")) > 0:
                covered_lines += 1

            if line_elem.attrib.get("branch") == "true":
                branch_covered, branch_total = parse_branch_coverage(line_elem)
                covered_branches += branch_covered
                total_branches += branch_total

    return (covered_lines, total_lines, covered_branches, total_branches)


def rate(covered: int, total: int) -> float:
    if total == 0:
        return 100.0
    return covered * 100.0 / total


def is_added_file(file_path: str, target_branch: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"origin/{target_branch}:{file_path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode != 0 and Path(file_path).is_file()


def load_added_targets(path: Path, target_branch: str) -> list[CoverageTarget]:
    if not path.is_file():
        return []

    targets: list[CoverageTarget] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        file_path = normalise_path(raw_line.strip())
        if not file_path.startswith("mindiesd/") or not file_path.endswith(".py"):
            continue
        if not is_added_file(file_path, target_branch):
            continue
        targets.append(CoverageTarget(file_path, file_path, LINE_THRESHOLD, BRANCH_THRESHOLD))
    return targets


def main() -> int:
    args = parse_args()
    xml_path = Path(args.xml)
    changed_files_path = Path(args.changed_files)
    if not xml_path.is_file():
        print(f"Coverage report not found: {xml_path}", file=sys.stderr)
        return 1

    targets = load_added_targets(changed_files_path, args.target_branch)
    if not targets:
        print("Coverage gate skipped: no added Python source files under mindiesd/.")
        return 0

    root = ET.parse(xml_path).getroot()
    failures: list[str] = []

    print("Coverage gate summary:")
    for target in targets:
        covered_lines, total_lines, covered_branches, total_branches = collect_file_metrics(
            root, target
        )
        line_rate = rate(covered_lines, total_lines)
        branch_rate = rate(covered_branches, total_branches)
        print(
            f"- {target.name}: "
            f"line {line_rate:.2f}% ({covered_lines}/{total_lines}), "
            f"branch {branch_rate:.2f}% ({covered_branches}/{total_branches})"
        )

        if line_rate < target.line_threshold:
            failures.append(
                f"{target.name} line coverage {line_rate:.2f}% is below {target.line_threshold:.2f}%."
            )
        if branch_rate < target.branch_threshold:
            failures.append(
                f"{target.name} branch coverage {branch_rate:.2f}% "
                f"is below {target.branch_threshold:.2f}%."
            )

    if failures:
        print("\nCoverage gate failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nCoverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
