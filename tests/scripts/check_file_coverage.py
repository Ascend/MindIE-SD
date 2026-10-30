# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

"""Require branch-aware coverage for every Python source file, including unimported files."""

import argparse
import json
from pathlib import Path

# mindiesd/ops is produced by build/build_ops.sh (copy_ops) from the kernel sources; it is build output, not source.
GENERATED_DIRS = {"ops"}


def check_coverage(report, source_root, minimum=80):
    if not report.get("meta", {}).get("branch_coverage"):
        raise ValueError("Coverage must be collected with branch=True")
    source_root = Path(source_root).resolve()
    rows = []
    files = report.get("files", {})
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root).as_posix()
        if relative.split("/", 1)[0] in GENERATED_DIRS:
            continue
        suffix = f"{source_root.name}/{relative}"
        matches = [
            value
            for name, value in files.items()
            if name.replace("\\", "/") == suffix or name.replace("\\", "/").endswith("/" + suffix)
        ]
        if len(matches) != 1:
            rows.append({"file": suffix, "passed": False, "reason": "missing or ambiguous coverage record"})
            continue
        summary = matches[0]["summary"]
        total = summary["num_statements"] + summary["num_branches"]
        covered = summary["covered_lines"] + summary["covered_branches"]
        percent = 100 * covered / total if total else 100.0
        rows.append(
            {"file": suffix, "percent": percent, "passed": percent >= minimum, "covered": covered, "total": total}
        )
    if not rows:
        raise ValueError("No Python source files found")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2] / "mindiesd")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = check_coverage(json.loads(args.report.read_text(encoding="utf-8")), args.source_root)
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        if not row["passed"]:
            detail = f'{row["percent"]:.2f}%' if "percent" in row else row["reason"]
            print(f'FAIL {row["file"]}: {detail}')
    failed = sum(not row["passed"] for row in rows)
    print(f"{len(rows)} source files checked; {failed} below the per-file 80% requirement")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
