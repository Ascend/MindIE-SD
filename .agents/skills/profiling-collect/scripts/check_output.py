#!/usr/bin/env python3
"""profiling 产出「完成检查门禁」——fail-closed 校验采集产物是否够用。

为什么需要它：融合范围与收益判定（`fusion-scope-analyze`）依赖**计算单元利用率**字段；
采集时若没开 `--task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization`，
这些字段会全部是 `N/A`。把 `N/A`/缺列当成 0 会把「无数据」读成「无内存瓶颈」——
判定看似完成、结论却是假的（静默失败）。故本门禁在采集完成时**拦住**不合格产物。

用法：

```bash
python check_output.py --dir {ASCEND_PROFILER_OUTPUT}   # 校验某次采集产物
python check_output.py --selftest                       # 负样本自测（验证门禁本身有效）
```

退出码：0 = 通过；1 = 不合格（缺列 / 关键列全 `N/A` / 无可用文件）；2 = 前置条件缺失（目录不存在）。
零网络、零模型、只读（`--selftest` 只在自己的夹具目录内读写，结束即清理）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

# 执行序必需列（任一别名命中即可）
ORDER_NAME = ("Name", "Op Name", "kernel_name")
ORDER_DURATION = ("Duration(us)", "Task Duration(us)", "duration(us)")
ORDER_TYPE = ("Task Type", "kernel_type")

# 判型必需列：四个单元族（前缀 aic/aiv 均可）。
# `memory_bound` 是**可算字段**（官方公式 `mte2_ratio / max(mac_ratio, vec_ratio)`），
# 实测导出常不含它 ⇒ 不作必需列，判型时按公式现算。
UTIL_EXACT = ()
UTIL_PATTERNS = (
    re.compile(r"^(aic|aiv)_vec_ratio$"),
    re.compile(r"^(aic|aiv)_mac_ratio$"),
    re.compile(r"^(aic|aiv)_mte2_ratio$"),
    re.compile(r"^(aic|aiv)_mte3_ratio$"),
)

NA_TOKENS = {"", "n/a", "na", "none", "null", "-"}


class Finding:
    """一条检查结论。"""

    def __init__(self, level: str, message: str) -> None:
        self.level = level
        self.message = message


def _read_header(path: Path) -> list[str] | None:
    """读 CSV 表头；不可解析返回 None。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            for row in reader:
                return [cell.strip().lstrip("\ufeff") for cell in row]
    except OSError:
        return None
    return None


def _ratio_columns(header: list[str]) -> list[str]:
    """header 中命中的判型列。"""
    hits = [name for name in header if name in UTIL_EXACT]
    for pattern in UTIL_PATTERNS:
        hits.extend(name for name in header if pattern.match(name))
    return hits


def _has_values(path: Path, columns: list[str]) -> tuple[bool, int]:
    """判型列里是否存在非 N/A、非空的取值；返回 (是否有值, 检查过的行数)。"""
    checked = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                checked += 1
                for column in columns:
                    value = str(row.get(column, "") or "").strip()
                    if value.lower() not in NA_TOKENS:
                        return True, checked
                if checked >= 200:  # 采样足够判定，避免读大文件
                    break
    except OSError:
        return False, checked
    return False, checked


def check_dir(root: Path) -> list[Finding]:
    """校验目录下的采集产物；返回结论清单。"""
    findings: list[Finding] = []
    csv_files = sorted(p for p in root.rglob("*.csv") if p.is_file())
    if not csv_files:
        return [Finding("error", f"目录内没有任何 CSV 产物（{root}）")]

    headers: dict[Path, list[str]] = {}
    for path in csv_files:
        header = _read_header(path)
        if header:
            headers[path] = header

    # ① 执行序可用
    order_ok = False
    for header in headers.values():
        has_name = any(c in header for c in ORDER_NAME)
        has_dur = any(c in header for c in ORDER_DURATION)
        if has_name and has_dur and any(c in header for c in ORDER_TYPE):
            order_ok = True
            break
    if not order_ok:
        findings.append(
            Finding(
                "error",
                "执行序不可用：找不到同时含 Name（或 Op Name）/ Duration(us)（或 Task Duration(us)）/ Task Type 的文件",
            ),
        )

    # ② 单元利用率可用（判型的唯一数据源）
    util_hits: list[tuple[Path, list[str]]] = [(p, _ratio_columns(h)) for p, h in headers.items()]
    util_hits = [(p, cols) for p, cols in util_hits if cols]
    if not util_hits:
        findings.append(
            Finding(
                "error",
                "单元利用率不可用：没有任何文件含 *_vec|mac|mte2|mte3_ratio 列"
                "（请带 --task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization 重采）",
            ),
        )
    else:
        best = max(util_hits, key=lambda item: len(item[1]))
        path, columns = best
        families_ok = all(any(pattern.match(c) for c in columns) for pattern in UTIL_PATTERNS)
        if not families_ok:
            findings.append(
                Finding(
                    "error",
                    f"{path.name}: 判型列不全（需 vec/mac/mte2/mte3 四族；memory_bound 可缺、按公式现算），"
                    f"实得 {sorted(columns)}",
                ),
            )
        has_value, rows = _has_values(path, columns)
        if not has_value:
            findings.append(
                Finding(
                    "error",
                    f"{path.name}: 判型列在抽样的 {rows} 行内全为 N/A 或空 ⇒ 判采集不合格"
                    "（不可把 N/A 当 0；请按 PipeUtilization 档重采）",
                ),
            )
    return findings


def _selftest(base: Path | None = None) -> int:
    """负样本自测：合格产物必须通过，四类不合格必须被抓住。

    夹具用确定性路径（Windows 上 mkdtemp 目录 DACL 过严，嵌套建目录会被拒）。
    """
    failures: list[str] = []
    root = (base or Path(__file__).resolve().parents[1]) / ".check_output_selftest"
    shutil.rmtree(root, ignore_errors=True)

    good_order = "Name,Duration(us),Task Type\nk1,10.0,AI_VECTOR_CORE\nk2,20.0,AI_CORE\n"
    good_util = (
        "Op Name,aic_vec_ratio,aic_mac_ratio,aic_mte2_ratio,aic_mte3_ratio,memory_bound,cube_utilization(%)\n"
        "k2,0.05,0.80,0.40,0.20,0.47,72.5\n"
    )
    na_util = "Op Name,aic_vec_ratio,aic_mac_ratio,aic_mte2_ratio,aic_mte3_ratio,memory_bound\nk2,N/A,N/A,N/A,N/A,N/A\n"
    # 真实 CANN 导出常不含 memory_bound（可算字段）：四族齐全即应通过
    no_mb_util = "Op Name,aic_mac_ratio,aic_mte2_ratio,aic_mte3_ratio,aiv_vec_ratio,cube_utilization(%)\nk2,0.80,0.40,0.20,0.05,72.5\n"

    def build(name: str, files: dict[str, str]) -> Path:
        case = root / name
        case.mkdir(parents=True)
        for filename, text in files.items():
            (case / filename).write_text(text, encoding="utf-8")
        return case

    try:
        # ① 合格：必须 0 结论（含"无 memory_bound 列"的真实导出形态）
        for label, util_text in (("ok", good_util), ("ok_no_memory_bound", no_mb_util)):
            case = build(label, {"kernel_details.csv": good_order, "op_summary_0.csv": util_text})
            for item in check_dir(case):
                failures.append(f"合格产物被误报（{label}）：{item.message}")

        # ② 缺利用率文件 / ③ 利用率全 N/A / ④ 缺执行序 / ⑤ 空目录
        negatives = {
            "缺利用率档": {"kernel_details.csv": good_order},
            "利用率全 N/A": {"kernel_details.csv": good_order, "op_summary_0.csv": na_util},
            "缺执行序": {"op_summary_0.csv": good_util},
        }
        for label, files in negatives.items():
            case = build(label, files)
            if not check_dir(case):
                failures.append(f"负样本未触发：{label}")
        empty = root / "空目录"
        empty.mkdir()
        if not check_dir(empty):
            failures.append("负样本未触发：空目录")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("check_output --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        "check_output --selftest: PASS（2 个合格样例 0 误报，含「无 memory_bound 列」的真实导出形态"
        " + 4 类不合格逐类触发）"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="profiling 产出完成检查门禁")
    parser.add_argument("--dir", help="采集产物目录（如 ASCEND_PROFILER_OUTPUT）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结论")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.dir:
        print("check_output: 需要 --dir <采集产物目录>（或 --selftest）", file=sys.stderr)
        return 2

    root = Path(args.dir)
    if not root.is_dir():
        print(f"check_output: 前置条件缺失：目录不存在（{root}）", file=sys.stderr)
        return 2

    findings = check_dir(root)
    if args.json:
        print(json.dumps([{"level": f.level, "message": f.message} for f in findings], ensure_ascii=False, indent=2))
    if not findings:
        print(f"check_output: 通过（{root}：执行序 + 单元利用率齐备）")
        return 0
    for item in findings:
        print(f"check_output: {item.level}: {item.message}", file=sys.stderr)
    print(f"\ncheck_output: 采集不合格（{len(findings)} 处）——不得交下游；请按提示重采", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
