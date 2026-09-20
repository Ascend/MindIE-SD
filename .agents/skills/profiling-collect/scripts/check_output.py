#!/usr/bin/env python3
"""profiling 产出「完成检查门禁」—— fail-closed 校验采集产物是否够用。

为什么需要它：融合范围与收益判定（`fusion-scope-analyze`）依赖**计算单元利用率**字段；
采集时若没开 `--task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization`，
这些字段会全部是 `N/A`。把 `N/A`/缺列当成 0 会把「无数据」读成「无内存瓶颈」——
判定看似完成、结论却是假的（静默失败）。故本门禁在采集完成时**拦住**不合格产物。

第二条静默失败：**「命令返回 0 / 日志里有 `done`·`OK`」不等于产物存在**。本环境实测
进程内文本导出会**从不执行且异常被吞**（`analyse_profiling_data()` 被 `@no_exception_func()`
装饰）⇒ 因此本门禁另加两项**可判定性显式**检查：

* **新鲜度**：产物时间戳必须**晚于本次运行的开始标记**（`--start-marker <文件>` 或 `--start-epoch <秒>`）。
  **没给标记 ⇒ 报「无法判定」，绝不报「通过」**（把不可判定当通过正是要防的事）；
* **行数 > 0**：声明为数据表的产物（`kernel_details*.csv` / `op_summary*.csv`）至少要有一行**数据行**，
  **只有表头的文件判失败**。

用法：

```bash
python check_output.py --dir {ASCEND_PROFILER_OUTPUT} \
    --start-marker {run_dir}/.start_epoch          # 推荐：带运行开始标记
python check_output.py --dir {ASCEND_PROFILER_OUTPUT} --start-epoch 1737000000.0
python check_output.py --dir {ASCEND_PROFILER_OUTPUT}   # 未给标记 ⇒ 新鲜度报「无法判定」（退出码 3）
python check_output.py --selftest                       # 负样本自测（验证门禁本身有效）
```

**每一项检查各自独立报结论**（不合并成一个布尔）；报告逐项给出 `通过 / 失败 / 无法判定`。

退出码：`0` = 全部通过；`1` = 有检查**判失败**；`2` = 前置条件缺失（目录不存在 / 参数缺失）；
`3` = 无失败项但存在**无法判定**（当前即"未提供开始标记"或标记文件不存在）—— **3 与 0 必须被区分对待**。

零网络、零模型、只读（`--selftest` 只在自己的夹具目录内读写，结束即清理）。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from pathlib import Path

# 执行序必需列（任一别名命中即可）
# 注：新版 CANN 把执行序拆到两个文件——`kernel_details.csv`（Name + Duration(us)，无 Task Type）
# 与 `task_time.csv`（kernel_name + kernel_type + task_time(us)）——故 `task_time(us)` 必须计入
# duration 别名，否则两文件都不满足"三列同文件"，门禁会对合格产物误报（2026-09 实测）。
ORDER_NAME = ("Name", "Op Name", "kernel_name")
ORDER_DURATION = ("Duration(us)", "Task Duration(us)", "duration(us)", "task_time(us)")
# `Accelerator Core` = torch_npu `kernel_details.csv` 对 `Task Type` 的改名
# （实测：48 列 / 42 列同名 / 首行数据逐字节相同，只有 6 个表头被投影改名）。
# 少了它，**完全合格的 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 会被判失败**。
ORDER_TYPE = ("Task Type", "Accelerator Core", "kernel_type")

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

# 「数据表」产物名（行数检查只针对它们；其余 CSV 不判行数）
DATA_TABLE_PATTERNS = (
    re.compile(r"^kernel_details.*\.csv$", re.IGNORECASE),
    re.compile(r"^op_summary.*\.csv$", re.IGNORECASE),
)

NA_TOKENS = {"", "n/a", "na", "none", "null", "-"}

PASS, FAIL, UNKNOWN = "通过", "失败", "无法判定"


class Outcome:
    """一项检查的独立结论（三个检查各一条，不合并）。"""

    __slots__ = ("messages", "name", "verdict")

    def __init__(self, name: str, verdict: str, messages: list[str] | None = None) -> None:
        self.name = name
        self.verdict = verdict
        self.messages = messages or []

    def as_dict(self) -> dict[str, object]:
        return {"check": self.name, "verdict": self.verdict, "messages": self.messages}


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


def _count_data_rows(path: Path) -> int | None:
    """数据行数（不含表头、跳过全空行）；不可读返回 None。"""
    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            for index, row in enumerate(csv.reader(fh)):
                if index == 0:  # 表头不算数据行
                    continue
                if any(cell.strip() for cell in row):
                    count += 1
    except OSError:
        return None
    return count


def resolve_start_time(marker: str | None, epoch: float | None) -> tuple[float | None, str, str | None]:
    """解析"本次运行开始时刻"。

    返回 `(开始时刻, 来源说明, 不可判定原因)`；开始时刻为 None 时**必须**带原因，
    调用方据此报「无法判定」而不是「通过」。
    """
    if epoch is not None:
        return float(epoch), f"--start-epoch={epoch}", None
    if marker is None:
        return (
            None,
            "（未提供开始标记）",
            (
                "新鲜度无法判定：未提供开始标记。请传 --start-marker <运行开始时写入的标记文件> "
                "或 --start-epoch <秒>；**不可判定不等于通过**。"
            ),
        )
    path = Path(marker)
    if not path.exists():
        return None, f"（标记文件不存在：{marker}）", f"新鲜度无法判定：开始标记文件不存在（{marker}）。"
    return path.stat().st_mtime, f"{marker} 的 mtime", None


def check_dir(
    root: Path,
    start_time: float | None = None,
    start_desc: str = "",
    start_unknown_reason: str | None = None,
    tolerance: float = 0.0,
) -> list[Outcome]:
    """校验目录下的采集产物；返回**逐项独立**的结论清单（顺序固定）。"""
    outcomes: list[Outcome] = []
    csv_files = sorted(p for p in root.rglob("*.csv") if p.is_file())

    # ① 产物存在且可读
    headers: dict[Path, list[str]] = {}
    unreadable: list[str] = []
    for path in csv_files:
        header = _read_header(path)
        if header:
            headers[path] = header
        else:
            unreadable.append(path.name)
    if not csv_files:
        outcomes.append(Outcome("产物存在", FAIL, [f"目录内没有任何 CSV 产物（{root}）"]))
    elif unreadable:
        outcomes.append(Outcome("产物存在", FAIL, [f"{len(unreadable)} 个 CSV 不可读：{sorted(unreadable)[:5]}"]))
    else:
        outcomes.append(Outcome("产物存在", PASS, [f"{len(csv_files)} 个 CSV 可读"]))

    # ② 执行序可用
    order_ok = False
    for header in headers.values():
        has_name = any(c in header for c in ORDER_NAME)
        has_dur = any(c in header for c in ORDER_DURATION)
        if has_name and has_dur and any(c in header for c in ORDER_TYPE):
            order_ok = True
            break
    outcomes.append(
        Outcome(
            "执行序可用",
            PASS if order_ok else FAIL,
            []
            if order_ok
            else [
                (
                    "执行序不可用：找不到同时含 Name（或 Op Name）/ Duration(us)（或 Task Duration(us)）/ "
                    "Task Type（或 Accelerator Core）的文件"
                )
            ],
        )
    )

    # ③ 单元利用率可用（判型的唯一数据源）
    util_hits: list[tuple[Path, list[str]]] = [(p, _ratio_columns(h)) for p, h in headers.items()]
    util_hits = [(p, cols) for p, cols in util_hits if cols]
    if not util_hits:
        outcomes.append(
            Outcome(
                "单元利用率可用",
                FAIL,
                [
                    (
                        "单元利用率不可用：没有任何文件含 *_vec|mac|mte2|mte3_ratio 列"
                        "（请带 --task-time=l1 --aic-mode=task-based --aic-metrics=PipeUtilization 重采）"
                    )
                ],
            )
        )
        outcomes.append(Outcome("判型列有值（抽样）", UNKNOWN, ["无利用率文件 ⇒ 本项无从判定（见上一项）"]))
    else:
        best = max(util_hits, key=lambda item: len(item[1]))
        path, columns = best
        families_ok = all(any(pattern.match(c) for c in columns) for pattern in UTIL_PATTERNS)
        util_msgs: list[str] = []
        if not families_ok:
            util_msgs.append(
                f"{path.name}: 判型列不全（需 vec/mac/mte2/mte3 四族；memory_bound 可缺、按公式现算），"
                f"实得 {sorted(columns)}"
            )
        outcomes.append(Outcome("单元利用率可用", FAIL if util_msgs else PASS, util_msgs))

        # ④ 判型列有值（抽样；不与上一项合并）
        has_value, rows = _has_values(path, columns)
        outcomes.append(
            Outcome(
                "判型列有值（抽样）",
                PASS if has_value else FAIL,
                []
                if has_value
                else [
                    (
                        f"{path.name}: 判型列在抽样的 {rows} 行内全为 N/A 或空 ⇒ 判采集不合格"
                        "（不可把 N/A 当 0；请按 PipeUtilization 档重采）"
                    )
                ],
            )
        )

    # ⑤ 数据表行数 > 0（只有表头的文件判失败）
    tables = [p for p in csv_files if any(pat.match(p.name) for pat in DATA_TABLE_PATTERNS)]
    if not tables:
        outcomes.append(
            Outcome(
                "数据表行数 > 0",
                FAIL,
                ["目录内没有任何声明的数据表（kernel_details*.csv / op_summary*.csv）⇒ 产物不完整"],
            )
        )
    else:
        row_msgs: list[str] = []
        for path in tables:
            count = _count_data_rows(path)
            if count is None:
                row_msgs.append(f"{path.name}: 不可读 ⇒ 行数无从判定")
            elif count == 0:
                row_msgs.append(
                    f"{path.name}: **只有表头、零数据行** ⇒ 判失败"
                    "（导出/解析可能报成功但什么都没写——见 SKILL.md「导出类操作一律『无新文件即失败』」）"
                )
            else:
                row_msgs.append(f"{path.name}: {count} 行数据 ✓")
        failed = any("判失败" in m or "不可读" in m for m in row_msgs)
        outcomes.append(Outcome("数据表行数 > 0", FAIL if failed else PASS, [] if not failed else row_msgs))

    # ⑥ 新鲜度（不可判定必须显式报出）
    if start_time is None:
        reason = start_unknown_reason or "新鲜度无法判定：未提供可用的开始标记。"
        outcomes.append(
            Outcome(
                "产物新鲜度",
                UNKNOWN,
                [
                    (
                        f"{reason} 判据 = 产物时间戳晚于本次运行开始标记。"
                        "**不可判定不等于通过**；也不得以「命令返回 0 / 日志里有 done·OK」当通过依据。"
                    )
                ],
            )
        )
    else:
        stale = [p.name for p in csv_files if p.stat().st_mtime + tolerance < start_time]
        if stale:
            outcomes.append(
                Outcome(
                    "产物新鲜度",
                    FAIL,
                    [
                        (
                            f"{len(stale)} 个产物的时间戳早于开始标记（{start_desc}）⇒ 判失败："
                            f"本次运行**没有重新产出**它们（陈旧产物不得当本次结果）：{sorted(stale)[:5]}"
                        )
                    ],
                )
            )
        else:
            outcomes.append(Outcome("产物新鲜度", PASS, [f"{len(csv_files)} 个产物均晚于开始标记（{start_desc}）"]))

    return outcomes


def _verdict_of(outcomes: list[Outcome]) -> str:
    if any(o.verdict == FAIL for o in outcomes):
        return FAIL
    if any(o.verdict == UNKNOWN for o in outcomes):
        return UNKNOWN
    return PASS


def _exit_code(verdict: str) -> int:
    return {PASS: 0, FAIL: 1, UNKNOWN: 3}[verdict]


# ------------------------------------------------------------------ 自测


def _touch(path: Path, epoch: float) -> None:
    os.utime(path, (epoch, epoch))


def _selftest(base: Path | None = None) -> int:
    """负样本自测：每类夹具**逐条报结论**，并校验退出码语义（无法判定 ≠ 通过）。

    夹具用确定性路径（Windows 上 mkdtemp 目录 DACL 过严，嵌套建目录会被拒），
    时间戳用 `os.utime` 固定，避免依赖真实时钟。
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
    header_only = "Name,Duration(us),Task Type\n"  # 只有表头、零数据行

    t0 = 1_700_000_000.0  # 运行开始标记
    fresh = t0 + 60.0  # 本次运行产出
    stale = t0 - 3600.0  # 上一轮遗留

    def build(name: str, files: dict[str, str], mtimes: dict[str, float] | None = None, marker: bool = True):
        case = root / name
        case.mkdir(parents=True)
        for filename, text in files.items():
            target = case / filename
            target.write_text(text, encoding="utf-8")
            _touch(target, (mtimes or {}).get(filename, fresh))
        marker_path = None
        if marker:
            marker_path = case / ".start_epoch"
            marker_path.write_text(f"{t0}\n", encoding="utf-8")
            _touch(marker_path, t0)
        return case, (str(marker_path) if marker_path else None)

    def run(case: Path, marker: str | None):
        start_time, desc, reason = resolve_start_time(marker, None)
        outcomes = check_dir(case, start_time=start_time, start_desc=desc, start_unknown_reason=reason)
        return outcomes, _verdict_of(outcomes), _exit_code(_verdict_of(outcomes))

    # (标签, 期望结论, 期望退出码)
    expected: list[tuple[str, str, int]] = []
    reports: list[str] = []
    try:
        # ① 合格（新时间戳 + 有数据行）
        case, marker = build("ok", {"kernel_details.csv": good_order, "op_summary_0.csv": good_util})
        expected.append(("合格：新时间戳 + 有数据行", PASS, 0))
        reports.append(_render_selftest_case("合格：新时间戳 + 有数据行", case, marker, run))

        # ② 合格：真实导出形态（无 memory_bound 列）
        case, marker = build("ok_no_memory_bound", {"kernel_details.csv": good_order, "op_summary_0.csv": no_mb_util})
        expected.append(("合格：无 memory_bound 列（真实导出形态）", PASS, 0))
        reports.append(_render_selftest_case("合格：无 memory_bound 列（真实导出形态）", case, marker, run))

        # ③ 陈旧产物（时间戳早于开始标记）⇒ 判失败
        case, marker = build(
            "stale",
            {"kernel_details.csv": good_order, "op_summary_0.csv": good_util},
            mtimes={"kernel_details.csv": stale, "op_summary_0.csv": stale},
        )
        expected.append(("陈旧产物（早于开始标记）", FAIL, 1))
        reports.append(_render_selftest_case("陈旧产物（早于开始标记）", case, marker, run))

        # ④ 只有表头、零数据行 ⇒ 判失败
        case, marker = build("header_only", {"kernel_details.csv": header_only, "op_summary_0.csv": good_util})
        expected.append(("只有表头、零数据行", FAIL, 1))
        reports.append(_render_selftest_case("只有表头、零数据行", case, marker, run))

        # ⑤ 无开始标记 ⇒ 无法判定（≠ 通过）
        case, _ = build("no_marker", {"kernel_details.csv": good_order, "op_summary_0.csv": good_util}, marker=False)
        expected.append(("无开始标记", UNKNOWN, 3))
        reports.append(_render_selftest_case("无开始标记", case, None, run))

        # ⑥ 标记文件不存在 ⇒ 无法判定
        case, _ = build(
            "marker_missing", {"kernel_details.csv": good_order, "op_summary_0.csv": good_util}, marker=False
        )
        missing = str(case / ".start_epoch")
        expected.append(("标记文件不存在", UNKNOWN, 3))
        reports.append(_render_selftest_case("标记文件不存在", case, missing, run))

        # ⑦–⑩ 原有负样本（缺利用率档 / 利用率全 N/A / 缺执行序 / 空目录）
        for label, files in (
            ("缺利用率档", {"kernel_details.csv": good_order}),
            ("利用率全 N/A", {"kernel_details.csv": good_order, "op_summary_0.csv": na_util}),
            ("缺执行序", {"op_summary_0.csv": good_util}),
        ):
            case, marker = build(f"neg_{label}", files)
            expected.append((label, FAIL, 1))
            reports.append(_render_selftest_case(label, case, marker, run))
        empty = root / "空目录"
        empty.mkdir()
        expected.append(("空目录", FAIL, 1))
        reports.append(_render_selftest_case("空目录", empty, None, run))

        # 逐条核对（结论 + 退出码两件都对才算命中）
        got = {line.split("]")[0].lstrip("["): (verdict, code) for line, verdict, code in reports}
        for label, want_verdict, want_code in expected:
            if label not in got:
                failures.append(f"{label}: 夹具没跑起来")
                continue
            got_verdict, got_code = got[label]
            if got_verdict != want_verdict:
                failures.append(f"{label}: 结论应为 {want_verdict}，实为 {got_verdict}")
            if got_code != want_code:
                failures.append(f"{label}: 退出码应为 {want_code}，实为 {got_code}")
    finally:
        pass

    print("check_output --selftest：夹具逐条结论")
    for line, _, _ in reports:
        print(f"  {line}")
    shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("check_output --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(
        f"check_output --selftest: PASS（{len(expected)} 类夹具逐条命中：2 合格 / 6 判失败 / 2 无法判定；"
        "且「无法判定」退出码 3 与「通过」0 可区分，含「无 memory_bound 列」的真实导出形态）"
    )
    return 0


def _render_selftest_case(label: str, case: Path, marker: str | None, run) -> tuple[str, str, int]:
    """跑一个夹具，返回 `(摘要行, 结论, 退出码)` 供自测逐条核对。"""
    outcomes, verdict, code = run(case, marker)
    detail = " | ".join(f"{o.name}={o.verdict}" for o in outcomes)
    return f"[{label}] 结论={verdict} 退出码={code} :: {detail}", verdict, code


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):  # 非标准流（被重定向为非常规对象）时忽略
            pass

    parser = argparse.ArgumentParser(description="profiling 产出完成检查门禁")
    parser.add_argument("--dir", help="采集产物目录（如 ASCEND_PROFILER_OUTPUT）")
    parser.add_argument("--start-marker", help="本次运行开始时写入的标记文件（其 mtime 即开始时刻）")
    parser.add_argument("--start-epoch", type=float, help="本次运行开始时刻（unix 秒），与 --start-marker 二选一")
    parser.add_argument(
        "--freshness-tolerance",
        type=float,
        default=0.0,
        help="新鲜度容差（秒，默认 0）：文件 mtime + 容差 < 开始标记才判陈旧",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出逐项结论")
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

    start_time, start_desc, unknown_reason = resolve_start_time(args.start_marker, args.start_epoch)
    outcomes = check_dir(
        root,
        start_time=start_time,
        start_desc=start_desc,
        start_unknown_reason=unknown_reason,
        tolerance=args.freshness_tolerance,
    )
    verdict = _verdict_of(outcomes)
    code = _exit_code(verdict)

    # `--json` 时 stdout 只承载**一个** JSON 文档；人类可读行一律走 stderr
    # （否则「JSON + 摘要行」会让下游解析器报 Extra data ✗）
    human = sys.stderr if args.json else sys.stdout
    if args.json:
        print(
            json.dumps(
                {"dir": str(root), "verdict": verdict, "exit_code": code, "checks": [o.as_dict() for o in outcomes]},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"check_output: 逐项结论（{root}）")
        for outcome in outcomes:
            print(f"  [{outcome.verdict}] {outcome.name}")
            for message in outcome.messages:
                print(f"        - {message}")

    if verdict == PASS:
        print("check_output: 通过（全部检查通过且无不可判定项）", file=human)
        return 0
    if verdict == UNKNOWN:
        for outcome in outcomes:
            if outcome.verdict == UNKNOWN:
                for message in outcome.messages:
                    print(f"check_output: 无法判定: {message}", file=sys.stderr)
        print(
            "check_output: 无法判定（退出码 3）—— **不得当作通过**；补齐开始标记后重跑，或显式记录“本次未验证新鲜度”",
            file=sys.stderr,
        )
        return 3
    for outcome in outcomes:
        if outcome.verdict == FAIL:
            for message in outcome.messages:
                print(f"check_output: 失败: {message}", file=sys.stderr)
    print("check_output: 采集不合格——不得交下游；请按提示重采", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
