#!/usr/bin/env python3
"""参与度门：断言「新路径真的跑过」，并断言「没有静默回退」（口径见 `references/measurement-discipline.md` §9）。

**为什么需要它**：当一项优化在设计上应当与基线**逐字节相同**时，md5 一致无法区分
「跑了且精确」与「静默回退到基线」。实测形态：某融合臂 md5 / 字节数 / 耗时全部正常，
但新路径每次调用都抛异常（数千次）、框架捕获后回退到基线分支 —— 该臂等价于「什么都没开」。
本脚本把该门禁做成一条命令，避免每次都靠人工 `grep`（人工 grep 最容易犯的错是
**把参与计数与失败计数写进同一条模式**，从而把失败数读成参与数）。

用法：

```bash
# 开臂：参与计数 >= rank 数，且失败计数 == 0
python engagement_check.py --log serve.log \\
    --engaged "FUSED PRODUCER ENGAGED" --failed "producer fast path failed" --ranks 8

# 关臂：参与计数必须为 0（证明关闸路径没被误开）
python engagement_check.py --log serve_ctrl.log --engaged "FUSED PRODUCER ENGAGED" --ranks 8 --expect-absent

# 自证门禁本身有效（负样本必须被判 fail）
python engagement_check.py --selftest
```

退出码：0 = 通过；1 = 不通过；2 = 前置条件缺失（日志文件不存在）。

零依赖、只读、幂等；不联网、不需要 NPU。
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FAIL, PASS, PRECOND = 1, 0, 2

# 输出编码兜底：非 UTF-8 控制台上打印中文会抛 UnicodeEncodeError。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def count(path: Path, pattern: str) -> int:
    """按**行**计数（同一行出现多次只算一次；参与度计数是「每次调用一条」）。"""
    rx = re.compile(pattern)
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if rx.search(line):
                total += 1
    return total


def check(log: Path, engaged: str, failed: str | None, ranks: int, expect_absent: bool) -> tuple[int, list[str]]:
    """返回 (退出码, 输出行)。参与与失败**分别**计数，是本门禁的核心。"""
    if not log.is_file():
        return PRECOND, [f"engagement_check: 前置条件缺失：日志不存在（{log}）"]

    hits = count(log, engaged)
    lines = [
        f"engaged  pattern {engaged!r}: {hits} 行",
    ]
    problems: list[str] = []

    if expect_absent:
        lines.append("模式：--expect-absent（应关闸的对照臂）")
        if hits:
            problems.append(f"关闸臂出现了 {hits} 行参与日志 ⇒ 门控没有真正关闭（该臂不能当对照）")
    else:
        if ranks > 0:
            lines.append(f"要求：engaged >= ranks = {ranks}")
            if hits < ranks:
                problems.append(
                    f"参与计数 {hits} < ranks {ranks} ⇒ 新路径没有在每个 rank 上生效"
                    "（若为单进程/单卡实验，用 --ranks 1 或 --ranks 0 显式声明）"
                )
        else:
            lines.append("要求：engaged >= 1（--ranks 0 表示不按 rank 数断言）")
            if hits < 1:
                problems.append("参与计数为 0 ⇒ 新路径一次都没生效（该臂无信息量）")

    if failed:
        bad = count(log, failed)
        lines.append(f"failed   pattern {failed!r}: {bad} 行")
        if bad:
            problems.append(f"失败计数 {bad} != 0 ⇒ 存在被捕获的静默回退，该臂的结果来自基线路径而不是新路径")
    else:
        lines.append("failed   pattern: (未提供) ⇒ 无法排除静默回退；建议始终提供 --failed")

    if problems:
        lines.append("VERDICT: FAIL")
        lines.extend(f"  - {p}" for p in problems)
        return FAIL, lines
    lines.append("VERDICT: PASS（参与已证明，且无静默回退）")
    return PASS, lines


def selftest() -> int:
    """负样本自测：每种缺口都必须被判 fail，合规样例必须 pass。"""
    # 夹具用**确定性路径**：`mkdtemp` 建的目录权限过严，在本环境往里写文件会被拒
    # （同因处理见 `.agents/scripts/run_evals.py --selftest`）。
    root = Path(__file__).resolve().parents[1] / ".engagement_check_selftest"
    failures: list[str] = []
    shutil.rmtree(root, ignore_errors=True)
    try:
        root.mkdir(parents=True, exist_ok=True)
        good = root / "good.log"
        good.write_text(
            "\n".join(f"[h3_p2] rank={r} FUSED PRODUCER ENGAGED call=1 chunks=7" for r in range(4)) + "\n",
            encoding="utf-8",
        )
        silent = root / "silent.log"  # 参与有、但每次调用都回退
        silent.write_text(
            "[h3_p2] rank=0 FUSED PRODUCER ENGAGED call=1\n"
            + "\n".join("[h3_p2] producer fast path failed: ModuleNotFoundError" for _ in range(50))
            + "\n",
            encoding="utf-8",
        )
        absent = root / "absent.log"
        absent.write_text("rank=0 nothing happened\n", encoding="utf-8")
        ctrl_open = root / "ctrl_open.log"
        ctrl_open.write_text("[x] rank=0 FUSED PRODUCER ENGAGED\n", encoding="utf-8")

        cases = [
            (
                "合规（4 rank 参与、0 失败）",
                good,
                "FUSED PRODUCER ENGAGED",
                "producer fast path failed",
                4,
                False,
                PASS,
            ),
            ("静默回退（失败计数非 0）", silent, "FUSED PRODUCER ENGAGED", "producer fast path failed", 1, False, FAIL),
            ("零参与（新路径没生效）", absent, "FUSED PRODUCER ENGAGED", None, 4, False, FAIL),
            ("关闸臂却出现参与日志", ctrl_open, "FUSED PRODUCER ENGAGED", None, 4, True, FAIL),
            ("关闸臂确实无参与（合规）", absent, "FUSED PRODUCER ENGAGED", None, 4, True, PASS),
        ]
        for label, log, eng, fail_pat, ranks, expect_absent, want in cases:
            code, _ = check(log, eng, fail_pat, ranks, expect_absent)
            if code != want:
                failures.append(f"{label}: 期望 rc={want}，实得 rc={code}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("engagement_check --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return FAIL
    print(f"engagement_check --selftest: PASS（{len(cases)} 类样例逐类判定正确）")
    return PASS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", help="服务/臂日志路径")
    ap.add_argument("--engaged", help="参与度日志的模式（唯一、每 rank 至少一次）")
    ap.add_argument("--failed", help="静默回退/被捕获异常的模式（要求计数为 0）")
    ap.add_argument("--ranks", type=int, default=0, help="期望的 rank 数（参与计数下限）；0=只要求 >=1")
    ap.add_argument("--expect-absent", action="store_true", help="对照臂：参与计数必须为 0")
    ap.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not (a.log and a.engaged):
        ap.error("需要 --log 与 --engaged（或用 --selftest）")

    code, lines = check(Path(a.log), a.engaged, a.failed, a.ranks, a.expect_absent)
    for line in lines:
        print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
