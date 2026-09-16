#!/usr/bin/env python3
"""融合交付件门禁：逐列校验「融合范围与收益交付表」，fail-closed。

为什么需要：交付表是**强制交付件**，下游（选点/开工）依赖它的每一列；列内容一旦混入注释
（括号说明、"须按…确认"、规则名等），列就不可机判、也会误导下游。故在交付时**按列独立校验**，
不合格即拒绝交付（**只约束编排与选点链**；实现类技能在用户输入下不受此门禁约束）。

列契约（顺序固定）：
    | 序号 | 融合kernel名 | 融合kernel明细 | 次数 | 收益预期 | 是否建议融合 | 说明 |
    - 序号：连续正整数（跨表递增）
    - 融合kernel名 / 融合kernel明细：算子在列表中的 `_` 尾段名（`a` / `a+b+c`，每 5 个可用 <br> 换行），
      **不得含任何注释**（括号、省略号、"规则/须/见/表"等字样）
    - 次数：正整数
    - 收益预期：`0%` 或 `a%-b%`（块内总收益；表 2 为增量）
    - 是否建议融合：`建议` / `不建议`
    - 说明：非空；规则、理由、成员/种子数、待确认项**只能在这里**

用法：
    python check_fusion_scope.py --file {run_results_dir}/fusion_scope.md
    python check_fusion_scope.py --selftest

退出码：0 = 通过；1 = 不合格（列内容/格式违规）；2 = 前置条件缺失（文件不存在）。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

COLUMNS = ["序号", "融合kernel名", "融合kernel明细", "次数", "收益预期", "是否建议融合", "说明"]
BAD_TOKENS = ["（", "）", "(", ")", "【", "】", "「", "」", "…", "**", "规则", "须", "见表", "待确认"]
NAME_RE = r"^[A-Za-z0-9_.\-]+$"
DETAIL_RE = r"^[A-Za-z0-9_.\-\+]+$"
PCT_RE = r"^(0%|\d+(?:\.\d+)?%-\d+(?:\.\d+)?%)$"


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    if "[探索]" not in text and "验收态" not in text:
        errors.append("缺口径声明：必须标注 `[探索]` 或 `验收态`（数字纪律）")

    expected_seq = 0
    seen_header = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = _cells(line)
        if cells[:2] == ["序号", "融合kernel名"]:
            seen_header += 1
            if cells != COLUMNS:
                errors.append(f"第 {lineno} 行：表头列不符（应为 {' | '.join(COLUMNS)}）")
            continue
        if set("".join(cells)) <= set("-: "):  # 分隔行
            continue
        if not cells[0].isdigit():  # 汇总行/分节行，跳过
            continue
        num, name, det, cnt, pct, advice, why = (cells + [""] * 7)[:7]
        expected_seq += 1
        if int(num) != expected_seq:
            errors.append(f"第 {lineno} 行：序号应为 {expected_seq}，实为 {num}")
        for label, value in (("融合kernel名", name.strip("`")), ("融合kernel明细", det)):
            if any(tok in value for tok in BAD_TOKENS):
                errors.append(f"第 {lineno} 行：{label} 含注释信息（注释只能进「说明」列）")
            for part in value.replace("<br>", "+").split("+"):
                if part and not __import__("re").match(NAME_RE, part.strip()):
                    errors.append(f"第 {lineno} 行：{label} 片段 `{part}` 不是合法算子名")
        if not __import__("re").match(r"^\d+$", cnt) or int(cnt) < 1:
            errors.append(f"第 {lineno} 行：次数应为正整数，实为 `{cnt}`")
        if not __import__("re").match(PCT_RE, pct):
            errors.append(f"第 {lineno} 行：收益预期应为 `0%` 或 `a%-b%`，实为 `{pct}`")
        if advice not in ("建议", "不建议"):
            errors.append(f"第 {lineno} 行：是否建议融合应为 `建议`/`不建议`，实为 `{advice}`")
        if not why:
            errors.append(f"第 {lineno} 行：说明列为空")
        if __import__("re").search(r"\d+(\.\d+)?\s*(us|ms)\b", name + det):
            errors.append(f"第 {lineno} 行：名/明细列出现绝对耗时（数字纪律）")
    if seen_header < 2:
        errors.append(f"应包含两张表（表 1 融合单元 / 表 2 递进融合），实际找到 {seen_header} 个表头")
    return errors


def _selftest(base: Path | None = None) -> int:
    """负样本自测：合格样例 0 误报 + 5 类列违规逐类触发。"""
    failures: list[str] = []
    root = (base or Path(__file__).resolve().parents[1]) / ".check_fusion_scope_selftest"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    head = [
        "# 交付表",
        "",
        "- 口径：占用口径；状态 **[探索]**",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|---|---|---|---|---|---|---|",
    ]
    good_row = "| 1 | `Fused_RmsNorm_Slice` | RmsNorm+Slice<br>Tile+Slice | 3 | 4.8%-9.5% | 建议 | 规则 0 规范族强制项；成员 9 个 |"
    tail = [
        "",
        "## 表 2",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|---|---|---|---|---|---|---|",
        "| 2 | `Fused_A_B` | A+B | 1 | 0.9%-1.7% | 建议 | 递进融合（增量） |",
    ]

    cases = {
        "ok": head + [good_row] + tail,
        "名含注释": head + ["| 1 | `Fused_RmsNorm(+相邻)` | RmsNorm+Slice | 1 | 1.0%-2.0% | 建议 | x |"] + tail,
        "收益非百分比": head + ["| 1 | `Fused_A_B` | A+B | 1 | 约一成 | 建议 | x |"] + tail,
        "次数非整数": head + ["| 1 | `Fused_A_B` | A+B | 多次 | 1.0%-2.0% | 建议 | x |"] + tail,
        "说明为空": head + ["| 1 | `Fused_A_B` | A+B | 1 | 1.0%-2.0% | 建议 |  |"] + tail,
        "缺口径声明": [
            "# t",
            "",
            "| " + " | ".join(COLUMNS) + " |",
            "|---|---|---|---|---|---|---|",
            good_row,
            "",
            "| " + " | ".join(COLUMNS) + " |",
            "|---|---|---|---|---|---|---|",
            "| 2 | `Fused_A_B` | A+B | 1 | 1.0%-2.0% | 建议 | x |",
        ],
    }
    try:
        for label, lines in cases.items():
            path = root / f"{label}.md"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            errors = check_file(path)
            if label == "ok" and errors:
                failures.append(f"合格样例被误报：{errors[0]}")
            if label != "ok" and not errors:
                failures.append(f"负样本未触发：{label}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if failures:
        print("check_fusion_scope --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("check_fusion_scope --selftest: PASS（1 合格样例 0 误报 + 5 类列违规逐类触发）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="融合交付件门禁（逐列校验）")
    parser.add_argument("--file", help="交付表路径（{run_results_dir}/fusion_scope.md）")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.file:
        print("check_fusion_scope: 需要 --file <交付表>（或 --selftest）", file=sys.stderr)
        return 2
    path = Path(args.file)
    if not path.is_file():
        print(f"check_fusion_scope: 前置条件缺失：交付件不存在（{path}）", file=sys.stderr)
        return 2
    errors = check_file(path)
    if args.json:
        print(json.dumps({"file": str(path), "errors": errors}, ensure_ascii=False, indent=2))
    if not errors:
        print(f"check_fusion_scope: 通过（{path.name}：7 列齐备且列内容洁净）")
        return 0
    for item in errors:
        print(f"check_fusion_scope: error: {item}", file=sys.stderr)
    print(f"\ncheck_fusion_scope: 交付件不合格（{len(errors)} 处）——不得进入选点/开工", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
