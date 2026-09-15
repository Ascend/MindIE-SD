#!/usr/bin/env python3
"""report_lint.py —— overview_report.md 总览表结构校验（close 前置 · 零 NPU）。

校验（对照 model-auto-optimization/references/overview-report.md §2/§7）：
1. 主表表头 = 契约 8 列**按序匹配**：
   `优化类型 | 特性名 | e2e 耗时 | 首步耗时 | 步数 | 加速比 | 质量数据 | 说明`
   （容忍表头单元格带单位注记如 `e2e 耗时 (s)`，主名须匹配；**容许可选的前置「序号」列**——
   该列已在 overview-report.md §2 登记，8 列顺序不变）
2. 每行（主表区域，非 §5 子表）：
   - 优化类型 ∈ {基线/无损优化/免训练有损优化/训练感知优化/其他}
   - 特性名非空
   - e2e/首步耗时/步数/加速比列非空（数值或 `[估算]`/待核验标记 合规值）
   - 锚点行 e2e 禁 `[估算]`（§1 锚点 = 基线行 / 三元组合行 / 最终推荐行）。判定分两路：
     ① 结构判定——特性名为 §2.2 三元固定名词（同时含 `Cache`+`量化`+`稀疏`）或优化类型为基线；
     ② 标记判定——整行文本（含「说明」列，§7 锚点词禁入特性名）含锚点标记
     （`三元` / `最强组合` / `最终推荐`）。**只查特性名关键词会漏判**：§2.1/§7 的固定名词
     白名单不允许在特性名里塞「三元」「最终推荐」等词，故标记只能落在说明列
3. 质量数据列：无损行=输出一致；有损行含数值（SSIM/PSNR/质量门结论）非空
4. 至少存在一行数据（否则报「基线行缺失」）。**不校验**首行是否即基线行（§2 有此要求，
   但首行身份由评审把关，本 lint 不做结构判定）

用法:
    python report_lint.py <overview_report.md>
退出码：0 = 通过；1 = 存在 error（不得 close）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

EXPECTED_HEADER = ["优化类型", "特性名", "e2e", "首步", "步数", "加速比", "质量", "说明"]
VALID_TYPES = {"基线", "无损优化", "免训练有损优化", "训练感知优化", "其他"}
# 锚点标记词：§2.1/§7 禁其进入特性名列，故只能扫整行（含说明列）。
ANCHOR_KEYWORDS = ("三元", "最强组合", "最终推荐")
# §2.2 三元组合固定名词：Cache + 量化(修饰符) + 稀疏（特性名按构成判定，不靠关键词）。
ANCHOR_COMBO = ("Cache", "量化", "稀疏")
LOSSY_TYPES = {"免训练有损优化", "训练感知优化"}


def _anchor_kind(opt: str, feat: str, row_text: str) -> str:
    """返回该行的锚点类型（非锚点返回空串）。锚点行 e2e 必须实测（§1/§1.1）。"""
    if opt.startswith("基线"):
        return "基线行"
    if all(k in feat for k in ANCHOR_COMBO):
        return "三元组合行（Cache+量化+稀疏）"
    for k in ANCHOR_KEYWORDS:
        if k in row_text:
            return f"{k}行"
    return ""


def _norm_header(cell: str) -> str:
    cell = re.sub(r"\s*\(.*?\)\s*", "", cell).strip()
    cell = re.sub(r"\s+", "", cell)
    return cell


def check_table(text: str) -> list[str]:
    errors: list[str] = []
    lines = text.splitlines()
    table_start = None
    header_cells: list[str] = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and "优化类型" in ln and "特性名" in ln:
            header_cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            table_start = i
            break
    if table_start is None:
        return ["未找到总览主表（含「优化类型|特性名」表头）——报表须含 §2 固定 8 列主表"]

    # 表头列名校验（前缀匹配，容忍单位括注）
    norm = [_norm_header(c) for c in header_cells]
    # User-requested extension: an optional leading 序号 (index) column is allowed;
    # the 8 contract columns must follow it in order.
    has_idx = bool(norm) and norm[0].startswith("序号")
    if has_idx:
        norm = norm[1:]
    for expect, actual in zip(EXPECTED_HEADER, norm):
        if not actual.startswith(expect):
            col_no = EXPECTED_HEADER.index(expect) + 1
            errors.append(
                f"表头列序不符：期望第 {col_no} 列「{expect}…」，实际「{actual}」"
            )
    if len(norm) != 8:
        errors.append(f"表头列数 ≠ 8：实际 {len(norm)} 列（不含序号列）{header_cells}")

    # 逐行解析主表（到第一个非 | 行或 §5 子表标题为止）
    ncol = len(norm)
    row_no = 0
    for ln in lines[table_start + 1:]:
        s = ln.strip()
        if not s.startswith("|"):
            break
        if re.match(r"^\|[\s\-|:]+\|?$", s):  # 分隔行
            continue
        cells = [c.strip() for c in s.strip().strip("|").split("|")]
        if has_idx and cells:
            cells = cells[1:]
        if len(cells) < ncol:
            # 可能行尾竖线缺失，补齐空
            cells += [""] * (ncol - len(cells))
        row_no += 1
        opt, feat, e2e, first, steps, speed, qual, _note = cells[:8]
        if not feat:
            errors.append(f"主表第 {row_no} 行：特性名为空")
        if not opt:
            errors.append(f"主表第 {row_no} 行：优化类型为空")
        elif opt not in VALID_TYPES and not opt.startswith("基线"):
            errors.append(f"主表第 {row_no} 行：优化类型非法「{opt}」（枚举见 §7.1）")

        # e2e/首步/步数/加速比非空
        for col_name, val in (("e2e 耗时", e2e), ("首步耗时", first),
                              ("步数", steps), ("加速比", speed)):
            if not val or val == "—":
                errors.append(
                    f"主表第 {row_no} 行（{feat}）：{col_name} 列为空/「—」"
                    "（§7 每行必填单值）"
                )

        # 锚点行禁估算（锚点 = 基线 / 三元组合 / 最终推荐；判定见 _anchor_kind）
        anchor = _anchor_kind(opt, feat, " | ".join(cells))
        if anchor and "[估算]" in e2e:
            errors.append(
                f"主表第 {row_no} 行（{feat}）：锚点行 e2e 禁 [估算]（§1 必须实测；"
                f"锚点={anchor}）"
            )

        # 质量列：无损 vs 有损
        if opt in VALID_TYPES and opt != "基线":
            if opt == "无损优化":
                if qual and qual != "—" and "输出一致" not in qual and "输出" not in qual:
                    # 宽松：无损质量列须提及输出一致性
                    errors.append(
                        f"主表第 {row_no} 行（{feat}）：无损行质量列应表输出一致性，"
                        f"实际「{qual}」"
                    )
            elif opt in LOSSY_TYPES:
                if not qual or qual == "—":
                    errors.append(
                        f"主表第 {row_no} 行（{feat}）：有损行质量列空"
                        "（§4 须给数值/结论）"
                    )
                elif not re.search(r"SSIM|PSNR|质量门|pass|fail|inconclusive|0\.\d", qual):
                    errors.append(
                        f"主表第 {row_no} 行（{feat}）：有损行质量列缺数值/结论"
                        f"「{qual}」"
                    )

    if row_no == 0:
        errors.append("主表无数据行（基线行缺失）——报表须含基线行")
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path)
    args = ap.parse_args(argv)
    if not args.report.exists():
        print(f"[error] 报表不存在：{args.report}")
        return 1
    text = args.report.read_text(encoding="utf-8-sig")
    errors = check_table(text)
    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)}（{args.report.name}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
