#!/usr/bin/env python3
"""audit_report.py —— 总览/细分报表的**表格结构审计**与**数值自洽审计**（close 前置 · 零 NPU）。

与 `report_lint.py` 的分工：lint 管"主表**写法**合规"（列序/枚举/必填/锚点/禁估算/质量列语义），
本脚本管"**改表之后的机械复核**"——结构是否还渲染得出来、每个数字是否与自己表里的分母自洽。
契约出处：`references/report-contract.md` §4（参考项与主表同测量须逐字段一致）、
§6（格式事故清单）、§7（lint 与审计脚本）、§8（复核流程："修完跑 lint + 结构审计 + 数值审计"）。

结构审计
  S1 每张表：表头列数 = 分隔行列数 = 每个数据行列数（切分按**未转义**竖线 `(?<!\\\\)\\|`）
  S2 表体连续性：表头 + 分隔行之后的数据行必须连续；出现"孤立表行"（前面是空行/非表格行）
     报错——这类行不属于任何表，会渲染成孤立片段（真实事故）
  S3 有竖线的行组却缺分隔行（不是合法表格）报错
  S4 单元格加粗：真嵌套（外层 `**` 对内层 `**` 对）报错；`**` 个数为奇数（未闭合）报错。
     配对按**块**做（表行 / 段落 / 列表项 / 标题），并**剔除围栏代码块与行内代码**——因为
     ① 强调可以跨软换行成对（逐行数标记会把 `**…` 换行 `…**` 误报成未闭合）；
     ② 代码里的 `` `**` `` 是字面文本，不是标记（两处都是本脚本踩过的假阳性）。
  S5 **主表报告必带章节**（`overview-report.md` §2.8）：只对"含 `优化类型`+`特性名` 主表"的文件判，
      要求有「最佳路径下的特性详解」详情节与「附录」，附录内须含「开关对照表」/「内部代号对照表」/「证据指针」三小节，
      且详情节在附录之前——报告"该有的章节没有"是结构缺陷（提交前即被打回），不是风格问题。
      非报告类 md 不受此判（通常没有主表）；标题关键字判据与 `report_lint.py` 同源同改。

数值自洽审计（分母判定见下，所用分母**始终打印**）
  N1 有 e2e + 加速比 → 校验 `分母 ÷ e2e ≈ 加速比`（跳过 `[估算]`/区间/`—`/`?` 行并逐条列出）
  N2 有 diffuse + 每步 → 校验 `diffuse ÷ 每步` 是合理整数（只看是否整数，跨世代不误判）
  N3 有 实测 + 单点连乘 + 比值 → 校验 `实测 ÷ 连乘 ≈ 比值`
  N4 "相对上一行" → 先判它是**百分比**还是**倍数**，再校验参照行是"上一行"还是"第 0 行"
     （支持双参照写法 `-5.8%（vs 行 0）/ -2.2%（vs 行 1）`：任一命中即通过）
  N5 同一次测量出现在两张表（e2e 相同）且两表分母相同 → e2e/首步/步数/加速比须逐字段一致

**分母判定（关键，决定 N1 是否为真阳性）**：加速比的分母是"报告级基线"，不是"表内那条叫基线的
参考行"（参考行自身也相对报告基线，例如 CP2USP4 参考行在同表内是 0.90×）。因此按候选顺序拟合：
  1) `--baseline`；2) **报告级基线** = 主表（表头含「优化类型」+「特性名」）中第一行「基线」的 e2e；
  3) 本表内第一行「基线」的 e2e。逐表取**与全部数值行相容**的第一个候选；无相容候选 → 报错。
  表内另有基线行却不作分母时，打印 info 说明（避免"用了哪个分母"被误读）。

输出与纪律
  * 每条发现按"**位置 / 原值 / 应为 / 依据**"四列打印（report-contract §8 要求的复核清单格式）；
  * **幂等**：只读不改文件；同样输入永远同样结论；
  * **锚点缺失显式报错**（"有竖线行但凑不出合法表格"/"无分母可定" → 报错并给出补法，绝不静默跳过）；
    但**不含表格的文件**（一行竖线都没有，例如纯说明文档）返回 exit 0 + 说明——那是"无可审计内容"，
    不是缺陷（审计任意 md 时不该据此误判）；
  * 被跳过的行/表**打印原因**，并打印每张表所用的分母口径，便于复查审计覆盖面。

用法:
    python audit_report.py <report.md> [--baseline {baseline}] [--only structure|numeric] [--strict]
    python audit_report.py <report.md> --selftest     # 负样本自测：在真实报表上注入已知事故并断言被抓到

退出码：0 = 干净（或仅 warn/info 且未开 --strict）；1 = 存在 error 级发现；2 = 前置条件缺失
（有竖线行但无合法表格 / 无法确定分母）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Windows consoles default to a legacy code page (GBK here), which cannot encode the
# characters these reports use (⊆, ❓, CJK). A gate that dies on a character is
# indistinguishable from one that never ran, so force UTF-8 on both streams.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SPLIT_UNESCAPED = re.compile(r"(?<!\\)\|")
DELIM_ROW = re.compile(r"^\s*\|?[\s:\-|]+\|?\s*$")
FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
INLINE_CODE = re.compile(r"`+[^`]*`+")
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
PCT = re.compile(r"[-−+]?\d+(?:\.\d+)?\s*%")
MULT = re.compile(r"[-+]?\d+(?:\.\d+)?\s*[×x]")
KNOWN_STEP_SETS = {1, 2, 4, 8, 16, 32, 50, 100}
#: 这些标记出现在 e2e/加速比 单元格时，该行不参与 N1（契约允许的估算/区间/缺失写法）
SKIP_MARKS = ("[估算]", "[探索]", "—", "?", "❓", "待测", "未测")

# --- S5 必带章节（与 report_lint.py 的 REQUIRED_SECTIONS 同源同改：overview-report.md §2.8） ---
#: 附录三小节须齐；详情节位置不限（编号随报表），但必须在附录之前。
REQUIRED_SECTIONS = (
    ("最佳路径下的特性详解", ("最佳路径", "特性详解")),
    ("开关对照表", ("开关", "对照")),
    ("内部代号对照表", ("内部代号", "对照")),
    ("证据指针", ("证据", "指针")),
)
APPENDIX_HEADING_RE = re.compile(r"附录|appendix", re.IGNORECASE)
HEADING_SPACERS = " \t\u3000*`|｜/\\."


def norm_heading(heading: str) -> str:
    """标题归一化：去 `#`/加粗/空白/分隔符，按词判「是不是那个小节」（不绑定编号）。"""
    text = re.sub(r"^#+\s*", "", heading.strip()).replace("**", "")
    for ch in HEADING_SPACERS:
        text = text.replace(ch, "")
    return text


def required_section_heading_hits(lines):
    """必带章节判定（S5）。返回 (missing_errors, appendix_line)。

    只判「主表报告」：调用方须已确认本文件含 `优化类型`+`特性名` 主表。判据与
    `report_lint.py` 的 `check_doc_structure` 一致（同源同改），此处只做结构与次序。
    """
    hits = [(i, norm_heading(ln)) for i, ln in enumerate(lines, 1) if ln.lstrip().startswith("#")]
    if not hits:
        return (["报表无任何标题（§2.8：须有「最佳路径下的特性详解」详情节与「附录」）"], None)
    appendix = [i for i, h in hits if APPENDIX_HEADING_RE.search(h)]
    detail = [i for i, h in hits if all(k in h for k in REQUIRED_SECTIONS[0][1])]
    errs = []
    if not detail:
        errs.append("缺必带小节「最佳路径下的特性详解」（overview-report.md §2.8）")
    if not appendix:
        errs.append("缺必带「附录」节（须含 开关对照表 / 内部代号对照表 / 证据指针，§2.8）")
        return errs, None
    first_appendix = min(appendix)
    if detail and max(detail) > first_appendix:
        errs.append("「最佳路径下的特性详解」位于附录之后（§2.8：正文在前、附录收尾）")
    for name, aliases in REQUIRED_SECTIONS[1:]:
        if not [i for i, h in hits if all(k in h for k in aliases) and i > first_appendix]:
            errs.append(f"附录缺必带小节「{name}」（§2.8）")
    return errs, first_appendix


class Finding:
    __slots__ = ("basis", "expect", "level", "old", "where")

    def __init__(self, level, where, old, expect, basis):
        self.level, self.where, self.old, self.expect, self.basis = level, where, old, expect, basis

    def fmt(self):
        return (
            f"[{self.level}] {str(self.where)[:30]:<30} 原值={str(self.old)[:20]:<20} "
            f"应为={str(self.expect)[:24]:<24} 依据={self.basis}"
        )


def cells(line: str) -> list:
    t = line.strip()
    t = t.removeprefix("|")
    t = t.removesuffix("|")
    return [c.strip() for c in SPLIT_UNESCAPED.split(t)]


def is_pipe(line: str) -> bool:
    return line.strip().startswith("|")


def norm_head(cell: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"\s*[（(].*?[)）]\s*", "", cell))


def fnum(s):
    m = NUM.search((s or "").replace(",", ""))
    return float(m.group(0)) if m else None


LEAD_JUNK = re.compile(r"^[\s`*_~≈＜＞<>≤≥=＝约（(]{0,6}")


def lead_num(s):
    """单元格的**主数值**：剥掉加粗/反引号/约等号等前缀后，必须以数字开头。

    防两类错误：
      * 假阳性："见 §6" 这类文字单元格被 `fnum` 抽出 `6`；
      * **假阴性（更危险）**：`**34.32**（实测）` 这种加粗单元格被判为"无数值"而**静默排除**在
        N1/N2/N3/N4 之外——报表大量使用加粗，会造成整片检查盲区（负样本自测抓到了这一点）。
    """
    if s is None:
        return None
    t = s.replace(",", "")
    for _ in range(3):  # 允许 `**…`、`≈**…` 这类叠加前缀
        t2 = LEAD_JUNK.sub("", t)
        if t2 == t:
            break
        t = t2
    if not re.match(r"[-−+]?\d", t):
        return None
    return fnum(t)


def decimals(s: str) -> int:
    m = re.search(r"[-+]?\d+\.(\d+)", s or "")
    return len(m.group(1)) if m else 0


def near(got: float, exp: float, shown: str, rel: float = 0.02) -> bool:
    """容差 = 显示精度的半个末位（放大 1.5 倍留余量）+ 相对项（治 1605.5/15.05=106.68 写成 106.3 的舍入）。"""
    tol = max(0.5 * 10 ** (-decimals(shown)) * 1.5, abs(exp) * rel)
    return abs(got - exp) <= tol


def collect_tables(lines):
    tables, orphans, i = [], [], 0
    while i < len(lines):
        if is_pipe(lines[i]):
            j = i
            while j < len(lines) and is_pipe(lines[j]):
                j += 1
            group = list(range(i, j))
            if len(group) >= 2 and DELIM_ROW.match(lines[group[1]]) and not DELIM_ROW.match(lines[group[0]]):
                tables.append(
                    (group[0] + 1, cells(lines[group[0]]), group[1] + 1, [(n + 1, cells(lines[n])) for n in group[2:]])
                )
            else:
                orphans.append((group[0] + 1, group[-1] + 1, len(group)))
            i = j
            continue
        i += 1
    return tables, orphans


BLOCK_START = re.compile(r"^\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+[.)]\s)")


def _fence_spans(lines):
    """围栏代码块覆盖的行号集合（1-based）：其中的 `**` 是字面文本，不参与加粗配对。"""
    spans, fence, start = set(), None, None
    for i, text_line in enumerate(lines, 1):
        m = FENCE.match(text_line)
        if not m:
            continue
        if fence is None:
            fence, start = m.group(2)[0], i
        elif m.group(2).startswith(fence) and not m.group(3).strip():
            spans.update(range(start, i + 1))
            fence = None
    if fence is not None:
        spans.update(range(start, len(lines) + 1))
    return spans


def bold_blocks(lines, fenced=frozenset()):
    """加粗标记按**块**配对，块边界 = 空行 / 表行 / 标题行 / 新列表项 / **围栏代码块**。

    为什么必须按块：Markdown 的强调可以跨**软换行**成对（`**…` 换行 `…**`），但**不能跨**
    空行、标题、列表项或代码块（各自是独立叶子块）。按单行数 `**` 个数会把合法的跨行加粗误报成
    "未闭合"（本脚本第一版就踩了这个假阳性）；反过来，把标题/多个列表项并成一块又会掩盖
    真正的未闭合。故取二者之间：块内配对、块间不跨越。
    """
    blocks, i, n = [], 0, len(lines)
    while i < n:
        if (i + 1) in fenced or not lines[i].strip():
            i += 1
            continue
        if is_pipe(lines[i]):
            blocks.append((i + 1, i + 1, lines[i]))  # 表行：单元格是行内内容，各自成块
            i += 1
            continue
        j = i + 1
        while (
            j < n
            and lines[j].strip()
            and not is_pipe(lines[j])
            and not FENCE.match(lines[j])
            and (j + 1) not in fenced
            and not BLOCK_START.match(lines[j])
        ):
            j += 1
        blocks.append((i + 1, j, "\n".join(lines[i:j])))
        i = j
    return blocks


def audit_bold(lines, out):
    for a, b, text in bold_blocks(lines, _fence_spans(lines)):
        where = f"L{a} 单元格" if a == b else f"L{a}–L{b} 段落"
        text = INLINE_CODE.sub("", text)  # 行内代码里的 `**` 是字面文本，不是加粗标记
        marks = [m.start() for m in re.finditer(r"(?<!\\)\*\*", text)]
        if len(marks) % 2:
            out.append(
                Finding("error", where, f"`**` 共 {len(marks)} 个（奇数）", "成对闭合", "§6：加粗标记未闭合 → 渲染错乱")
            )
            continue
        spans = [(marks[k], marks[k + 1]) for k in range(0, len(marks), 2)]
        for x, y in spans:  # 真嵌套：本对内还完整包含另一对
            if any(x < x2 and y2 < y for x2, y2 in spans):
                out.append(
                    Finding(
                        "error",
                        where,
                        "嵌套加粗 `**…**…**…**`",
                        "拆成两个独立加粗",
                        "§6：外层 `**` 对内嵌内层 → 渲染错乱",
                    )
                )
                break


def audit_structure(lines, tables, orphans, out):
    for start, hdr, delim, rows in tables:
        n = len(hdr)
        d = cells(lines[delim - 1])
        if len(d) != n:
            out.append(
                Finding(
                    "error",
                    f"L{delim} 表(L{start}) 分隔行",
                    f"{len(d)} 列",
                    f"{n} 列（= 表头）",
                    "§6：分隔行列数 ≠ 表头 → 整表不渲染",
                )
            )
        for ln, c in rows:
            if len(c) != n:
                out.append(
                    Finding(
                        "error",
                        f"L{ln} 表(L{start}) 数据行",
                        f"{len(c)} 列",
                        f"{n} 列",
                        "§6/§7：列数须与表头一致（转义竖线 \\| 不切分）",
                    )
                )
    for a, b, cnt in orphans:
        if cnt == 1:
            out.append(
                Finding(
                    "error",
                    f"L{a} 孤立表行",
                    "单行 `|…|`（无分隔行）",
                    "并入所属表或删除",
                    "§6：行被插到空行之后 → 渲染成孤立片段",
                )
            )
        else:
            out.append(
                Finding("error", f"L{a}–L{b} 竖线行组", f"{cnt} 行无分隔行", "补分隔行使其成为合法表格", "§6/§7")
            )
    audit_bold(lines, out)


def col_index(hdr, *keys):
    for k in keys:
        for idx, h in enumerate(hdr):
            if k in norm_head(h):
                return idx
    return None


def baseline_row(rows, i_type, i_e2e):
    """返回表内第一行「基线」的 (行号, e2e)。"""
    if i_type is None or i_e2e is None:
        return None
    for ln, c in rows:
        if max(i_type, i_e2e) < len(c) and c[i_type].startswith("基线"):
            v = lead_num(c[i_e2e])
            if v is not None:
                return ln, v
    return None


def numeric_rows(rows, i_e2e, i_sp):
    """参与 N1 的行 (行号, e2e, 加速比原文, e2e原文)；跳过行单独返回。"""
    keep, skip = [], []
    for ln, c in rows:
        if max(i_e2e, i_sp) >= len(c):
            continue
        raw_e, raw_s = c[i_e2e], c[i_sp]
        if any(k in raw_e for k in SKIP_MARKS) or any(k in raw_s for k in SKIP_MARKS):
            skip.append((ln, raw_e, raw_s))
            continue
        e, sp = lead_num(raw_e), lead_num(raw_s)
        if e and sp is not None:
            keep.append((ln, e, raw_s, raw_e))
        else:
            skip.append((ln, raw_e, raw_s))
    return keep, skip


def choose_baseline(cands, keep):
    """取与全部数值行相容的第一个候选（顺序即优先级：CLI > 报告基线 > 本表基线行）。"""
    for src, val in cands:
        if val and all(
            lead_num(raw_s) is not None and near(lead_num(raw_s), val / e, raw_s) for _, e, raw_s, _ in keep
        ):
            return src, val
    return (cands[0][0], cands[0][1]) if cands else (None, None)


def _audit_ratio(start, delim, hdr, rows, i_e2e, i_sp, i_type, report_base, cli_base, out, notes, skips):
    """N1：`分母 ÷ e2e ≈ 加速比`。分母按候选拟合（CLI > 报告基线 > 本表基线行），并始终打印所用分母。"""
    own = baseline_row(rows, i_type, i_e2e)
    keep, skip = numeric_rows(rows, i_e2e, i_sp)
    for ln, re_, rs in skip:
        skips.append(f"L{ln} 未参与 N1（e2e={str(re_)[:16]}，加速比={str(rs)[:16]}）")
    if not keep:
        notes.append(f"表 L{start}：无可用数值行（跳过 N1）")
        return
    cands = []
    if cli_base is not None:
        cands.append(("--baseline", cli_base))
    if report_base is not None:
        cands.append(("报告基线", report_base))
    if own is not None:
        cands.append((f"本表基线行 L{own[0]}", own[1]))
    src, val = choose_baseline(cands, keep)
    if val is None:
        out.append(
            Finding(
                "error",
                f"L{delim} 表(L{start}) 分母",
                "无分母候选（表内无「基线」行）",
                "补基线行或传 --baseline",
                "§7：锚点缺失须显式报错，不得静默跳过",
            )
        )
        return
    notes.append(f"表 L{start}：分母={val:.2f}（{src}），核算 {len(keep)} 行")
    if own is not None and abs(own[1] - val) > 0.01:
        notes.append(
            f"表 L{start}：另有参考行 L{own[0]}「基线」={own[1]:.2f}，**非本表分母**（其加速比也相对 {val:.2f}）"
        )
    for ln, e, raw_s, raw_e in keep:
        exp = val / e
        if not near(lead_num(raw_s), exp, raw_s):
            out.append(
                Finding(
                    "error",
                    f"L{ln} 表(L{start}) 列「加速比」",
                    raw_s,
                    f"{exp:.2f}×",
                    f"{src} {val:.2f} ÷ e2e {raw_e}",
                )
            )


def audit_numeric(tables, report_base, cli_base, out, notes, skips):
    """N1~N4 **逐表各自独立判定**：某表没有 e2e/加速比 列，不影响它的 N2/N3/N4 被检查。

    （第一版把这些检查写在"N1 前置列缺失 → continue"之后，导致 `实测/单点连乘/比值`
    这类没有 e2e 列的表整表被静默跳过——负样本自测抓到了这个漏检。）
    """
    for start, hdr, delim, rows in tables:
        i_e2e = col_index(hdr, "e2e")
        i_sp = col_index(hdr, "加速比")
        i_type = col_index(hdr, "优化类型")
        # ---- N1：e2e 与 加速比 的分母自洽（本表无这两列则跳过 N1，不跳过整表）----
        if i_e2e is None or i_sp is None:
            notes.append(f"表 L{start}：无 e2e/加速比 列（跳过 N1）")
        else:
            _audit_ratio(start, delim, hdr, rows, i_e2e, i_sp, i_type, report_base, cli_base, out, notes, skips)

        i_d, i_step = col_index(hdr, "diffuse"), col_index(hdr, "每步")
        if i_d is not None and i_step is not None:
            for ln, c in rows:
                if max(i_d, i_step) >= len(c):
                    continue
                d, st = lead_num(c[i_d]), lead_num(c[i_step])
                if not d or not st:
                    continue
                implied = d / st
                if abs(implied - round(implied)) > 0.03 * implied or round(implied) < 1:
                    out.append(
                        Finding(
                            "error",
                            f"L{ln} 表(L{start}) 列「每步」",
                            c[i_step],
                            "使 diffuse÷每步 为整数",
                            f"隐含步数={implied:.3f}（非整数）",
                        )
                    )
                elif round(implied) not in KNOWN_STEP_SETS:
                    notes.append(f"表 L{start} 行 L{ln}：隐含步数={round(implied)}（不在常见集合，确认世代即可）")

        i_m, i_p, i_r = col_index(hdr, "实测"), col_index(hdr, "单点连乘"), col_index(hdr, "比值")
        if None not in (i_m, i_p, i_r):
            for ln, c in rows:
                if max(i_m, i_p, i_r) >= len(c):
                    continue
                m, p, r = lead_num(c[i_m]), lead_num(c[i_p]), lead_num(c[i_r])
                if not (m and p and r):
                    continue
                if not near(r, m / p, c[i_r], rel=0.03):
                    out.append(
                        Finding(
                            "error",
                            f"L{ln} 表(L{start}) 列「比值」",
                            c[i_r],
                            f"{m / p:.3f}",
                            f"实测 {m:.3f} ÷ 连乘 {p:.3f}",
                        )
                    )

        i_up = col_index(hdr, "相对上一行")
        if i_up is not None:
            # 被比较的指标列：优先显式 e2e 列；否则取「相对上一行」左侧最近的数值列（推断须显式说明）
            i_val = i_e2e if i_e2e is not None else (i_up - 1 if i_up > 0 else None)
            if i_val is None:
                notes.append(f"表 L{start}：有「相对上一行」但无法确定被比较列（跳过 N4）")
            else:
                if i_e2e is None:
                    notes.append(
                        f"表 L{start}：N4 的被比较列按「相对上一行」左侧列推断 = 第 {i_val + 1} 列"
                        f"「{hdr[i_val] if i_val < len(hdr) else '?'}」"
                    )
                prev = first = None
                for ln, c in rows:
                    if max(i_up, i_val) >= len(c):
                        continue
                    e = lead_num(c[i_val])
                    cell = c[i_up]
                    if e is None:
                        continue
                    if first is None:
                        first = e
                    refs = []
                    for m in PCT.finditer(cell):
                        refs.append(
                            (
                                "pct",
                                float(m.group(0).replace("−", "-").rstrip("% ").strip()),
                                "行 0" if re.search(r"行\s*0", cell) else "上一行",
                            )
                        )
                    for m in MULT.finditer(cell):
                        refs.append(
                            (
                                "mult",
                                float(NUM.search(m.group(0)).group(0)),
                                "行 0" if re.search(r"行\s*0", cell) else "上一行",
                            )
                        )
                    if prev is not None and refs:
                        ok = False
                        for kind, v, ref in refs:
                            b = first if ref == "行 0" else prev
                            if not b:
                                continue
                            if kind == "pct" and abs(v - (e / b - 1) * 100) <= max(0.6, abs((e / b - 1) * 100) * 0.05):
                                ok = True
                            if kind == "mult" and near(v, b / e, f"{v:.2f}", rel=0.03):
                                ok = True
                        if not ok:
                            out.append(
                                Finding(
                                    "error",
                                    f"L{ln} 表(L{start}) 列「相对上一行」",
                                    cell,
                                    f"{((e / prev - 1) * 100):+.1f}%（上一行 {prev:.3f}）或 "
                                    f"{(((e / first - 1) * 100) if first else 0.0):+.1f}%（行 0 {(first or 0.0):.3f}）",
                                    "§7：先判百分比/倍数与参照行（上一行 or 行 0）",
                                )
                            )
                    elif prev is None and refs:
                        notes.append(f"表 L{start} 行 L{ln}：有「相对上一行」但本行是首行（参照在本表之外，跳过）")
                    prev = e


def audit_cross_table(tables, out, notes):
    """N5：同一报告内 e2e 相同的行，若两表分母也相同 → 加速比/首步/步数须一致（§4）。"""

    def base_of(hdr, rows):
        i_type, i_e2e = col_index(hdr, "优化类型"), col_index(hdr, "e2e")
        r = baseline_row(rows, i_type, i_e2e)
        return r[1] if r else None

    seen = {}
    for start, hdr, delim, rows in tables:
        b = base_of(hdr, rows)
        i_e2e, i_sp = col_index(hdr, "e2e"), col_index(hdr, "加速比")
        i_first, i_steps = col_index(hdr, "首步"), col_index(hdr, "步数")
        if i_e2e is None:
            continue
        for ln, c in rows:
            if i_e2e >= len(c):
                continue
            e = lead_num(c[i_e2e])
            if e is None:
                continue
            sig = tuple(c[i] if (i is not None and i < len(c)) else None for i in (i_sp, i_first, i_steps))
            key = round(e, 3)
            if key not in seen:
                seen[key] = (b, ln, sig)
                continue
            pb, pln, psig = seen[key]
            if b is None or pb is None:
                continue
            if abs(b - pb) > 0.01:
                notes.append(
                    f"同测量行 L{ln} 与 L{pln}：两表分母不同（{b:.2f} vs {pb:.2f}）→ 加速比不可比，按 §4 跨血缘标注"
                )
                continue
            for name, a, x in (("加速比", psig[0], sig[0]), ("首步耗时", psig[1], sig[1]), ("步数", psig[2], sig[2])):
                if a is None or x is None or a == x:
                    continue
                if lead_num(a) is not None and lead_num(x) is not None and near(lead_num(x), lead_num(a), a, rel=0.02):
                    continue
                out.append(
                    Finding(
                        "error",
                        f"L{ln}（与 L{pln} 同一次测量）",
                        f"{name}={x}",
                        f"与 L{pln} 一致：{a}",
                        "§4：参考项与主表同测量行须逐字段一致",
                    )
                )


def audit_text(text: str, only: str = "all", cli_base=None):
    """审计一段报表文本，返回 (res, msg)。res = None 表示前置条件缺失（msg 为原因）。

    只接受文本而不落盘：既保证"只读、幂等"，也让 `--selftest` 能在内存里注入事故（无需临时文件）。
    """
    lines = text.split("\n")
    tables, orphans = collect_tables(lines)
    if not tables:
        pipes = sum(1 for text_line in lines if is_pipe(text_line))
        if pipes == 0:
            # 文件本身不含表格：无可审计内容（不是错误）——审计任意 md 时不应报假警报
            return (
                {
                    "errs": [],
                    "warns": [],
                    "notes": ["本文件不含表格（无竖线行），无可审计内容"],
                    "skips": [],
                    "tables": 0,
                    "orphans": 0,
                    "base": None,
                    "no_table": True,
                    "head_warns": [],
                },
                "",
            )
        return None, (
            f"有 {pipes} 行竖线但没有任何合法表格（表头 + 分隔行）——无法审计。依据：report-contract "
            "§6/§7（分隔行缺失 → 整表不渲染）。请确认路径，或补上分隔行。"
        )
    main = next((t for t in tables if any("优化类型" in x for x in t[1]) and any("特性名" in x for x in t[1])), None)
    report_base = None
    head_warns = []
    if main is None:
        head_warns.append(
            "未找到含「优化类型 | 特性名」的主表：按通用表格审计继续（细分报表属正常），报告级基线不可用。"
        )
    else:
        br = baseline_row(main[3], col_index(main[1], "优化类型"), col_index(main[1], "e2e"))
        report_base = br[1] if br else None
        if report_base is None:
            head_warns.append(f"主表(L{main[0]}) 无「基线」行 → 报告级基线缺失；各表需自备基线行或 --baseline。")
    out, notes, skips = [], [], []
    if only in ("structure", "all"):
        audit_structure(lines, tables, orphans, out)
        if main is not None:
            # S5 只对「主表报告」判——非报告类 md（无 优化类型+特性名 表）不在此范围
            for msg in required_section_heading_hits(lines)[0]:
                out.append(Finding("error", "报表章节（S5）", "缺少/次序错", msg, "§2.8 必带详情节与附录"))
    if only in ("numeric", "all"):
        audit_numeric(tables, report_base, cli_base, out, notes, skips)
        audit_cross_table(tables, out, notes)
    return (
        {
            "errs": [f for f in out if f.level == "error"],
            "warns": [f for f in out if f.level == "warn"],
            "notes": notes,
            "skips": skips,
            "tables": len(tables),
            "orphans": len(orphans),
            "base": report_base,
            "head_warns": head_warns,
        },
        "",
    )


def run_audit(path: Path, only: str = "all", cli_base=None):
    return audit_text(path.read_text(encoding="utf-8-sig"), only, cli_base)


def run_audit_text(text: str, cli_base=None):
    return audit_text(text, "all", cli_base)


def render(path: Path, res) -> int:
    if res.get("no_table"):
        print(f"== 审计 {path.name}：本文件不含表格，无可审计内容（exit 0，非错误）")
        return 0
    base_shown = f'{res["base"]:.2f}' if res["base"] else "未取到"
    print(f"== 审计 {path.name}：{res['tables']} 张表（另 {res['orphans']} 个竖线行组非表格）；报告级基线={base_shown}")
    for w in res["head_warns"]:
        print("  [warn] " + w)
    for f in res["errs"] + res["warns"]:
        print("  " + f.fmt())
    if res["notes"]:
        print("  -- 分母口径与说明 --")
        for n in res["notes"]:
            print("     " + n)
    if res["skips"]:
        print("  -- 未参与 N1 的行（显式列出，避免覆盖面被误读）--")
        for s in res["skips"][:15]:
            print("     " + s)
        if len(res["skips"]) > 15:
            print(f"     …其余 {len(res['skips']) - 15} 条")
    print(
        f"结论：error={len(res['errs'])} warn={len(res['warns'])}"
        f"（说明 {len(res['notes'])} 条，跳过 {len(res['skips'])} 行）"
    )
    return 1 if res["errs"] else 0


def _corruptions(text: str):
    """在真实报表上**动态注入**已知事故，用于 `--selftest`（每条都必须被抓到）。"""
    lines = text.split("\n")
    tabs, _ = collect_tables(lines)
    cases = []

    def find_num_row():
        for start, hdr, delim, rows in tabs:
            i_e, i_s = col_index(hdr, "e2e"), col_index(hdr, "加速比")
            if i_e is None or i_s is None:
                continue
            keep, _ = numeric_rows(rows, i_e, i_s)
            if keep:
                ln, _e, raw_s, _ = keep[0]
                return ln, i_s, raw_s
        return None

    hit = find_num_row()
    if hit:
        ln, _i_s, raw_s = hit
        v = lead_num(raw_s)
        cases.append(
            (
                "加速比与分母不符",
                "\n".join(lines[: ln - 1] + [lines[ln - 1].replace(raw_s, f"{v * 3:.2f}×", 1)] + lines[ln:]),
                "列「加速比」",
            )
        )
    if tabs:
        start, hdr, delim, rows = tabs[0]
        cases.append(
            (
                "分隔行列数少 1",
                "\n".join(
                    lines[: delim - 1] + [lines[delim - 1].rstrip().rstrip("|").rsplit("|", 1)[0] + "|"] + lines[delim:]
                ),
                "分隔行",
            )
        )
        if rows:
            ln = rows[0][0]
            cells_ = cells(lines[ln - 1])
            cases.append(
                (
                    "数据行少 1 列",
                    "\n".join(lines[: ln - 1] + ["| " + " | ".join(cells_[:-1]) + " |"] + lines[ln:]),
                    "数据行",
                )
            )
            # 孤立表行：插到表块之后的空行之后
            tail = start
            while tail < len(lines) and is_pipe(lines[tail]):
                tail += 1
            cases.append(
                (
                    "孤立表行（空行后单行 |…|）",
                    "\n".join(lines[:tail] + ["", "| 孤立 | 行 |"] + lines[tail:]),
                    "孤立表行",
                )
            )
    fenced = _fence_spans(lines)
    for i, line in enumerate(lines):
        if line.strip() and not is_pipe(line) and (i + 1) not in fenced and not FENCE.match(line):
            # 只注入到**散文行**：围栏代码块 / 行内代码里的 `**` 是字面文本，按设计不参与配对
            cases.append(("加粗未闭合", "\n".join(lines[:i] + [line + " **未闭合"] + lines[i + 1 :]), "奇数"))
            break
    for start, hdr, delim, rows in tabs:
        i_m, i_p, i_r = col_index(hdr, "实测"), col_index(hdr, "单点连乘"), col_index(hdr, "比值")
        if None in (i_m, i_p, i_r):
            continue
        for ln, c in rows:
            if max(i_m, i_p, i_r) < len(c) and lead_num(c[i_r]):
                cases.append(
                    (
                        "比值 ≠ 实测÷连乘",
                        "\n".join(lines[: ln - 1] + [lines[ln - 1].replace(c[i_r], "9.999", 1)] + lines[ln:]),
                        "列「比值」",
                    )
                )
                break
        else:
            continue
        break
    for start, hdr, delim, rows in tabs:
        i_up = col_index(hdr, "相对上一行")
        if i_up is None:
            continue
        picked = False
        for idx, (ln, c) in enumerate(rows):
            if idx == 0 or i_up >= len(c) or not PCT.search(c[i_up]):
                continue  # 首行的"上一行"在本表之外 → 按设计不可验证，不能作为用例
            cases.append(
                (
                    "相对上一行百分比错",
                    "\n".join(
                        lines[: ln - 1]
                        + [lines[ln - 1].replace(PCT.search(c[i_up]).group(0), "+99.9%", 1)]
                        + lines[ln:]
                    ),
                    "列「相对上一行」",
                )
            )
            picked = True
            break
        if not picked:
            continue
        break
    # 锚点缺失：把**所有**「基线」行改名 → 报告级与表级分母全部消失，必须显式报错
    hits = []
    for start, hdr, delim, rows in tabs:
        i_type = col_index(hdr, "优化类型")
        if i_type is None:
            continue
        for ln, c in rows:
            if i_type < len(c) and c[i_type].startswith("基线"):
                hits.append((ln, c[i_type]))
    if hits:
        m = lines[:]
        for ln, val in hits:
            m[ln - 1] = m[ln - 1].replace(val, "参考", 1)
        cases.append(("全部基线行锚点被改名", "\n".join(m), "分母"))
    return cases


def selftest(path: Path) -> int:
    """负样本自测：注入已知事故 → 必须各自被抓到；同一输入两次运行结论必须一致（幂等）。

    全部在内存里做（不写临时文件）：注入用例在真实报表文本上动态构造，故随报表结构自适应。
    """
    src = path.read_text(encoding="utf-8-sig")
    base, msg = run_audit(path)
    if base is None:
        print(f"[error] 自测基准不可审计：{msg}")
        return 2
    cases = _corruptions(src)
    if not cases:
        print(f"[error] 未能在 {path.name} 上构造出任何注入用例（报表结构不足）")
        return 2
    ok = True
    print(f"== 负样本自测：{path.name}（基准 error={len(base['errs'])}）")
    for name, mutated, sig in cases:
        res, _ = run_audit_text(mutated)
        res2, _ = run_audit_text(mutated)

        # 签名可出现在任一字段（位置/原值/应为/依据）——例如"奇数""比值"出现在「原值」里
        def _hit(r, sig=sig):
            return [x for x in r["errs"] if sig in (x.where + x.old + x.expect + x.basis)]

        caught = res is not None and bool(_hit(res))
        idem = (
            res is not None and res2 is not None and [x.fmt() for x in res["errs"]] == [x.fmt() for x in res2["errs"]]
        )
        print(
            f"  [{'PASS' if (caught and idem) else 'FAIL'}] {name:<26} 期望命中 {sig:<18} "
            f"幂等={'OK' if idem else 'NO'}"
            f"{'' if caught else '  实际命中=' + (str([x.where for x in res['errs']][:3]) if res else 'N/A')}"
        )
        ok = ok and caught and idem
    print(f"自测结论：{'全部抓到且幂等' if ok else '有漏检'}（{len(cases)} 个注入用例）")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="报表结构 + 数值自洽审计（只读、幂等）")
    ap.add_argument("report", type=Path)
    ap.add_argument(
        "--baseline", type=float, default=None, help="覆盖分母（例如细分报表只给相对值、或跨血缘表要用 §1 基线）"
    )
    ap.add_argument("--only", choices=("structure", "numeric", "all"), default="all")
    ap.add_argument("--strict", action="store_true", help="warn 也视为失败")
    ap.add_argument("--selftest", action="store_true", help="负样本自测：注入已知事故并断言被抓到")
    a = ap.parse_args(argv)

    if not a.report.exists():
        print(f"[error] 报表不存在：{a.report}")
        return 2
    if a.selftest:
        return selftest(a.report)
    res, msg = run_audit(a.report, a.only, a.baseline)
    if res is None:
        print("[error] " + msg)
        return 2
    rc = render(a.report, res)
    return 1 if (rc or (a.strict and res["warns"])) else 0


if __name__ == "__main__":
    sys.exit(main())
