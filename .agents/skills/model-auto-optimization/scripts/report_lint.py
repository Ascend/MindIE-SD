#!/usr/bin/env python3
"""report_lint.py —— overview_report.md 总览表结构与**可读性**校验（close 前置 · 零 NPU）。

校验（对照 model-auto-optimization/references/overview-report.md §2/§2.8/§7，
契约单点 = references/report-contract.md §7）：
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
5. **可读性契约（2026-09 主表三轮打回后的机械化出口，§2.8）**：
   - 单元格**不带开关/环境变量名**（`OMNI_H3_*` / `MINDIESD_*` 式全大写带下划线 token）、
     **不带裸内部代号**（`O2` / `C2` / `C3b` / `E4` 式单字母+数字 token）——具体对象进「说明」，
     开关名与代号进**附录**（正文按工程语言写）；白名单只放 `H3` 这类模型名（见 CODE_ALLOW）。
   - 「说明」列**可见文本**长度 ≤ NOTE_CAP（markdown 标记不计）——治「越详细越安全」写出的
     600–900 字单元格；NOTE_TARGET 只为不阻断的更紧目标值（超了给 warn，不算 error）。
   - `特性名` 含 `kernel融合` 的行：说明必须同时给**位置标记**（「位置：」/「· 位置」等）与
     **性能标记**（`每 step` / `×ms/step` / `非逐步项 … s/请求` 等）——齐「融合了什么 / 什么位置 /
     性能变化」三段（§2.8 紧凑说明模板）。
6. **必带章节**（§2.8）：`最佳路径下的特性详解` 详情节 + `附录`，附录内须含
   `开关对照表`（开关与环境变量对照表）/ `内部代号对照表` / `证据指针` 三个小节；
   详情节必须在附录之前（正文→附录的阅读顺序）。
7. **正文开关名 warn**（不阻断）：附录之前的正文若仍出现 `OMNI_H3_*` / `MINDIESD_*` 式 token，
   逐条 warn 并指向附录——历史报表的「可追溯性引用」不算错，但新增引用应向附录收敛。

用法:
    python report_lint.py <overview_report.md>
    python report_lint.py --selftest          # 负样本自测（内置夹具，零文件依赖）
退出码：0 = 通过；1 = 存在 error（不得 close）。
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

EXPECTED_HEADER = ["优化类型", "特性名", "e2e", "首步", "步数", "加速比", "质量", "说明"]
VALID_TYPES = {"基线", "无损优化", "免训练有损优化", "训练感知优化", "其他"}
# 锚点标记词：§2.1/§7 禁其进入特性名列，故只能扫整行（含说明列）。
ANCHOR_KEYWORDS = ("三元", "最强组合", "最终推荐")
# §2.2 三元组合固定名词：Cache + 量化(修饰符) + 稀疏（特性名按构成判定，不靠关键词）。
ANCHOR_COMBO = ("Cache", "量化", "稀疏")
LOSSY_TYPES = {"免训练有损优化", "训练感知优化"}

# --- §2.8 可读性契约常量（与 overview-report.md §2.8 / report-contract.md §7 成对维护） ---
#: 开关 / 环境变量名的**已知开关族前缀**（主判据）。族前缀是刻意的降假阳性设计：
#: 报表/证据文件名式词串（`BASELINE_REBASE_20260920`、`AMENDMENT_8CARD_ROOTCAUSE`、
#: `CT_CONVTRANSPOSE3D_REPORT`）与"各段无数字"的简写（`MUX_NO_FILL`、`DO_DUMP`）一律不判 ——
#: **误报会让门禁失去信任**（同 kb_lint「误报就改规则」纪律），故宁漏不误报。
#: 新增开关族时在 `SWITCH_FAMILY_PREFIXES` 登记一行即可（登记处即规则处，勿在正文里绕）。
SWITCH_FAMILY_PREFIXES = ("OMNI_H3", "MINDIESD_")
#: 开关 token 的段形态：全大写段 + 下划线，**至少 3 段**（`OMNI_H3_*` / `MINDIESD_*`_段_段）。
_SWITCH_MIN_SEGS = 3
_SWITCH_SEG_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
#: **显式 env 赋值形态**（补充判据）：`MINDIESD_XXX=1` / `OMNI_H3_XXX=true` —— 无论前缀是否登记
#: 都判（段数放宽到 ≥2）。`(?<![A-Za-z0-9_])` + `(?![A-Za-z0-9_])` 要求整个名字完整（不截断长名）。
ASSIGN_SWITCH_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)"
    r"(?![A-Za-z0-9_])"
    r"\s*=\s*(?=[A-Za-z0-9\"'\[({])"
)
#: **路径 / 文件名证据指针**：族前缀式 token 后面紧跟文件扩展名时按文件名放行
#: （`OMNI_H3_FOO_REPORT.md` 是证据指针，不是开关引用）。已知扩展名列成白名单，避免
#: `=1.0` 这类数值场景误判（`1.0` 不在白名单里，故不受影响）。
_EVIDENCE_SUFFIXES = (
    ".md",
    ".markdown",
    ".py",
    ".json",
    ".sh",
    ".ps1",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".log",
    ".csv",
    ".png",
    ".jpg",
    ".mp4",
)


def _is_evidence_reference(text: str, end: int) -> bool:
    """token 结束位置之后是否紧跟文件扩展名（→ 证据指针，不是开关引用）。"""
    low = text[end : end + 6].lower()
    return any(low.startswith(ext) for ext in _EVIDENCE_SUFFIXES)


#: 内部代号 token：单字母大写 + 一位数字（`O2` / `C2` / `C3b` / `E4` / `T2`）。
#: `H3`（模型名）、`L1/L2/L3`（等价性等级与 host 段杠杆）、`S1–S6`（阶段号）是**契约内用法**，
#: 故走白名单 —— 宁少报不误报，误报会让门禁失去信任（同 kb_lint「误报就改规则」纪律）。
CODE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Z]\d(?![A-Za-z0-9])")
CODE_ALLOW = ("H3", "L1", "L2", "L3", "S1", "S2", "S3", "S4", "S5", "S6")
#: 「说明」列可见文本上限：**能靠机制抓到的上限**（历史缺陷单元格 600–900 字）。
NOTE_CAP = 350
#: 目标值：更紧的写作目标，超了只 warn（**不阻断**：已定稿报表里更长的说明列仍可 close）。
NOTE_TARGET = 180
#: 说明列的「位置」标记：模板形如 `**位置**：…` / `· 位置：…`。
POSITION_RE = re.compile(r"位置|站点|层间|层内|通路|作用对象")
#: 说明列的「性能变化」标记：逐步给 `ms/step`/`s/step`，非逐步给 `s/请求`（/step 已覆盖 →/step）。
PERF_RE = re.compile(r"每\s*step|每步|/step|/请求|非逐步项|同窗")
#: 必带章节：[小节名, 允许的别名写法]；附录三个小节须齐（见 check_doc_structure）。
REQUIRED_SECTIONS = (
    ("最佳路径下的特性详解", ("最佳路径", "特性详解")),
    ("开关对照表", ("开关", "对照")),
    ("内部代号对照表", ("内部代号", "对照")),
    ("证据指针", ("证据", "指针")),
)
APPENDIX_HEADING_RE = re.compile(r"附录|appendix", re.IGNORECASE)
FUSION_FEATURE = "kernel融合"
_SPACER_CHARS = " \t\u3000*`|｜/\\."


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


def cells_of(line: str) -> list[str]:
    """按竖线切分表格行（去首尾竖线；不处理转义竖线 —— 结构事故由 audit_report 管）。"""
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------- §2.8 可读性检查


def visible_text(cell: str) -> str:
    """单元格的**可见文本**：去掉加粗/反引号/斜体标记后计数（长度上限按可见文本判）。"""
    return cell.replace("**", "").replace("`", "").replace("*", "")


def switch_tokens(text: str) -> list[str]:
    """开关 / 环境变量名式 token（§2.8：不得出现在主表，须入附录）。

    判据 = **已知开关族前缀**（`SWITCH_FAMILY_PREFIXES`，如 `OMNI_H3_*` / `MINDIESD_*`）
    **或显式 env 赋值形态** `NAME=value`；族前缀式还要求至少 `_SWITCH_MIN_SEGS` 段全大写下划线。
    这样"长得像常量"的**报表/证据文件名**（`BASELINE_REBASE_20260920`、`AMENDMENT_8CARD_ROOTCAUSE`
    —— 既无族前缀也不是赋值形态）与后跟扩展名的族前缀式文件名（`OMNI_H3_FOO_REPORT.md`）都不判。
    **误报会让门禁失去信任**（同 kb_lint「误报就改规则」纪律），故宁漏不误报；新增开关族在
    `SWITCH_FAMILY_PREFIXES` 登记。

    实现按"**下划线分段**"做（单个正则在"族前缀 + 至少 3 段"上要回溯，读写都难验证）：
    从每个全大写段起点贪心吃下连续的 `_大写段`，取满足段数与族前缀的**最长**前缀；
    不成立则从下一个段起点继续。赋值形态（`…=1` / `…=true`）由 `ASSIGN_SWITCH_RE` 捕获。
    """
    out: list[str] = []
    for m in ASSIGN_SWITCH_RE.finditer(text):
        name = m.group(1)
        # 赋值形态是强信号，段数放宽到 ≥2（`MINDIESD_XXX=1`）；但文件名/md 代码里的 `...=1` 仍放行
        if name.count("_") >= 1 and name not in out and not _is_evidence_reference(text, m.end(1)):
            out.append(name)
    i, n = 0, len(text)
    while i < n:
        if text[i] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or (i > 0 and (text[i - 1].isalnum() or text[i - 1] == "_")):
            i += 1
            continue
        j = i + 1
        while j < n and text[j] in _SWITCH_SEG_ALPHABET:
            j += 1
        segs = [text[i:j]]
        k = j
        while k < n and text[k] == "_":
            p = k + 1
            q = p
            while q < n and text[q] in _SWITCH_SEG_ALPHABET:
                q += 1
            if q == p:
                break
            segs.append(text[p:q])
            k = q
        token = ""
        if len(segs) >= _SWITCH_MIN_SEGS:
            full = "_".join(segs)
            if not any(full.startswith(prefix) for prefix in SWITCH_FAMILY_PREFIXES):
                i = j
                continue
            if _is_evidence_reference(text, i + len(full)):
                # 族前缀式 token 后面跟扩展名 ⇒ 文件名证据指针，不是开关引用
                i += len(full)
                continue
            token = full
        if token:
            if token not in out:
                out.append(token)
            i += len(token)
        else:
            i = j
    return out


def code_tokens(text: str) -> list[str]:
    """裸内部代号 token（白名单外的单字母+数字）。"""
    return [tok for tok in CODE_TOKEN_RE.findall(text) if tok not in CODE_ALLOW]


def check_row_readability(row_no: int, opt: str, feat: str, note: str) -> list[str]:
    """主表**单行**的可读性契约（§2.8）。返回 error 列表。"""
    errors: list[str] = []
    for tok in switch_tokens(opt) + switch_tokens(feat):
        errors.append(
            f"主表第 {row_no} 行（{feat}）：特性名/优化类型含开关或环境变量名「{tok}」"
            "（§2.8 开关名进附录，本表只写工程语言）"
        )
    for tok in code_tokens(opt) + code_tokens(feat):
        errors.append(
            f"主表第 {row_no} 行（{feat}）：特性名/优化类型含裸内部代号「{tok}」"
            "（§2.8 代号进附录对照表，本表只写固定名词）"
        )
    for tok in code_tokens(note):
        errors.append(
            f"主表第 {row_no} 行（{feat}）：说明列含裸内部代号「{tok}」（§2.8 具体对象用工程语言写，代号进附录对照表）"
        )
    vis = visible_text(note)
    if len(vis) > NOTE_CAP:
        errors.append(
            f"主表第 {row_no} 行（{feat}）：说明列可见文本 {len(vis)} 字 > 上限 {NOTE_CAP}"
            "（§2.8 紧凑模板：融合/调整了什么 → 在什么位置 → 每 step 性能变化；"
            "长文移「最佳路径下的特性详解」详情节）"
        )
    if FUSION_FEATURE in feat:
        if not POSITION_RE.search(note):
            errors.append(
                f"主表第 {row_no} 行（{feat}）：kernel融合行说明缺**位置**标记"
                "（§2.8 须回答「在哪一段/哪个站点」，如 `**位置**：…`）"
            )
        if not PERF_RE.search(note):
            errors.append(
                f"主表第 {row_no} 行（{feat}）：kernel融合行说明缺**性能变化**标记"
                "（§2.8 逐步给 `ms/step`/`s/step`，非逐步项给 `s/请求`）"
            )
    return errors


def check_row_readability_warn(row_no: int, feat: str, note: str) -> list[str]:
    """不阻断的更紧目标（超 NOTE_TARGET 只提示，报表仍可 close）。"""
    vis = len(visible_text(note))
    if NOTE_CAP >= vis > NOTE_TARGET:
        return [
            (
                f"主表第 {row_no} 行（{feat}）：说明列 {vis} 字 > 目标 {NOTE_TARGET}"
                f"（§2.8 建议值；上限 {NOTE_CAP}）——能移入详情节的细节尽量下移"
            )
        ]
    return []


# ---------------------------------------------------------------- 文档结构（必带章节）


def _norm_heading(heading: str) -> str:
    """标题归一化：去 `#`/加粗/空白/分隔符，便于按词判「这是不是那个小节」（不绑定编号）。"""
    text = re.sub(r"^#+\s*", "", heading.strip()).replace("**", "")
    for ch in _SPACER_CHARS:
        text = text.replace(ch, "")
    return text


def _heading_index(text: str) -> list[tuple[int, str]]:
    """全部标题 → (行号, 归一化标题)。"""
    out: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            out.append((i, _norm_heading(line)))
    return out


def check_doc_structure(text: str) -> list[str]:
    """必带章节检查（§2.8）：详情节 + 附录（开关对照表 / 内部代号对照表 / 证据指针）。"""
    errors: list[str] = []
    hits = _heading_index(text)
    if not hits:
        return [
            (
                "未找到任何标题——报表须含「最佳路径下的特性详解」详情节与「附录」"
                "（开关对照表 / 内部代号对照表 / 证据指针，§2.8）"
            )
        ]
    appendix_lines = [ln for ln, h in hits if APPENDIX_HEADING_RE.search(h)]
    detail_lines = [ln for ln, h in hits if all(k in h for k in REQUIRED_SECTIONS[0][1])]
    if not detail_lines:
        errors.append(
            "缺必带小节「最佳路径下的特性详解」（§2.8：主表说明列放不下的长文（机制 / shape / "
            "限定条件 / 证据）统一落此节）"
        )
    if not appendix_lines:
        errors.append(
            "缺必带「附录」节（§2.8：开关对照表 / 内部代号对照表 / 证据指针三小节；开关名与代号不得留在正文）"
        )
        return errors
    first_appendix = min(appendix_lines)
    if detail_lines and max(detail_lines) > first_appendix:
        errors.append("「最佳路径下的特性详解」位于附录之后（§2.8：正文（主表→详情节）在前、附录收尾）")
    for name, aliases in REQUIRED_SECTIONS[1:]:
        found = [ln for ln, h in hits if all(k in h for k in aliases) and ln > first_appendix]
        if not found:
            errors.append(f"附录缺必带小节「{name}」（§2.8：开关名 / 内部代号 / 证据指针各有其位）")
    return errors


def check_body_switches(text: str) -> list[str]:
    """正文（附录之前）残留开关/env 名 → warn（历史报表的可追溯性引用不算错）。"""
    warns: list[str] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if APPENDIX_HEADING_RE.search(_norm_heading(line)) and line.lstrip().startswith("#"):
            break
        for tok in switch_tokens(line):
            warns.append(
                f"正文 L{i} 含开关/环境变量名「{tok}」——§2.8 建议收进附录（开关对照表）；"
                "此处按可追溯性引用放行（warn 不阻断）"
            )
    return warns


# ---------------------------------------------------------------- 主表结构与可读性


def check_table(text: str) -> tuple[list[str], list[str]]:
    """主表结构 + 可读性检查，返回 (errors, warns)。"""
    errors: list[str] = []
    warns: list[str] = []
    lines = text.splitlines()
    table_start = None
    header_cells: list[str] = []
    for i, ln in enumerate(lines):
        if ln.strip().startswith("|") and "优化类型" in ln and "特性名" in ln:
            header_cells = cells_of(ln)
            table_start = i
            break
    if table_start is None:
        return (["未找到总览主表（含「优化类型|特性名」表头）——报表须含 §2 固定 8 列主表"], warns)

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
            errors.append(f"表头列序不符：期望第 {col_no} 列「{expect}…」，实际「{actual}」")
    if len(norm) != 8:
        errors.append(f"表头列数 ≠ 8：实际 {len(norm)} 列（不含序号列）{header_cells}")

    # 逐行解析主表（到第一个非 | 行或 §5 子表标题为止）
    ncol = len(norm)
    row_no = 0
    for ln in lines[table_start + 1 :]:
        s = ln.strip()
        if not s.startswith("|"):
            break
        if re.match(r"^\|[\s\-|:]+\|?$", s):  # 分隔行
            continue
        cells = cells_of(s)
        if has_idx and cells:
            cells = cells[1:]
        if len(cells) < ncol:
            # 可能行尾竖线缺失，补齐空
            cells += [""] * (ncol - len(cells))
        row_no += 1
        opt, feat, e2e, first, steps, speed, qual, note = cells[:8]
        if not feat:
            errors.append(f"主表第 {row_no} 行：特性名为空")
        if not opt:
            errors.append(f"主表第 {row_no} 行：优化类型为空")
        elif opt not in VALID_TYPES and not opt.startswith("基线"):
            errors.append(f"主表第 {row_no} 行：优化类型非法「{opt}」（枚举见 §7.1）")

        # e2e/首步/步数/加速比非空
        for col_name, val in (("e2e 耗时", e2e), ("首步耗时", first), ("步数", steps), ("加速比", speed)):
            if not val or val == "—":
                errors.append(f"主表第 {row_no} 行（{feat}）：{col_name} 列为空/「—」（§7 每行必填单值）")

        # 锚点行禁估算（锚点 = 基线 / 三元组合 / 最终推荐；判定见 _anchor_kind）
        anchor = _anchor_kind(opt, feat, " | ".join(cells))
        if anchor and "[估算]" in e2e:
            errors.append(f"主表第 {row_no} 行（{feat}）：锚点行 e2e 禁 [估算]（§1 必须实测；锚点={anchor}）")

        # 单值契约：e2e 只允许 数值 或 数值[估算…]；首步/步数/加速比只允许纯数值或 数值[估算…]
        # （同屏对比的单位/百分比一律进「说明」列 —— overview-report.md §2 单位契约）
        def _strip_md(val: str) -> str:
            out = val.replace("**", "").replace("`", "").replace("≈", "").replace("×", "")
            return out.replace("\u3000", " ").strip()

        def _single_value(col_name: str, val: str, rn: int, ft: str) -> None:
            core = _strip_md(val)
            m = re.fullmatch(r"([\d.,]+)\s*(\[[^\[\]]*\])?", core)
            ok = bool(m) and (m.group(2) is None or "估算" in m.group(2))
            if not ok:
                errors.append(
                    f"主表第 {rn} 行（{ft}）：{col_name} 列非单值「{val}」"
                    "（§2 只允许 数值 或 数值[估算]，口径/区间/括号注释进说明列）"
                )

        _single_value("e2e 耗时", e2e, row_no, feat)
        for col_name, val in (("首步耗时", first), ("步数", steps), ("加速比", speed)):
            _single_value(col_name, val, row_no, feat)

        # 质量列：无损 vs 有损
        if opt in VALID_TYPES and opt != "基线":
            if opt == "无损优化":
                if qual and qual != "—" and "输出一致" not in qual and "输出" not in qual:
                    # 宽松：无损质量列须提及输出一致性
                    errors.append(f"主表第 {row_no} 行（{feat}）：无损行质量列应表输出一致性，实际「{qual}」")
            elif opt in LOSSY_TYPES:
                if not qual or qual == "—":
                    errors.append(f"主表第 {row_no} 行（{feat}）：有损行质量列空（§4 须给数值/结论）")
                elif not re.search(r"SSIM|PSNR|质量门|pass|fail|inconclusive|0\.\d", qual):
                    errors.append(f"主表第 {row_no} 行（{feat}）：有损行质量列缺数值/结论「{qual}」")

        # §2.8 可读性（开关/代号/长度/融合三段式）
        errors.extend(check_row_readability(row_no, opt, feat, note))
        warns.extend(check_row_readability_warn(row_no, feat, note))

    if row_no == 0:
        errors.append("主表无数据行（基线行缺失）——报表须含基线行")
    return errors, warns


def lint(text: str) -> tuple[list[str], list[str]]:
    """全部检查入口（主表 + 必带章节 + 正文开关 warn），返回 (errors, warns)。"""
    errors, warns = check_table(text)
    errors.extend(check_doc_structure(text))
    warns.extend(check_body_switches(text))
    return errors, warns


# ---------------------------------------------------------------- 负样本自测

_SAMPLE_HEAD = """# 样例总览报表（自测夹具）

| 优化类型 | 特性名 | e2e 耗时 | 首步耗时 | 步数 | 加速比 | 质量数据 | 说明 |
|---|---|---|---|---|---|---|---|
| 基线 | TP×2 未优化 | 444.5 | 7.4 | 60 | 1.00× | — | 入口基线：2 卡同拓扑未优化；每 step 7.4 s/step |
| 无损优化 | kernel融合 | 360.7 | 6.0 | 60 | 1.23× | 输出一致（逐字节） | **融合**：注意力前 RMSNorm + RoPE 合成一个 kernel。**位置**：注意力前处理。**每 step：7.40 → 6.10 s** |
| 免训练有损优化 | Cache | 87.1 | 1.5 | 60 | 5.10× | SSIM 0.867 | 单点：缓存复用（框架自带后端）；每 step 7.40 → 1.50 s |
| 免训练有损优化 | Cache + 量化(w8a8) + 稀疏 | 49.5 | 0.8 | 60 | 8.98× | SSIM 0.838 | 三元锚点：缓存 + 线性层 8 bit + 块稀疏 80%；每 step 7.40 → 0.80 s；实测 |

### 9.7 最佳路径下的特性详解

> 主表说明列放不下的机制 / shape / 限定条件 / 证据落此节。

## 12. 附录：开关、内部代号与证据指针

### 12.1 附录 A｜开关与环境变量对照表

| 开关 / 环境变量 | 它控制哪个优化点 | 对应主表行 | 默认 | 现行 | 回退方式 |
|---|---|---|---|---|---|
| `OMNI_H3_FA_OVERLAP` | 通信-计算掩盖 | 行 2 | 7 | 7 | 置 0 |

### 12.2 附录 B｜内部代号对照表

| 代号 | 含义 |
|---|---|
| C3b | Q/K 前缀融合候选 |

### 12.3 附录 C｜证据与产物指针

- 产物：`runs/demo/best.mp4`
"""


def _sample_ok() -> str:
    return _SAMPLE_HEAD


def _sample_switch_token() -> str:
    return _SAMPLE_HEAD.replace("| 无损优化 | kernel融合 |", "| 无损优化 | OMNI_H3_FUSED_FFN_MXQ |")


def _sample_code_token() -> str:
    return _SAMPLE_HEAD.replace("| 无损优化 | kernel融合 |", "| 无损优化 | 稀疏（O2） |")


def _sample_code_token_in_note() -> str:
    return _SAMPLE_HEAD.replace("**位置**：注意力前处理。", "**位置**：注意力前处理（C4）。")


def _sample_too_long() -> str:
    return _SAMPLE_HEAD.replace("**每 step：7.40 → 6.10 s**", "细节：" + "长" * 400)


def _sample_no_position() -> str:
    return _SAMPLE_HEAD.replace("**位置**：注意力前处理。", "")


def _sample_no_perf() -> str:
    return _SAMPLE_HEAD.replace("**每 step：7.40 → 6.10 s**", "")


def _sample_no_appendix() -> str:
    head, _, _ = _SAMPLE_HEAD.partition("## 12. 附录")
    return head


def _sample_no_detail() -> str:
    return _SAMPLE_HEAD.replace("### 9.7 最佳路径下的特性详解", "### 9.7 其他小节")


def _sample_no_switch_table() -> str:
    return _SAMPLE_HEAD.replace("### 12.1 附录 A｜开关与环境变量对照表", "### 12.1 附录 A｜其他表")


def _sample_no_code_table() -> str:
    return _SAMPLE_HEAD.replace("### 12.2 附录 B｜内部代号对照表", "### 12.2 附录 B｜其他表")


def _sample_no_evidence() -> str:
    return _SAMPLE_HEAD.replace("### 12.3 附录 C｜证据与产物指针", "### 12.3 附录 C｜其他表")


def _sample_body_switch_assign() -> str:
    """正文里显式注入 `OMNI_H3_XXX=1` / `MINDIESD_XXX=1`（§2.8 正文开关名 warn 必须触发）。"""
    return _SAMPLE_HEAD.replace(
        "## 12. 附录",
        "现行档由 `OMNI_H3_FA_OVERLAP=7` 与 `MINDIESD_QK_FAT_PROGRAM=1` 控制（本地改动）。\n\n## 12. 附录",
    )


def _sample_evidence_filenames() -> str:
    """证据指针文件名（`.md`，全大写下划线但**不是**开关族）不得被判成开关名。"""
    return _SAMPLE_HEAD.replace(
        "## 12. 附录",
        "证据指针：`BASELINE_REBASE_20260920.md`、`AMENDMENT_8CARD_ROOTCAUSE.md`、"
        "`CT_CONVTRANSPOSE3D_REPORT.md`（均为报告文件名）。\n\n## 12. 附录",
    )


#: (用例名, 生成函数, 期望命中的 error 片段, 期望命中的 warn 片段)。
#: error/warn 两栏分开断言：正文开关名是 warn（不阻断），主表内才是 error（§2.8）。
SELFTEST_CASES = (
    ("合规样例（error=0）", _sample_ok, "", ""),
    ("特性名含开关名", _sample_switch_token, "开关或环境变量名", ""),
    ("特性名含内部代号", _sample_code_token, "裸内部代号", ""),
    ("说明列含内部代号", _sample_code_token_in_note, "说明列含裸内部代号", ""),
    ("说明列超长", _sample_too_long, "可见文本", ""),
    ("fusion 行缺位置标记", _sample_no_position, "缺**位置**标记", ""),
    ("fusion 行缺性能标记", _sample_no_perf, "缺**性能变化**标记", ""),
    ("缺附录", _sample_no_appendix, "缺必带「附录」", ""),
    ("缺特性详解节", _sample_no_detail, "最佳路径下的特性详解", ""),
    ("附录缺开关对照表", _sample_no_switch_table, "开关对照表", ""),
    ("附录缺内部代号对照表", _sample_no_code_table, "内部代号对照表", ""),
    ("附录缺证据指针", _sample_no_evidence, "证据指针", ""),
    ("正文注入 env 开关（应 warn）", _sample_body_switch_assign, "", "含开关/环境变量名"),
    ("证据文件名（应完全不报）", _sample_evidence_filenames, "", ""),
)


def selftest() -> int:
    """负样本自测：每条新规则都必须被触发，合规样例必须 error=0/warn=0（门禁自身有效）。"""
    failures: list[str] = []
    print("== report_lint 负样本自测（主表可读性 §2.8）==")
    for name, make, expect_err, expect_warn in SELFTEST_CASES:
        errs, warns = lint(make())
        if not expect_err and not expect_warn:
            ok = not errs and not warns
            if errs:
                failures.append(f"{name}：合规样例误报 {len(errs)} 条（{errs[0]}）")
            for w in warns:
                failures.append(f"{name}：合规样例误 warn（{w}）")
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}（error={len(errs)} warn={len(warns)}）")
            continue
        if expect_err:
            hit = [e for e in errs if expect_err in e]
            if not hit and not [w for w in warns if expect_err in w]:
                failures.append(f"{name}：期望命中「{expect_err}」，实际 error={[e[:60] for e in errs][:2]}")
            print(f"  [{'PASS' if hit else 'FAIL'}] {name}：期望 error 命中「{expect_err}」")
        if expect_warn:
            hit = [w for w in warns if expect_warn in w]
            if not hit:
                failures.append(f"{name}：期望 warn 命中「{expect_warn}」，实际 warn={[w[:60] for w in warns][:2]}")
            print(f"  [{'PASS' if hit else 'FAIL'}] {name}：期望 warn 命中「{expect_warn}」")
    # 白名单回归：契约内标识符（模型名 / 等价性等级 / 阶段号 / 算子名 / 权重表达式）不得被判成代号
    allow_probe = (
        "**融合**：按 `W' = W + (α/r)·B·A` 合并适配器（H3 模型，L1 逐字节，S4 阶段）；"
        "**位置**：FFN 层内 fc1/RMSNorm。**每 step：3.46 → 3.28 s**；非逐步项：解码 5.08 s/请求"
    )
    errs = check_row_readability(1, "无损优化", "kernel融合", allow_probe)
    if errs:
        failures.append(f"白名单回归：契约内标识符被误报（{errs[0]}）")
    print(f"  [{'PASS' if not errs else 'FAIL'}] 白名单回归：H3 / L1 / S4 / fc1 / RMSNorm / W' 表达式不被误报")
    # 幂等：同一输入两次结论一致
    a, _ = lint(_SAMPLE_HEAD)
    b, _ = lint(_SAMPLE_HEAD)
    if a != b:
        failures.append("幂等性：同一输入两次结论不一致")
    if failures:
        print("report_lint --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"report_lint --selftest: PASS（{len(SELFTEST_CASES)} 个用例：合规样例 0 error，负样本逐条触发）")
    return 0


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, nargs="?")
    ap.add_argument("--selftest", action="store_true", help="负样本自测（内置夹具）并退出")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.report is None:
        ap.error("缺少报表路径（或用 --selftest）")
    if not args.report.exists():
        print(f"[error] 报表不存在：{args.report}")
        return 1
    text = args.report.read_text(encoding="utf-8-sig")
    errors, warns = lint(text)
    for e in errors:
        print(f"[error] {e}")
    for w in warns:
        print(f"[warn] {w}")
    print(f"结论：error={len(errors)} warn={len(warns)}（{args.report.name}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
