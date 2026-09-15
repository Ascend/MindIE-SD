#!/usr/bin/env python3
"""知识层结构门禁（`.agents/` 技能库）——13 条规则，把「文档化纪律」变成「可执行门禁」。

把 `.agents/` 当**有向图**（节点 = SKILL.md / references / scripts / evals，边 = 相对路径指针），
校验图的不变量与接线契约；**不判结论对错**（那属语义层，需模型或人）——只让「没接线、指向不存在、
规范缺失」fail（与 `stage_gate.py` 同构：门禁校验证据与结构齐备性，不校验判断本身）。
零依赖、只读、幂等，可安全进 CI。

规则（`--list-rules` 可列出）：

| 规则 | 级别 | 判什么 |
|------|------|--------|
| KB001 | error | 相对引用必须可解析（悬空引用/死链） |
| KB002 | error | `references/*.md` 必须在其所属 SKILL.md 登记（防孤儿） |
| KB003 | error | 技能目录名与 reference 文件名须 kebab-case（连字符、不用下划线） |
| KB004 | error | SKILL.md frontmatter 必含 `name` / `description` / `compatibility` |
| KB005 | error | `evals/evals.json` 存在、可解析、≥2 条含 `prompt` 与 `expectations` |
| KB006 | error | >300 行的 reference 必须带目录（progressive disclosure 可导航） |
| KB007 | error | README 声明的技能计数须与磁盘目录一致（「唯一权威」不得自相矛盾） |
| KB009 | error | 仓内坐标（`docs/` `mindiesd/` …）必须可解析 |
| KB010 | warn | reference 应含「维护与更新」章节（写明更新触发条件） |
| KB011 | warn | SKILL.md 应含「维护与更新」章节 |
| KB012 | warn | SKILL.md 登记 reference 时应写明「加载时机」（只列路径不算接线） |
| KB013 | error | 仓内消费方指向 `.agents/**` 的引用必须可解析（反向边） |
| KB014 | warn | 携带经验结论的文件须含「复核方法 / 失效信号」（结论有寿命） |

退出码：0 = 无 error（warn 不阻断）；1 = 存在 error；2 = 前置条件缺失（`.agents/` 不存在）。
`--strict` 时 warn 也计为失败。

用法：

```bash
python .agents/scripts/kb_lint.py                    # 全量检查
python .agents/scripts/kb_lint.py --list-rules       # 列出规则
python .agents/scripts/kb_lint.py --rule KB001       # 只跑一条规则
python .agents/scripts/kb_lint.py --format json      # 机器可读
python .agents/scripts/kb_lint.py --selftest         # 负样本自测（验证门禁本身有效）
```

背景：本仓的引用完整性与接线纪律此前**只是文档约定**（悬空引用靠人工复扫）。本脚本是这些约定的
机械化出口。**规则细节、豁免机制、历史理由、已知盲区与积压登记见同目录 `README.md`**（与脚本同源同改）；
规则清单与级别以本文件 `RULES` 为准，`.agents/README.md` §7 只给当前口径与用法。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------- 规则注册表

RULES: dict[str, tuple[str, str]] = {
    "KB001": ("error", "相对引用必须可解析（悬空引用/死链）"),
    "KB002": ("error", "references/*.md 必须在其所属 SKILL.md 登记（防孤儿）"),
    "KB003": ("error", "技能目录名与 reference 文件名须 kebab-case"),
    "KB004": ("error", "SKILL.md frontmatter 必含 name / description / compatibility"),
    "KB005": ("error", "evals/evals.json 存在、可解析、≥2 条含 prompt 与 expectations"),
    "KB006": ("error", ">300 行的 reference 必须带目录"),
    "KB007": ("error", "README 声明的技能计数须与磁盘目录一致"),
    "KB009": ("error", "仓内坐标（docs/ mindiesd/ …）必须可解析"),
    "KB010": ("warn", "reference 应含「维护与更新」章节"),
    "KB011": ("warn", "SKILL.md 应含「维护与更新」章节"),
    "KB012": ("warn", "SKILL.md 登记 reference 时应写明「加载时机」"),
    "KB013": ("error", "仓内消费方指向 .agents/** 的引用必须可解析（反向边）"),
    "KB014": ("warn", "携带经验结论的文件须含『复核方法/失效信号』（结论有寿命）"),
}

ORDER = list(RULES)

# ---------------------------------------------------------------- 常量

REF_EXTS = (".md", ".py", ".sh", ".ps1", ".json", ".toml", ".yaml", ".yml", ".txt")

KEBAB_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

BACKTICK_RE = re.compile(r"`([^`\n]+)`")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")

# 含这些字符的 token 一律视为占位符/模式/外部坐标，不做解析
SKIP_CHARS = set("{}<>*|$\\\"'…@")

# 运行期产物与仓内简写坐标：不在本仓树内（或真源在别处），不该被解析。
# 每条都要写清豁免理由；新增条目须同时在此注释与 README §7 说明。
IGNORE_PREFIXES = (
    # 运行期产物目录（真源 references/artifact-layout.md，不入 git）
    "runs/",
    "./runs/",
    "evidence/",
    "output/",
    "profiles/",
    # 构建/依赖产物
    "build/",
    "dist/",
    "node_modules/",
    # 外部技能库（真源在 $CANNBOT_SKILLS_DIR，非本仓路径）
    "cannbot",
    # mindiesd 包内部路径简写（真源 mindiesd/compile/passes/，prose 中按包内相对写）
    "passes/",
)

# token 必须以「段 + /」开头才算路径（挡掉裸文件名与 `.sh` 这类后缀片段）
PATH_HEAD_RE = re.compile(r"^[.A-Za-z0-9_-]+/")

# 末段主干是占位符的 token 不判（`references/x.md`、`tests/test_xxx.py` 这类是示例写法）
PLACEHOLDER_STEM_RE = re.compile(r"^(x+|xx+|foo|bar|baz|test|test_.*|example|sample|placeholder|tbd)$", re.IGNORECASE)

# KB 内部坐标：只有这些首段才进 KB001 的判定范围。
# 范围界定理由：本门禁校验的是 **KB 自身的链接图**（技能间指针、reference/script/evals 接线）。
# 仓内其它坐标（docs/、mindiesd/、tests/）与外部框架仓源码路径（lightx2v_platform/、
# diffsynth_engine/、torch/）、运行期产物（runs/）各有其真源与治理，不在本图内 —— 纳入只会
# 制造大量「合法但不在本仓」的假违规，让门禁失去信号。
KB_DIRS = ("references", "scripts", "workflows", "evals", "assets")

# KB 自身的根级文件（`.agents/README.md` 是 L1 索引，技能文件大量反向引用它的 §2/§4/§7；
# 缺了这一段，README 被删除/改名时不会有任何门禁报警）
KB_ROOTS = (".agents",)

# 仓内坐标（KB009）：技能正文会引用本仓其它目录（如 `docs/zh/feat*.md`、`mindiesd/**`）。
# 这些坐标**不属于** KB 链接图，但会因仓库重组而腐烂（实测：治理技能曾教人检查已不存在的
# `docs/index.md` 与 `menu_user_manual.md`）。只登记**本仓真实存在的顶层目录**，
# 外部框架仓路径（lightx2v_platform/、diffsynth_engine/ …）不在其列，故天然不判。
REPO_COORD_DIRS = (
    "docs",
    "mindiesd",
    "tests",
    "benchmarks",
    "examples",
    "csrc",
    "docker",
    "pre-commit",
    ".gitcode",
)

# 外部仓坐标（本仓确实不含）：按 token 前缀整段豁免
EXTERNAL_REFS = (
    # 外部 skill-creator 仓的 evals schema 文档（dev-workflow「新增 Skill 规范」引用）
    "references/schemas.md",
)

# 仓内「消费方」文件：它们引用 `.agents/**`（**反向边**）。KB001/KB009 只从 .agents 往外看，
# 反向边此前无门禁 —— 实测起点：产品侧 `evals/README.md` 曾指向已迁走的
# `performance-optimization/references/quality-gate.md`。
#
# ⚠️ **依赖方向（强制）**：**skills 引用 evals，反之不成立**。因此**不把 `evals/**` 纳入本清单** ——
# 那等于让 skills 层的门禁去治理产品侧目录（反向依赖）；正确解法是**消除跨层引用**（产品侧
# `evals/` 已改为不引用 `.agents/` 路径），而不是给它加一道由 skills 侧施加的门禁。
# 本清单只保留**产品侧根级入口**（README / README.en 是让人发现 skills 层的入口，属导航而非依赖）。
CONSUMER_FILES = ("README.md", "README.en.md")

# 「携带经验结论」的文件类（KB014）：只有这几类才有「结论会不会过期」的问题。
# 机制/模板/清单/表（run-state、artifact-layout、pattern-templates、mismatch-catalog、
# dispatch-table、analysis-flow 等）没有可失效的结论，**不该**被要求写失效信号——
# 实测：不分类地要求「每个 reference 都有失效信号」会产出 38 条、其中绝大多数是误报。
CONCLUSION_SUFFIXES = ("-method", "-enablement", "-notes", "-case")
CONCLUSION_PREFIXES = ("troubleshooting-",)
LIFETIME_RE = re.compile(r"失效信号|复核|重测|重新核对|仍存在|是否仍成立|再核")


# 三层架构的层成员（KB007 用）。README 自称「架构（三层）与技能清单唯一权威」——
# 计数一旦与磁盘不一致，权威性即失效（实测曾同时出现 18/21/23 三套计数）。
# 分层调整时须同步本表与 README「架构速览」。
LAYER_L1 = ("model-auto-optimization", "dev-workflow")
LAYER_L2 = ("performance-optimization",)

# README 里**声明计数**的规范句式（只判这些固定句式，避免把「某次会话里其余 17 个技能」
# 这类历史叙述也算成当前声明）。期望值由磁盘目录数推导。
# 注：第三/四个句式分别锚定「L3 数」与「域入口+能力层数」，二者同出于 total，故 total 本身
# 已被隐式钉住；`**N 个技能**` 句式原挂在状态清单（P1-4 已删），保留以覆盖再次出现的写法。
README_COUNT_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (r"L3 能力（(\d+)）", "l3", "架构速览：L3 能力数"),
    (r"优化域入口与能力层（(\d+)）", "non_l1", "§2 标题：域入口+能力层数"),
    (r"能力层 (\d+)（L3）", "l3", "分层说明：能力层数"),
    (r"\*\*(\d+) 个技能\*\*", "total", "技能总数声明"),
)

# 历史引用标记（本仓既有约定）：迁移/更名/删除记录里出现旧名是**有意**的，不算悬空。
# 见 `.agents/README.md`「compilation-dev SKILL 改为指引现有 GraphPatternEntry 指南并标「已退役」」。
HISTORICAL_MARKERS = (
    "已退役",
    "已删除",
    "删除",
    "git rm",
    "已合并",
    "已迁",
    "迁移记录",
    "原名",
    "旧名",
    "历史名",
    "已废弃",
    "废弃",
    "已移除",
    "从未提交",
    "未合入",
    # 「不存在 / 本仓无」：**说明某路径缺失**的行本身就是在记录缺失（如历史理由表里列出
    # 「曾教人检查 5 个不存在的文件」），属有意缺失 —— 按「误报就改规则」扩词，不改文字。
    "不存在",
)

# 外部坐标标记：prose 里已显式声明「不是本仓文件」的引用，跳过（本仓既有写法，如
# `$CANNBOT_SKILLS_DIR/...（非本仓路径，不可按本文件相对解析）`、catlass「外部库」`scripts/build.sh`）。
EXTERNAL_MARKERS = (
    "外部库",
    "非本仓",
    "外部仓",
    "外部技能库",
    "非本文件相对解析",
    "不在本仓",
    "本仓不含",
    "本仓无",
)

TOC_RE = re.compile(r"内容索引|\*\*目录\*\*|^#+\s*目录|本文件目录|目录 ·|^\|\s*章节\s*\|", re.MULTILINE)

MAINTENANCE_RE = re.compile(r"^#+.*维护与更新", re.MULTILINE)

FRONTMATTER_KEYS = ("name", "description", "compatibility")

LARGE_REF_LINES = 300

_SKILLS_DIRNAME = "skills"
_AGENTS_DIRNAME = ".agents"


class Finding:
    """一条门禁发现。"""

    __slots__ = ("line", "message", "path", "rule", "severity")

    def __init__(self, rule: str, path: str, line: int, message: str) -> None:
        self.rule = rule
        self.severity = RULES[rule][0]
        self.path = path
        self.line = line
        self.message = message

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.path,
            "line": self.line,
            "message": self.message,
        }


# ---------------------------------------------------------------- 基础设施


def read_text(path: Path) -> str:
    """UTF-8 读取（含中文文件必须显式指定编码，勿依赖平台默认）。"""
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def line_of(text: str, needle: str) -> int:
    idx = text.find(needle)
    if idx < 0:
        return 1
    return text.count("\n", 0, idx) + 1


def marker_scope(lines: list[str], idx: int) -> str:
    """返回判定「历史/外部标记」时用的文本范围。

    - **表格行**：只看本行（行是独立记录，不跨行继承标记）；
    - **其余**（列表项/段落/引用块）：看整个**连续非空块** —— 列表项常跨多行书写，
      标记写在首行、路径写在次行是常态（首版只看同行，实测漏判 2 处）。
    """
    if lines[idx].lstrip().startswith("|"):
        return lines[idx]
    start = idx
    while start > 0 and lines[start - 1].strip():
        start -= 1
    end = idx
    while end + 1 < len(lines) and lines[end + 1].strip():
        end += 1
    return "\n".join(lines[start : end + 1])


def is_exempt(lines: list[str], lineno: int) -> bool:
    """该行是否带「历史/外部」标记（迁移记录、已移除清单、外部库标注等）。"""
    idx = lineno - 1
    if not 0 <= idx < len(lines):
        return False
    scope = marker_scope(lines, idx)
    return any(mark in scope for mark in HISTORICAL_MARKERS + EXTERNAL_MARKERS)


def skill_dirs(agents: Path) -> list[Path]:
    base = agents / _SKILLS_DIRNAME
    if not base.is_dir():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir())


def reference_files(skill: Path) -> list[Path]:
    refs = skill / "references"
    if not refs.is_dir():
        return []
    return sorted(p for p in refs.rglob("*.md") if p.is_file())


# ---------------------------------------------------------------- KB001


def _looks_like_path(token: str) -> bool:
    token = token.strip().rstrip(".,;:)")
    if not token or any(ch in SKIP_CHARS for ch in token):
        return False
    if token.startswith(("http://", "https://", "mailto:", "/", "~")):
        return False
    if token.startswith(IGNORE_PREFIXES):
        return False
    if "://" in token or token.endswith("/"):
        return False
    if not PATH_HEAD_RE.match(token):
        return False  # 裸文件名与后缀片段不是路径引用，不判
    if not token.endswith(REF_EXTS):
        return False
    stem = token.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    # 示例/占位写法（`references/x.md`、`tests/test_xxx.py`）不是真实引用
    return not (PLACEHOLDER_STEM_RE.match(stem) or "xxx" in token.lower())


def _candidate_tokens(text: str) -> list[str]:
    tokens = [m.group(1) for m in BACKTICK_RE.finditer(text)]
    tokens += [m.group(1) for m in LINK_RE.finditer(text)]
    out = []
    for raw in tokens:
        token = raw.split("#", 1)[0].strip()
        if _looks_like_path(token):
            out.append(token)
    return out


def _kb_path_index(agents: Path) -> set[str]:
    """`.agents` 下全部文件的 posix 相对路径（供后缀兜底匹配，只建一次）。"""
    cached = _PATH_INDEX_CACHE.get(agents)
    if cached is None:
        cached = {p.relative_to(agents).as_posix() for p in agents.rglob("*") if p.is_file()}
        _PATH_INDEX_CACHE[agents] = cached
    return cached


_PATH_INDEX_CACHE: dict[Path, set[str]] = {}


def _resolve(token: str, source: Path, skill: Path | None, agents: Path) -> bool:
    bases = [source.parent]
    if skill is not None:
        bases.append(skill)
    # 跨技能引用两种写法都要认：`../other-skill/...`（相对本技能）与
    # `other-skill/references/...`（相对 skills/，本仓大量使用后者）
    bases.append(agents / _SKILLS_DIRNAME)
    bases.append(agents)
    bases.append(agents.parent)
    for base in bases:
        try:
            if (base / token).exists():
                return True
        except OSError:
            continue
    # 后缀兜底：KB 内部也常见「只写技能内相对路径」（references/x.md、scripts/y.py），
    # 无论写在哪个技能/README 里都指向同一个文件 —— 图里存在即算可达。
    tail = "/" + token.lstrip("./")
    return any(path.endswith(tail) for path in _kb_path_index(agents))


def _in_kb_scope(token: str, skill_names: set[str]) -> bool:
    """token 是否属 KB 自身链接图（见 KB_DIRS 的范围界定理由）。"""
    if token.startswith("../"):
        return True
    first = token.split("/", 1)[0]
    return first in KB_DIRS or first in KB_ROOTS or first in skill_names


def rule_kb001(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    skill_names = {s.name for s in skills}
    for path in sorted(agents.rglob("*.md")):
        text = read_text(path)
        lines = text.splitlines()
        skill = None
        for parent in path.parents:
            if parent.parent == agents / _SKILLS_DIRNAME:
                skill = parent
                break
        for token in sorted(set(_candidate_tokens(text))):
            if not _in_kb_scope(token, skill_names) or token in EXTERNAL_REFS:
                continue
            if _resolve(token, path, skill, agents):
                continue
            lineno = line_of(text, token)
            if is_exempt(lines, lineno):
                continue  # 迁移/退役/删除记录与「外部库」标注里出现旧名是有意的
            findings.append(Finding("KB001", rel(path, agents.parent), lineno, f"KB 内相对引用无法解析：{token}"))
    return sorted(findings, key=lambda f: (f.path, f.line))


# ---------------------------------------------------------------- KB002 / KB012


def rule_kb002(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        skill_md = skill / "SKILL.md"
        body = read_text(skill_md) if skill_md.is_file() else ""
        for ref in reference_files(skill):
            if ref.name in body:
                continue
            findings.append(
                Finding("KB002", rel(ref, agents.parent), 1, f"未在 {skill.name}/SKILL.md 登记（孤儿 reference）")
            )
    return findings


REF_SECTION_RE = re.compile(r"^#+.*(reference files|references 清单|参考文件)", re.IGNORECASE)

# 「加载时机」的等价写法：本仓实际用过 加载时机 / 何时读 / 时读（「要 X 时读它」）/ Phase N
TIMING_RE = re.compile(r"加载时机|何时读|读取时机|时读|Phase\s*\d")

# 显式声明「登记在别处、此处不重复」的行：是**指针**不是登记，不该按登记判
DEFER_MARKERS = ("不重复登记", "单点登记于", "此处不重复", "不在本文件")


def _reference_section_lines(text: str) -> list[tuple[int, str]]:
    """取出「Reference Files」章节内的非空行（行号, 内容）。

    只查这一区，不查全文 —— 正文里顺带提到的 `references/x.md` 属上下文引用，
    不是「接线」声明，纳入判定会把信号淹掉（首版实测 120 处里多数是误报）。
    """
    out: list[tuple[int, str]] = []
    inside = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            inside = bool(REF_SECTION_RE.match(line))
            continue
        if inside and line.strip():
            out.append((lineno, line))
    return out


def _timing_column_index(header: str) -> int | None:
    """表头里表达「时机」的列下标；无则 None。"""
    cells = [c.strip() for c in header.strip().strip("|").split("|")]
    for idx, cell in enumerate(cells):
        if TIMING_RE.search(cell):
            return idx
    return None


def rule_kb012(agents: Path, skills: list[Path]) -> list[Finding]:
    """接线区登记须带「加载时机」语义。

    判定按**语义来源**分三种（首版只认行内字面量「加载时机」，实测 23 处**全部**是误报）：
    ① **表格**：表头列名表达时机（`加载时机` / `何时读`）时，**该列的单元格内容即时机的值** ——
       单元格非空者放行，**空单元格仍要报**（表头不能替空值背书）；表头无时机列则退回按行判（单行）；
    ② **列表项/段落**：在**连续非空块**内认「加载时机 / 何时读 / 时读 / Phase N」等写法
       （条目常跨行：路径在首行、时机在次行）；
    ③ **显式让位**：写出「单点登记于 §X / 此处不重复登记」的行是指针，不按登记判。
    **已知代价**：`时读` 这类宽写法会带来少量漏判（宁漏不误报——规则要先能被信任）。
    """
    findings: list[Finding] = []
    for skill in skills:
        skill_md = skill / "SKILL.md"
        if not skill_md.is_file():
            continue
        text = read_text(skill_md)
        all_lines = text.splitlines()
        section = _reference_section_lines(text)
        if not section:
            continue
        header = next((line for _, line in section if line.lstrip().startswith("|")), "")
        timing_idx = _timing_column_index(header) if header else None

        for lineno, line in section:
            if not re.search(r"references/[a-z0-9-]+\.md", line):
                continue
            if line.lstrip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if timing_idx is not None:
                    if len(cells) > timing_idx and cells[timing_idx] not in ("", "-"):
                        continue  # ① 时机列有值
                elif TIMING_RE.search(line):
                    continue
            elif TIMING_RE.search(marker_scope(all_lines, lineno - 1)):
                continue  # ② 块内有时机写法
            if any(mark in line for mark in DEFER_MARKERS):
                continue  # ③ 显式让位
            findings.append(Finding("KB012", rel(skill_md, agents.parent), lineno, "接线区登记缺口「加载时机」语义"))
    return findings


# ---------------------------------------------------------------- KB003


def rule_kb003(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        if not KEBAB_RE.match(skill.name):
            findings.append(Finding("KB003", rel(skill, agents.parent), 1, f"技能目录名非 kebab-case：{skill.name}"))
        for ref in reference_files(skill):
            if KEBAB_RE.match(ref.stem):
                continue
            findings.append(Finding("KB003", rel(ref, agents.parent), 1, f"reference 文件名非 kebab-case：{ref.name}"))
    return findings


# ---------------------------------------------------------------- KB004 / KB011


def rule_kb004(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        skill_md = skill / "SKILL.md"
        if not skill_md.is_file():
            findings.append(Finding("KB004", rel(skill, agents.parent), 1, "缺 SKILL.md"))
            continue
        text = read_text(skill_md)
        head = text.split("\n---", 1)[0] if text.startswith("---") else text[:2000]
        for key in FRONTMATTER_KEYS:
            if re.search(rf"(?m)^{key}:", head):
                continue
            findings.append(
                Finding(
                    "KB004",
                    rel(skill_md, agents.parent),
                    1,
                    f"frontmatter 缺 `{key}`（见 dev-workflow「新增 Skill 规范」）",
                )
            )
    return findings


def rule_kb011(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        skill_md = skill / "SKILL.md"
        if skill_md.is_file() and not MAINTENANCE_RE.search(read_text(skill_md)):
            findings.append(Finding("KB011", rel(skill_md, agents.parent), 1, "缺「维护与更新」章节"))
    return findings


# ---------------------------------------------------------------- KB005


def rule_kb005(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        evals = skill / "evals" / "evals.json"
        loc = rel(evals, agents.parent)
        if not evals.is_file():
            findings.append(Finding("KB005", rel(skill, agents.parent), 1, "缺 evals/evals.json"))
            continue
        try:
            data = json.loads(read_text(evals))
        except json.JSONDecodeError as exc:
            findings.append(Finding("KB005", loc, exc.lineno, f"evals.json 解析失败：{exc.msg}"))
            continue
        cases = data.get("evals") if isinstance(data, dict) else None
        if not isinstance(cases, list) or len(cases) < 2:
            findings.append(Finding("KB005", loc, 1, "evals 用例数 < 2"))
            continue
        for idx, case in enumerate(cases, start=1):
            if not isinstance(case, dict):
                findings.append(Finding("KB005", loc, 1, f"第 {idx} 条用例不是对象"))
                continue
            missing = [k for k in ("prompt", "expectations") if not case.get(k)]
            if missing:
                findings.append(Finding("KB005", loc, 1, f"第 {idx} 条用例缺字段：{', '.join(missing)}"))
    return findings


# ---------------------------------------------------------------- KB007


def _layer_counts(skills: list[Path]) -> dict[str, int] | None:
    """由磁盘目录推导各层计数；层成员缺失时返回 None（不判，避免误报）。"""
    names = {s.name for s in skills}
    if not set(LAYER_L1) <= names or not set(LAYER_L2) <= names:
        return None
    total = len(names)
    l1 = len(set(LAYER_L1) & names)
    l2 = len(set(LAYER_L2) & names)
    return {"total": total, "l1": l1, "l2": l2, "l3": total - l1 - l2, "non_l1": total - l1}


def rule_kb007(agents: Path, skills: list[Path]) -> list[Finding]:
    """README 是「技能清单唯一权威」——计数声明必须与磁盘一致，否则权威性失效。"""
    readme = agents / "README.md"
    counts = _layer_counts(skills)
    if not readme.is_file() or counts is None:
        return []
    findings: list[Finding] = []
    for lineno, line in enumerate(read_text(readme).splitlines(), start=1):
        for pattern, key, label in README_COUNT_PATTERNS:
            for match in re.finditer(pattern, line):
                declared = int(match.group(1))
                if declared == counts[key]:
                    continue
                findings.append(
                    Finding(
                        "KB007",
                        rel(readme, agents.parent),
                        lineno,
                        f"{label}声明为 {declared}，磁盘实为 {counts[key]}",
                    )
                )
    return findings


# ---------------------------------------------------------------- KB009


def rule_kb009(agents: Path, _skills: list[Path]) -> list[Finding]:
    """仓内坐标（`docs/` `mindiesd/` `tests/` …）必须可解析 —— 防仓库重组后指令腐烂。"""
    findings: list[Finding] = []
    for path in sorted(agents.rglob("*.md")):
        text = read_text(path)
        lines = text.splitlines()
        for token in sorted(set(_candidate_tokens(text))):
            if token.split("/", 1)[0] not in REPO_COORD_DIRS:
                continue
            if (agents.parent / token).exists():
                continue
            lineno = line_of(text, token)
            if is_exempt(lines, lineno):
                continue
            findings.append(Finding("KB009", rel(path, agents.parent), lineno, f"仓内坐标不存在：{token}"))
    return sorted(findings, key=lambda f: (f.path, f.line))


# ---------------------------------------------------------------- KB013


def rule_kb013(agents: Path, _skills: list[Path]) -> list[Finding]:
    """仓内消费方指向 `.agents/**` 的反向引用必须可解析（KB001 只从 .agents 往外看）。"""
    findings: list[Finding] = []
    root = agents.parent
    for name in CONSUMER_FILES:
        path = root / name
        if not path.is_file():
            continue
        text = read_text(path)
        lines = text.splitlines()
        for token in sorted(set(_candidate_tokens(text))):
            if not token.startswith(_AGENTS_DIRNAME + "/"):
                continue
            if (root / token).exists():
                continue
            lineno = line_of(text, token)
            if is_exempt(lines, lineno):
                continue
            findings.append(Finding("KB013", name, lineno, f"指向 .agents 的引用不存在：{token}"))
    return findings


# ---------------------------------------------------------------- KB014


def _is_conclusion_class(ref: Path) -> bool:
    stem = ref.stem
    return stem.endswith(CONCLUSION_SUFFIXES) or stem.startswith(CONCLUSION_PREFIXES)


def rule_kb014(agents: Path, skills: list[Path]) -> list[Finding]:
    """携带经验结论的文件必须写「如何判定它仍存在」——把 §7「结论有寿命」机械化。

    这是 P1-5 原设计（新增 `registry.yaml` 记 `scope`/`last_checked`）的**替代实现**：
    `scope` 已由四分类文件名编码（重复真源），日期式 `last_checked` 在本仓无信号
    （实测所有文件 git 日期同为一次重构提交）；而**条件式失效**（出现什么现象即过期）
    比日期更可执行 —— 故只判「有没有写复核/失效信号」，不引入新元数据。
    """
    findings: list[Finding] = []
    for skill in skills:
        for ref in reference_files(skill):
            if not _is_conclusion_class(ref):
                continue
            if LIFETIME_RE.search(read_text(ref)):
                continue
            findings.append(Finding("KB014", rel(ref, agents.parent), 1, "结论类文件缺『复核方法/失效信号』表述"))
    return findings


# ---------------------------------------------------------------- KB006 / KB010


def rule_kb006(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        for ref in reference_files(skill):
            text = read_text(ref)
            if text.count("\n") + 1 <= LARGE_REF_LINES:
                continue
            if TOC_RE.search(text):
                continue
            findings.append(
                Finding(
                    "KB006",
                    rel(ref, agents.parent),
                    1,
                    f">{LARGE_REF_LINES} 行但无目录（内容索引/目录）",
                )
            )
    return findings


def rule_kb010(agents: Path, skills: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for skill in skills:
        for ref in reference_files(skill):
            if MAINTENANCE_RE.search(read_text(ref)):
                continue
            findings.append(Finding("KB010", rel(ref, agents.parent), 1, "缺「维护与更新」章节"))
    return findings


# ---------------------------------------------------------------- 执行

RUNNERS = {
    "KB001": rule_kb001,
    "KB002": rule_kb002,
    "KB003": rule_kb003,
    "KB004": rule_kb004,
    "KB005": rule_kb005,
    "KB006": rule_kb006,
    "KB007": rule_kb007,
    "KB009": rule_kb009,
    "KB010": rule_kb010,
    "KB011": rule_kb011,
    "KB012": rule_kb012,
    "KB013": rule_kb013,
    "KB014": rule_kb014,
}


def run(agents: Path, selected: list[str]) -> list[Finding]:
    skills = skill_dirs(agents)
    findings: list[Finding] = []
    for rule in selected:
        findings.extend(RUNNERS[rule](agents, skills))
    findings.sort(key=lambda f: (ORDER.index(f.rule), f.path, f.line))
    return findings


def render_text(findings: list[Finding], root: Path) -> str:
    if not findings:
        return "kb_lint: 0 违规（error=0, warn=0）"
    lines = []
    errors = sum(1 for f in findings if f.severity == "error")
    warns = len(findings) - errors
    for f in findings:
        lines.append(f"{f.path}:{f.line}: {f.severity} {f.rule} {f.message}")
    lines.append(f"kb_lint: error={errors}, warn={warns}（root={root}）")
    return "\n".join(lines)


def render_grouped(findings: list[Finding]) -> str:
    if not findings:
        return "kb_lint: 0 违规（error=0, warn=0）"
    lines = []
    for rule in ORDER:
        group = [f for f in findings if f.rule == rule]
        if not group:
            continue
        severity, desc = RULES[rule]
        lines.append(f"## {rule} ({severity}) {desc} —— {len(group)} 处")
        for f in group:
            lines.append(f"  - {f.path}:{f.line} {f.message}")
    errors = sum(1 for f in findings if f.severity == "error")
    lines.append(f"\nkb_lint: error={errors}, warn={len(findings) - errors}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 自测


def _write_minimal_skill(skills_root: Path, name: str) -> None:
    """写一个各规则都合规的最小技能（供 KB007 夹具凑出层成员）。"""
    skill = skills_root / name
    (skill / "evals").mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\ncompatibility: c\n---\n## Reference Files\n\n## 维护与更新\n",
        encoding="utf-8",
    )
    cases = [{"id": i, "prompt": "p", "expectations": ["e"]} for i in (1, 2)]
    (skill / "evals" / "evals.json").write_text(json.dumps({"skill_name": name, "evals": cases}), encoding="utf-8")


def _build_fixture(agents: Path, broken: bool) -> None:
    """构造一份技能树夹具。broken=True 时**每条规则**都埋入一个违规。

    负样本覆盖表（broken=True）：
      KB001 `references/missing.md` 不存在 | KB002 `references/orphan.md` 未登记
      KB003 目录 `Bad_Skill` + 文件 `Bad_Name.md` | KB004 缺 `compatibility`
      KB005 evals 仅 1 条用例 | KB006 `big.md` >300 行无目录
      KB007 README 三层计数全写错 | KB010 `not-maintained.md` 无维护章节
      KB011 SKILL.md 无维护章节 | KB012 `big.md` 登记行缺「加载时机」

    KB007 需要真实层成员目录（`LAYER_L1`/`LAYER_L2`）才会启用，故两种夹具都补齐层成员。
    """
    skills_root = agents / _SKILLS_DIRNAME
    name = "Bad_Skill" if broken else "demo-skill"
    skill = skills_root / name
    (skill / "references").mkdir(parents=True, exist_ok=True)
    (skill / "evals").mkdir(parents=True, exist_ok=True)

    for layer_skill in (*LAYER_L1, *LAYER_L2):
        _write_minimal_skill(skills_root, layer_skill)

    (skill / "references" / "good.md").write_text("# good\n\n## 维护与更新\n", encoding="utf-8")
    (skill / "references" / "not-maintained.md").write_text(
        "# no maintenance\n" if broken else "# second\n\n## 维护与更新\n", encoding="utf-8"
    )
    # KB014 负样本：结论类文件名（`-method`）但无「复核/失效信号」表述
    (skill / "references" / "demo-method.md").write_text(
        "# demo method\n\n## 维护与更新\n"
        if broken
        else "# demo method\n\n失效信号：出现 X 即本条过期。\n\n## 维护与更新\n",
        encoding="utf-8",
    )

    rows = [
        "| 文件 | 加载时机 |",
        "|------|---------|",
        "| `references/good.md` | 总是 |",
        "| `references/not-maintained.md` | 总是 |",
        "| `references/demo-method.md` | 总是 |",
        # 占位/示例写法必须**不**被判为悬空引用（回归保护：占位符判定）
        "| `references/x.md` | 示例占位 |",
    ]
    frontmatter = "---\nname: demo\ndescription: d\ncompatibility: c\n---\n"

    if broken:
        (skill / "references" / "Bad_Name.md").write_text("# bad name\n\n## 维护与更新\n", encoding="utf-8")
        (skill / "references" / "orphan.md").write_text("# orphan\n\n## 维护与更新\n", encoding="utf-8")
        (skill / "references" / "big.md").write_text("# big\n" + "x\n" * 320, encoding="utf-8")
        rows += [
            "| `references/missing.md` | 从不 |",
            "| `references/big.md` |",
            "| `references/Bad_Name.md` | 坏名字 |",
        ]
        frontmatter = "---\nname: demo\ndescription: d\n---\n"

    tail = "" if broken else "## 维护与更新\n"
    if broken:
        # 负样本：仓内坐标不存在（KB009）。**措辞里不能带任何豁免标记词** ——
        # 首版写「… `docs/missing.md` 不存在」，被随后新增的「不存在」标记豁免，负样本失效。
        extra = "- 负样本：引用一个仓库坐标 `docs/missing.md`（KB009 应报）\n"
    else:
        # 回归保护：① 跨行豁免标记（标记在首行、路径在次行）；
        # ② 「时读」等价写法（首版只认字面量「加载时机」，实测 23 处全误报）；
        # ③ 「不存在」缺失说明（历史理由表会列出缺失路径）
        extra = (
            "- 已移除 `references/gone.md`（历史留档，勿按清单复原）\n"
            "  与仓库坐标 `docs/zzz.md`（**本仓不含**）：两者都不得被判为悬空。\n"
            "- 逆向量化契约时读 `references/not-maintained.md`（等价写法回归保护）\n"
            "- 另一处 `docs/yyy.md` 同为**不存在**的示例（「不存在」标记回归保护）\n"
        )
    (skill / "SKILL.md").write_text(
        frontmatter + "## Reference Files\n" + "\n".join(rows) + "\n" + tail + extra, encoding="utf-8"
    )

    cases = [{"id": 1, "prompt": "p", "expectations": ["e"]}]
    if not broken:
        cases.append({"id": 2, "prompt": "p", "expectations": ["e"]})
    (skill / "evals" / "evals.json").write_text(json.dumps({"skill_name": "demo", "evals": cases}), encoding="utf-8")

    # 夹具技能 = demo-skill/Bad_Skill + LAYER_L1(2) + LAYER_L2(1) ⇒ total=4, l3=1, non_l1=2
    counts = _layer_counts(sorted(d for d in skills_root.iterdir() if d.is_dir()))
    assert counts is not None
    readme_counts = {"total": counts["total"], "l3": counts["l3"], "non_l1": counts["non_l1"]}
    if broken:  # 三层计数全写错
        readme_counts = {k: v + 90 for k, v in readme_counts.items()}
    (agents / "README.md").write_text(
        "# fixture\n\n"
        f"> L3 能力（{readme_counts['l3']}）：占位\n\n"
        f"## 2. 优化域入口与能力层（{readme_counts['non_l1']}）\n\n"
        f"- ✅ **{readme_counts['total']} 个技能**：占位\n",
        encoding="utf-8",
    )

    # 反 向 边（KB013 读的是 **.agents 之外**的消费方文件）：broken 指向不存在的技能文件
    consumer_ref = ".agents/skills/demo-skill/references/nope.md" if broken else f".agents/skills/{name}/SKILL.md"
    (agents.parent / "README.md").write_text(f"# fixture root\n\n- 技能入口见 `{consumer_ref}`\n", encoding="utf-8")


def _selftest(base_dir: Path | None = None) -> int:
    """负样本自测：每条规则都要能被触发，干净样例必须 0 违规。

    夹具用确定性路径（`mkdtemp` 在 Windows 上建的目录 DACL 过严，嵌套建目录会被拒）。
    """
    failures: list[str] = []
    base = (base_dir or Path(__file__).resolve().parents[1]) / ".kb_lint_selftest"
    broken_root = base / "broken"
    clean_root = base / "clean"
    shutil.rmtree(base, ignore_errors=True)
    try:
        _build_fixture(broken_root / _AGENTS_DIRNAME, broken=True)
        _build_fixture(clean_root / _AGENTS_DIRNAME, broken=False)

        expected = set(RULES)
        got = {f.rule for f in run(broken_root / _AGENTS_DIRNAME, ORDER)}
        for rule in sorted(expected - got):
            failures.append(f"负样本未触发 {rule} {RULES[rule][1]}")
        for rule in sorted(got - expected):
            failures.append(f"负样本误触发 {rule}")

        residue = run(clean_root / _AGENTS_DIRNAME, ORDER)
        for finding in residue:
            failures.append(f"干净样例误报 {finding.rule} {finding.path}: {finding.message}")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print("kb_lint --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"kb_lint --selftest: PASS（{len(RULES)} 条规则：负样本逐条触发，干净样例 0 违规）")
    return 0


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="仓库根（默认当前目录）")
    parser.add_argument("--rule", action="append", default=[], help="只跑指定规则（可重复）")
    parser.add_argument("--format", choices=("text", "grouped", "json"), default="text", help="输出格式")
    parser.add_argument("--strict", action="store_true", help="warn 也视为失败")
    parser.add_argument("--list-rules", action="store_true", help="列出规则后退出")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.list_rules:
        for rule in ORDER:
            severity, desc = RULES[rule]
            print(f"{rule}  {severity:<5}  {desc}")
        return 0

    if args.selftest:
        return _selftest()

    root = Path(args.root).resolve()
    agents = root / _AGENTS_DIRNAME
    if not agents.is_dir():
        print(f"kb_lint: 前置条件缺失：{agents} 不存在", file=sys.stderr)
        return 2

    selected = args.rule or ORDER
    unknown = [r for r in selected if r not in RULES]
    if unknown:
        parser.error(f"未知规则：{', '.join(unknown)}")

    findings = run(agents, selected)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "root": str(agents),
                    "errors": sum(1 for f in findings if f.severity == "error"),
                    "warns": sum(1 for f in findings if f.severity == "warn"),
                    "findings": [f.as_dict() for f in findings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.format == "grouped":
        print(render_grouped(findings))
    else:
        print(render_text(findings, agents))

    errors = [f for f in findings if f.severity == "error"]
    warns = [f for f in findings if f.severity == "warn"]
    if errors or (args.strict and warns):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
