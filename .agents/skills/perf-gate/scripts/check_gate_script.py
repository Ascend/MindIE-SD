#!/usr/bin/env python3
"""交付门禁脚本的「平台归属 + 可运行自证」机械检查（measurement-discipline §11.1）。

为什么需要这个门禁
------------------
2026-09-20 现场：一条 lane 把 `gate_byte_neutral.py` 当作验收 gate 交付，而该 gate

* 把 Windows 仓库路径写成**唯一默认值**（`Path(r"D:\\framework\\vllm-omni")`），换台机器就静默检查
  另一个对象；
* 需要的 **POSIX `SharedMemory` 语义在该平台上根本不成立**（框架自身的 shm 往返在 Windows 上
  打补丁与不打补丁都一样失败）⇒ gate 在任何它够得着的主机上都跑不起来；
* 交付时**没有任何一次运行记录**，于是"未证明"被读成了"未通过"，甚至被读成"通过"。

判据是 `../references/measurement-discipline.md` §11.1 的四项：平台归属 / 互相依赖 / 可运行自证 /
路径可解析。本脚本机械检查**第 1、3、4 项**并输出 `error` / `warn`；**第 2 项（外部依赖与
"依赖不可达时判无法判定"）只有人能读**，脚本把它列为「无法机检项」，不装作检查过。

检查项
------
G1 平台归属（error）：声明区必须写出平台语义 —— `平台` / `platform` / `POSIX` / `Linux` /
   `Windows` / `sys.platform` / `os.name`。
G2 互相依赖（warn）：声明区应出现外部前提字样（需要 / 依赖 / 必须 / 前提 / 不可达 / 无法判定 /
   requires / depends / needs），否则提示补"依赖什么、不可达时怎么办"。
G3 路径可解析（error）：不得把 Windows 盘符字面量（`r"D:\\..."` / `"C:/..."`）当**唯一**路径；
   同行带 `environ` / `argv` 兜底即合规。**文档字符串与夹具字面量内部的举例不算**（多行字符串
   内部的行按「文本」跳过）——否则门禁会自报假阳性（2026-09-20 实测：本脚本被自己抓到 3 条，
   全是 docstring 举例与负样本夹具）。真正写在代码行里的默认值、以及负样本夹具**落盘成文件后**，
   仍照常被抓到。
G4 可运行自证（warn）：声明区应给一条可复制的用法行（`用法：` / `python <脚本>.py`），
   它就是回执「运行证据」里的最小用例。

用法
----
    python check_gate_script.py <脚本.py | 目录> [...]
    python check_gate_script.py --selftest

退出码：0 = 无 error；1 = 有 error（或 `--strict` 下有 warn）；2 = 前置条件缺失（目标不存在）。
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
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

# G1：平台归属声明（交付脚本必须写出它属于哪个平台）
PLATFORM_TOKENS = (
    "平台",
    "platform",
    "sys.platform",
    "os.name",
    "POSIX",
    "posix",
    "Linux",
    "Windows",
)

# G2：外部依赖声明
DEP_TOKENS = (
    "需要",
    "依赖",
    "必须",
    "前提",
    "不可达",
    "无法判定",
    "requires",
    "Requires",
    "depends",
    "needs",
)

# G3：Windows 盘符字面量（原始串或普通串皆命中）
WINPATH_RE = re.compile(r"""["'][rRbB]?["']?[A-Za-z]:[\\/]""")

# G4：可运行用法行
USAGE_RE = re.compile(r"(用法\s*[:：]|Usage\s*[:：]|python3?\s+\S*\.py)")

# 宿主探测（出现在代码里也算平台声明）
HOST_PROBE_RE = re.compile(r"(sys\.platform|os\.name|platform\.system\(\))")


def _declaration_region(path: Path, text: str) -> str:
    """返回用于判定 G1/G2/G4 的「声明区」文本：模块 docstring + 文件前 60 行 + 宿主探测痕迹。"""
    parts: list[str] = []
    try:
        doc = ast.get_docstring(ast.parse(text))
    except SyntaxError:
        doc = None
    if doc:
        parts.append(doc)
    parts.extend(text.splitlines()[:60])
    if HOST_PROBE_RE.search(text):
        parts.append("host probe present: sys.platform")
    return "\n".join(parts)


def _literal_text_lines(text: str) -> set[int]:
    """多行字符串字面量**内部**的行号（首尾行除外）。

    这些行是文档/夹具文本，不是「可执行的唯一默认值」：把 Windows 盘符写在 docstring 的举例里、
    或写在 `_BAD` 这类负样本夹具字符串里，都不该被判成写死路径。真正的默认值落在代码行上，
    仍会被抓到；负样本夹具**落盘成独立文件后**同样会被抓到（那时它那行就是代码）。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", 0)
            if end > start:
                lines.update(range(start + 1, end))
    return lines


def check_file(path: Path) -> tuple[list[str], list[str], list[str]]:
    """返回 (errors, warns, cannot_check)。"""
    errors: list[str] = []
    warns: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    region = _declaration_region(path, text)
    literal_lines = _literal_text_lines(text)

    if not any(tok in region for tok in PLATFORM_TOKENS):
        errors.append(
            "G1 缺平台归属声明：写明该脚本只能/必须在哪个平台跑"
            "（POSIX / Linux 容器 / Windows；是否需要 /dev/shm、NPU、特定框架）"
        )

    if not any(tok in region for tok in DEP_TOKENS):
        warns.append(
            "G2 未声明外部依赖：写出它依赖的前置（目标主机可达 / 某目录已暂存 / 某包已安装），"
            "并写明依赖不可达时判「无法判定」而不是「通过」"
        )

    for lineno, line in enumerate(text.splitlines(), start=1):
        if lineno in literal_lines:
            continue  # 文档/夹具文本里的举例不是「唯一默认值」
        if not WINPATH_RE.search(line):
            continue
        if "environ" in line or "argv" in line:
            continue  # 有环境变量/参数兜底 ⇒ 路径可解析
        errors.append(
            f"G3 第 {lineno} 行：把本机绝对路径当唯一默认值 ⇒ 换台机器会静默检查另一个对象；"
            "改从 CLI 参数或环境变量取，缺失即报错退出"
        )

    if not USAGE_RE.search(region):
        warns.append("G4 未给可运行用法行：补一条最小用例命令（回执「运行证据」要用它）")

    cannot_check = [
        (
            "第 2 项 互相依赖的可达性（需人读）：目标主机是否可达、目标平台语义是否成立、"
            "依赖不可达时是否显式判「无法判定」"
        ),
        "该脚本是否**真的在目标平台跑通过一次**（回执里的命令 + 退出码 + 环境指纹）",
    ]
    return errors, warns, cannot_check


def _python_files(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in targets:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(q for q in p.rglob("*.py") if "__pycache__" not in q.parts))
        else:
            out.append(p)
    return out


# ---------------------------------------------------------------- 自测

_GOOD = '''#!/usr/bin/env python3
"""某个门禁脚本（合规样例）。

平台：POSIX / Linux 容器（需要 /dev/shm 与 POSIX 段名存活语义；Windows 上本 gate 不适用）。
依赖：目标主机的仓库已暂存；依赖不可达时判「无法判定」，不得降级为通过。

用法：
    python demo_gate.py --repo /path/to/repo
"""
import os
import sys

REPO = os.environ.get("DEMO_REPO") or (sys.argv[1] if len(sys.argv) > 1 else None)
if REPO is None:
    raise SystemExit("缺 --repo / DEMO_REPO：不给路径就退出，不猜默认值")
print(sys.platform, REPO)
'''

_BAD = '''#!/usr/bin/env python3
"""L1 patch gate (negative fixture: no host statement, no usage line, path pinned)."""
from pathlib import Path

REPO = Path(r"D:\\framework\\vllm-omni")
print(REPO)
'''


def _selftest() -> int:
    # 夹具用**确定性路径**：`mkdtemp` 建的目录权限过严，在本环境往里写文件会被拒
    # （同因处理见 `.agents/scripts/run_evals.py --selftest`）。
    base = Path(__file__).resolve().parents[1] / ".check_gate_script_selftest"
    failures: list[str] = []
    shutil.rmtree(base, ignore_errors=True)
    try:
        base.mkdir(parents=True, exist_ok=True)
        good = base / "good_gate.py"
        bad = base / "bad_gate.py"
        good.write_text(_GOOD, encoding="utf-8")
        bad.write_text(_BAD, encoding="utf-8")

        g_err, g_warn, _ = check_file(good)
        if g_err or g_warn:
            failures.append(f"合规样例误报：errors={g_err} warns={g_warn}")

        b_err, _, _ = check_file(bad)
        if not any("G1" in e for e in b_err):
            failures.append("负样本未被抓到 G1（平台归属缺失）")
        if not any("G3" in e for e in b_err):
            failures.append("负样本未被抓到 G3（写死盘符路径）")

        # 回归：门禁对**自身**运行不得因 docstring 举例 / 负样本夹具字面量而误报 G3
        self_err, _, _ = check_file(Path(__file__).resolve())
        if any("G3" in e for e in self_err):
            failures.append(f"回归：对自身文档/夹具里的盘符举例误报 G3：{self_err}")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print("check_gate_script --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("check_gate_script --selftest: PASS（合规样例 0 违规；负样本 G1/G3 逐条被触发；自身举例不误报）")
    return 0


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="交付门禁脚本的平台归属与可运行自证检查（measurement-discipline §11.1）"
    )
    parser.add_argument("targets", nargs="*", help="待检查的 .py 文件或目录")
    parser.add_argument("--strict", action="store_true", help="warn 也计为失败")
    parser.add_argument("--selftest", action="store_true", help="负样本自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.targets:
        parser.error("需要至少一个 .py 文件或目录（或 --selftest）")

    files = _python_files(args.targets)
    if not files:
        print("check_gate_script: 前置条件缺失：未找到 .py 文件", file=sys.stderr)
        return 2

    errors = 0
    warns = 0
    cannot: list[str] = []
    for path in files:
        if not path.is_file():
            print(f"{path.as_posix()}: 前置条件缺失：文件不存在", file=sys.stderr)
            return 2
        errs, wrns, cn = check_file(path)
        cannot.extend(cn)
        for msg in errs:
            errors += 1
            print(f"{path.as_posix()}: error {msg}")
        for msg in wrns:
            warns += 1
            print(f"{path.as_posix()}: warn {msg}")

    print("\n无法机检项（必须人读，不得当成已检查）：")
    for item in dict.fromkeys(cannot):
        print(f"- {item}")

    print(f"\ncheck_gate_script: error={errors}, warn={warns}（检查 {len(files)} 个文件）")
    if errors or (args.strict and warns):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
