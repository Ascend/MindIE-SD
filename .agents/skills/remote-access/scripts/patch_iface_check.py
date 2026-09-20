#!/usr/bin/env python
"""patch_iface_check.py — 跨文件补丁的**接口一致性**静态门禁（调用点关键字 ⊆ 被调方形参）。

为什么需要它（2026-09-20 现场）
-------------------------------
一条"把 mask 构建挪到侧流、与设备 FA 重叠"的补丁跨两个文件：

* `mindiesd/parallel/_comm.py`（transport，**调用点**）按关键字把侧流上建好的句柄传下去 ——
  `consume(gi, rq, rk, rv, out_dst=…, prep=prep_h[gi])`；
* `mindiesd/parallel/mask/rf_v2.py`（consumer，**被调方**）的闭包仍写作
  `def consume(gi, rq, rk, rv, out_dst=None)` —— **没有 `prep` 形参**。

八卡上四次请求全 `HTTP 500`（`TypeError: … consume() got an unexpected keyword argument 'prep'`），
而三道常见防线全部漏掉：

| 防线 | 为什么漏 |
|---|---|
| `py_compile` / 语法检查 | 关键字不匹配是**运行期**错误，编译期完全合法 |
| 私有 harness / 单卡烟测 | harness 自己实现调度、**不走该调用点**；单卡又被下游前置条件静默禁用 |
| 归档哈希对得上 | 归档的正是这个坏形态 ⇒ 哈希只证明"没传错文件" |

所以：**在花任何设备时间之前**，先把这件事静态查掉。

判据（narrow、零依赖、纯 `ast`，**不是类型检查器**）
----------------------------------------------------
对每个被调函数：**调用点传的关键字集合 ⊆ 被调函数形参集合**（或该函数接受 `**kwargs`）。

用法
----
    # 两文件模式（同 O5 形态：调用点文件 + 被调方文件）
    python patch_iface_check.py <callsite.py> <callee.py> [--callee NAME]

    # 单文件模式（被调函数与调用点在同一文件；自动逐个交叉校验）
    python patch_iface_check.py <file.py> [--callee NAME]

    # 只看本次改动涉及的行（行号来自 diff，多段用逗号）
    python patch_iface_check.py <file.py> --lines 379,386-388

    python patch_iface_check.py --selftest      # 正控 + 负控自测

退出码：0 = 一致；1 = 不一致（逐条打印）；2 = 前置条件缺失（文件不存在/用法不对）。
`--selftest` 只允许 0。

为什么把规则做成脚本而不是更多散文
----------------------------------
`remote-access/references/arm-driver-traps.md` §12 早已写明这条纪律，但它在 2026-09-20 之前
**没有机械出口**，于是"补丁能跑吗"只能靠人想起那条散文。本脚本把同一判据变成一条可执行命令；
**负控内建**（§8.6：门禁必须先证明它会失败）。
"""

from __future__ import annotations

import argparse
import ast
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


# ---------------------------------------------------------------- AST 工具


def _params(fn: ast.AST) -> tuple[set[str], bool]:
    """(具名形参集合, 是否接受 **kwargs)。"""
    a = fn.args  # type: ignore[attr-defined]
    names = {p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)}
    if a.vararg:
        names.add(a.vararg.arg)
    plain = set(names)
    return plain, a.kwarg is not None


def _qualname(node: ast.AST) -> str | None:
    """`f` / `mod.f` / `self.f` 形态的函数名；取不到返回 None。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _qualname(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _collect_defs(tree: ast.AST) -> dict[str, list[ast.AST]]:
    """文件内定义的函数（含嵌套/类方法），按名字归组（重名全部保留）。"""
    out: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, []).append(node)
    return out


def _calls_in_file(tree: ast.AST) -> list[tuple[int, str, set[str], bool, bool]]:
    """全文件的调用点：(行号, 被调名, 关键字集合, 有 ** 展开, 该行是否是 def 行)。"""
    out: list[tuple[int, str, set[str], bool, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _qualname(node.func)
        if not name:
            continue
        kws = {k.arg for k in node.keywords if k.arg is not None}
        star = any(k.arg is None for k in node.keywords)
        out.append((node.lineno, name.split(".")[-1], kws, star, False))
    return out


def _parse(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---------------------------------------------------------------- 检查核心


def check_pair(
    callsite_tree: ast.AST, callsite_name: str, callee_tree: ast.AST, callee_name: str, lines: set[int] | None = None
) -> tuple[list[str], list[str], int]:
    """返回 (errors, notes, 检查的调用点数)。"""
    errors: list[str] = []
    notes: list[str] = []
    defs = _collect_defs(callee_tree)
    if callee_name not in defs:
        return (
            [f"{callee_name}: 在被调方文件里找不到 `def {callee_name}` —— 无法校验该回调接口"],
            [],
            0,
        )
    param_variants = []
    for fn in defs[callee_name]:
        params, has_kwarg = _params(fn)
        param_variants.append((fn.lineno, params, has_kwarg))
    variant_desc = " | ".join(f"L{ln} params={sorted(p)} **kwargs={kw}" for ln, p, kw in param_variants)
    notes.append(f"  callee  : {callee_name} {variant_desc}")

    checked = 0
    for lineno, name, kws, star, _ in _calls_in_file(callsite_tree):
        if name != callee_name:
            continue
        if lines is not None and lineno not in lines:
            continue
        checked += 1
        if star:
            notes.append(f"    L{lineno:<5d} 含 ** 展开 —— 跳过（静态不可判）")
            continue
        # 同一名字可能有多个定义（真函数 + 测试替身）：任一接受即算通过，
        # 全部不接受才报，且把各自的形参列出来便于定位。
        ok_variant = None
        for ln, params, has_kwarg in param_variants:
            if has_kwarg or not (kws - params):
                ok_variant = ln
                break
        if ok_variant is not None:
            notes.append(f"    L{lineno:<5d} kwargs={sorted(kws)}  -> ok")
            continue
        missing = sorted(set().union(*[kws - p for _, p, _ in param_variants]))
        defined_at = ", ".join(f"L{ln}" for ln, _, _ in param_variants)
        accepted = sorted(param_variants[0][1])
        errors.append(
            f"{callsite_name} L{lineno}: `{callee_name}(...)` 传了关键字 {missing}，"
            f"而被调方（{defined_at}）只接受 {accepted}"
        )
        notes.append(f"    L{lineno:<5d} kwargs={sorted(kws)}  -> MISMATCH {missing}")
    return errors, notes, checked


def auto_pairs(tree: ast.AST) -> list[str]:
    """单文件模式：找出「本文件既定义、又被调用」的名字（排除自身定义行）。"""
    defs = {n for n in _collect_defs(tree)}
    called = {name for _, name, _, _, _ in _calls_in_file(tree)}
    return sorted(defs & called)


# ---------------------------------------------------------------- 自测

_GOOD_CALLER = '''def run(handle, gi):
    return consume(gi, out_dst=handle, prep=None)


def consume(gi, out_dst=None, prep=None):
    return gi
'''

_BAD_CALLER = '''def run(handle, gi):
    return consume(gi, out_dst=handle, prep=None)


def consume(gi, out_dst=None):
    return gi
'''

_GOOD_TWO_FILE = '''def consume(gi, rq, rk, rv, out_dst=None, prep=None):
    return gi
'''

_BAD_TWO_FILE = '''def consume(gi, rq, rk, rv, out_dst=None):
    return gi
'''

_SITE = "consume(gi, rq, rk, rv, out_dst=dst, prep=prep_h[gi])\n"


def _selftest() -> int:
    # 夹具用**确定性路径**：`mkdtemp` 建的目录权限过严，在本环境往里写文件会被拒
    # （同因处理见 `.agents/scripts/run_evals.py --selftest`）。
    base = Path(__file__).resolve().parents[1] / ".patch_iface_check_selftest"
    failures: list[str] = []
    shutil.rmtree(base, ignore_errors=True)
    try:
        base.mkdir(parents=True, exist_ok=True)
        good = base / "good.py"
        bad = base / "bad.py"
        good.write_text(_GOOD_CALLER, encoding="utf-8")
        bad.write_text(_BAD_CALLER, encoding="utf-8")

        g_err, _, g_n = check_pair(_parse(good), good.name, _parse(good), "consume")
        if g_err or g_n == 0:
            failures.append(f"正控误报 / 未检查到调用点：errors={g_err} checked={g_n}")

        b_err, _, b_n = check_pair(_parse(bad), bad.name, _parse(bad), "consume")
        if not b_err or b_n == 0:
            failures.append("负控未报错（门禁是空门）")

        # 两文件模式：调用点与定义分居两文件（O5 的真实形态）
        site = base / "_comm.py"
        site.write_text(_SITE, encoding="utf-8")
        good_callee = base / "rf_v2_good.py"
        bad_callee = base / "rf_v2_bad.py"
        good_callee.write_text(_GOOD_TWO_FILE, encoding="utf-8")
        bad_callee.write_text(_BAD_TWO_FILE, encoding="utf-8")

        ok_err, _, ok_n = check_pair(_parse(site), site.name, _parse(good_callee), "consume")
        if ok_err or ok_n != 1:
            failures.append(f"两文件正控误报：errors={ok_err} checked={ok_n}")

        ng_err, _, ng_n = check_pair(_parse(site), site.name, _parse(bad_callee), "consume")
        if not ng_err or ng_n != 1:
            failures.append("两文件负控未报错（O5 缺陷形态未被抓到）")

        # --lines 过滤：跳过该行即不该报
        skip_err, _, skip_n = check_pair(_parse(site), site.name, _parse(bad_callee), "consume", lines={999})
        if skip_err or skip_n != 0:
            failures.append(f"--lines 过滤失效：errors={skip_err} checked={skip_n}")
    finally:
        shutil.rmtree(base, ignore_errors=True)

    if failures:
        print("patch_iface_check --selftest: FAIL")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("patch_iface_check --selftest: PASS（正控 0 误报；负控单文件/两文件均被触发；--lines 生效）")
    return 0


# ---------------------------------------------------------------- CLI


def _parse_lines(spec: str) -> set[int]:
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="跨文件补丁的接口一致性静态门禁（调用点关键字 ⊆ 被调方形参）")
    parser.add_argument("files", nargs="*", help="一个文件（自动交叉校验）或两个文件（调用点 被调方）")
    parser.add_argument("--callee", default=None, help="只校验这个名字的被调函数（默认自动）")
    parser.add_argument("--lines", default=None, help="只看这些行（如 379,386-388）")
    parser.add_argument("--selftest", action="store_true", help="正控/负控自测并退出")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.files or len(args.files) > 2:
        parser.error("需要一个或两个 .py 文件（或 --selftest）")
    for raw in args.files:
        if not Path(raw).is_file():
            print(f"[error] 前置条件缺失：文件不存在 {raw}", file=sys.stderr)
            return 2

    lines = _parse_lines(args.lines) if args.lines else None
    errors: list[str] = []
    notes: list[str] = []
    total = 0
    print("== 跨文件接口一致性检查（调用点关键字 ⊆ 被调方形参）==")

    if len(args.files) == 2:
        callsite, callee = Path(args.files[0]), Path(args.files[1])
        tree_c, tree_k = _parse(callsite), _parse(callee)
        names = [args.callee] if args.callee else auto_pairs(tree_c) or auto_pairs(tree_k)
        if not names:
            print("[error] 两个文件里没有「既定义、又被调用」的函数名；用 --callee 指定", file=sys.stderr)
            return 2
        for name in names:
            errs, nt, n = check_pair(tree_c, callsite.name, tree_k, name, lines)
            errors.extend(errs)
            notes.extend(nt)
            total += n
    else:
        path = Path(args.files[0])
        tree = _parse(path)
        names = [args.callee] if args.callee else auto_pairs(tree)
        if not names:
            print(f"[error] {path.name} 里没有「既定义、又被调用」的函数名；用 --callee 指定", file=sys.stderr)
            return 2
        for name in names:
            errs, nt, n = check_pair(tree, path.name, tree, name, lines)
            errors.extend(errs)
            notes.extend(nt)
            total += n

    for line in notes:
        print(line)
    for e in errors:
        print(f"[error] {e}")
    print(f"检查调用点 {total} 个，被调函数 {len(names)} 个；结论：error={len(errors)}")
    if total == 0:
        print("[warn] 没有检查到任何调用点 —— 该门禁本次没有信号（勿当成通过）")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
