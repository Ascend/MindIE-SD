#!/usr/bin/env python3
"""stage_gate.py —— 编排层阶段推进门禁（零 NPU、零数据）。

做：解析 run-state.md（见 ../references/run-state.md）的「阶段推进表」→ 校验目标阶段
状态为 done 且声明的验收证据路径真实存在；close 阶段额外强制声明 overview_report.md /
detail_report.md（缺任一视为未闭环），并联动 report_lint.py（主表**写法**）+ audit_report.py
（表**结构**与**数值自洽**）+ evals/scripts/check_profile.py（profile 强校验）。不做：真实运行/耗时/
质量判定（那些在能力技能门禁内）。

用法：
    python stage_gate.py --stage S0 --run-dir <工作目录>/agentic
    python stage_gate.py --stage close --run-dir <工作目录>/agentic
退出码：0 = 通过；1 = 存在 error（不得进入下一阶段 / 不得宣称闭环）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STAGES = ("S0", "S1", "S3", "S4", "S5", "S6", "close")
VALID_STATUS = {"done", "in_progress", "blocked"}
RUN_STATE_NAME = "run-state.md"
# close 阶段的强制交付双报表（存在性以推进表声明为准，不猜测产物目录）
CLOSE_REQUIRED_BASENAMES = ("overview_report.md", "detail_report.md")


def parse_stage_rows(text: str) -> list[dict]:
    """从 run-state.md 提取推进表行：| 阶段 | 状态 | 验收证据 | 备注 |。"""
    rows: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if re.search(r"^\|[\s\-:|]+\|?$", stripped):  # 表头分隔行
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 3:
            continue
        stage, status = cells[0], cells[1]
        if stage not in STAGES:
            continue
        evidence = cells[2]
        rows.append(
            {"stage": stage, "status": status, "evidence": evidence, "note": cells[3] if len(cells) > 3 else ""}
        )
    return rows


def split_evidence(raw: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[，,;；、]", raw) if p.strip()]
    return parts


def coverage_checklist_check(text: str) -> list[str]:
    """close：校验「特性覆盖清单」逐特性判定已收口（无空/无"分析后做"残留）。

    「触发判定」列**按表头名定位**（模板见 ../references/run-state.md）——此前按 cells[1]
    取列，而模板第 2 列是「框架档位」，导致该校验长期空转（校的是恒非空的档位列）。
    """
    errors: list[str] = []
    m = re.search(r"^##\s*特性覆盖清单.*?$\n(.*?)(?=^##\s|\Z)", text, re.MULTILINE | re.DOTALL)
    if not m:
        errors.append("close: run-state 缺「特性覆盖清单」——闭环前须对固定特性全集逐项判定收口")
        return errors
    decision_idx: int | None = None
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.search(r"^\|[\s\-:|]+\|?$", s):  # 表头分隔行
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if decision_idx is None:
            # 表头行：定位「触发判定」列（容忍括注/词序差异）
            hit = next((i for i, c in enumerate(cells) if "触发判定" in c or "判定" in c), None)
            if hit is None:
                errors.append(
                    "close: 覆盖清单表头未含「触发判定」列，无法校验收口（模板见 references/run-state.md，须含该列）"
                )
                return errors
            decision_idx = hit
            continue
        if len(cells) <= decision_idx:
            continue
        decision = cells[decision_idx]
        if not decision or "分析后做" in decision or "未裁决" in decision:
            errors.append(
                f"close: 覆盖清单「{cells[0]}」判定未收口"
                f"（{decision if decision else '空'}）——闭环前须为 做/不做（带理由）"
            )
    return errors


def check_row(
    row: dict, run_dir: Path, is_close: bool, task_id: str | None = None, task_start_mtime: float | None = None
) -> tuple[list[str], list[str]]:
    """返回 (errors, warns)。"""
    errors: list[str] = []
    warns: list[str] = []
    stage = row["stage"]

    if row["status"] not in VALID_STATUS:
        errors.append(f"{stage}: 状态非法「{row['status']}」（可选 {sorted(VALID_STATUS)}）")
        return errors, warns
    if row["status"] != "done":
        errors.append(f"{stage}: 状态为 {row['status']}，未 done，不可推进/宣称完成")
        return errors, warns

    evidence = split_evidence(row["evidence"])
    if not evidence:
        errors.append(f"{stage}: 验收证据为空——先回写推进表（状态 done + 证据路径）再跑门禁")
        return errors, warns

    declared_basenames: set[str] = set()
    for entry in evidence:
        if "<" in entry or ">" in entry:
            warns.append(f"{stage}: 证据含占位符「{entry}」，跳过存在性校验")
            continue
        target = Path(entry)
        if not target.is_absolute():
            target = run_dir / target
        declared_basenames.add(target.name)
        if not target.exists():
            errors.append(f"{stage}: 验收证据不存在：{target}")
        else:
            # 任务归属校验：evidence/ 下的证据必须含当前 task_id 前缀（任务隔离，防旧残留冒充）
            ev_txt = str(target).replace("\\", "/")
            if "/evidence/" in ev_txt and task_id and f"/evidence/{task_id}/" not in ev_txt:
                errors.append(
                    f"{stage}: 证据不在本任务 evidence/{task_id}/ 下（任务隔离违规）：{ev_txt}"
                    "——复用历史须显式 ../runs/{旧task_id}/... 引用并标注，禁止旧文件充当本轮证据"
                )

    if is_close:
        missing = [name for name in CLOSE_REQUIRED_BASENAMES if name not in declared_basenames]
        if missing:
            errors.append(
                f"close: 强制交付双报表未在推进表声明：{', '.join(missing)}"
                "（缺任一视为未闭环，见 optimization-flow.md 闭环复验）"
            )
    return errors, warns


# ---------------------------------------------------------------- 多 agent 契约
# 单点：../references/agent-roles-and-handoff.md（§3 交付件 / §4 交接单 / §5 并行控制）
# 触发条件：run-dir 下存在 agentic/dispatch/（派发单目录）= 多 agent 模式；不存在则**跳过**
# —— 默认单 agent 自执行必须保持绿（不因"没开多 agent"而报错），故为条件校验并打印 [skip]。
DISPATCH_DIR = "dispatch"
HANDOFF_NAME = "handoff.md"
HANDOFF_FIELDS = ("状态", "最后完成动作", "未决问题", "下一步", "产物指针", "口径指纹", "资源占用", "预算余量")
RECEIPT_FIELDS = ("结论", "命令", "退出码", "证据")
POINTER_RE = re.compile(r"`([^`\n]*?/[^`\n]*?\.(?:md|log|csv|txt|json|toml|py|sh|ya?ml))`")


def latest_handoff_entry(text: str) -> str | None:
    """取最后一条交接单条目（模板：每条以 '## ' 开头、追加在末尾，见 §4）。"""
    blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)[1:]
    return blocks[-1] if blocks else None


def handoff_check(run_dir: Path) -> list[str]:
    """多 agent 模式：交接单存在且**最新一条**字段齐全（换 session 接力的凭据）。"""
    path = run_dir / HANDOFF_NAME
    if not path.exists():
        return [
            (
                f"多 agent 模式（存在 dispatch/）但缺 {HANDOFF_NAME}——每个角色收尾必须写交接单"
                "（换 session 接力凭据，八字段见 references/agent-roles-and-handoff.md §4）"
            )
        ]
    entry = latest_handoff_entry(path.read_text(encoding="utf-8-sig"))
    if not entry:
        return [f"{HANDOFF_NAME}: 无可解析条目（每条以 '## ' 开头，最新在末尾）"]
    missing = [f for f in HANDOFF_FIELDS if f not in entry]
    if missing:
        return [f"{HANDOFF_NAME}: 最新交接单缺字段 {missing}（字段集见 references/agent-roles-and-handoff.md §4）"]
    return []


def pointer_tokens(text: str) -> list[str]:
    """回执里声明的证据指针（带目录的相对路径）；占位符 `<…>` 跳过。"""
    return [t for t in POINTER_RE.findall(text) if "<" not in t and ">" not in t]


def receipt_chain_check(run_dir: Path, dispatch_dir: Path) -> list[str]:
    """派发单 → 回执（四要素）→ 证据指针链可达；≥2 单元并行须有资源租约（§3/§5）。"""
    errs: list[str] = []
    dispatches = sorted(dispatch_dir.glob("dispatch-*.md"))
    if not dispatches:
        return ["dispatch/ 下无 dispatch-*.md（多 agent 模式须有派发单，见 §3）"]
    evidence_root = run_dir / "evidence"
    for d in dispatches:
        feature = d.stem[len("dispatch-") :]
        found = sorted(evidence_root.rglob(f"receipt-{feature}.md")) if evidence_root.exists() else []
        if not found:
            errs.append(
                f"派发单 {d.name} 无对应回执 receipt-{feature}.md"
                "（回执落 evidence/{task_id}/{stage}/{feature}/，见 §3）"
            )
            continue
        receipt = found[0]
        text = receipt.read_text(encoding="utf-8-sig")
        missing = [f for f in RECEIPT_FIELDS if f not in text]
        if missing:
            errs.append(
                f"{receipt.name}: 回执缺要素 {missing}（结论 / 命令 / 退出码 / 证据；四行格式见 dispatch-templates）"
            )
        for token in pointer_tokens(text):
            target = Path(token)
            if not target.is_absolute():
                target = run_dir / target
            if not target.exists():
                errs.append(f"{receipt.name}: 指针链断链——声明的证据不存在：{token}（断链的行不得进总表，见 §3）")
    if len(dispatches) >= 2 and evidence_root.exists() and not list(evidence_root.rglob("lease-*.md")):
        errs.append(
            "多单元（≥2 派发单）未见资源租约 lease-*.md——并行须由部署与资源分配角色签发"
            "（一次一实验 / 卡组互斥 / 超时回收，见 §5）"
        )
    return errs


def agent_contract_check(run_dir: Path) -> tuple[list[str], list[str]]:
    """多 agent 契约总入口；非多 agent 模式返回 (无 error, [skip 提示])。"""
    dispatch_dir = run_dir / DISPATCH_DIR
    if not dispatch_dir.is_dir():
        return [], ["未发现 agentic/dispatch/ —— 单 agent 自执行模式，跳过多 agent 契约校验"]
    return handoff_check(run_dir) + receipt_chain_check(run_dir, dispatch_dir), []


def _selftest_root() -> Path:
    """自测夹具目录：系统临时目录 + 普通 mkdir。

    为什么不用 `tempfile.mkdtemp`：它对新建目录做 `chmod(0o700)`，在拒绝 chmod 的受限
    环境（受限沙箱）下直接 PermissionError —— 门禁自测必须能在受限环境跑，否则等于没回归。
    """
    import os
    import tempfile

    root = Path(tempfile.gettempdir()) / f"stage_gate_selftest_{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def selftest() -> int:
    """负样本自测：多 agent 契约的每条检查都必须能被触发（防"门禁静默失效"）。"""
    import shutil

    root = _selftest_root()
    shutil.rmtree(root, ignore_errors=True)
    task = "20260916_drill_optimization"

    def fixture(name: str, *, dispatch: bool = True, units: int = 1) -> Path:
        d = root / name
        (d / "evidence" / task / "S0" / "cache").mkdir(parents=True, exist_ok=True)
        if dispatch:
            (d / DISPATCH_DIR).mkdir(parents=True, exist_ok=True)
            for i in range(units):
                (d / DISPATCH_DIR / f"dispatch-cache{i or ''}.md").write_text(
                    "工作目录 / 角色: developer / 必须使用 skill: dit-perf-opt\n", encoding="utf-8"
                )
        return d

    handoff = (
        "# 交接单\n\n## 2026-09-16 10:00 S0 · 结果分析 · agent-A\n"
        "- 状态: done\n"
        f"- 最后完成动作: 采集完成 `evidence/{task}/S0/cache/collect.log`\n"
        "- 未决问题: 无\n- 下一步: 分析 kernel 序列\n"
        f"- 产物指针: `evidence/{task}/S0/cache/collect.log`\n"
        "- 口径指纹: manifest.baseline.toml / 20 步\n- 资源占用: 卡组 0-1，已释放\n- 预算余量: 3/5\n"
    )
    receipt = (
        f"## 回执 cache\n- 结论: 采集完成\n- 命令: python collect.py 2>&1，退出码 0\n"
        "- 退出码: 0\n"
        f"- 证据: `evidence/{task}/S0/cache/collect.log`\n- 异常: 无\n"
    )
    cases: list[tuple[str, bool, callable]] = []

    def add(name: str, expect_error: bool, fn) -> None:
        cases.append((name, expect_error, fn))

    def c_single_agent() -> tuple[list[str], list[str]]:
        d = fixture("single", dispatch=False)
        return agent_contract_check(d)

    def c_complete() -> tuple[list[str], list[str]]:
        d = fixture("complete")
        (d / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "receipt-cache.md").write_text(receipt, encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "collect.log").write_text("ok\n", encoding="utf-8")
        return agent_contract_check(d)

    def c_no_handoff() -> tuple[list[str], list[str]]:
        d = fixture("no-handoff")
        return agent_contract_check(d)

    def c_handoff_missing_field() -> tuple[list[str], list[str]]:
        d = fixture("handoff-field")
        (d / HANDOFF_NAME).write_text(handoff.replace("- 预算余量: 3/5\n", ""), encoding="utf-8")
        return agent_contract_check(d)

    def c_no_receipt() -> tuple[list[str], list[str]]:
        d = fixture("no-receipt")
        (d / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
        return agent_contract_check(d)

    def c_receipt_missing_field() -> tuple[list[str], list[str]]:
        d = fixture("receipt-field")
        (d / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "collect.log").write_text("ok\n", encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "receipt-cache.md").write_text(
            receipt.replace("- 结论: 采集完成\n", ""), encoding="utf-8"
        )
        return agent_contract_check(d)

    def c_broken_pointer() -> tuple[list[str], list[str]]:
        d = fixture("broken-pointer")
        (d / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "receipt-cache.md").write_text(
            receipt, encoding="utf-8"
        )  # collect.log 故意不建
        return agent_contract_check(d)

    def c_no_lease() -> tuple[list[str], list[str]]:
        d = fixture("no-lease", units=2)
        (d / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
        (d / "evidence" / task / "S0" / "cache" / "collect.log").write_text("ok\n", encoding="utf-8")
        for feat in ("cache", "cache0"):
            (d / "evidence" / task / "S0" / "cache" / f"receipt-{feat}.md").write_text(receipt, encoding="utf-8")
        return agent_contract_check(d)

    add("单 agent 模式：契约校验跳过（不报 error + 有 skip 提示）", False, c_single_agent)
    add("多 agent 完整夹具：无 error", False, c_complete)
    add("缺交接单 → 报错", True, c_no_handoff)
    add("交接单缺字段 → 报错", True, c_handoff_missing_field)
    add("派发单无对应回执 → 报错", True, c_no_receipt)
    add("回执缺要素 → 报错", True, c_receipt_missing_field)
    add("回执指针断链 → 报错", True, c_broken_pointer)
    add("多单元无资源租约 → 报错", True, c_no_lease)

    failures: list[str] = []
    for name, expect_error, fn in cases:
        errs, _warns = fn()
        if expect_error and not errs:
            failures.append(f"{name}：期望报错但未触发（门禁静默失效）")
        if not expect_error and errs:
            failures.append(f"{name}：不应报错却报错 {errs}")
        print(f"[{'FAIL' if (expect_error and not errs) or (not expect_error and errs) else 'ok'}] {name}")
    shutil.rmtree(root, ignore_errors=True)
    if failures:
        for f in failures:
            print(f"[error] {f}")
        print(f"selftest: FAIL（{len(failures)}/{len(cases)}）")
        return 1
    print(f"selftest: PASS（{len(cases)} 个用例：正样本无 error、负样本逐条被触发）")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGES), help="要校验的阶段（close=闭环）")
    parser.add_argument("--run-dir", default=".", type=Path, help="run-state.md 所在目录（默认当前目录）")
    parser.add_argument(
        "--task-id",
        default=None,
        help="任务 id（= runs 目录名，如 20260908_minimax-h3_optimization）；"
        "提供后校验 evidence/ 证据归属本任务（任务隔离）",
    )
    args = parser.parse_args(argv)

    run_state = args.run_dir / RUN_STATE_NAME
    errors: list[str] = []
    warns: list[str] = []

    if not run_state.exists():
        errors.append(f"run-state 不存在：{run_state}（先按 references/run-state.md 模板创建）")
        print(f"结论：error={len(errors)}")
        return 1

    content = run_state.read_text(encoding="utf-8-sig")
    rows = parse_stage_rows(content)
    if not rows:
        errors.append(f"{RUN_STATE_NAME} 推进表无有效行（期望表头：| 阶段 | 状态 | 验收证据 | 备注 |）")
        print(f"结论：error={len(errors)}")
        return 1

    for row in rows:
        tag = "ok" if row["status"] == "done" else "warn"
        print(f"[{tag}] {row['stage']:<5} 状态={row['status']:<10} 证据={(split_evidence(row['evidence']) or ['—'])}")

    target = next((r for r in rows if r["stage"] == args.stage), None)
    if target is None:
        errors.append(f"推进表未登记阶段 {args.stage}——先回写 run-state 再跑门禁")
    else:
        e, w = check_row(target, args.run_dir, is_close=(args.stage == "close"), task_id=args.task_id)
        errors += e
        warns += w

    # 前序阶段缺失仅提示，不阻断（S1/S3 等可因任务分支合法跳过）
    target_idx = STAGES.index(args.stage)
    for prev in STAGES[:target_idx]:
        if not any(r["stage"] == prev for r in rows):
            warns.append(f"前序阶段 {prev} 未登记（如按任务分支合法跳过可忽略，否则回补）")

    # close：覆盖清单逐特性判定收口校验（L1 总览收口的机械兜底）
    if args.stage == "close":
        errors += coverage_checklist_check(content)
        # 无条件跑（此前以 --task-id 传参为条件，按文档命令执行会静默跳过两个强校验）
        errors += _run_close_tools(args.run_dir)

    # 多 agent 契约（条件校验：run-dir 下有 dispatch/ 才算多 agent 模式，判据单点见
    # ../references/agent-roles-and-handoff.md §3/§4/§5；单 agent 自执行打印 [skip] 不报错）
    contract_errors, contract_warns = agent_contract_check(args.run_dir)
    errors += contract_errors
    warns += contract_warns
    if (args.run_dir / DISPATCH_DIR).is_dir():
        units = len(list((args.run_dir / DISPATCH_DIR).glob("dispatch-*.md")))
        print(f"[info] 多 agent 契约校验：{units} 个单元（交接单八字段 / 回执四要素 / 指针链 / 租约）已校")

    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)}（stage={args.stage}）")
    return 1 if errors else 0


def _run_close_tools(run_dir: Path) -> list[str]:
    """close 前置工具：报表 lint + 表结构/数值审计 + profile 强校验（缺工具/失败均记 error）。

    不接收 task_id：任务隔离校验在 check_row（evidence/ 前缀）内完成，这里只处理产物。
    """
    import os
    import subprocess
    import sys

    def run_tool(cmd: list[str]) -> subprocess.CompletedProcess:
        """跑子工具；两侧都钉 UTF-8，保证中文诊断文本可读、不抛解码异常。

        为什么两侧都要钉：`text=True` 不带 `encoding` 时**父侧按 locale 解码**，而子进程
        的管道 stdout **也按 locale 编码**——同 locale 时偶然一致，一旦父进程处于 UTF-8
        模式（`python -X utf8`）或两侧 `PYTHONIOENCODING` 不一致，就会父侧按 UTF-8 解
        locale 字节：轻则中文乱码，重则 `UnicodeDecodeError`（此时 `r.stdout` 变 None，
        后面 `r.stdout[-2000:]` 直接 `TypeError` 崩掉整个 close 门禁）。
        子侧用 `PYTHONIOENCODING`（**只影响 stdio**，不改 `open()` 默认编码，故不改变各工具
        自身的读写语义）；父侧 `errors="replace"` 兜底，任何字节都能得到可读文本而不抛异常。
        退出码语义不变（仍是 `returncode`）。
        """
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,  # 非 0 退出码由调用方查 returncode 判定，语义与默认值一致
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

    errs: list[str] = []
    here = Path(__file__).resolve().parent
    scripts = here / "report_lint.py"
    # check_profile/quality_compare 位于仓库根 evals/——多路径探测
    # （dev-skills 与 product 仓分离时均可定位）
    candidates = [here.parents[3] / "evals", here.parents[2].parent.parent / "evals"]
    repo_evals = next(
        (p for p in candidates if (p / "scripts" / "check_profile.py").exists()),
        candidates[0],
    )
    # 报表路径：run-dir/../runs/{task_id}_{model}_optimization/overview_report.md
    # （由 close 行解析更稳）
    rows = parse_stage_rows((run_dir / RUN_STATE_NAME).read_text(encoding="utf-8-sig"))
    close_row = next((r for r in rows if r["stage"] == "close"), None)
    overview = None
    detail = None
    if close_row:
        for entry in split_evidence(close_row["evidence"]):
            p = Path(entry)
            if not p.is_absolute():
                p = run_dir / p
            if p.name == "overview_report.md":
                overview = p
            elif p.name == "detail_report.md":
                detail = p

    if overview and scripts.exists():
        r = run_tool([sys.executable, str(scripts), str(overview)])
        if r.returncode != 0:
            errs.append(f"close: report_lint 失败（{overview.name}）：\n{r.stdout[-2000:]}")
    elif overview:
        errs.append(f"close: report_lint.py 缺失（应随 skills 提供）：{scripts}")
    else:
        errs.append("close: 推进表 close 行未声明 overview_report.md（lint 无法定位报表）")

    # 结构与数值审计（契约 §7/§8）：lint 管"主表写法合规"，审计管"改表后结构是否还渲染得出来、
    # 每个数字是否与自己表里的分母自洽"（含"锚点缺失须显式报错"与"分母口径"打印）；
    # 缺工具/非 0 均记 error——lint + 结构审计 + 数值审计三者都干净才算闭环。
    audit = here / "audit_report.py"
    if overview and audit.exists():
        r = run_tool([sys.executable, str(audit), str(overview)])
        if r.returncode != 0:
            errs.append(f"close: audit_report 失败（{overview.name}）：\n{r.stdout[-2000:]}")
    elif overview:
        errs.append(f"close: audit_report.py 缺失（应随 skills 提供）：{audit}")

    # 双报表契约：close 行须同时声明 detail_report.md（report_lint 只 lint overview）
    if not detail:
        errs.append("close: 推进表 close 行未声明 detail_report.md（双报表缺一视为未闭环）")

    chk = repo_evals / "scripts" / "check_profile.py"
    if chk.exists() and overview:
        task_dir = overview.parent
        # model 名从产物目录名推导：runs/{task_id}_{model}_optimization
        model = _model_from_task_dir(task_dir)
        r = run_tool([sys.executable, str(chk), "--model", model, "--task-dir", str(task_dir)])
        if r.returncode != 0:
            errs.append(f"close: check_profile 失败：\n{r.stdout[-1500:]}")
    elif overview:
        errs.append(f"close: check_profile.py 缺失（应随 evals 提供）：{chk}")
    return errs


def _model_from_task_dir(task_dir: Path) -> str:
    """runs/20260908_minimax-h3_optimization → minimax-h3（中间段，取 task_id 后首个非日期段）。"""
    name = task_dir.name
    # 去掉 runs 前缀（若有）与 _optimization 后缀
    name = name.removeprefix("runs_")
    name = name.removesuffix("_optimization")
    # 20260908_minimax-h3 → 去首段日期后取 minimax-h3（去尾 _optimization 已做）
    parts = name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 8:
        return parts[1]
    return name


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())
    sys.exit(main())
