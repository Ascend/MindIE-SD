#!/usr/bin/env python3
"""stage_gate.py —— 编排层阶段推进门禁（零 NPU、零数据）。

做：解析 run-state.md（见 ../references/run-state.md）的「阶段推进表」→ 校验目标阶段
状态为 done 且声明的验收证据路径真实存在；close 阶段额外强制声明 overview_report.md /
detail_report.md（缺任一视为未闭环）。不做：真实运行/耗时/质量判定（那些在能力技能门禁内）。

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

STAGES = ("S0", "S1", "S3", "S4", "S5", "close")
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
        rows.append({"stage": stage, "status": status, "evidence": evidence,
                     "note": cells[3] if len(cells) > 3 else ""})
    return rows


def split_evidence(raw: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[，,;；、]", raw) if p.strip()]
    return parts


def coverage_checklist_check(text: str) -> list[str]:
    """close：校验「特性覆盖清单」逐特性判定已收口（无空/无"分析后做"残留）。"""
    errors: list[str] = []
    m = re.search(r"^##\s*特性覆盖清单.*?$\n(.*?)(?=^##\s|\Z)", text, re.M | re.S)
    if not m:
        errors.append("close: run-state 缺「特性覆盖清单」——闭环前须对固定特性全集逐项判定收口")
        return errors
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s.startswith("|"):
            continue
        if re.search(r"^\|[\s\-:|]+\|?$", s):  # 表头分隔行
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 2 or cells[0] in ("特性", "特性/能力", "特性/实现"):
            continue
        decision = cells[1]
        if not decision or "分析后做" in decision or "未裁决" in decision:
            errors.append(
                f"close: 覆盖清单「{cells[0]}」判定未收口"
                f"（{decision if decision else '空'}）——闭环前须为 做/不做（带理由）"
            )
    return errors


def check_row(row: dict, run_dir: Path, is_close: bool, task_id: str | None = None,
              task_start_mtime: float | None = None) -> tuple[list[str], list[str]]:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=list(STAGES),
                        help="要校验的阶段（close=闭环）")
    parser.add_argument("--run-dir", default=".", type=Path,
                        help="run-state.md 所在目录（默认当前目录）")
    parser.add_argument("--task-id", default=None,
                        help="任务 id（= runs 目录名，如 20260908_minimax-h3_optimization）；"
                             "提供后校验 evidence/ 证据归属本任务（任务隔离）")
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
        errors.append(
            f"{RUN_STATE_NAME} 推进表无有效行"
            "（期望表头：| 阶段 | 状态 | 验收证据 | 备注 |）"
        )
        print(f"结论：error={len(errors)}")
        return 1

    for row in rows:
        tag = "ok" if row["status"] == "done" else "warn"
        print(f"[{tag}] {row['stage']:<5} 状态={row['status']:<10} "
              f"证据={(split_evidence(row['evidence']) or ['—'])}")

    target = next((r for r in rows if r["stage"] == args.stage), None)
    if target is None:
        errors.append(f"推进表未登记阶段 {args.stage}——先回写 run-state 再跑门禁")
    else:
        e, w = check_row(target, args.run_dir, is_close=(args.stage == "close"),
                         task_id=args.task_id)
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
        if args.task_id:
            errors += _run_close_tools(args.task_id, args.run_dir)

    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)}（stage={args.stage}）")
    return 1 if errors else 0


def _run_close_tools(task_id: str, run_dir: Path) -> list[str]:
    """close 前置工具：报表结构 lint + profile 强校验（缺工具/失败均记 error）。"""
    import subprocess
    import sys
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
        r = subprocess.run([sys.executable, str(scripts), str(overview)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            errs.append(f"close: report_lint 失败（{overview.name}）：\n{r.stdout[-2000:]}")
    elif overview:
        errs.append(f"close: report_lint.py 缺失（应随 skills 提供）：{scripts}")
    else:
        errs.append("close: 推进表 close 行未声明 overview_report.md（lint 无法定位报表）")

    # 双报表契约：close 行须同时声明 detail_report.md（report_lint 只 lint overview）
    if not detail:
        errs.append("close: 推进表 close 行未声明 detail_report.md（双报表缺一视为未闭环）")

    chk = repo_evals / "scripts" / "check_profile.py"
    if chk.exists() and overview:
        task_dir = overview.parent
        # model 名从产物目录名推导：runs/{task_id}_{model}_optimization
        model = _model_from_task_dir(task_dir)
        r = subprocess.run([sys.executable, str(chk), "--model", model,
                            "--task-dir", str(task_dir)],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            errs.append(f"close: check_profile 失败：\n{r.stdout[-1500:]}")
    elif overview:
        errs.append(f"close: check_profile.py 缺失（应随 evals 提供）：{chk}")
    return errs


def _model_from_task_dir(task_dir: Path) -> str:
    """runs/20260908_minimax-h3_optimization → minimax-h3（中间段，取 task_id 后首个非日期段）。"""
    name = task_dir.name
    # 去掉 runs 前缀（若有）与 _optimization 后缀
    if name.startswith("runs_"):
        name = name[5:]
    if name.endswith("_optimization"):
        name = name[: -len("_optimization")]
    # 20260908_minimax-h3 → 去首段日期后取 minimax-h3（去尾 _optimization 已做）
    parts = name.split("_", 1)
    if len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 8:
        return parts[1]
    return name


if __name__ == "__main__":
    sys.exit(main())
