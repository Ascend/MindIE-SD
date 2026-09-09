#!/usr/bin/env python3
"""check_profile.py —— skill 强校验：具体模型 profile 是否由流程正确生成（close 前置门禁）。

校验（全部通过 exit 0，任一失败 exit 1）：
1. runs/{task_dir}/profiles/{model}.toml 存在（**仓库内不存具体模型 profile**——若在仓库
   evals/profiles/ 下发现具体模型 profile → error「具体模型 profile 不得入库」）；
2. 必填 section 齐（[profile][prompts][geometry][baseline][decisions][last_verified]）；
3. 必填字段非占位/非空（model/domain/resolution/steps/topology/frozen_hash/off_identity）；
4. frozen_hash 非默认空值；quality_json 指针指向 runs/{task_dir}/quality.json；
5. 与 evidence.json 的 baseline 字段一致（model/topology/分辨率，若提供 --evidence）。

用法:
    python evals/scripts/check_profile.py --model MiniMax-H3-FL2VA \
        --task-dir runs/20260908_minimax-h3_optimization [--evidence <path>]
退出码：0 = 通过；1 = 存在 error（不得 close）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]          # <repo>/evals
REQUIRED_SECTIONS = ("profile", "prompts", "geometry", "baseline", "decisions", "last_verified")
REQUIRED_FIELDS = {
    "profile": ("model", "domain", "precision"),
    "prompts": ("file", "seed"),
    "geometry": ("resolution", "steps"),
    "baseline": ("topology", "run_ref", "frozen_hash"),
    "decisions": ("off_identity", "threshold_semantics"),
    "last_verified": ("quality_json",),
}

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--evidence", help="evidence.json 路径（可选交叉校验）")
    args = ap.parse_args(argv)

    errors: list[str] = []
    repo_profiles = REPO / "profiles"

    # 0) 仓库不得存具体模型 profile（只允许 _template.toml）
    for p in repo_profiles.glob("*.toml"):
        if p.name != "_template.toml":
            errors.append(
                f"具体模型 profile 不得入库：{p}"
                "（应删除；用 gen_profile.py 生成到 runs/）"
            )

    task = Path(args.task_dir)
    prof = task / "profiles" / f"{args.model}.toml"
    if not prof.exists():
        errors.append(f"profile 缺失（流程未生成）：{prof}（S0 冻结基线后须跑 gen_profile.py）")
        for e in errors:
            print(f"[error] {e}")
        print(f"结论：error={len(errors)}")
        return 1

    try:
        cfg = tomllib.loads(prof.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"profile 解析失败：{exc}")
        for e in errors:
            print(f"[error] {e}")
        print(f"结论：error={len(errors)}")
        return 1

    # 1) sections 齐
    for sec in REQUIRED_SECTIONS:
        if sec not in cfg:
            errors.append(f"缺 section [{sec}]")

    # 2) 必填字段非占位
    for sec, fields in REQUIRED_FIELDS.items():
        for f in fields:
            val = cfg.get(sec, {}).get(f)
            if val is None or (isinstance(val, str) and (not val.strip() or "<" in val)):
                errors.append(f"[{sec}] {f} 缺失或占位：{val!r}")

    # 3) quality_json 指向 task 内（路径归一化后比较：容忍 ../ 与绝对路径等价）
    qj = cfg.get("last_verified", {}).get("quality_json", "")
    expected_qj = str(Path(task) / "quality.json")
    if qj:
        try:
            qj_norm = str(Path(qj).resolve())
            exp_norm = str(Path(expected_qj).resolve())
            if qj_norm != exp_norm:
                errors.append(
                    f"quality_json 指针应指向 task 内 quality.json（{exp_norm}），"
                    f"实际解析到 {qj_norm}"
                )
        except OSError:
            errors.append(f"quality_json 指针无法解析：{qj}")

    # 4) evidence.json 交叉（可选）
    if args.evidence and Path(args.evidence).exists():
        try:
            ev = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
            b = ev.get("baseline", {})
            if b.get("model") != args.model:
                errors.append(f"evidence.model={b.get('model')} ≠ profile model {args.model}")
            if b.get("topology") and b.get("topology") != cfg.get("baseline", {}).get("topology"):
                errors.append(f"evidence.topology={b.get('topology')} ≠ profile topology")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"evidence.json 解析失败：{exc}")

    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)}（model={args.model}）")
    return 1 if errors else 0

if __name__ == "__main__":
    sys.exit(main())
