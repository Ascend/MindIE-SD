#!/usr/bin/env python3
"""check_profile.py —— skill 强校验：具体模型 profile 是否由流程正确生成（close 前置门禁）。

校验（全部通过 exit 0，任一 error exit 1）：
1. runs/{task_dir}/profiles/{model}.toml 存在（**仓库内不存具体模型 profile**——若在仓库
   evals/profiles/ 下发现具体模型 profile → error「具体模型 profile 不得入库」）；
2. 必填 section 齐（[profile][prompts][geometry][baseline][decisions][last_verified]）；
3. 必填字段非占位/非空（model/precision/resolution/steps/topology/frozen_hash/off_identity/
   threshold_semantics）；
4. **域契约强校验**（`_check_domain_contract`，变更理由 2026-09-15）：[profile] domain 必须存在
   且取自固定枚举（域**无默认值也不可推断**——视频域不设绝对门槛、图像域可设绝对门槛，缺省套用
   视频域语义会让图像任务静默用错阈值）；[decisions] threshold_policy 必须与该 domain 对应、
   threshold_source 必须指向**本目录**的阈值契约（不得指向 `.agents/`——skills 引用 evals，反向不成立）、
   threshold_semantics 必须声明所属域。**旧版 profile（无域字段）→ 明确报错要求用 gen_profile.py
   重新生成**（不崩溃、不静默通过）；域与阈值语义错配一律 error；
5. frozen_hash 非默认空值；quality_json 指针指向 runs/{task_dir}/quality.json；
6. 与 evidence.json 的 baseline 字段一致（model/topology/域，若提供 --evidence）；
7. 预期域交叉校验：预期域只来自 --expect-domain 或 evidence.json 登记的 domain（**本脚本不取
   默认域、不推断域**）；两者都缺时打印 [warn] 说明「未做预期域交叉校验」（域取值合法、域与阈值
   语义自洽仍为强校验）——close 自动门（model-auto-optimization/scripts/stage_gate.py）只传
   --model/--task-dir，故此处不 fail-closed，但也**绝不静默**。

用法:
    python evals/scripts/check_profile.py --model MiniMax-H3-FL2VA \
        --task-dir runs/20260908_minimax-h3_optimization \
        [--evidence <path>] [--expect-domain video_chaos|image_seed_deterministic]
退出码：0 = 通过（可能带 warn）；1 = 存在 error（不得 close）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]  # <repo>/evals
REQUIRED_SECTIONS = ("profile", "prompts", "geometry", "baseline", "decisions", "last_verified")
# 域/阈值字段不列在此表：由 _check_domain_contract 强校验（需额外做枚举与 domain 与 policy 一致性判定）
REQUIRED_FIELDS = {
    "profile": ("model", "precision"),
    "prompts": ("file", "seed"),
    "geometry": ("resolution", "steps"),
    "baseline": ("topology", "run_ref", "frozen_hash"),
    "decisions": ("off_identity", "threshold_semantics"),
    "last_verified": ("quality_json",),
}

# 阈值真源（唯一）：绝对质量分值/阈值不入库、不入脚本；域 → 阈值语义映射须与 gen_profile.py 的
# DOMAIN_CONTRACTS 逐字一致（改一处必须同步另一处）。
THRESHOLD_SOURCE = "evals/profiles/README.md + evals/README.md"

# 方向约束（强制）：阈值契约归本目录；**不得指向 `.agents/`** ——
# 依赖方向是 skills → evals（技能引用本目录的契约与工具）。反向引用会让产品侧资产
# 随技能树重组而腐烂（历史实例：本目录曾指向已迁走的 skills 文件，迁移后即成死链）。
if ".agents/" in THRESHOLD_SOURCE:
    raise SystemExit(
        "[error] THRESHOLD_SOURCE 不得指向 `.agents/`：阈值契约归 evals 目录（依赖方向 skills → evals，反向不成立）"
    )
DOMAIN_CONTRACTS = {
    "video_chaos": {
        "threshold_policy": "baseline_regression_only",
        "threshold_semantics": (
            "video_chaos 域（视频/混沌轨迹）：定量指标（psnr/ssim）只作基线登记与回归对比，"
            "不设绝对门槛；主判据 visual_artifact + off_identity"
        ),
    },
    "image_seed_deterministic": {
        "threshold_policy": "absolute_threshold_allowed",
        "threshold_semantics": (
            "image_seed_deterministic 域（图像/21-seed 同 seed 像素对）：同 seed 像素质量域显著高于"
            "视频域，可设绝对门槛；域阈值不可跨任务类型迁移"
        ),
    },
}
POLICIES = ("baseline_regression_only", "absolute_threshold_allowed")


def _is_blank(val: object) -> bool:
    """占位/空值判定（与 REQUIRED_FIELDS 检查同一口径）。"""
    return val is None or (isinstance(val, str) and (not val.strip() or "<" in val))


def _norm_ref(ref: str) -> str:
    """指针归一化（仅用于比较）：统一分隔符 + 去前导 ./。"""
    return ref.strip().replace("\\", "/").removeprefix("./")


def _evidence_domain(ev: dict) -> str | None:
    """从 evidence.json 取已登记的域（三处可选位置，先到先得）。"""
    for holder in (ev, ev.get("baseline"), ev.get("evals")):
        if isinstance(holder, dict):
            val = holder.get("domain")
            if isinstance(val, str) and val.strip():
                return val.strip()
    return None


def _check_domain_contract(cfg: dict, expect_domain: str | None, errors: list[str]) -> str | None:
    """强校验「域 + 该域阈值语义」；返回 profile 声明的域（无法判定时为 None）。"""
    domain = cfg.get("profile", {}).get("domain")
    if _is_blank(domain):
        errors.append(
            "缺 [profile] domain（旧版 profile 无域字段）：域不可推断、更不可默认——视频域不设绝对门槛、"
            "图像域可设绝对门槛，缺省套用视频域语义会让图像任务静默用错阈值 → 必须重生成："
            "python evals/scripts/gen_profile.py --domain video_chaos|image_seed_deterministic ..."
        )
        return None
    if domain not in DOMAIN_CONTRACTS:
        errors.append(
            f"[profile] domain 取值非法：{domain!r}；合法取值："
            + " | ".join(DOMAIN_CONTRACTS)
            + "（域决定取样协议与阈值语义，不接受自由文本/别名）"
        )
        return None
    decisions = cfg.get("decisions", {})
    contract = DOMAIN_CONTRACTS[domain]
    policy = decisions.get("threshold_policy")
    if _is_blank(policy):
        errors.append(
            f"[decisions] threshold_policy 缺失：域阈值语义必须与域成对登记"
            f"（domain={domain} → threshold_policy={contract['threshold_policy']}）——旧脚本生成的 profile 请重生成"
        )
    elif policy not in POLICIES:
        errors.append(f"[decisions] threshold_policy 取值非法：{policy!r}；合法取值：" + " | ".join(POLICIES))
    elif policy != contract["threshold_policy"]:
        errors.append(
            f"[decisions] threshold_policy={policy!r} 与 domain={domain!r} 错配：该域应为 "
            f"{contract['threshold_policy']!r}——域与阈值语义错配即跨域套阈值（禁止）"
        )
    source = decisions.get("threshold_source")
    if _is_blank(source):
        errors.append(f"[decisions] threshold_source 缺失：阈值真源必须指向 {THRESHOLD_SOURCE}（阈值数字不入库）")
    elif _norm_ref(str(source)) != _norm_ref(THRESHOLD_SOURCE):
        errors.append(
            f"[decisions] threshold_source={source!r} 不是阈值真源：应为 {THRESHOLD_SOURCE}"
            "（真源只有一处，不得指向副本/别处）"
        )
    semantics = decisions.get("threshold_semantics")
    if isinstance(semantics, str) and semantics.strip() and domain not in semantics:
        errors.append(
            f"[decisions] threshold_semantics 未声明所属域 {domain!r}：疑似从其他域复制的阈值语义"
            "（跨域套阈值的典型形态）"
        )
    if expect_domain and domain != expect_domain:
        errors.append(
            f"域与预期不一致：profile domain={domain!r} ≠ 预期域 {expect_domain!r}"
            "（预期域来自 --expect-domain / evidence.json；域阈值不可跨任务类型迁移）"
        )
    return domain


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--evidence", help="evidence.json 路径（可选交叉校验）")
    ap.add_argument(
        "--expect-domain",
        choices=sorted(DOMAIN_CONTRACTS),
        default=None,
        help="本次任务应属的域；不取默认值，缺省时只告警而不做预期域交叉校验",
    )
    args = ap.parse_args(argv)

    errors: list[str] = []
    warns: list[str] = []
    repo_profiles = REPO / "profiles"

    # 0) 仓库不得存具体模型 profile（只允许 _template.toml）
    for p in repo_profiles.glob("*.toml"):
        if p.name != "_template.toml":
            errors.append(f"具体模型 profile 不得入库：{p}（应删除；用 gen_profile.py 生成到 runs/）")

    task = Path(args.task_dir)
    prof = task / "profiles" / f"{args.model}.toml"
    if not prof.exists():
        errors.append(f"profile 缺失（流程未生成）：{prof}（S0 冻结基线后须跑 gen_profile.py）")
        for e in errors:
            print(f"[error] {e}")
        print(f"结论：error={len(errors)}, warn=0（model={args.model}）")
        return 1

    try:
        cfg = tomllib.loads(prof.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"profile 解析失败：{exc}")
        for e in errors:
            print(f"[error] {e}")
        print(f"结论：error={len(errors)}, warn=0（model={args.model}）")
        return 1

    # 1) sections 齐
    for sec in REQUIRED_SECTIONS:
        if sec not in cfg:
            errors.append(f"缺 section [{sec}]")

    # 2) 必填字段非占位
    for sec, fields in REQUIRED_FIELDS.items():
        for f in fields:
            val = cfg.get(sec, {}).get(f)
            if _is_blank(val):
                errors.append(f"[{sec}] {f} 缺失或占位：{val!r}")

    # 3) evidence.json 交叉（可选）：先读——预期域可由其登记的 domain 提供
    evidence_domain: str | None = None
    if args.evidence and Path(args.evidence).exists():
        try:
            ev = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
            b = ev.get("baseline", {})
            if b.get("model") != args.model:
                errors.append(f"evidence.model={b.get('model')} ≠ profile model {args.model}")
            if b.get("topology") and b.get("topology") != cfg.get("baseline", {}).get("topology"):
                errors.append(f"evidence.topology={b.get('topology')} ≠ profile topology")
            evidence_domain = _evidence_domain(ev)
            if evidence_domain is None:
                warns.append("evidence.json 未登记 domain：无法用它作预期域交叉校验（可在 baseline.domain 登记）")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"evidence.json 解析失败：{exc}")
    elif args.evidence:
        warns.append(f"evidence.json 不存在，跳过交叉校验：{args.evidence}")

    # 4) 预期域（不取默认）：--expect-domain 优先，其次 evidence.json 登记的 domain
    if args.expect_domain and evidence_domain and args.expect_domain != evidence_domain:
        errors.append(
            f"预期域冲突：--expect-domain={args.expect_domain!r} ≠ evidence.domain={evidence_domain!r}"
            "（两者必须一致；域不得默认，也不得两处写法不一）"
        )
    expect_domain = args.expect_domain or evidence_domain

    # 5) 域契约强校验：存在 + 取值合法 + 与预期域一致 + 与阈值语义自洽
    declared_domain = _check_domain_contract(cfg, expect_domain, errors)
    if expect_domain is None:
        warns.append(
            "未提供预期域（--expect-domain 与 evidence.json 的 domain 均缺）：本次只强校验域取值合法 + "
            "域与阈值语义自洽，未做「与任务预期域一致」交叉校验；如需该门请传 --expect-domain 或在 "
            "evidence.json 登记 domain——不取默认域是刻意的（域不可推断）"
        )

    # 6) quality_json 指向 task 内（路径归一化后比较：容忍 ../ 与绝对路径等价）
    qj = cfg.get("last_verified", {}).get("quality_json", "")
    expected_qj = str(Path(task) / "quality.json")
    if qj:
        try:
            qj_norm = str(Path(qj).resolve())
            exp_norm = str(Path(expected_qj).resolve())
            if qj_norm != exp_norm:
                errors.append(f"quality_json 指针应指向 task 内 quality.json（{exp_norm}），实际解析到 {qj_norm}")
        except OSError:
            errors.append(f"quality_json 指针无法解析：{qj}")

    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)}, warn={len(warns)}（model={args.model}, domain={declared_domain or '未声明'}）")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
