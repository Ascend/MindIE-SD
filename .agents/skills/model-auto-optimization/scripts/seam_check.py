#!/usr/bin/env python3
"""seam_check.py —— 特性组合的静态 seam/能力冲突检查（组合前门禁，零 NPU）。

判定规则（对应 combination-search.md 的 seam 表）：
- capability：候选特性 requires_model_capabilities 必须由模型能力提供，否则拒；
- exclusive seam：同一互斥 seam（precision/attention_backend/attention_precision/step_decision/
  token_semantics）内至多一个 writer；>1 即冲突——
  其中 step_decision/token_semantics 属「窗口可分」seam，输出 warning（可错开窗口/只留最强档），
  其余输出 error（不允许双 writer）；
- conflicts_with：显式互斥对（如 DiTCache × AttentionCache）输出 error。

用法：
    python seam_check.py --features quant_w8a8_dynamic,sparse_rf_v2,cache_dit
    python seam_check.py --features cache_dit,cache_attention --model minimax-h3-vllm-omni
    python seam_check.py --required-combos quant_w8a8_mxfp8,sparse_rf_v2,cache_dit --model minimax-h3-vllm-omni
        # S4-2 组合前：由已过 gate 的单点 frontier 推导 [MUST] 必测组合清单
        # （跨族两两 + 三元 Cache+量化+稀疏，行内带 seam 预判）
退出码：0 = 无 error（可有 warning）；1 = 存在 error。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DECL_PATH = Path(__file__).resolve().parent / "feature_declarations.json"


def load_declarations(path: Path = DECL_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check(candidates: list[str], model: str | None, decl: dict) -> tuple[list[str], list[str]]:
    """返回 (errors, warnings)。"""
    errors: list[str] = []
    warnings: list[str] = []
    by_id = {f["id"]: f for f in decl["features"]}
    unknown = [c for c in candidates if c not in by_id]
    if unknown:
        errors.append(f"未知特性：{unknown}（可用：{sorted(by_id)}）")
        return errors, warnings

    caps = set(decl["models"][model]["capabilities"]) if model and model in decl["models"] else None
    if model and model not in decl["models"]:
        errors.append(f"未知模型适配：{model}")
        return errors, warnings

    chosen = [by_id[c] for c in candidates]

    # 1) capability
    for f in chosen:
        if caps is None:
            continue
        missing = [cap for cap in f["requires_model_capabilities"] if cap not in caps]
        if missing:
            errors.append(f"{f['id']} 需模型能力 {missing}（模型 {model}）")

    # 2) exclusive seam
    warning_seams = set(decl["window_conflict_warning_seams"])
    for seam in decl["exclusive_seams"]:
        writers = [f["id"] for f in chosen if seam in f["seams"]]
        if len(writers) > 1:
            msg = f"互斥 seam '{seam}' 多 writer：{sorted(writers)}"
            if seam in warning_seams:
                warnings.append(f"{msg}（同窗口需错开/只留最强档，见 combination-search）")
            else:
                errors.append(msg)

    # 3) explicit conflicts（成对检查，只报一次）
    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            if b["id"] in a["conflicts_with"] or a["id"] in b["conflicts_with"]:
                errors.append(f"{a['id']} 与 {b['id']} 显式互斥")

    return errors, warnings


def required_combos(single_ids: list[str], model: str | None, decl: dict) -> list[dict]:
    """由「已通过 quality gate 的单点 frontier」推导 S4-2 必测组合（[MUST] 行）。

    族判定：family 前缀 quant → 量化族；sparse → 稀疏族；cache → Cache 族。
    规则（combination-search.md「必测组合覆盖集」）：
    - 参与的族 ≥2 → 跨族两两全测；
    - 量化/稀疏/Cache 三族齐备 → 追加三元 Cache+量化+稀疏（强制实测）。
    每族只应传「该族最强档」单点 id（多档时取最强，其余进 §5 子表）。
    返回 [{combo, ids, families, must, errors, warnings}]；每行 seam 判定复用 check()。
    """
    by_id = {f["id"]: f for f in decl["features"]}
    family_map = {"quant": "量化", "sparse": "稀疏", "cache": "Cache"}
    get_family = lambda i: by_id[i]["family"]
    present = {}
    for i in single_ids:
        if i not in by_id:
            raise ValueError(f"未知特性：{i}")
        fam = get_family(i)
        present.setdefault(family_map.get(fam, fam), []).append(i)
    fams = sorted(present)
    lines: list[dict] = []
    for a in range(len(fams)):
        for b in range(a + 1, len(fams)):
            ids = [present[fams[a]][0], present[fams[b]][0]]
            e, w = check(ids, model, decl)
            lines.append({"combo": f"{fams[a]} × {fams[b]}", "ids": ids,
                          "families": [fams[a], fams[b]], "must": True,
                          "errors": e, "warnings": w})
    if {"量化", "稀疏", "Cache"} <= set(fams):
        ids = [present["量化"][0], present["稀疏"][0], present["Cache"][0]]
        e, w = check(ids, model, decl)
        lines.append({"combo": "Cache + 量化 + 稀疏（三元）", "ids": ids,
                      "families": ["量化", "稀疏", "Cache"], "must": True,
                      "errors": e, "warnings": w})
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        help="候选特性逗号分隔，如 quant_w8a8_dynamic,sparse_rf_v2",
    )
    parser.add_argument(
        "--required-combos",
        metavar="IDS",
        help="已过 quality gate 的单点 frontier 特性逗号分隔（每族最强档），推导 [MUST] 必测组合",
    )
    parser.add_argument("--model", help="模型适配名（feature_declarations.json 的 models 键）")
    parser.add_argument("--decl", default=str(DECL_PATH), help="声明文件路径（默认随脚本）")
    args = parser.parse_args(argv)

    decl = load_declarations(Path(args.decl))

    if args.required_combos:
        single_ids = [c.strip() for c in args.required_combos.split(",") if c.strip()]
        try:
            lines = required_combos(single_ids, args.model, decl)
        except ValueError as exc:
            print(f"[error] {exc}")
            return 1
        errors = 0
        for ln in lines:
            for w in ln["warnings"]:
                print(f"[warn]  {ln['combo']}: {w}")
            for e in ln["errors"]:
                errors += 1
                print(f"[error] {ln['combo']}: {e}")
            status = "error" if ln["errors"] else ("warn" if ln["warnings"] else "ok")
            print(f"[MUST] {ln['combo']}  ids={ln['ids']}  seam={status}")
        print(f"结论：[MUST] 行 {len(lines)}（无 error 即可按序实测；warning 行按窗口可分处理）")
        return 1 if errors else 0

    if not args.features:
        parser.error("--features 与 --required-combos 必须提供其一")

    candidates = [c.strip() for c in args.features.split(",") if c.strip()]
    errors, warnings = check(candidates, args.model, decl)

    for w in warnings:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[error] {e}")
    print(f"结论：error={len(errors)} warning={len(warnings)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
