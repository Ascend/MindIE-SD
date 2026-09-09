#!/usr/bin/env python3
"""gen_profile.py —— 由流程生成具体模型 profile（不入库，落 runs/{task_id}/profiles/）。

S0 冻结基线后调用：生成物骨架由本脚本内嵌定义（与 `profiles/_template.toml` 字段契约一致——
该文件是给人看的**字段语义文档**（含说明非纯 toml），脚本仅以其存在性作门禁；改字段须同步
脚本骨架 + _template.toml + check_profile.py 的 REQUIRED_*）。**不预填可推导数值**
（psnr/ssim 由 quality_compare.py 现算）。

用法:
    python evals/scripts/gen_profile.py --model MiniMax-H3-FL2VA \
        --task-dir runs/20260908_minimax-h3_optimization \
        --domain video_chaos --resolution 1024x576 --frames 124 --steps 60 \
        --topology "TP2 (2 cards)" --frozen-hash <md5> --seed 1101 \
        --visual inconclusive --off-identity "<说明>" \
        [--prompts <file>] [--entry <占位>]
输出: runs/{task_dir}/profiles/{model}.toml
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # <repo>/evals
TEMPLATE = REPO / "profiles" / "_template.toml"

REQUIRED_HEADERS = (
    "[profile]", "[prompts]", "[geometry]", "[baseline]",
    "[decisions]", "[last_verified]",
)

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--task-dir", required=True,
        help="runs/{task_id}_{model}_optimization（相对仓库或绝对）",
    )
    ap.add_argument("--domain", required=True, choices=["video_chaos", "image_seed_deterministic"])
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--topology", required=True)
    ap.add_argument("--frozen-hash", required=True, help="baseline 帧集 md5（一致性锚）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visual", default="inconclusive（无 VLM，并排存证）")
    ap.add_argument("--off-identity", required=True)
    ap.add_argument("--prompts", default="<提示词集文件或官方引用>")
    ap.add_argument("--entry", default="<推理脚本或服务端点占位>")
    ap.add_argument("--precision", default="bfloat16")
    ap.add_argument("--lossless-e2e", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not TEMPLATE.exists():
        print(f"[error] 模板缺失：{TEMPLATE}（仓库须含 profiles/_template.toml）")
        return 1

    task = Path(args.task_dir)
    out_dir = task / "profiles"
    out = out_dir / f"{args.model}.toml"

    if args.dry_run:
        print(f"[dry-run] 将生成 {out}（model={args.model}, domain={args.domain}, "
              f"steps={args.steps}, frozen_hash={args.frozen_hash[:12]}…）")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    run_ref = str(task).replace("\\", "/")
    if args.domain == "video_chaos":
        threshold_semantics = (
            "video_chaos 域：定量仅登记回归不设绝对门槛；主判据 visual+off_identity"
        )
    else:
        threshold_semantics = (
            "image 域：21-seed 同 seed 像素对，可设绝对门槛（域阈值不跨任务迁移）"
        )
    content = f"""# Profile: {args.model}（流程生成 · 不入库 · 数值现算）
# 生成: gen_profile.py @ {today}；契约/判定/指针入此处，可推导数值以 quality.json 为准。

[profile]
model = "{args.model}"
entry = "{args.entry}"
precision = "{args.precision}"
official_config = true
domain = "{args.domain}"

[prompts]
file = "{args.prompts}"
seed = {args.seed}

[geometry]
resolution = "{args.resolution}"
frames = {args.frames}
steps = {args.steps}

[baseline]
topology = "{args.topology}"
lossless_e2e_seconds = {args.lossless_e2e}
run_ref = "{run_ref}/"
frozen_hash = "{args.frozen_hash}"

[decisions]
visual_artifact = "{args.visual}"
off_identity = "{args.off_identity}"
threshold_semantics = "{threshold_semantics}"

[last_verified]
date = "{today}"
run_ref = "{run_ref}/"
quality_json = "{run_ref}/quality.json"
note = "profile 只保证契约/判定/指针与 runs 产物一致；数值以 quality.json 为准"
"""
    out.write_text(content, encoding="utf-8")
    print(f"[ok] 生成 {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
