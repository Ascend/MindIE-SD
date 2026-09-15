#!/usr/bin/env python3
"""gen_profile.py —— 由流程生成具体模型 profile（不入库，落 runs/{task_id}/profiles/）。

S0 冻结基线后调用：生成物骨架由本脚本内嵌定义（与 `profiles/_template.toml` 字段契约一致——
该文件是给人看的**字段语义文档**（含说明非纯 toml），脚本以其存在性 + 域/阈值字段标记作门禁；
改字段须同步 脚本骨架 + _template.toml + check_profile.py 的 REQUIRED_*/域契约校验）。
**不预填可推导数值**（psnr/ssim 由 quality_compare.py 现算）。

域（domain）必须显式给出、**不接受默认值**（变更理由，2026-09-15）：质量阈值**不可跨域迁移**——
视频域同 prompt 异配置的帧级相似度天然偏低（混沌轨迹）→ 定量指标只作基线登记与回归对比、
不设绝对门槛；图像域用同 seed 像素对口径、质量域显著高于视频 → 可设绝对门槛。原先「无域字段」
的 profile 形态会让图像任务**静默套用视频域阈值**（判错也不报错），故本脚本对 --domain
缺失/非法一律 fail-closed（非 0 退出、不落盘），错误信息说明「为何不许默认」。

阈值语义由 domain 唯一决定；本脚本只写**字段 + 契约指针**，不写任何阈值数字——阈值契约归**本目录**
（`evals/README.md` + `evals/profiles/README.md`）。**依赖方向：skills 引用 evals，本目录不反向引用
`.agents/` 路径**（本仓数字纪律：绝对质量分值不入库）。

用法:
    python evals/scripts/gen_profile.py --model MiniMax-H3-FL2VA \
        --task-dir runs/20260908_minimax-h3_optimization \
        --domain video_chaos --resolution 1024x576 --frames 124 --steps 60 \
        --topology "TP2 (2 cards)" --frozen-hash <md5> --seed 1101 \
        --visual inconclusive --off-identity "<说明>" \
        [--prompts <file>] [--entry <占位>]
输出: runs/{task_dir}/profiles/{model}.toml
退出码: 0 = 生成/dry-run 成功；1 = 模板或骨架自检失败；2 = --domain 缺失/非法（fail-closed）。
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]  # <repo>/evals
TEMPLATE = REPO / "profiles" / "_template.toml"

REQUIRED_HEADERS = (
    "[profile]",
    "[prompts]",
    "[geometry]",
    "[baseline]",
    "[decisions]",
    "[last_verified]",
)

# _template.toml 必须表达的字段标记（域 + 该域阈值语义）：缺任一即字段契约漂移 → 拒绝生成。
TEMPLATE_FIELD_MARKERS = ("domain", "threshold_policy", "threshold_source")

# 阈值真源（唯一）：本文件只放字段与指针，不放任何阈值数字。域 → 阈值语义映射须与
# check_profile.py 的 DOMAIN_CONTRACTS（及 POLICIES）逐字一致——改一处必须同步另一处。
THRESHOLD_SOURCE = "evals/profiles/README.md + evals/README.md"
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
DOMAINS = tuple(DOMAIN_CONTRACTS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--task-dir",
        required=True,
        help="runs/{task_id}_{model}_optimization（相对仓库或绝对）",
    )
    ap.add_argument(
        "--domain",
        default=None,
        help="必填且**无默认**：video_chaos | image_seed_deterministic（缺失/非法一律报错退出）",
    )
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

    # --domain 缺失/非法 → fail-closed（域没有默认值，理由见模块 docstring：跨域套阈值会静默判错）
    if args.domain is None:
        print(
            "[error] 必须显式指定 --domain：**域没有默认值**。视频域不设绝对质量门槛、图像域可设绝对"
            "门槛，两域阈值与数值域不可跨任务类型迁移；缺省套用视频域语义会让图像任务静默用错阈值。"
            "请显式二选一：" + " | ".join(DOMAINS)
        )
        return 2
    if args.domain not in DOMAIN_CONTRACTS:
        print(
            f"[error] 非法 --domain={args.domain!r}：域只能取自固定枚举（"
            + " | ".join(DOMAINS)
            + "），不接受自由文本/别名——它同时决定取样协议与阈值语义，写错即等于跨域套阈值。"
        )
        return 2
    contract = DOMAIN_CONTRACTS[args.domain]

    if not TEMPLATE.exists():
        print(f"[error] 模板缺失：{TEMPLATE}（仓库须含 profiles/_template.toml）")
        return 1
    template_text = TEMPLATE.read_text(encoding="utf-8")
    missing_markers = [m for m in TEMPLATE_FIELD_MARKERS if m not in template_text]
    if missing_markers:
        print(
            f"[error] _template.toml 未声明域/阈值语义字段 {missing_markers}：字段契约漂移——"
            "先同步 _template.toml 与本脚本骨架（域是 profile 必填契约，不得只在一侧存在）。"
        )
        return 1

    task = Path(args.task_dir)
    out_dir = task / "profiles"
    out = out_dir / f"{args.model}.toml"

    # 占位值不静默：check_profile 的 REQUIRED_FIELDS 会把含 "<" 的值判为「缺失或占位」并拦下 close 门
    # （既有缺口：--prompts 的默认值即占位值 → 缺省生成的 profile 过不了门；此处只告警，不改 CLI 契约）
    if "<" in args.prompts:
        print(
            f"[warn] [prompts] file 仍是占位值：{args.prompts!r}——close 前置 check_profile 会判"
            "「缺失或占位」拦下，请用 --prompts <真实提示词集文件> 重生成（本次仍按原样落盘）。"
        )

    if args.dry_run:
        print(
            f"[dry-run] 将生成 {out}（model={args.model}, domain={args.domain}, "
            f"threshold_policy={contract['threshold_policy']}, steps={args.steps}, "
            f"frozen_hash={args.frozen_hash[:12]}…）"
        )
        return 0

    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    run_ref = str(task).replace("\\", "/")
    content = f"""# Profile: {args.model}（流程生成 · 不入库 · 数值现算）
# 生成: gen_profile.py @ {today}；domain={args.domain} 为显式指定（域无默认值，不可缺省）。
# 契约/判定/指针入此处，可推导数值以 quality.json 为准。

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
threshold_policy = "{contract['threshold_policy']}"
threshold_source = "{THRESHOLD_SOURCE}"
threshold_semantics = "{contract['threshold_semantics']}"

[last_verified]
date = "{today}"
run_ref = "{run_ref}/"
quality_json = "{run_ref}/quality.json"
note = "profile 只保证契约/判定/指针与 runs 产物一致；数值以 quality.json 为准"
"""
    missing_headers = [h for h in REQUIRED_HEADERS if h not in content]
    if missing_headers:
        print(f"[error] 内嵌骨架缺 section {missing_headers}（脚本骨架损坏，拒绝落盘）")
        return 1
    try:
        tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        print(
            f"[error] 生成物不是合法 TOML（多为入参含引号/反斜杠）：{exc}——修正入参后重跑；"
            "本次不落盘，以免留下坏 profile。"
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"[ok] 生成 {out}（domain={args.domain}, threshold_policy={contract['threshold_policy']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
