#!/usr/bin/env python3
"""manifest_dryrun.py —— manifest.toml 的 NPU 前结构性门禁（dry-run）。

做：schema/枚举校验 → 特性 seam 检查（复用同目录 seam_check.py）→ 路径只读检查 →
    按框架渲染启动计划（vllm-omni / lightx2v，其余 unsupported）。
不做：真实启动/耗时（与 dummy-run 的区别见 references/manifest-schema.md）。

用法：
    python manifest_dryrun.py --manifest runs_plan.toml
    python manifest_dryrun.py --manifest runs_plan.toml --framework lightx2v
退出码：0 = 通过（可有 seam warning）；1 = 存在 error。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    sys.exit("需要 python >= 3.11（tomllib 标准库）")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seam_check  # noqa: E402

VALID_KIND = {"optimization", "smoke", "control"}
VALID_PURPOSE = {"closed_loop", "frontier", "evidence", "blocker_probe"}
KNOWN_FRAMEWORKS = {"vllm-omni", "lightx2v"}

RENDER_HINTS = {
    "vllm-omni": "vllm-omni 推理入口 + FA/稀疏/缓存 env（姿势见 framework cases）",
    "lightx2v": "LightX2V 入口 + use_compile/seq_p_a2a_backend env（见 lightx2v case）",
}


def render_plan(cfg: dict) -> str:
    run = cfg.get("run", {})
    feats = cfg.get("features", {}).get("enable", [])
    env = cfg.get("env", {})
    meta = cfg.get("meta", {})
    renderer = RENDER_HINTS.get(meta.get("framework"), "unsupported-framework")
    lines = [
        f"run plan: model={meta.get('model')} framework={meta.get('framework')}",
        f"topology={run.get('topology')} gpus={run.get('gpus')} "
        f"baseline={run.get('baseline_run')}",
        f"features={feats}",
        f"renderer: {renderer}",
    ]
    env_str = ", ".join(f"{k}={v}" for k, v in sorted(env.items()))
    lines.append(f"resolved env: {env_str or '(无)'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.toml 路径")
    parser.add_argument("--decl", default=str(seam_check.DECL_PATH), help="特性声明文件")
    args = parser.parse_args(argv)

    errors: list[str] = []
    try:
        cfg = tomllib.loads(args.manifest.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] manifest 解析失败：{exc}")
        return 1

    meta = cfg.get("meta", {})
    kind, purpose = meta.get("kind"), meta.get("purpose")
    fw = meta.get("framework")
    if kind not in VALID_KIND:
        errors.append(f"kind 非法：{kind}（可选 {sorted(VALID_KIND)}）")
    if purpose not in VALID_PURPOSE:
        errors.append(f"purpose 非法：{purpose}（可选 {sorted(VALID_PURPOSE)}）")
    if fw not in KNOWN_FRAMEWORKS:
        errors.append(
            f"framework 渲染器未覆盖：{fw}（当前 {sorted(KNOWN_FRAMEWORKS)}，"
            "不做静默猜测）"
        )

    feats = cfg.get("features", {}).get("enable", [])
    if not isinstance(feats, list):
        errors.append("features.enable 非法（应为数组；基线档填 enable=[]）")
    elif not feats:
        is_baseline = bool(cfg.get("run", {}).get("baseline_run", False))
        if is_baseline:
            print("[info] features.enable=[] 且 baseline_run=true：按基线档处理（无 seam 可查，跳过）")
        else:
            errors.append(
                "features.enable 为空且非基线档（非基线档必须声明 enable 列表；"
                "基线档请 baseline_run=true + enable=[]）"
            )
    else:
        try:
            decl = seam_check.load_declarations(Path(args.decl))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"声明加载失败：{exc}")
            decl = {
                "features": [],
                "exclusive_seams": [],
                "window_conflict_warning_seams": [],
                "models": {},
            }
        e2, w2 = seam_check.check(feats, meta.get("model_adapter"), decl)
        errors += e2
        for w in w2:
            print(f"[warn] {w}")

    # 路径只读检查（manifest 内 file/ 字段，相对 manifest 目录；不产生数据）
    for key in ("profile_dir", "output_media"):
        p = cfg.get("artifacts", {}).get(key)
        if p and not str(p).startswith("<"):
            target = args.manifest.parent / p
            if not target.exists():
                errors.append(f"artifacts.{key} 不存在（只读检查）：{target}")

    for e in errors:
        print(f"[error] {e}")
    print(render_plan(cfg))
    print(f"结论：error={len(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
