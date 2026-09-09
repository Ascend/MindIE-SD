#!/usr/bin/env python3
"""帧级定量质量对照：baseline 帧目录 vs config 帧目录（有损优化端到端质量门禁）。

指标：
- psnr / ssim : 纯 numpy 实现，无外部权重依赖（默认，CPU 可跑；依赖 numpy + Pillow）
- lpips       : 可选。需要 torch + lpips 包；AlexNet 权重首次使用会联网下载，
                无网络或未安装时该指标自动跳过并提示

用法：
    python evals/scripts/quality_compare.py --baseline <dir> --config <dir> \
        --metric ssim --metric psnr --threshold ssim=0.95 --output quality.json

约定（见 evals/profiles/README.md）：
- 两个目录须分辨率一致、帧索引对齐（同名文件优先按名配对，否则按排序索引配对）
- --threshold 支持 fail-closed：任一指标低于阈值 → exit 1（供后续 CI/验收门禁复用）

输出：JSON（各指标 mean/median/min + per_frame），退出码 0/1。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

# ---------------------------------------------------------------------------
# 图像读取：优先 Pillow；缺失时报错并提示（NPU 推理环境一般自带）
# ---------------------------------------------------------------------------
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def load_image(path: Path) -> Image.Image:
    if Image is None:
        sys.exit("需要 Pillow：pip install Pillow（或在已装 diffusers/torchvision 的环境运行）")
    with Image.open(path) as im:
        return im.convert("RGB")


def to_luma_gray(np_img: np.ndarray) -> np.ndarray:
    """RGB -> 亮度灰度（Rec.601），返回 float64 0..1 二维数组。"""
    r = np_img[..., 0].astype(np.float64)
    g = np_img[..., 1].astype(np.float64)
    b = np_img[..., 2].astype(np.float64)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


# ---------------------------------------------------------------------------
# 指标实现
# ---------------------------------------------------------------------------
def _psnr_pair(ref: np.ndarray, tgt: np.ndarray) -> float:
    mse = float(np.mean((ref.astype(np.float64) - tgt.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(255.0 * 255.0 / mse))


def _ssim_map(ref_gray: np.ndarray, tgt_gray: np.ndarray, win_size: int = 11) -> np.ndarray:
    """滑动均值窗 SSIM（灰度），返回逐像素 ssim 图。"""
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    x = np.pad(ref_gray * 255.0, win_size // 2, mode="reflect")
    y = np.pad(tgt_gray * 255.0, win_size // 2, mode="reflect")

    def local_sum(a: np.ndarray) -> np.ndarray:
        # 滑动窗口和：对 padded 图做 win x win 均值滤波（stride_tricks 实现，确定性）
        s = as_strided(
            a,
            shape=(a.shape[0] - win_size + 1, a.shape[1] - win_size + 1, win_size, win_size),
            strides=a.strides + a.strides,
        )
        return s.sum(axis=(2, 3))

    mu_x, mu_y = local_sum(x), local_sum(y)
    n = float(win_size * win_size)
    mu_x, mu_y = mu_x / n, mu_y / n
    sigma_xx = local_sum(x * x) / n - mu_x * mu_x
    sigma_yy = local_sum(y * y) / n - mu_y * mu_y
    sigma_xy = local_sum(x * y) / n - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_xx + sigma_yy + c2)
    )
    return ssim


def _ssim_pair(img_a: np.ndarray, img_b: np.ndarray) -> float:
    ga, gb = to_luma_gray(img_a), to_luma_gray(img_b)
    return float(np.mean(_ssim_map(ga, gb)))


def _lpips_pair(img_a: Path, img_b: Path) -> float:
    """LPIPS（可选依赖）；首次运行需联网下载 AlexNet 权重。"""
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lpips 指标需要 pip install lpips torch（权重首次联网下载）") from exc

    loss_fn = lpips.LPIPS(net="alex")  # noqa: S301 —— 官方包，权重由包管理
    a = lpips.im2tensor(lpips.load_image(str(img_a)))  # type: ignore[attr-defined]
    b = lpips.im2tensor(lpips.load_image(str(img_b)))  # type: ignore[attr-defined]
    with torch.no_grad():
        return float(loss_fn(a, b).item())


# ---------------------------------------------------------------------------
# 帧配对
# ---------------------------------------------------------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def pair_frames(base_dir: Path, cfg_dir: Path) -> list[tuple[Path, Path]]:
    def collect(d: Path) -> dict[str, Path]:
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            sys.exit(f"目录无图片帧（png/jpg/webp/bmp）：{d}")
        return {p.name: p for p in files}

    base_map, cfg_map = collect(base_dir), collect(cfg_dir)
    common = sorted(set(base_map) & set(cfg_map))
    if len(common) == len(base_map) == len(cfg_map):
        return [(base_map[name], cfg_map[name]) for name in common]
    # 同名不全：按排序索引配对（须长度一致）
    if len(base_map) != len(cfg_map):
        sys.exit(
            f"帧数不一致且同名不齐：baseline={len(base_map)} config={len(cfg_map)}；"
            "请按 profiles/README.md 对齐帧（同分辨率/同帧数/同帧索引）"
        )
    return list(zip(sorted(base_map.values()), sorted(cfg_map.values())))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="baseline 帧目录")
    parser.add_argument("--config", required=True, type=Path, help="config（优化后）帧目录")
    parser.add_argument(
        "--metric",
        action="append",
        default=None,
        choices=["psnr", "ssim", "lpips"],
        help="指标（可多次）；默认 psnr,ssim",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=None,
        metavar="metric=value",
        help="fail-closed 阈值，如 --threshold ssim=0.95；任一低于阈值 exit 1",
    )
    parser.add_argument("--output", type=Path, help="写 JSON 报告路径")
    args = parser.parse_args(argv)

    metrics = args.metric or ["psnr", "ssim"]
    thresholds: dict[str, float] = {}
    for item in args.threshold or []:
        name, _, raw = item.partition("=")
        try:
            thresholds[name] = float(raw)
        except ValueError:
            parser.error(f"--threshold 需为 metric=value 形式：{item}")

    for d in (args.baseline, args.config):
        if not d.is_dir():
            sys.exit(f"目录不存在：{d}")

    pairs = pair_frames(args.baseline, args.config)
    results: dict[str, dict] = {}
    lpips_available = True

    for metric in metrics:
        per_frame: list[float] = []
        for bp, cp in pairs:
            img_a, img_b = load_image(bp), load_image(cp)
            if img_a.size != img_b.size:
                sys.exit(
                    f"分辨率不一致（{bp.name}: {img_a.size} vs {cp.name}: {img_b.size}）；"
                    "质量对照要求同分辨率（见 evals/profiles/README.md）"
                )
            if metric == "psnr":
                value = _psnr_pair(np.asarray(img_a), np.asarray(img_b))
            elif metric == "ssim":
                value = _ssim_pair(np.asarray(img_a), np.asarray(img_b))
            else:  # lpips
                if not lpips_available:
                    per_frame.append(float("nan"))
                    continue
                try:
                    value = _lpips_pair(bp, cp)
                except (RuntimeError, ImportError) as exc:
                    print(f"[warn] lpips 跳过（{exc}）", file=sys.stderr)
                    lpips_available = False
                    per_frame.append(float("nan"))
                    continue
            per_frame.append(value)

        finite = [v for v in per_frame if math.isfinite(v)]
        if not finite:
            results[metric] = {"skipped": True, "reason": "指标不可用（lpips 依赖/网络缺失）"}
            continue
        arr = np.asarray(finite)
        results[metric] = {
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "min": round(float(arr.min()), 6),
            "n_valid": int(len(finite)),
            "n_total": int(len(per_frame)),
            "per_frame": [round(v, 6) if math.isfinite(v) else None for v in per_frame],
        }

    # fail-closed 判定
    failures = []
    for metric, value in thresholds.items():
        if metric not in results or results[metric].get("skipped"):
            failures.append(f"{metric}: 无数据，无法通过阈值")
            continue
        got = results[metric]["median"]
        if got < value:
            failures.append(f"{metric}: median {got} < 阈值 {value}")

    report = {
        "baseline_dir": str(args.baseline),
        "config_dir": str(args.config),
        "n_pairs": len(pairs),
        "metrics": results,
        "threshold_result": "pass" if not failures else "fail",
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
