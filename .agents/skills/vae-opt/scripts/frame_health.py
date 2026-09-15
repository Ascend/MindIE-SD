#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闸③：帧级健康指标 —— 定位“从第几帧开始坏”，并给出合格判据。

指标（默认每 24 帧一桶，按帧率可调）
  * hf       高频能量 = (mean|Δx| + mean|Δy|)/2（0-255 单位）。后半段**塌陷**=画面糊成一片。
  * d_prev   帧间差 = mean|frame_t - frame_{t-1}|。频闪=反复尖峰；冻结=长串接近 0。
  * 冻结段   d_prev < 0.5 的连续段计数（健康解码应为 0 段）。

用法
  python frame_health.py --input out.mp4                # 看成品
  python frame_health.py --input latent_decoded.pt      # 看原始解码张量（不走 h264）
  python frame_health.py --selftest                     # 自证：合成的“尾段塌陷”必须被标出
  python frame_health.py --input a.mp4 --input b.mp4 --bucket 24

合格判据（写进交付物时照抄）
  1) 各桶 hf 无“后半段断崖”：末桶 hf ≥ 首桶 hf × 0.5；
  2) 冻结段 0 段；d_prev 无 > 首桶 10 倍的孤立尖峰；
  3) 与同 latent 的对照解码（真值）比桶曲线形状一致。

注意（数字纪律）：**单看绝对值无法判断好坏**——必须把缺陷样本与**已知良好的参照**一起跑、比量级。
健康样本的量级随分辨率/内容/编码器变化，所以本文件不带任何“应该等于多少”的读数；
若把某次观测的数字写进交付物，请连同口径（机器、形状、窗口、权重）一起写，并按证据卡标注失效信号。
"""
from __future__ import annotations

import argparse
import glob
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass


# ------------------------------------------------------------------ loading
def _to_gray_frames(arr: np.ndarray) -> np.ndarray:
    """归一化成 (T, H, W) float 0-255。接受 (N,C,T,H,W)/(T,H,W,C)/(T,C,H,W)/(N,T,H,W)。"""
    a = np.asarray(arr)
    if a.ndim == 5:                      # (N,C,T,H,W) 或 (N,T,C,H,W)
        a = a[0]
    if a.ndim == 4:
        # (C,T,H,W) / (T,C,H,W) / (T,H,W,C)
        if a.shape[0] <= 4 and a.shape[1] > 4:          # (C,T,H,W)
            a = np.transpose(a, (1, 2, 3, 0))           # -> (T,H,W,C)
        elif a.shape[1] <= 4 and a.shape[0] > 4:        # (T,C,H,W)
            a = np.transpose(a, (0, 2, 3, 1))           # -> (T,H,W,C)
        if a.ndim == 4:                                 # (T,H,W,C)
            a = a[..., :3].mean(-1)
    if a.ndim != 3:
        raise ValueError("无法识别的张量形状: %s" % (a.shape,))
    a = a.astype(np.float32, copy=False)
    if a.max() <= 1.5:                                  # 浮点 0-1
        a = a * 255.0
    return a


def load_input(path: str) -> np.ndarray:
    low = path.lower()
    if low.endswith((".pt", ".pth")):
        import torch
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            obj = next(v for v in obj.values() if hasattr(v, "shape"))
        return _to_gray_frames(obj.detach().cpu().numpy() if hasattr(obj, "detach") else obj)
    if low.endswith(".npy"):
        return _to_gray_frames(np.load(path))
    if low.endswith((".mp4", ".mov", ".mkv")):
        try:
            import av
        except ModuleNotFoundError:
            raise SystemExit("读 mp4 需要 PyAV（pip install av）；或先落成 .npy/.pt 再看")
        frames = []
        with av.open(path) as c:
            for f in c.decode(video=0):
                frames.append(f.to_ndarray(format="rgb24").astype(np.float32).mean(-1))
        return np.stack(frames)
    raise SystemExit("不支持的输入类型: %s（支持 .mp4/.npy/.pt）" % path)


# ------------------------------------------------------------------ metrics
def hf(a: np.ndarray) -> float:
    return float((np.abs(a[:, 1:] - a[:, :-1]).mean() + np.abs(a[1:, :] - a[:-1, :]).mean()) / 2.0)


def analyze(frames: np.ndarray, bucket: int) -> dict:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim == 4 and frames.shape[1] == 1:      # (T,1,H,W) -> (T,H,W)
        frames = frames[:, 0]
    if frames.ndim == 4:                               # (T,H,W,C) -> 灰度
        frames = frames[..., :3].mean(-1)
    if frames.ndim != 3:
        raise ValueError("analyze 需要 (T,H,W)，得到 %s" % (frames.shape,))
    t = frames.shape[0]
    hfs, dps = [], []
    prev = None
    for i in range(t):
        hfs.append(hf(frames[i]))
        dps.append(float(np.abs(frames[i] - prev).mean()) if prev is not None else 0.0)
        prev = frames[i]
    hfs = np.asarray(hfs)
    dps = np.asarray(dps)
    nb = max(1, t // bucket)
    buckets = []
    for b in range(nb):
        lo, hi = b * bucket, min((b + 1) * bucket, t)
        buckets.append((b, lo, hi, float(hfs[lo:hi].mean()), float(hfs[lo:hi].max()), float(dps[lo + 1 : hi + 1].mean()) if hi - lo > 1 else 0.0))
    # 冻结段（d_prev < 0.5）
    runs, cur = 0, 0
    for d in dps[1:]:
        if d < 0.5:
            cur += 1
        else:
            runs += 1 if cur >= 4 else 0
            cur = 0
    runs += 1 if cur >= 4 else 0
    first_bad = None
    if hfs[0] > 0:
        idx = np.where(hfs < 0.5 * hfs[:bucket].mean())[0]
        idx = idx[idx > bucket]                      # 忽略首桶自身
        if len(idx):
            first_bad = int(idx[0])
    # 帧间差尖峰：区分「孤立尖峰（内容场景切换，各解码器都会有）」与「反复尖峰（频闪/坏帧）」
    med = float(np.median(dps[1:])) if t > 1 else 0.0
    thr = max(20.0, 5.0 * med)
    spike_idx = [int(i) for i in np.where(dps > thr)[0]]
    return {"t": t, "buckets": buckets, "frozen_runs": runs,
            "hf_first": float(hfs[:bucket].mean()), "hf_last": float(hfs[-bucket:].mean()),
            "first_collapse_frame": first_bad, "d_prev_max": float(dps[1:].max() if t > 1 else 0),
            "d_prev_med": med, "spike_frames": spike_idx, "spike_thr": thr}


def report(path: str, res: dict, bucket: int, fps: float) -> bool:
    print("== %s  (帧数=%d, 每桶 %d 帧)" % (path, res["t"], bucket))
    print("   %-14s %8s %8s %9s" % ("桶(帧区间)", "秒", "hf均值", "d_prev"))
    for b, lo, hi, hm, hx, dm in res["buckets"]:
        print("   %-14s %8.2f %8.3f %9.3f" % ("%d-%d" % (lo, hi - 1), lo / fps, hm, dm))
    ok_tail = res["hf_last"] >= 0.5 * res["hf_first"]
    ok_frozen = res["frozen_runs"] == 0
    n_spike = len(res["spike_frames"])
    ok_jump = n_spike <= max(2, int(0.05 * res["t"]))        # 允许 1-2 帧的孤立内容切换
    verdict = []
    if not ok_tail:
        verdict.append("末桶 hf(%.2f) < 首桶(%.2f) 的 50%% -> 尾段塌陷" % (res["hf_last"], res["hf_first"]))
    if not ok_frozen:
        verdict.append("冻结段 %d 段（d_prev<0.5 连续≥4 帧）" % res["frozen_runs"])
    if not ok_jump:
        verdict.append("帧间差反复尖峰 %d 帧（阈值 %.1f，中位 %.2f）-> 疑似频闪/坏帧" % (n_spike, res["spike_thr"], res["d_prev_med"]))
    if res["first_collapse_frame"] is not None:
        verdict.append("首个塌陷帧 = %d（%.2f s）" % (res["first_collapse_frame"], res["first_collapse_frame"] / fps))
    print("   首桶 hf=%.3f 末桶 hf=%.3f 冻结段=%d 帧间差 中位=%.2f 峰值=%.2f 超阈帧数=%d%s" % (
        res["hf_first"], res["hf_last"], res["frozen_runs"], res["d_prev_med"], res["d_prev_max"], n_spike,
        "（孤立尖峰@帧 %s，内容切换则正常）" % res["spike_frames"][:3] if 0 < n_spike <= 3 else ""))
    print("   判定: %s" % ("健康 [PASS]" if not verdict else "异常 [FAIL] —— " + "；".join(verdict)))
    print()
    return not verdict


# ------------------------------------------------------------------ selftest
def _synth(t=72, h=96, w=160, bad_from=48, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, w)[None, :] + np.linspace(0, 255, h)[:, None]   # (h, w)
    tex = rng.normal(0, 9, size=(h, w))                       # 细节纹理（hf 由它决定）
    frames = []
    for i in range(t):
        f = base + 20 * np.sin(2 * np.pi * (i / 12.0)) + tex + rng.normal(0, 2, (h, w))
        if i >= bad_from:
            f = base * 0 + base.mean() + rng.normal(0, 0.2, (h, w))   # 塌陷：近乎平坦
        frames.append(np.clip(f, 0, 255))
    return np.stack(frames)


def main() -> int:
    ap = argparse.ArgumentParser(description="帧级健康指标（hf / 帧间差 / 冻结段 / 塌陷起点）")
    ap.add_argument("--input", action="append", default=[], help="可重复：.mp4 / .npy / .pt")
    ap.add_argument("--bucket", type=int, default=24, help="桶大小（帧），默认 24")
    ap.add_argument("--fps", type=float, default=24.0, help="用于秒数换算")
    ap.add_argument("--selftest", action="store_true", help="用合成序列自证指标有效")
    args = ap.parse_args()

    ok_all = True
    if args.selftest:
        good = _synth(bad_from=10**9)
        bad = _synth(bad_from=48)
        print("---- 自证：健康序列（应 PASS）----")
        ok_all &= report("<synthetic-healthy>", analyze(good, args.bucket), args.bucket, args.fps)
        print("---- 自证：尾段塌陷序列（应 FAIL 且指出塌陷帧 ~48）----")
        ok_all &= not report("<synthetic-collapsed>", analyze(bad, args.bucket), args.bucket, args.fps)

    for pat in args.input:
        for p in sorted(glob.glob(pat)) or [pat]:
            ok_all &= report(p, analyze(load_input(p), args.bucket), args.bucket, args.fps)

    if not args.input and not args.selftest:
        ap.print_help()
        return 2
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
