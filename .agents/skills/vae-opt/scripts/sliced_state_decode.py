#!/usr/bin/env python3
"""状态携带切分骨架：把带记忆块（MemBlock/past）的递归解码器按潜帧边界切成若干段，
逐段顺序解码并在段间搬运“每个记忆块输入处的末帧激活”，使结果与整段解码逐位等价。

为什么要这样做（详见 references/sliced-state-decode.md）：
  * 记忆块的 past 取的是“上一帧在该块 **输入** 处”的激活，逐层递推 → 第 t 帧依赖整段前缀
    → 时间轴**不能**等分；但只有记忆块跨帧，其余层逐帧独立 → 把状态显式搬过边界即可精确。
  * 顺带的收益：每次算子调用的 batch 变小 → 峰值显存下降，并可绕开“大 batch 设备缺陷”
    （实测 CANN nn.Upsample(nearest) 在 batch > ~165 时少写尾部）。

两种用法
  1) 直接抄 `apply_sliced()`：把 `is_mem_block` / `post_process` 两个 ADAPTER 换成你自己的实现；
  2) `python sliced_state_decode.py --demo`：用内置小网络自证“整段 vs 分段”逐位一致。

ADAPTER（**必须由使用者替换，本文件里是示例/伪代码级的占位**）：
  * `is_mem_block(block) -> bool`：判断某层是否是记忆块。真实实现通常按类名/属性判定，
    例如 `isinstance(b, MemBlock)` 或 `type(b).__name__ == "_MemBlock"`。
  * `post_process(flat, n, t) -> Tensor`：整段实现里“所有块之后”的收尾（如 pixel_shuffle、
    reshape 回 (N,T,C,H,W)）。不同网络的收尾不同，故留成钩子而非常量逻辑。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- core
def _slice_frames(
    model: nn.Sequential, x: torch.Tensor, post_process: Callable, is_mem_block: Callable, state: dict, keep_state: bool
):
    """解码一段帧序列，返回 (输出, 新状态)。

    x: (N, T, C, H, W)。
    state[i] = 上一段**末帧在 MemBlock i 输入处**的激活，shape (N, 1, C, H, W)。
    首段 state 为空 → 用 zeros pad 复现整段语义（mem[0] = 0）。
    """
    n, t, c, h, w = x.shape
    flat = x.reshape(n * t, c, h, w)
    nxt: dict = {}
    for idx, block in enumerate(model):
        if is_mem_block(block):
            bt, cc, hh, ww = flat.shape
            tt = bt // n
            view = flat.reshape(n, tt, cc, hh, ww)
            prev = state.get(idx)
            if prev is None:
                mem = F.pad(view, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :tt]
            else:
                mem = torch.cat([prev.to(view.dtype), view[:, : tt - 1]], dim=1)
            if keep_state:
                nxt[idx] = view[:, -1:].detach().clone()  # 该块 **输入** 的末帧
            flat = block(flat, mem.reshape(flat.shape))
        else:
            flat = block(flat)
    return post_process(flat, n, n * 0 + (flat.shape[0] // n)), nxt


def apply_whole(model: nn.Sequential, x: torch.Tensor, post_process: Callable, is_mem_block: Callable) -> torch.Tensor:
    """整段解码（对照基准，与线上整段实现同语义）。"""
    out, _ = _slice_frames(model, x, post_process, is_mem_block, {}, keep_state=False)
    return out


def apply_sliced(
    model: nn.Sequential, x: torch.Tensor, post_process: Callable, is_mem_block: Callable, slices: int
) -> torch.Tensor:
    """按潜帧边界切 `slices` 段顺序解码，段间携带状态。

    slices < 2（或帧数不够）时自动退化为整段 —— 退化分支**只依赖 env 与 shape**，
    不依赖 rank，避免“部分 rank 进入集体通信”导致的死锁。
    """
    n_t = int(x.shape[1])
    if slices < 2 or n_t < 2 * slices:
        return apply_whole(model, x, post_process, is_mem_block)
    bounds = [(i * n_t) // slices for i in range(slices + 1)]
    state: dict = {}
    outs = []
    for i in range(slices):
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        part, state = _slice_frames(model, x[:, lo:hi], post_process, is_mem_block, state, keep_state=True)
        outs.append(part)
    return torch.cat(outs, dim=1)


def slices_from_env(env_name: str = "DECODER_SHARD_SLICES", default: int = 2) -> int:
    """段数只能由 env 推导（全 rank 一致），**不要**用 rank 决定是否分段。"""
    raw = os.environ.get(env_name, str(default)).strip()
    try:
        return max(0, int(raw or default))
    except ValueError:
        return default


# ----------------------------------------------------------------- demo (self-check)
class _MemBlock(nn.Module):
    """cat([x, past]) -> 3 convs + skip（与实测真实网络同构）。"""

    def __init__(self, n_in: int, n_out: int, act: nn.Module) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_in * 2, n_out, 3, padding=1),
            act,
            nn.Conv2d(n_out, n_out, 3, padding=1),
            act,
            nn.Conv2d(n_out, n_out, 3, padding=1),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    """时间上采样：1x1 conv 把 stride 展开到时间轴（逐帧映射、不跨帧）。"""

    def __init__(self, n_f: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _nt, c, h, w = x.shape
        return self.conv(x).reshape(-1, c, h, w)


def _demo():
    torch.manual_seed(0)
    act = nn.ReLU(inplace=True)
    model = nn.Sequential(
        nn.Conv2d(4, 16, 3, padding=1),
        act,
        _MemBlock(16, 16, act),
        _MemBlock(16, 16, act),
        nn.Upsample(scale_factor=2, mode="nearest"),
        _TGrow(16, 2),
        nn.Conv2d(16, 8, 3, padding=1),
        _MemBlock(8, 8, act),
        nn.Conv2d(8, 3 * 2 * 2, 3, padding=1),
    ).eval()

    def post(flat: torch.Tensor, n: int, t: int) -> torch.Tensor:
        flat = F.pixel_shuffle(flat, 2)  # 空间 x2（逐帧）
        bt, c, h, w = flat.shape
        return flat.view(n, bt // n, c, h, w)

    def is_mem_block(b) -> bool:
        return isinstance(b, _MemBlock)

    x = torch.randn(1, 12, 4, 8, 8)
    with torch.no_grad():
        ref = apply_whole(model, x, post, is_mem_block)
        print("整段输出:", tuple(ref.shape))
        for pieces in (2, 3, 4, 6):
            got = apply_sliced(model, x, post, is_mem_block, pieces)
            d = (got - ref).abs().max().item()
            print(f"  {pieces} 段: max|d| = {d:.3e}  {'bitwise 等价' if d == 0 else '**不等价**'}")
        # 半帧切分（错误示范）：段数与潜帧数不整除时仍应保持一致（这里用不整除的 5 段验证）
        got = apply_sliced(model, x, post, is_mem_block, 5)
        print(f"  5 段(不整除): max|d| = {(got - ref).abs().max().item():.3e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="状态携带切分骨架 / 自证")
    ap.add_argument("--demo", action="store_true", help="用内置小网络自证整段 vs 分段逐位一致")
    args = ap.parse_args()
    if args.demo:
        _demo()
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
