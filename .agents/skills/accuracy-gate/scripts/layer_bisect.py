#!/usr/bin/env python3
"""Per-layer CPU-vs-NPU divergence finder (skeleton).

Why this shape: choosing *which* layer first diverges is the single most valuable step in a silent
wrong-output hunt -- it turns "the whole model is wrong" into "this one op, at this position".  The
trick that makes it cheap is comparing the same module run twice, on the same input, on two devices,
recording every submodule's output with hooks and printing two *relative* measures:

  * rel  = mean|NPU - CPU| / mean|CPU|          -> grows when a layer starts to diverge
  * amax = max|NPU| / max|CPU|                  -> ~1.000 means "same scale", far from 1 means
                                                  the layer is writing something else entirely

Adaptation points are marked ADAPT: instantiate your model, load the SAME weights twice (one per
device), and feed the SAME input tensor.  Keep fp32 on both sides unless dtype is the variable under
test -- mixing dtype into this step destroys attribution.

Usage:
  python layer_bisect.py --split 0.75        # report head (0..split) and tail (split..1) separately

Watch for the *segmented* signature: a layer whose head matches to ~1e-5 while its tail differs by
O(0.1--1) is almost always a single device call that failed to write part of its output (see
references/silent-failure-localization.md §11.1), not a precision problem.
"""

from __future__ import annotations

import argparse
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover
    sys.exit(f"需要 torch ({exc})")


def collect_outputs(model: torch.nn.Module, x: torch.Tensor) -> dict:
    """Run `model(x)` with forward hooks capturing every submodule output."""
    out: dict[str, torch.Tensor] = {}

    def hook(name):
        def _f(_m, _i, o):
            if isinstance(o, torch.Tensor):
                out[name] = o.detach()

        return _f

    handles = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]
    try:
        with torch.inference_mode():
            model(x)
    finally:
        for h in handles:
            h.remove()
    return out


def compare(cpu: dict, npu: dict, split: float, dim_for_time: int = -1) -> None:
    print(f"   {'module':<46} {'rel(head)':>10} {'rel(tail)':>10} {'amax_ratio':>10}")
    first_bad = None
    for name, c_raw in cpu.items():
        if name not in npu:
            continue
        c = c_raw.float().cpu()
        n = npu[name].float().cpu()
        if c.shape != n.shape:
            print(f"   {name:<46}  形状不一致: {tuple(c.shape)} vs {tuple(n.shape)}")
            continue
        flat_c = c.reshape(c.shape[0], -1) if c.dim() > 1 else c.reshape(1, -1)
        flat_n = n.reshape(n.shape[0], -1) if n.dim() > 1 else n.reshape(1, -1)
        cut = max(1, int(flat_c.shape[0] * split))

        def _rel(a, b):
            base = float(a.abs().mean()) or 1.0
            return float((a - b).abs().mean()) / base

        rel_h = _rel(flat_c[:cut], flat_n[:cut])
        rel_t = _rel(flat_c[cut:], flat_n[cut:])
        amax = float(flat_n.abs().max()) / (float(flat_c.abs().max()) or 1.0)
        flag = ""
        if rel_t > 20 * max(rel_h, 1e-6) and rel_t > 1e-2:
            flag = "  <== 分段式分叉（头段一致、尾段坏）"
            first_bad = first_bad or name
        elif rel_t > 1e-2 or rel_h > 1e-2:
            flag = "  <== 分叉"
            first_bad = first_bad or name
        print(f"   {name[:46]:<46} {rel_h:10.2e} {rel_t:10.2e} {amax:10.3f}{flag}")
    print("\n首个分叉模块: %s" % (first_bad or "未发现（两者一致）"))
    if first_bad:
        print(
            "下一步：把上一层输出单独喂给该模块（算子隔离），并做 batch/shape 阈值扫描\n"
            "        （scripts/op_threshold_sweep.py 给出可直接改用的骨架）。"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=float, default=0.75, help="头/尾切分比例（按第 1 维）")
    a = ap.parse_args()

    # ADAPT: ---------------------------------------------------------------
    # 1) 取同一份权重（例如从 checkpoint 加载两次，或 deepcopy 后各自 .to(device)）
    # 2) 取同一份真实输入样本（优先真实数据 dump，而不是随机张量）
    # 3) 两端都用 fp32；不要在这一步混入 dtype 变量
    # model_cpu = build().eval().to("cpu").float()
    # model_npu = build().eval().to("npu").float()
    # model_npu.load_state_dict(model_cpu.state_dict())
    # x_cpu = sample().float().cpu()
    # x_npu = x_cpu.to("npu")
    # ----------------------------------------------------------------------
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("当前环境没有可用 NPU（torch.npu 不可用）。请在本文件 ADAPT 段落接入你的模型后，")
        print("在装有 torch_npu 的机器上运行；CPU 真值可以在任何机器上产生。")
        return 0

    print("请先完成脚本 ADAPT 段落的接线（模型构造 + 权重同步 + 真实输入样本）。")
    print("接线后的调用方式：")
    print("    cpu_out = collect_outputs(model_cpu, x_cpu)")
    print("    npu_out = collect_outputs(model_npu, x_npu)")
    print(f"    compare(cpu_out, npu_out, split={a.split:.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
