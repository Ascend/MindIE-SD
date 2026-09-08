#!/usr/bin/env python
"""三方框架标准 CANN profiler 采集补丁模板。

用法（以 LightX2V MiniMax-H3 为例）：
    1) 把本文件放到远端 /home/<user>/collect_patch.py
    2) 按框架把 _ORIG_ATTR 指向顶层推理方法（类名.方法名）
    3) torchrun 启动时把它作为入口脚本（框架 CLI 参数原样透传）

环境变量：
    H3_WARMUP_STEPS   profiler 外 warmup 步数（默认 5；compile/首次 JIT 场景建议 >=10）
                      —— 少步快速采集：设 1 = 1 步预热 + 采第 2 步 1 步（eager / 图已编译稳定
                      的重复采集；单步即代表算子形态与 kernel 序，kernel diff 用 1 步数据足够）。
                      ⚠️ 少 step 是 profiling 手段非优化目标：只改采集配置、采完还原；
                      compile/首次 JIT 必须预热覆盖编译（>=10）后再采，勿用 1 步预热。
    H3_CANN_PROF_OUT  CANN 输出目录（默认 <model_dir>/h3_cann_prof）

产出：<PROF_OUT>/localhost.localdomain_*_ascend_pt/ASCEND_PROFILER_OUTPUT/
      kernel_details.csv + trace_view.json + step_trace_time.csv
      （performance-analysis 的 analyze_trace.py / compare_traces.py 直接消费）
"""
import os

os.environ.setdefault("PLATFORM", "ascend_npu")
os.environ.setdefault("DTYPE", "BF16")
os.environ.setdefault("SENSITIVE_LAYER_DTYPE", "BF16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")

PROF_OUT = os.environ.get("H3_CANN_PROF_OUT", "<model_dir>/h3_cann_prof")
WARMUP = int(os.environ.get("H3_WARMUP_STEPS", "5"))

import torch
import torch.distributed as dist
import torch_npu

# ── 框架适配区 ────────────────────────────────────────────────
# 把下面两行替换为实际框架的顶层推理类与方法：
#   以 LightX2V 为例：
#   from lightx2v.models.networks.minimax_h3.infer.transformer_infer import (
#       MiniMaxH3TransformerInfer,
#   )
#   FRAMEWORK_CLASS = MiniMaxH3TransformerInfer
#   METHOD = "infer"
FRAMEWORK_CLASS = None  # 替换：框架推理类
METHOD = "infer"        # 替换：顶层推理方法名
# ──────────────────────────────────────────────────────────────

os.makedirs(PROF_OUT, exist_ok=True)

_state = {"step": 0}


def _make_patch():
    _orig = getattr(FRAMEWORK_CLASS, METHOD)

    def _profiled(self, *args, **kwargs):
        _state["step"] += 1
        n = _state["step"]
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0 and n == WARMUP + 1:
            handler = torch_npu.profiler.tensorboard_trace_handler(PROF_OUT)
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                profile_memory=False,
                on_trace_ready=handler,
            ) as prof:
                out = _orig(self, *args, **kwargs)
                torch.npu.synchronize()
            print(f"[CANN] profiled step {n}; output in {PROF_OUT}", flush=True)
            return out
        return _orig(self, *args, **kwargs)

    setattr(FRAMEWORK_CLASS, METHOD, _profiled)


def main():
    _make_patch()
    # 框架 CLI 入口，参数原样透传（以 LightX2V 为例）：
    from lightx2v.infer import main as cli_main  # noqa: E402  # 替换为框架入口

    cli_main()


if __name__ == "__main__":
    main()