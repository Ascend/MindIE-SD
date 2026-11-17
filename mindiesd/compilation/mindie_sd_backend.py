#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import inspect
import os
from collections.abc import Callable, Sequence

import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.freezing import freeze

from ._custom_decomposition import select_custom_decomp_table

try:
    from torch.fx.passes.graph_transform_observer import GraphTransformObserver
except ImportError:

    class GraphTransformObserver:
        def __init__(self, gm, passname, subsystem=None, log_url=None):
            self.gm = gm
            self.passname = passname
            self.subsystem = subsystem
            self.log_url = log_url

        def apply_gm_pass(self, pass_obj):
            pass_obj(self.gm)

        def apply_graph_pass(self, pass_func):
            pass_func(self.gm.graph)


from ..utils.logs.logging import logger
from .aclgraph_backend import (
    create_aclgraph_backend,
    create_segmented_aclgraph_backend,
    npu_graph_available,
)
from .compiliation_config import BackendMode, CompilationConfig
from .npugraph_ex_backend import create_npugraph_ex_backend, npugraph_ex_available
from .passes import activate_pattern_once
from .passes.redundant_node_elimination_pass import ReduandantNodeEliminationPass
from .passes.register_pattern_to_pass import patterns

DEBUG_LOG_LEVEL = 10


def decompose_auto_functionalized(graph: fx.Graph):
    try:
        from torch._inductor.fx_passes.post_grad import (
            decompose_auto_functionalized as original_decompose,
        )

        return original_decompose(graph)
    except ImportError:
        for node in list(graph.nodes):
            if node.op == 'call_function' and 'auto_functionalized' in str(node.target):
                orig_name = node.target.__name__.replace('_auto_functionalized', '')
                if hasattr(torch, orig_name):
                    node.target = getattr(torch, orig_name)
        graph.eliminate_dead_code()
        graph.lint()
        return graph


_dynamo_triton_patch_done = False


def _dynamo_compat_patches_enabled() -> bool:
    """进程级 Dynamo 兼容补丁的开关。

    默认开启：Wan2.2 在 trace 期读取 env、以及 trace 期的 triton 设备探测，都需要
    在 backend 调用之前消除崩溃/碎图。设置 ``MINDIE_SD_DYNAMO_COMPAT_PATCHES=0``
    可整体关闭，用于与 torch_npu 官方补丁共存，或在 CUDA 机器上排除本补丁影响。
    关闭后若仍触发探测崩溃，会显式编译失败而不是静默降级。
    """
    return os.environ.get("MINDIE_SD_DYNAMO_COMPAT_PATCHES", "1") != "0"


def _patch_dynamo_triton_device_probes() -> None:
    """消除 NPU 环境 dynamo 内部 triton 设备探测的报错（不影响编译结果）。

    ``torch.utils._triton`` 在非 CUDA 环境做设备能力查询会炸，已知两个调用点：

    1. torch 2.9：每个 frame 编译结束后 ``record_compilation_metrics`` 经
       ``torch._dynamo.utils.has_triton`` 调 ``cuda_extra_check``，NPU 下
       ``major`` 为 None 触发 ``TypeError: '>=' not supported between
       NoneType and int``，被 metrics_context 捕获后以 error 级刷整段
       traceback（每 frame 一次）。仅日志噪音。
    2. 新版 torch：``VariableBuilder._wrap`` 调 ``has_triton_experimental_host_tma``
       → ``_device_supports_tma`` → ``torch.cuda.get_device_capability() >= (9, 0)``，
       NPU 下返回 None 与 tuple 比较直接抛 ``InternalTorchDynamoError``，
       **发生在 Dynamo trace 过程中（backend 调用之前），会中断编译**。

    因此补丁必须同时：(a) 覆盖多个符号——在各自命名空间探测，仅在真实抛异常时
    置为恒 False（语义 = 设备不支持 triton/TMA，NPU 下正确）；(b) 在模块导入时
    执行——trace 期爆点早于 backend ``__call__``，入口补丁来不及。

    探测目标：
    - ``torch.utils._triton._device_supports_tma``（源头；包装函数体内按模块全局
      查找它，源头补丁覆盖所有动态调用者）
    - ``torch.utils._triton.cuda_extra_check`` / ``has_triton_experimental_host_tma``
      （按版本存在与否）
    - ``torch._dynamo.utils.has_triton``、``torch._dynamo.variables.builder.
      has_triton_experimental_host_tma``（直接 import 绑定的命名空间）
    - 2.6/2.7 与 2.8/2.9 各自的 TMA 探测符号 ``has_triton_tma`` /
      ``has_triton_tma_bulk_copy`` / ``has_triton_stable_tma_api``（按版本存在与否；
      仅对无必填实参者试调，避免把需要入参的符号误置为 ``lambda: False``）

    torch 版本结构变化（无该符号）时静默跳过；CUDA 正常环境探测通过则不动。
    """
    global _dynamo_triton_patch_done
    if _dynamo_triton_patch_done or not _dynamo_compat_patches_enabled():
        return
    _dynamo_triton_patch_done = True

    def _probe_and_patch(module, name: str) -> None:
        target = getattr(module, name, None)
        if target is None or not callable(target):
            return
        # 只对“无必填实参”的探测函数试调用；需要实参的符号无法在导入期安全判定，
        # 跳过而不是盲目置 False（否则会改变其真实语义）。
        try:
            params = inspect.signature(target).parameters.values()
        except (TypeError, ValueError):
            params = ()
        if any(
            p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in params
        ):
            return
        try:
            target()
        except Exception:  # noqa: BLE001
            setattr(module, name, lambda: False)
            logger.info(
                "[MindIE-SD/compilation] patched %s.%s to False on this NPU env. "
                "issue=device capability probe raised (cuda properties unavailable); "
                "impact=disables triton/TMA host-side paths only, compilation unaffected.",
                module.__name__,
                name,
            )

    try:
        import torch.utils._triton as torch_triton
    except ImportError:
        torch_triton = None
    if torch_triton is not None:
        _probe_and_patch(torch_triton, "_device_supports_tma")
        _probe_and_patch(torch_triton, "cuda_extra_check")
        _probe_and_patch(torch_triton, "has_triton_experimental_host_tma")
        # 2.6/2.7 与 2.8/2.9 的 TMA 探测符号名不同，逐个探测；不存在的静默跳过。
        for tma_symbol in ("has_triton_tma", "has_triton_tma_bulk_copy", "has_triton_stable_tma_api"):
            _probe_and_patch(torch_triton, tma_symbol)

    try:
        import torch._dynamo.utils as dynamo_utils
    except ImportError:
        dynamo_utils = None
    if dynamo_utils is not None:
        _probe_and_patch(dynamo_utils, "has_triton")
        for tma_symbol in ("has_triton_tma", "has_triton_stable_tma_api"):
            _probe_and_patch(dynamo_utils, tma_symbol)

    try:
        import torch._dynamo.variables.builder as dynamo_builder
    except ImportError:
        dynamo_builder = None
    if dynamo_builder is not None:
        _probe_and_patch(dynamo_builder, "has_triton_experimental_host_tma")
        for tma_symbol in ("has_triton_tma", "has_triton_stable_tma_api"):
            _probe_and_patch(dynamo_builder, tma_symbol)


_torchnpu_getenv_restore_done = False


def _restore_native_getenv() -> None:
    """恢复原生 ``os.getenv``（若被 torch_npu patch_getenv 包装）。

    新版 torch_npu 的 ``torch_npu/utils/patch_getenv.py`` 会包装 ``os.getenv`` 记录
    环境变量读取；包装内的 ``logger.info`` 被 Dynamo 以 "Logger not supported" 拒绝，
    被 trace 代码里**每次** ``os.getenv`` 调用都触发一次 graph break。Wan2.2
    model.py 在 attention 路径内联读取 ALGO/ROPE_OPT（self-attn 3 处、cross-attn
    1 处），等于每个 block 断 4 次、~120 断/step，是新版 torch_npu 下 aclgraph
    碎图的主要新增来源。

    恢复优先级：patch_getenv 模块保存的原始引用 > ``os.environ.get``（与
    ``os.getenv`` 语义一致：后者即 ``environ.get(key, default)``）。仅当
    ``os.getenv`` 确实来自 torch_npu 时才动作；该补丁只做日志记录，恢复不影响
    torch_npu 功能。幂等。
    """
    global _torchnpu_getenv_restore_done
    if _torchnpu_getenv_restore_done or not _dynamo_compat_patches_enabled():
        return
    _torchnpu_getenv_restore_done = True

    current = os.getenv
    if not current.__module__.startswith("torch_npu"):
        return

    restored = None
    try:
        from torch_npu.utils import patch_getenv as _pge

        for candidate in vars(_pge).values():
            if callable(candidate) and candidate is not current and candidate.__module__ == "os":
                restored = candidate
                break
    except Exception:  # noqa: BLE001
        restored = None
    if restored is None:
        restored = os.environ.get

    os.getenv = restored
    logger.info(
        "[MindIE-SD/compilation] restored native os.getenv (was torch_npu patch_getenv wrapper). "
        "issue=patched getenv logs every env read and Dynamo graph-breaks on the logger call in "
        "traced code; impact=env reads inside traced regions no longer fragment the graph."
    )


class MindieSDBackend:
    """
    The compilation backend for 'torch.compile'.
    It is used to process the FX graph and perform custom operation fusing etc.
    """

    def __call__(self, graph: fx.GraphModule, example_inputs) -> Callable:
        """路由到具体的编译后端。

        通过 :meth:`CompilationConfig.resolved_backend_mode` 单点解析最终后端，
        再分发到 default / aclgraph / npugraph_ex。NPU 或目标后端不可用时降级到
        default 后端并告警，保证可用性。

        Args:
            graph (fx.GraphModule): Dynamo 产出的 FX 图模块。
            example_inputs (optional): 示例输入，用于编译与图捕获。

        Returns:
            Callable: 编译后的可调用对象。
        """
        _patch_dynamo_triton_device_probes()
        _restore_native_getenv()
        mode = CompilationConfig.resolved_backend_mode()

        if mode == BackendMode.ACLGRAPH_WITH_COMPILE and npu_graph_available:
            if CompilationConfig.aclgraph_breakpoints:
                logger.warning(
                    "[MindIE-SD/compilation] Breakable Graph segmentation is not supported in "
                    "aclgraph_with_compile mode yet; fall back to single-segment. "
                    "Troubleshooting: use backend_mode=aclgraph_only for breakpoint segmentation.",
                )
            graph = self.compile(graph, example_inputs)
            return create_aclgraph_backend()(graph, example_inputs)

        if mode == BackendMode.ACLGRAPH_ONLY and npu_graph_available:
            if CompilationConfig.aclgraph_enable_patterns:
                from .passes import apply_lightweight_decomp
                from .passes.gelu_quant_pass import GeluQuantFusionPass
                from .passes.rope_cast_pass import RopeCastFusionPass

                graph = apply_lightweight_decomp(graph, example_inputs)
                self.__class__.apply_pattern_match_passes(graph, example_inputs)
                # P2/P4 与 npugraph_ex 路径同一开关（FusionPatterns），不接入 with_compile，
                # 保持其 A/B 基线干净。放在 pattern pass 之后：fast_gelu 模板先落图，
                # P4 对 aten.gelu / npu_fast_gelu 两种链头均可命中。
                fusion_cfg = CompilationConfig.fusion_patterns
                if fusion_cfg.enable_gelu_quant_fusion:
                    GraphTransformObserver(
                        gm=graph,
                        passname="gelu_quant_fusion",
                        subsystem="fusion_passes",
                        log_url=CompilationConfig.graph_log_url,
                    ).apply_gm_pass(GeluQuantFusionPass(approximate=fusion_cfg.gelu_quant_approximate))
                if fusion_cfg.enable_rope_cast_fusion:
                    GraphTransformObserver(
                        gm=graph,
                        passname="rope_cast_fusion",
                        subsystem="fusion_passes",
                        log_url=CompilationConfig.graph_log_url,
                    ).apply_gm_pass(RopeCastFusionPass())
            breakpoints = CompilationConfig.aclgraph_breakpoints
            if breakpoints:
                from .graph_splitter import split_graph_by_breakpoints

                _, seg_meta, out_spec = split_graph_by_breakpoints(graph, breakpoints)
                if seg_meta is not None:
                    logger.debug(
                        "[MindIE-SD/compilation] Segmented ACLGraph backend selected. mode=%s, segments=%d.",
                        mode.value,
                        len(seg_meta),
                    )
                    return create_segmented_aclgraph_backend(graph, seg_meta, out_spec)
                logger.debug("[MindIE-SD/compilation] breakpoints did not split graph, fall back to single-segment.")
            logger.debug(
                "[MindIE-SD/compilation] ACLGraph backend selected. mode=%s, npu_graph_available=%s.",
                mode.value,
                npu_graph_available,
            )
            return create_aclgraph_backend()(graph, example_inputs)

        if mode == BackendMode.NPUGRAPH_EX:
            if not npugraph_ex_available:
                logger.warning(
                    "[MindIE-SD/compilation] npugraph_ex backend unavailable, fall back to default backend. "
                    "possible_cause=torch_npu.npugraph_ex.compile_fx not found. "
                    "Troubleshooting: install a torch_npu version that provides npugraph_ex.",
                )
            else:
                if CompilationConfig.aclgraph_enable_patterns:
                    # 与 aclgraph_only 分支保持同一先后：先跑 pattern matcher，再由
                    # npugraph_ex 的 fw_compiler 施加 P2/P4。npugraph_ex 本身走完整
                    # AOT 分解，不需要 apply_lightweight_decomp 的 make_fx 预展开。
                    self.__class__.apply_pattern_match_passes(graph, example_inputs)
                logger.debug(
                    "[MindIE-SD/compilation] npugraph_ex backend selected. mode=%s.",
                    mode.value,
                )
                return create_npugraph_ex_backend()(graph, example_inputs)

        # default backend；包含 aclgraph 在 NPU 不可用、或 npugraph_ex 不可用时的降级。
        if mode in (BackendMode.ACLGRAPH_ONLY, BackendMode.ACLGRAPH_WITH_COMPILE) and not npu_graph_available:
            logger.warning(
                "[MindIE-SD/compilation] ACLGraph requested but NPU unavailable, fall back to default backend. "
                "mode=%s, npu_graph_available=%s.",
                mode.value,
                npu_graph_available,
            )
        graph = self.compile(graph, example_inputs)
        return graph

    @classmethod
    def apply_redundant_node_elimination_pass(cls, graph: fx.GraphModule, inputs):
        GraphTransformObserver(
            gm=graph,
            passname="redundant_node_elimination_pass",
            subsystem="redundant_node_elimination_pass",
            log_url=CompilationConfig.graph_log_url,
        ).apply_gm_pass(ReduandantNodeEliminationPass())
        logger.debug("Graph after redundant node elimination pass:")
        if logger.isEnabledFor(DEBUG_LOG_LEVEL):
            logger.debug(graph.print_readable(print_output=False))

    @classmethod
    def apply_pattern_match_passes(cls, graph: fx.GraphModule, inputs):
        activate_pattern_once()
        GraphTransformObserver(
            gm=graph,
            passname="pattern_match_pass",
            subsystem="pattern_match_passes",
            log_url=CompilationConfig.graph_log_url,
        ).apply_gm_pass(patterns)
        logger.debug("Graph after pattern matching:")
        if logger.isEnabledFor(DEBUG_LOG_LEVEL):
            logger.debug(graph.print_readable(print_output=False))

    @classmethod
    def apply_decompose_auto_functionalized_pass(cls, graph: fx.GraphModule):
        GraphTransformObserver(
            gm=graph,
            passname="decompose_auto_functionalized",
            subsystem="decompose_auto_functionalized_pass",
            log_url=CompilationConfig.graph_log_url,
        ).apply_graph_pass(decompose_auto_functionalized)

    def compile(
        self,
        gm: fx.GraphModule,
        example_inputs,
        **kwargs,
    ) -> Callable:
        def freezing_compile(compile_inner, aot_autograd_gm, example_inputs):
            # Freeze the graph first before passing to AOT Autograd.
            frozen_gm, preserved_arg_indices = freeze(gm, aot_autograd_gm, example_inputs)
            example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
            optimized_function = compile_inner(frozen_gm, example_inputs)

            def wrapper(args: list[object]) -> Sequence[torch.Tensor]:
                args_new = [args[i] for i in preserved_arg_indices]
                args.clear()
                return optimized_function(*args_new)

            wrapper._boxed_call = True  # type: ignore[attr-defined]

            return wrapper

        def graph_rewrite_before_freezing(fx_graph, inputs):
            logger.debug("Graph before compiling:")
            if logger.isEnabledFor(DEBUG_LOG_LEVEL):
                logger.debug(fx_graph.print_readable(print_output=False))
            self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
            self.__class__.apply_pattern_match_passes(fx_graph, inputs)
            return fx_graph

        def graph_rewrite_after_freezing(fx_graph, inputs):
            self.__class__.apply_redundant_node_elimination_pass(fx_graph, inputs)
            # make sure we add freezing passes after constant folding
            self.__class__.apply_decompose_auto_functionalized_pass(fx_graph)
            # 与 npugraph_ex 路径同一落盘开关：抓 default/with_compile 最终喂给
            # 捕获的图，用于和 npugraph_ex dump 对比量化链（cast/quant）的去向。
            from .npugraph_ex_backend import _dump_fx_graph

            _dump_fx_graph(fx_graph, "compiled_fx")
            logger.debug("Graph after compiling:")
            if logger.isEnabledFor(DEBUG_LOG_LEVEL):
                logger.debug(fx_graph.print_readable(print_output=False))
            return fx_graph

        def compile_inner(fx_graph, inputs):
            # we split the rewrite into two phases: before and after freezing
            # since freezing would do CSE which might break some assumptions in
            # the rewrite rules.
            graph_rewrite_before_freezing(fx_graph, inputs)
            if CompilationConfig.enable_freezing:
                return freezing_compile(graph_rewrite_after_freezing, fx_graph, inputs)
            else:
                return graph_rewrite_after_freezing(fx_graph, inputs)

        decompositions = select_custom_decomp_table()
        # Use AOT Autograd to handle the forward compilation.
        return aot_autograd(
            fw_compiler=compile_inner,
            decompositions=decompositions,
        )(gm, example_inputs)


# 导入时执行：新版 torch 的 triton 设备探测在 Dynamo trace 过程中（backend
# 调用之前）就会触发，晚于导入的补丁来不及。幂等，见函数 docstring。
_patch_dynamo_triton_device_probes()
# 同理在导入期恢复原生 os.getenv：Wan2.2 在 attention 路径内联读 env，首个
# frame 的 trace 早于任何 backend __call__。幂等，见函数 docstring。
_restore_native_getenv()
