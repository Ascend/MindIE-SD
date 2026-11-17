#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""npugraph_ex 编译后端。

封装 ``torch.npu.npugraph_ex.compile_fx`` 作为 ``torch.compile`` 的可调用后端，
外层用 ``aot_module_simplified`` 把 Dynamo 原始 gm 规范化为全 fake 图后再编译。

两种工作模式（由 options 的 ``force_eager`` 决定）：
- 默认（无 ``force_eager``）：走 npugraph_ex 全管线（fx2acl 降级 + 其自身捕获）。
- ``force_eager=True``：npugraph_ex 仅做编译期 FX 优化（融合/reinplace pass），不降级
  ACL IR 也不成图（NPU 实测单独使用退化为纯 eager）；此时本框架用 aclgraph 捕获
  兜底，等价 vLLM Ascend 的生产架构——编译期 FX 优化 + 框架自研 ACLGraph 捕获回放。

注意：``compile_fx`` 的确切签名随 torch_npu 版本变化，本模块按
``compile_fx(gm, example_inputs, options)`` 调用，需在目标 torch_npu 版本上验证。
"""

import inspect
import itertools
import os
from collections.abc import Callable
from typing import Any

from ..utils.logs.logging import logger
from .compiliation_config import CompilationConfig

# npugraph_ex 可用性检测，与 aclgraph_backend.npu_graph_available 的检测风格一致。
# 用别名导入：函数内另有局部 import torch，模块级同名绑定会触发 pylint 重名告警。
# 注意探测对象是 ``torch.npu.npugraph_ex``（真实调用点同此路径）；写成
# ``torch.npugraph_ex`` 时在 torch_npu 环境下 hasattr 的实参求值会抛 AttributeError。
npugraph_ex_available = False
try:
    import torch as _torch_probe

    if (
        hasattr(_torch_probe, "npu")
        and hasattr(_torch_probe.npu, "npugraph_ex")
        and hasattr(_torch_probe.npu.npugraph_ex, "compile_fx")
    ):
        npugraph_ex_available = True
except (ImportError, AttributeError):
    npugraph_ex_available = False


def _resolve_npu_cast_op(torch):
    """返回 npugraph_ex fx2acl 可直接解析的 NPU 原生 cast 算子，取不到时返回 None。

    torch_npu 在不同版本提供 ``npu_dtype_cast``（2.6）或 ``_npu_dtype_cast``。
    这里按属性存在性探测而不是匹配 ``torch.__version__`` 字符串——官方 wheel 的版本串
    可能带 ``+cpu``/``rc`` 等后缀，字符串包含判断会在这些版本上选错算子。
    """
    npu_ops = getattr(torch.ops, "npu", None)
    if npu_ops is None:
        return None
    for name in ("npu_dtype_cast", "_npu_dtype_cast"):
        op_packet = getattr(npu_ops, name, None)
        if op_packet is not None and hasattr(op_packet, "default"):
            return op_packet.default
    return None


def _patch_type_raw_for_npugraph_ex(gm):
    """将 torch_npu 注入的 ``type_raw`` call_method 改写为 NPU 原生 cast。

    torch_npu 将 ``Tensor.type`` monkey-patch 为 ``type_raw``；Dynamo trace 后 gm 中出现
    ``call_method[target=type_raw]``，而 npugraph_ex 的 fx2acl 转换器不支持该私有方法
    （报 'str' object is not callable）。等价改写为 ``npu_dtype_cast``/``_npu_dtype_cast``
    以消除。改写失败时原样返回 gm 并告警（由 npugraph_ex 自身报错定位），不再静默吞掉
    异常——静默失败会让后续编译错误失去上下文。
    """
    try:
        import torch

        cast_op = _resolve_npu_cast_op(torch)
        if cast_op is None:
            logger.warning(
                "[MindIE-SD/compilation] npugraph_ex type_raw patch skipped: neither "
                "torch.ops.npu.npu_dtype_cast nor torch.ops.npu._npu_dtype_cast is available. "
                "possible_cause=torch_npu version mismatch. gm is passed through unchanged."
            )
            return gm

        graph = gm.graph
        modified = False
        for node in list(graph.nodes):
            if node.op != "call_method" or node.target != "type_raw":
                continue
            args = list(node.args)
            kwargs = dict(node.kwargs)
            x = args[0] if len(args) > 0 else kwargs.pop("self", None)
            dtype = args[1] if len(args) > 1 else kwargs.pop("dtype", None)
            if x is None or dtype is None:
                logger.warning(
                    "[MindIE-SD/compilation] npugraph_ex type_raw patch skipped node '%s': "
                    "missing input or dtype argument. The node is kept as-is.",
                    node.name,
                )
                continue
            # npugraph_ex fx2acl 直接支持 NPU 原生 cast（其内置 add_rms_norm_cast pattern
            # 亦用此 op）；aten._to_copy 未注册 parse，走通用 meta 推断会因真实 tensor
            # 触发 FakeTensor 断言失败。
            with graph.inserting_before(node):
                new_node = graph.call_function(cast_op, args=(x, dtype))
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
            modified = True
        if modified:
            graph.eliminate_dead_code()
            graph.lint()
            gm.recompile()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[MindIE-SD/compilation] npugraph_ex type_raw patch failed, keep original gm. "
            "error=%s. The graph is still handed to compile_fx unchanged.",
            exc,
            exc_info=True,
        )
        return gm
    return gm


def _call_npugraph_ex_compile_fx(torch, gm, example_inputs, opts):
    """调用 ``torch.npu.npugraph_ex.compile_fx``，兼容不同 torch_npu 版本的签名。

    torch_npu 的接口在不同版本间有工厂式与 ``compile_fx(gm, example_inputs, options)``
    两种形态。优先按关键字 ``options`` 判断，再回退到 ``**opts``，避免版本差异直接
    导致整个编译流程失败；回退时会记录实际使用的签名。
    """
    compile_fx = torch.npu.npugraph_ex.compile_fx
    try:
        params = inspect.signature(compile_fx).parameters
    except (TypeError, ValueError):
        params = None

    # 入口显式声明 options 参数：整体作为位置实参传入。
    if params is not None and "options" in params:
        return compile_fx(gm, example_inputs, opts)

    # 其余形态（options 拆成若干关键字参数，或签名不可取）先按关键字展开传入，
    # 失败再退回位置实参。
    try:
        return compile_fx(gm, example_inputs, **opts)
    except TypeError as exc:
        logger.warning(
            "[MindIE-SD/compilation] npugraph_ex.compile_fx keyword options call failed "
            "(error=%s); retrying with a positional options dict. possible_cause=torch_npu "
            "signature differs across versions.",
            exc,
        )
        return compile_fx(gm, example_inputs, opts)


# npugraph_ex 的默认 option（见 _resolve_npugraph_ex_options 与文档「npugraph_ex 后端」）：
# - reuse_graph_pool_in_same_fx=False：多图共享内存池会覆写跨断点张量；
# - clone_output=True：不克隆时，同一张图先后两次调用返回同一块静态缓冲，前一次的结果会被
#   后一次覆盖（CFG 的 cond/uncond 即此形态，引导项会被抹平）。
_POOL_REUSE_OPTION = "reuse_graph_pool_in_same_fx"
_CLONE_OUTPUT_OPTION = "clone_output"
_option_hint_done = set()


def _hint_once(tag: str, level: str, msg: str, *args) -> None:
    """同一进程内同一 tag 只打印一次，避免每张图/每次编译刷屏。"""
    if tag in _option_hint_done:
        return
    _option_hint_done.add(tag)
    getattr(logger, level)(msg, *args)


def _resolve_npugraph_ex_options(options: dict | None) -> dict:
    """在用户 options 之上补齐 npugraph_ex 的安全默认值，并对不安全取值给出一次性提示。

    ``reuse_graph_pool_in_same_fx``：torch_npu 侧默认 True，即同一 FX 内的多张 ACLGraph
    共享一个内存池。进程内出现第二张图（多 prompt / shape 变化导致重捕获）后，跨图断点
    传递的张量（``tolist()`` 断图后的 shape 张量、模型返回张量）会被覆写成指针量级的
    垃圾值，表现为偶发的 shape/size 报错。

    ``clone_output``：torch_npu 侧默认 False，replay 直接返回静态输出缓冲。同一步内先后
    两次调用同一张图时（CFG 的 cond/uncond、双流、teacher forcing），第二次 replay 会覆盖
    第一次返回的张量，调用方拿到的两个结果完全相同；CFG 的引导项随之归零（等效
    ``guide_scale=1``），表现为不报错但生成结果的动态与文本响应明显变差。

    两项均在未显式配置时取安全值；用户显式配置时尊重其选择，但打印一次风险提示。
    不修改调用方传入的字典。
    """
    opts = dict(options) if options else {}
    if _POOL_REUSE_OPTION not in opts:
        opts[_POOL_REUSE_OPTION] = False
        _hint_once(
            "pool_reuse_default",
            "info",
            "[MindIE-SD/compilation] npugraph_ex option %s defaulted to False. "
            "reason=sharing one pool across multiple ACLGraphs in the same FX can overwrite "
            "tensors crossing a graph break. consequence=one memory pool per ACLGraph, "
            "peak memory grows with graph count. Set it to True explicitly to share a pool. "
            "See docs/zh/features/compilation.md",
            _POOL_REUSE_OPTION,
        )
    elif opts[_POOL_REUSE_OPTION]:
        _hint_once(
            "pool_reuse_override",
            "warning",
            "[MindIE-SD/compilation] npugraph_ex option %s is explicitly set to True, "
            "graph memory pools are shared across ACLGraphs in the same FX. "
            "risk=in a process with more than one graph, tensors crossing a graph break "
            "(shape tensors consumed in eager code, model outputs) can be overwritten with "
            "pointer-magnitude garbage, giving intermittent errors like \"shape "
            "'[5833846557024, ...]' is invalid for input of size ...\" or \"size of tensor a "
            "(5736774568800) must match the size of tensor b (0)\". "
            "suggestion=unset it (defaults to False) or set it to False. "
            "See docs/zh/features/compilation.md",
            _POOL_REUSE_OPTION,
        )

    if _CLONE_OUTPUT_OPTION not in opts:
        opts[_CLONE_OUTPUT_OPTION] = True
        _hint_once(
            "clone_output_default",
            "info",
            "[MindIE-SD/compilation] npugraph_ex option %s defaulted to True. "
            "reason=without it, successive replays of the same graph return the same static "
            "output buffer, so an earlier result is overwritten by a later call. consequence="
            "one extra output copy per replay and one extra output-sized buffer. Set it to "
            "False explicitly only if the caller never holds a result across two replays. "
            "See docs/zh/features/compilation.md",
            _CLONE_OUTPUT_OPTION,
        )
    elif not opts[_CLONE_OUTPUT_OPTION]:
        _hint_once(
            "clone_output_override",
            "warning",
            "[MindIE-SD/compilation] npugraph_ex option %s is explicitly set to False, "
            "replays of the same graph return the same static output buffer without cloning. "
            "risk=when the model is called twice per step with identical shapes (classifier-free "
            "guidance cond/uncond, dual-stream, teacher forcing), the second call overwrites the "
            "first result; with CFG this silently zeroes the guidance term (equivalent to "
            "guide_scale=1) and the generated video loses most of its motion, with no error. "
            "suggestion=unset it (defaults to True) or set it to True. "
            "See docs/zh/features/compilation.md",
            _CLONE_OUTPUT_OPTION,
        )
    return opts


_FX_DUMP_SEQ = itertools.count()


def _apply_fusion_patterns(gm):
    """按 FusionPatterns 开关施加 P2/P4 自定义融合 pass（dump 之后、compile_fx 之前）。

    与 aclgraph_only 分支的接线保持同一开关集合；刻意不接入 with_compile 路径，
    保持其 A/B 基线干净。
    """
    fusion_cfg = CompilationConfig.fusion_patterns
    if fusion_cfg.enable_gelu_quant_fusion:
        # 融合 pass 按需惰性导入：与其开关同源，未启用时不引入额外依赖。
        from .passes.gelu_quant_pass import GeluQuantFusionPass  # pylint: disable=no-name-in-module

        GeluQuantFusionPass(approximate=fusion_cfg.gelu_quant_approximate)(gm)
    if fusion_cfg.enable_rope_cast_fusion:
        from .passes.rope_cast_pass import RopeCastFusionPass  # pylint: disable=no-name-in-module

        RopeCastFusionPass()(gm)
    return gm


def _dump_fx_graph(gm, tag: str) -> None:
    """把进入 npugraph_ex pattern pass 前的 fx 图落盘（开发辅助，默认关闭）。

    设置 ``MINDIE_SD_DUMP_FX_DIR`` 后，每次调用 ``compile_fx`` 前把 gm 的
    forward 源码写入该目录。该 gm 正是 ``_optimize_fx``（pattern_fusion_pass）
    的输入，源码中的算子调用即图上的真实 ``node.target``——自定义 pattern 的
    匹配谓词（CallFunction 树）必须引用完全相同的 target 才能命中。
    """
    dump_dir = os.environ.get("MINDIE_SD_DUMP_FX_DIR")
    if not dump_dir:
        return
    try:
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"{tag}_{next(_FX_DUMP_SEQ)}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(gm.code)
        logger.info("[MindIE-SD/compilation] dumped fx graph (pre pattern pass) to %s", path)
    except OSError:
        logger.warning("[MindIE-SD/compilation] failed to dump fx graph to %s", dump_dir, exc_info=True)


def create_npugraph_ex_backend(options: dict | None = None) -> Callable:
    """创建 npugraph_ex 后端可调用对象。

    返回 ``npugraph_ex_backend(gm, example_inputs) -> compiled_fn``，
    签名与 :func:`aclgraph_backend.create_aclgraph_backend` 对齐，便于在
    :class:`MindieSDBackend` 路由中统一分发。

    Args:
        options: 透传给 ``compile_fx`` 的选项字典；为 None 时读取
            :class:`CompilationConfig.npugraph_ex_options`，仍为 None 时使用空字典。
            返回前会经 :func:`_resolve_npugraph_ex_options` 补齐安全默认值
            （``reuse_graph_pool_in_same_fx=False``、``clone_output=True``）。
    """

    def npugraph_ex_backend(gm: Any, example_inputs) -> Callable:
        opts = options if options is not None else CompilationConfig.npugraph_ex_options
        # 默认关闭跨图内存池复用：共享池会在多图场景下覆写跨断点张量，见
        # _resolve_npugraph_ex_options。用户显式配置的取值优先。
        opts = _resolve_npugraph_ex_options(opts)
        import torch
        from torch._functorch.aot_autograd import aot_module_simplified

        # npugraph_ex 须经 aot_autograd 规范化为全 fake 图后再编译：其 fx2acl 解释器对
        # 每个节点做 fake meta 推断，若 gm 含真实 tensor 输入（Dynamo 原始产物）会触发
        # FakeTensor 断言失败（cast/unsqueeze 等任何作用于真实输入的 op 均会中招）。
        # default/aclgraph 路径本就有 aot_autograd 这层故能整图；npugraph_ex 同样需要
        # （见 compilation-dev backend-comparison：npugraph_ex 在 torch 2.9 走 aot_autograd）。
        # 采用 aot_module_simplified 包装 compile_fx，是 torchair 官方推荐的 callable 接入
        # 模式，不引入 graph break，保留整图。
        # keep_inference_input_mutations=True 对齐 npugraph_ex 自身默认（其 npu_fx_compiler
        # 构造 aot_kwargs 时默认置 True），使输入 mutation 留在图内、与 npugraph_ex 的
        # copy 消减 pass（input_inplace_pass 默认启用）配套工作。
        def fw_compiler(compiled_gm, compiled_inputs):
            compiled_gm = _patch_type_raw_for_npugraph_ex(compiled_gm)
            _dump_fx_graph(compiled_gm, "npugraph_ex_fx")
            # P2/P4 融合 pass 落在 dump 之后：dump 保持“进入 pass 前的原始图”语义，
            # 与 pattern 谓词基准一致；融合结果由编译日志/性能回归观察。
            compiled_gm = _apply_fusion_patterns(compiled_gm)
            return _call_npugraph_ex_compile_fx(torch, compiled_gm, compiled_inputs, opts)

        compiled = aot_module_simplified(
            gm,
            example_inputs,
            fw_compiler=fw_compiler,
            keep_inference_input_mutations=True,
        )

        # force_eager=True（vLLM/SGLang 生产姿势）时 npugraph_ex 只做编译期 FX 优化：
        # 不降级 ACL IR、也不成图。此处用本框架的 aclgraph 捕获兜底，构成两段式架构
        # ——FX 优化后的 callable 交给框架自研 NPUGraph 捕获/回放。create_aclgraph_backend
        # 仅依赖 callable 契约（gm(*static_inputs)），可直接捕获该优化后的 callable。
        if opts.get("force_eager"):
            from .aclgraph_backend import create_aclgraph_backend, npu_graph_available

            if not npu_graph_available:
                logger.warning(
                    "[MindIE-SD/compilation] npugraph_ex force_eager without aclgraph capture. "
                    "issue=torch.npu.NPUGraph is unavailable, the FX-optimized callable will "
                    "run in pure eager mode (no graph replay)."
                )
                return compiled
            # 先以克隆输入预热一次：让 force_eager 的执行路径完成懒初始化（算子选择、
            # workspace 分配）后再进捕获。克隆是为避免 keep_inference_input_mutations 的
            # 图内原地写污染调用方传入的 example 张量。
            warmup_inputs = [t.detach().clone() if isinstance(t, torch.Tensor) else t for t in example_inputs]
            compiled(*warmup_inputs)
            return create_aclgraph_backend()(compiled, example_inputs)

        return compiled

    return npugraph_ex_backend
