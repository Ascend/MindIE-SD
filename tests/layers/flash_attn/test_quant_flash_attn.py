#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Interface and NPU tests for FP8/MXFP8/MXFP4 attention and legacy compatibility."""

import importlib
import sys
import unittest
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
import torch_npu
from mindiesd import quant_attention
from mindiesd.quantization.mode import FP8FAMode
from mindiesd.utils.exception import ParametersInvalid
from mindiesd.utils.get_platform import is_a5_device

fp8 = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")
mxfp4 = importlib.import_module("mindiesd.layers.flash_attn.quant_attention_mxfp4")


class TestQuantFlashAttn(unittest.TestCase):
    def test_entry_import_without_fp8_dtype(self):
        modules = (
            "mindiesd.layers.quant.block_quant",
            "mindiesd.layers.flash_attn.fused_infer_attention_score",
            "mindiesd.layers.flash_attn.quant_flash_attn",
        )
        root = Path(__file__).resolve().parents[3]
        # Reload the real import chain; cached modules could hide a dtype lookup.
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            for name in modules:
                spec = importlib.util.spec_from_file_location(name, root.joinpath(*name.split(".")).with_suffix(".py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
            self.assertTrue(callable(module.quant_attention))
            with self.assertRaisesRegex(RuntimeError, "float8_e4m3fn"):
                sys.modules[modules[0]].fa_block_quant_preprocess(torch.ones(1, 1, 4, 8))

    def setUp(self):
        self.q = torch.randn(1, 2, 17, 64)

    def test_fp8_uses_its_operator_layout(self):
        with (
            patch.object(fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))),
            patch.object(
                torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)
            ) as execute,
        ):
            result = quant_attention(self.q, self.q, self.q)
            torch.testing.assert_close(result, self.q)
            self.assertEqual(execute.call_args.kwargs["input_layout"], "BNSD")

    def test_rotation_is_applied_once_before_quantization(self):
        with (
            patch.object(
                fp8,
                "_fa_block_quant_preprocess",
                side_effect=lambda tensor, **kw: (tensor, torch.ones(1)),
            ) as quantize,
            patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)),
        ):
            quant_attention(self.q, self.q, self.q, precision="fp8", q_rot=2 * torch.eye(64))
            torch.testing.assert_close(quantize.call_args_list[0].args[0], 2 * self.q)
            self.assertIs(quantize.call_args_list[1].args[0], self.q)

    def test_rotation_is_cast_to_input_dtype_before_dispatch(self):
        entry = importlib.import_module("mindiesd.layers.flash_attn.quant_flash_attn")
        dtypes = (torch.float16, torch.bfloat16, torch.float32)
        for precision, dtype, rotation_dtype in product(("fp8", "mxfp8", "mxfp4"), dtypes, dtypes):
            query = self.q.to(dtype=dtype)
            q_rot = 2 * torch.eye(64, dtype=rotation_dtype)
            k_rot = 3 * torch.eye(64, dtype=rotation_dtype)
            with (
                self.subTest(precision=precision, dtype=dtype, rotation_dtype=rotation_dtype),
                patch.object(entry, f"_{precision}_attention_forward", return_value=query) as execute,
            ):
                result = quant_attention(query, query, query, precision=precision, q_rot=q_rot, k_rot=k_rot)
                self.assertIs(result, query)
                execute.assert_called_once()
                for name, original, factor in (("q_rot", q_rot, 2), ("k_rot", k_rot, 3)):
                    rotation = execute.call_args.kwargs[name]
                    self.assertEqual(rotation.dtype, dtype)
                    self.assertEqual(rotation.device, query.device)
                    torch.testing.assert_close(torch.matmul(query, rotation), factor * query)
                    self.assertEqual(original.dtype, rotation_dtype)
                    torch.testing.assert_close(original, factor * torch.eye(64, dtype=rotation_dtype))
                    if dtype == rotation_dtype:
                        self.assertIs(rotation, original)

    def test_float_rejected_before_execution(self):
        with (
            patch.object(torch_npu, "npu_fusion_attention", create=True) as floating,
            patch.object(fp8, "_fa_block_quant_preprocess") as quantize,
            patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True) as execute,
        ):
            with self.assertRaisesRegex(ValueError, "attention_forward"):
                quant_attention(self.q, self.q, self.q, precision="float")
            floating.assert_not_called()
            quantize.assert_not_called()
            execute.assert_not_called()

    def test_explicit_scale_and_window_reach_operator(self):
        for options, expected in (({}, 0.125), ({"scale": 0.25}, 0.25), ({"scale": 0.0}, 0.0)):
            with (
                self.subTest(options=options),
                patch.object(fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))),
                patch.object(
                    torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)
                ) as execute,
            ):
                quant_attention(self.q, self.q, self.q, pre_tokens=16, next_tokens=0, **options)
                self.assertEqual(execute.call_args.kwargs["softmax_scale"], expected)
                self.assertEqual(execute.call_args.kwargs["pre_tokens"], 16)
                self.assertEqual(execute.call_args.kwargs["next_tokens"], 0)

    def test_removed_softmax_scale_alias_rejected_before_quantization(self):
        with patch.object(fp8, "_fa_block_quant_preprocess") as quantize:
            for options in ({"softmax_scale": 0.5}, {"scale": 0.25, "softmax_scale": 0.5}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, "softmax_scale"):
                    quant_attention(self.q, self.q, self.q, **options)
            quantize.assert_not_called()

    def test_operator_error_propagates_without_float_retry(self):
        with (
            patch.object(fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))),
            patch.object(
                torch.ops.mindiesd,
                "fused_infer_attention_score_v2",
                create=True,
                side_effect=RuntimeError("operator failed"),
            ) as execute,
            patch.object(torch_npu, "npu_fusion_attention", create=True) as floating,
        ):
            with self.assertRaisesRegex(RuntimeError, "operator failed"):
                quant_attention(self.q, self.q, self.q)
            execute.assert_called_once()
            floating.assert_not_called()

    def test_unknown_precision_rejected(self):
        with self.assertRaises(ValueError):
            quant_attention(self.q, self.q, self.q, precision="int4")

    def test_unknown_options_rejected_before_quantization(self):
        with patch.object(fp8, "_fa_block_quant_preprocess") as quantize:
            for name in ("precisoin", "fp8_fa_mdoe", "mode", "mxfp4_scale_alg"):
                with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                    quant_attention(self.q, self.q, self.q, fp8_fa_mode="HIGH_PRECISION", **{name: "unused"})
            quantize.assert_not_called()

    def test_empty_dimensions_rejected(self):
        for axis in range(4):
            shape = list(self.q.shape)
            shape[axis] = 0
            query = torch.empty(shape)
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, "non-empty"):
                quant_attention(query, query, query, precision="fp8")

    def test_rotation_shape_rejected_before_matmul(self):
        for name in ("q_rot", "k_rot"):
            with self.subTest(name=name), patch.object(torch, "matmul") as rotate:
                with self.assertRaisesRegex(ValueError, name + ".*shape") as error:
                    quant_attention(self.q, self.q, self.q, **{name: torch.ones(64, 32)})
                self.assertIn("(64, 64)", str(error.exception))
                self.assertIn("got (64, 32)", str(error.exception))
                rotate.assert_not_called()

    def test_rotation_device_error_reports_both_devices(self):
        for name in ("q_rot", "k_rot"):
            with self.subTest(name=name), patch.object(torch, "matmul") as rotate:
                with self.assertRaisesRegex(ValueError, name + ".*device") as error:
                    quant_attention(self.q, self.q, self.q, **{name: torch.eye(64, device="meta")})
                self.assertIn(f"{name}.device=meta", str(error.exception))
                self.assertIn("input.device=cpu", str(error.exception))
                rotate.assert_not_called()

    def test_common_input_errors_stop_before_fp8_dispatch(self):
        entry = importlib.import_module("mindiesd.layers.flash_attn.quant_flash_attn")
        q = self.q
        cases = (
            (None, q, q, {}),
            (q, q, q, {"layout": "TND"}),
            (q, q[:, :, :8], q, {}),
            (q, q.half(), q.half(), {}),
            (q, q.to("meta"), q.to("meta"), {}),
            (q, q[..., :32], q[..., :32], {}),
            (q, q[:, :1].expand(-1, 3, -1, -1), q[:, :1].expand(-1, 3, -1, -1), {}),
        )
        with patch.object(entry, "_fp8_attention_forward") as execute:
            for index, (query, key, value, options) in enumerate(cases):
                with self.subTest(index=index), self.assertRaises(ValueError):
                    quant_attention(query, key, value, **options)
            execute.assert_not_called()

    def test_quantized_path_validates_common_inputs_once(self):
        entry = importlib.import_module("mindiesd.layers.flash_attn.quant_flash_attn")
        block = importlib.import_module("mindiesd.layers.quant.block_quant")
        with (
            patch.object(fp8, "fused_infer_attention_score_v2", wraps=fp8.fused_infer_attention_score_v2) as fia,
            patch.object(
                entry, "_validate_quant_attention_inputs", wraps=entry._validate_quant_attention_inputs
            ) as validate,
            patch.object(block, "fa_block_quant_preprocess", side_effect=AssertionError("duplicate block validation")),
            patch.object(
                torch_npu, "npu_dynamic_block_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
            ) as quantize,
            patch.object(
                torch.ops.mindiesd,
                "fused_infer_attention_score_v2",
                create=True,
                side_effect=lambda t, *args, **kw: (t,),
            ) as execute,
        ):
            output = quant_attention(self.q, self.q, self.q)
            torch.testing.assert_close(output, self.q)
            validate.assert_called_once()
            fia.assert_called_once()
            self.assertEqual(quantize.call_count, 3)
            execute.assert_called_once()

    def test_invalid_key_rotation_is_rejected_before_query_rotation(self):
        with (
            patch.object(torch, "matmul") as rotate,
            patch.object(torch_npu, "npu_dynamic_block_quant", create=True) as quant,
        ):
            with self.assertRaisesRegex(ValueError, "k_rot.*shape"):
                quant_attention(self.q, self.q, self.q, q_rot=torch.eye(64), k_rot=torch.ones(64, 32))
            rotate.assert_not_called()
            quant.assert_not_called()

    def test_fp8_implementation_rotates_before_quantization(self):
        for q_rot, k_rot in ((2 * torch.eye(64), None), (None, 3 * torch.eye(64))):
            with (
                self.subTest(q_rot=q_rot is not None, k_rot=k_rot is not None),
                patch.object(
                    fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))
                ) as quantize,
                patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)),
            ):
                fp8._fp8_attention_forward(
                    self.q,
                    self.q,
                    self.q,
                    layout="BNSD",
                    scale=0.125,
                    pre_tokens=2147483647,
                    next_tokens=2147483647,
                    mode="HIGH_PRECISION",
                    q_rot=q_rot,
                    k_rot=k_rot,
                )
                torch.testing.assert_close(quantize.call_args_list[0].args[0], self.q if q_rot is None else 2 * self.q)
                torch.testing.assert_close(quantize.call_args_list[1].args[0], self.q if k_rot is None else 3 * self.q)
                self.assertIs(quantize.call_args_list[2].args[0], self.q)

    def test_block_quant_default_and_explicit_dtype(self):
        block = importlib.import_module("mindiesd.layers.quant.block_quant")
        import torch_npu

        for dst_type in (None, 123):
            with patch.object(
                torch_npu, "npu_dynamic_block_quant", create=True, return_value=(self.q[0], torch.ones(1))
            ) as op:
                block.fa_block_quant_preprocess(self.q, dst_type=dst_type)
                self.assertEqual(
                    op.call_args.kwargs["dst_type"], torch_npu.float8_e4m3fn if dst_type is None else dst_type
                )

    def test_fp8_rejects_batch_before_quantization(self):
        query = self.q.expand(2, -1, -1, -1)
        with patch.object(fp8, "_fa_block_quant_preprocess") as quantize:
            with self.assertRaisesRegex(ValueError, "batch"):
                quant_attention(query, query, query, precision="fp8")
            quantize.assert_not_called()

    def test_fp8_specific_validation_precedes_quantization(self):
        for query, mode, error in (
            (self.q.expand(2, -1, -1, -1), None, ValueError),
            (self.q, "invalid", ParametersInvalid),
        ):
            with self.subTest(mode=mode), patch.object(fp8, "_fa_block_quant_preprocess") as quantize:
                with self.assertRaises(error):
                    fp8._fp8_attention_forward(
                        query, query, query, layout="BNSD", scale=0.125, pre_tokens=16, next_tokens=0, mode=mode
                    )
                quantize.assert_not_called()

    def test_legacy_mxfp4_fp8_fallback_calls_native_operator(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        model = layer.MXFP4QuantFA()
        with (
            patch.object(model.timestep_config, "get_strategy", return_value="FP8"),
            (
                patch.object(
                    torch_npu, "npu_dynamic_block_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
                )
            ),
            patch.object(
                torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)
            ) as op,
        ):
            torch.testing.assert_close(model(self.q, self.q, self.q, softmax_scale=0.5), self.q)
            self.assertEqual(op.call_args.kwargs["softmax_scale"], 0.5)
            op.assert_called_once()

    def test_fp8_bsnd_gqa_and_output_crop(self):
        q = self.q.transpose(1, 2)
        kv = q[:, :, :1]

        def quantize(tensor, **kwargs):
            return tensor.transpose(1, 2), torch.ones(1)

        with (
            patch.object(fp8, "_fa_block_quant_preprocess", side_effect=quantize),
            patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True) as execute,
        ):
            execute.return_value = (torch.zeros(1, 2, 32, 64),)
            out = quant_attention(q, kv, kv, layout="BSND", scale=0.5)
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(execute.call_args.kwargs["num_key_value_heads"], 1)
            self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.5)

    def test_fp8_modes_keep_quantization_and_kernel_in_sync(self):
        for mode, v_block, v_mode in (
            (None, 256, 7),
            ("HIGH_PRECISION", 256, 7),
            (FP8FAMode.HIGH_PRECISION, 256, 7),
            ("c8v16_tiling512", 512, 12),
            (FP8FAMode.C8V16_TILING512, 512, 12),
        ):
            with (
                self.subTest(mode=mode),
                patch.object(
                    fp8,
                    "_fa_block_quant_preprocess",
                    side_effect=lambda tensor, **kw: (tensor, torch.ones(1)),
                ) as quantize,
                patch.object(torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True) as execute,
            ):
                execute.return_value = (self.q,)
                quant_attention(self.q, self.q, self.q, fp8_fa_mode=mode)
                self.assertEqual(quantize.call_args_list[2].kwargs["block_size"], v_block)
                self.assertEqual(execute.call_args.kwargs["value_quant_mode"], v_mode)

    def test_legacy_fp8_ignores_unconsumed_options(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        weights = SimpleNamespace(keys=lambda: ("a.q_rot", "a.k_rot"), get_tensor=lambda _: torch.eye(64))
        model = layer.FP8RotateQuantFA(prefix="a", weights=weights)
        with (
            patch.object(fp8, "_fa_block_quant_preprocess", side_effect=lambda t, **kw: (t, torch.ones(1))),
            patch.object(
                torch.ops.mindiesd, "fused_infer_attention_score_v2", create=True, return_value=(self.q,)
            ) as execute,
        ):
            result = model(
                self.q,
                self.q,
                self.q,
                metadata=object(),
                softmax_scale=0.75,
                precision="unused",
                q_rot="unused",
                fp8_fa_mode="unused",
            )
            torch.testing.assert_close(result, self.q)
            # The legacy class only consumed layout at forward time.
            self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.125)

    def test_legacy_fp8_delegates_to_public_entry(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        weights = SimpleNamespace(keys=lambda: ("a.q_rot", "a.k_rot"), get_tensor=lambda _: torch.eye(64))
        model = layer.FP8RotateQuantFA(prefix="a", weights=weights)
        with patch.object(layer, "quant_attention", return_value=self.q) as execute:
            self.assertIs(model(self.q, self.q, self.q), self.q)
            self.assertEqual(execute.call_args.kwargs["precision"], "fp8")
            self.assertIs(execute.call_args.kwargs["q_rot"], model.q_rot)
            self.assertIs(execute.call_args.kwargs["k_rot"], model.k_rot)


class TestQuantAttentionMxfp4(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 2, 17, 64)

    def test_mxfp4_query_scales_are_head_first_for_both_input_layouts(self):
        for layout, paired in (("BSND", False), ("BNSD", False), ("BSND", True), ("BNSD", True)):
            with self.subTest(layout=layout, paired=paired):
                query = torch.zeros(1, 17, 2, 64)
                if layout == "BNSD":
                    query = query.transpose(1, 2).contiguous()
                scales = []

                def quantize(tensor, **kwargs):
                    # Distinct per-head/per-token scale values expose a swap
                    # that an all-ones scale fixture cannot detect.
                    scale_shape = (*tensor.shape[:-1], 1, 2) if paired else (*tensor.shape[:-1], 2)
                    scale = torch.arange(tensor.numel() // 32).reshape(scale_shape)
                    scales.append(scale)
                    return tensor, scale

                with (
                    patch.object(torch_npu, "npu_dynamic_mx_quant", side_effect=quantize, create=True),
                    patch.object(mxfp4, "_reshape_mxfp4_v_scale_for_fa", side_effect=lambda s, layout: s),
                    patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True) as execute,
                ):
                    execute.side_effect = lambda q, *args, **kwargs: (torch.zeros_like(q), None)
                    quant_attention(query, query, query, precision="mxfp4", layout=layout, metadata=object())
                    expected = scales[0].transpose(1, 2).contiguous() if layout == "BSND" else scales[0]
                    torch.testing.assert_close(execute.call_args.args[3], expected)
                    self.assertTrue(execute.call_args.args[3].is_contiguous())
                    # K follows layout_kv; it must not acquire Q's special layout.
                    self.assertIs(execute.call_args.args[4], scales[1])

    def test_mxfp4_explicit_kv_heads_must_match_tensor(self):
        with patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quant:
            for heads in (1, True, 2.0):
                with self.subTest(heads=heads), self.assertRaisesRegex(ValueError, "num_key_value_heads"):
                    quant_attention(self.q, self.q, self.q, precision="mxfp4", num_key_value_heads=heads)
            quant.assert_not_called()

    def test_options_are_rejected_before_rotation_or_quantization(self):
        with (
            patch.object(torch, "matmul") as rotate,
            patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quant,
        ):
            for options in ({"fp8_fa_mode": None}, {"softmax_scale": 0.5}, {"unknown": 1}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, next(iter(options))):
                    quant_attention(self.q, self.q, self.q, precision="mxfp4", q_rot=torch.eye(64), **options)
            rotate.assert_not_called()
            quant.assert_not_called()

    def test_module_import_without_optional_mx_dtypes(self):
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            spec = importlib.util.spec_from_file_location(mxfp4.__name__ + "_probe", mxfp4.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module._mxfp4_attention_forward))

    def test_mxfp4_padding_axes_metadata_and_output_layout(self):
        metadata = object()
        kv = torch.randn(1, 23, 1, 64)
        recorded = []

        def quantize(tensor, **kwargs):
            recorded.append((tensor.shape, kwargs))
            return tensor, torch.ones(1)

        with (
            patch.object(torch_npu, "npu_dynamic_mx_quant", side_effect=quantize, create=True),
            patch.object(mxfp4, "_reshape_mxfp4_v_scale_for_fa", side_effect=lambda scale, layout: scale),
            patch.object(torch.ops.mindiesd, "quant_flash_attn", create=True) as execute,
        ):
            execute.return_value = (torch.zeros(1, 512, 2, 64), None)
            out = quant_attention(
                self.q,
                kv,
                kv,
                precision="mxfp4",
                layout_kv="BSND",
                layout_out="BSND",
                metadata=metadata,
                mxfp4_scale_alg=2,
                mxfp4_dst_type_max=7.25,
                scale=0.0,
                win_left=0,
                win_right=7,
                return_softmax_lse=1,
            )
            self.assertEqual(out.shape, (1, 17, 2, 64))
            self.assertEqual([item[1]["axis"] for item in recorded], [-1, -1, 1])
            self.assertEqual(recorded[0][0], (1, 2, 512, 64))
            self.assertEqual(recorded[1][0], (1, 512, 1, 64))
            self.assertTrue(all(item[1]["dst_type_max"] == 7.25 for item in recorded))
            self.assertIs(execute.call_args.kwargs["metadata"], metadata)
            self.assertEqual(execute.call_args.kwargs["return_softmax_lse"], 1)
            self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.0)
            self.assertEqual((execute.call_args.kwargs["win_left"], execute.call_args.kwargs["win_right"]), (0, 7))

    def test_rotation_scale_packing_and_metadata_contract(self):
        for layout in ("BNSD", "BSND"):
            for paired in (False, True):
                q = self.q.expand(2, -1, -1, -1)
                if layout == "BSND":
                    q = q.transpose(1, 2)
                scales = []

                def quantize(tensor, *, axis, dst_type):
                    shape = list(tensor.shape)
                    shape[axis] //= 32
                    scale = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
                    if paired and axis != -1:
                        scale = mxfp4._reshape_mxfp4_v_scale_for_fa(scale, layout)
                    scales.append(scale)
                    return tensor, scale

                with (
                    self.subTest(layout=layout, paired=paired),
                    patch.object(torch_npu, "npu_dynamic_mx_quant", side_effect=quantize, create=True) as quant,
                    patch.object(
                        torch.ops.mindiesd, "quant_flash_attn_metadata", create=True, return_value=torch.ones(1)
                    ) as metadata,
                    patch.object(
                        torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=lambda t, *a, **kw: (t, None)
                    ) as execute,
                ):
                    output = quant_attention(
                        q, q, q, precision="mxfp4", layout=layout, q_rot=2 * torch.eye(64), pre_tokens=13, next_tokens=0
                    )
                    torch.testing.assert_close(output, 2 * q)
                    opts = execute.call_args.kwargs
                    for name in ("q", "k", "v"):
                        self.assertEqual(opts[name + "_quant_mode"], 3)
                        self.assertEqual(opts[name + "_dtype"], torch_npu.float4_e2m1fn_x2)
                        self.assertEqual(opts[name + "_descale_dtype"], torch_npu.float8_e8m0fnu)
                    self.assertEqual(
                        [c.kwargs["axis"] for c in quant.call_args_list], [-1, -1, 2 if layout == "BNSD" else 1]
                    )
                    self.assertEqual(opts["seqused_q"].dtype, torch.int32)
                    self.assertEqual((opts["win_left"], opts["win_right"]), (13, 0))
                    for name in ("layout_q", "layout_kv", "layout_out", "mask_mode", "win_left", "win_right"):
                        self.assertEqual(opts[name], metadata.call_args.kwargs[name])
                    self.assertIs(opts["seqused_q"], metadata.call_args.kwargs["seqused_q"])
                    v_scale = execute.call_args.args[5]
                    expected = (2, 2, 8, 64, 2) if layout == "BNSD" else (2, 8, 2, 64, 2)
                    self.assertEqual(v_scale.shape, expected)
                    if paired:
                        self.assertIs(v_scale, scales[2])
                    else:
                        # The paired axis holds adjacent sequence-block scales.
                        for block in (0, 3):
                            for pair in (0, 1):
                                actual = (
                                    v_scale[:, :, block, :, pair] if layout == "BNSD" else v_scale[:, block, :, :, pair]
                                )
                                original = (
                                    scales[2][:, :, 2 * block + pair, :]
                                    if layout == "BNSD"
                                    else scales[2][:, 2 * block + pair, :, :]
                                )
                                torch.testing.assert_close(actual, original)

    def test_invalid_options_fail_before_quantization(self):
        for options in (
            {"q_rot": torch.ones(64, 32)},
            {"k_rot": torch.eye(64, device="meta")},
            {"layout_out": "TND"},
        ):
            with self.subTest(options=options), patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quant:
                with self.assertRaises(ValueError):
                    quant_attention(self.q, self.q, self.q, precision="mxfp4", **options)
                quant.assert_not_called()
        with self.assertRaisesRegex(TypeError, "metadata"):
            quant_attention(self.q, self.q, self.q, precision="fp8", metadata=object())

    def test_native_error_propagates_without_retry(self):
        with (
            patch.object(
                torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
            ),
            patch.object(mxfp4, "_reshape_mxfp4_v_scale_for_fa", side_effect=lambda s, layout: s),
            patch.object(
                torch.ops.mindiesd, "quant_flash_attn", create=True, side_effect=RuntimeError("native failure")
            ) as execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                quant_attention(self.q, self.q, self.q, precision="mxfp4", metadata=object())
            execute.assert_called_once()


class TestQuantAttentionMxfp8(unittest.TestCase):
    def setUp(self):
        self.q = torch.randn(1, 2, 17, 64)

    def test_unequal_lengths_and_heads_preserve_tnd_inputs_and_query_output(self):
        cases = product(("BNSD", "BSND"), (1, 2), ((17, 8, 2, 2), (8, 17, 4, 2), (128, 256, 4, 2)))
        for layout, batch, (q_len, kv_len, q_heads, kv_heads) in cases:
            q = torch.randn(batch, q_heads, q_len, 64)
            k = torch.randn(batch, kv_heads, kv_len, 64) + 10
            v = torch.randn_like(k) + 20
            packed_q = (2 * q).transpose(1, 2).reshape(batch * q_len, q_heads, 64)
            packed_k = k.transpose(1, 2).reshape(batch * kv_len, kv_heads, 64)
            packed_v = v.transpose(1, 2).reshape(batch * kv_len, kv_heads, 64)
            if layout == "BSND":
                q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
            with (
                self.subTest(layout=layout, batch=batch, lengths=(q_len, kv_len), heads=(q_heads, kv_heads)),
                patch.object(
                    torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
                ) as quantize,
                patch.object(
                    torch_npu,
                    "npu_fused_infer_attention_score_v2",
                    create=True,
                    side_effect=lambda t, *args, **kw: (t + 1,),
                ) as execute,
            ):
                output = quant_attention(q, k, v, precision="mxfp8", layout=layout, q_rot=2 * torch.eye(64))
                torch.testing.assert_close(output, 2 * q + 1)
                for call, packed, axis in zip(quantize.call_args_list, (packed_q, packed_k, packed_v), (-1, -1, 0)):
                    torch.testing.assert_close(call.args[0], packed)
                    self.assertEqual(call.kwargs["axis"], axis)
                for actual, expected in zip(execute.call_args.args, (packed_q, packed_k, packed_v)):
                    torch.testing.assert_close(actual, expected)
                options = execute.call_args.kwargs
                self.assertEqual(options["input_layout"], "TND")
                self.assertEqual(options["num_query_heads"], q_heads)
                self.assertEqual(options["num_key_value_heads"], kv_heads)
                self.assertEqual(options["actual_seq_qlen"], [q_len] if batch == 1 else [q_len, 2 * q_len])
                self.assertEqual(options["actual_seq_kvlen"], [kv_len] if batch == 1 else [kv_len, 2 * kv_len])

    def test_options_are_rejected_before_rotation_or_quantization(self):
        with (
            patch.object(torch, "matmul") as rotate,
            patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quant,
        ):
            for options in ({"fp8_fa_mode": None}, {"softmax_scale": 0.5}, {"mxfp4_scale_alg": 2}, {"unknown": 1}):
                with self.subTest(options=options), self.assertRaisesRegex(TypeError, next(iter(options))):
                    quant_attention(self.q, self.q, self.q, precision="mxfp8", q_rot=torch.eye(64), **options)
            rotate.assert_not_called()
            quant.assert_not_called()

    def test_module_import_without_optional_mx_dtypes(self):
        impl = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")
        with patch.dict(sys.modules, {"torch_npu": ModuleType("torch_npu")}):
            spec = importlib.util.spec_from_file_location(impl.__name__ + "_probe", impl.__file__)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertTrue(callable(module._mxfp8_attention_forward))

    def test_mxfp8_tnd_sequence_metadata(self):
        q = self.q.expand(2, -1, -1, -1)
        with (
            patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
            patch.object(torch_npu, "npu_fused_infer_attention_score_v2", create=True) as execute,
        ):
            quantize.side_effect = lambda tensor, **kwargs: (tensor, torch.ones(1))
            execute.side_effect = lambda query, *args, **kwargs: (query,)
            out = quant_attention(q, q, q, precision="mxfp8")
            self.assertEqual(out.shape, q.shape)
            self.assertEqual(execute.call_args.kwargs["input_layout"], "TND")
            self.assertEqual(execute.call_args.kwargs["actual_seq_qlen"], [17, 34])
            self.assertEqual(execute.call_args.kwargs["value_quant_mode"], 8)

    def test_layout_rotation_scales_and_dtype_at_operator_boundary(self):
        for layout in ("BNSD", "BSND"):
            q = self.q.expand(2, -1, -1, -1)
            if layout == "BSND":
                q = q.transpose(1, 2)
            scales = [torch.tensor([index]) for index in range(3)]
            with (
                self.subTest(layout=layout),
                patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
                patch.object(torch_npu, "npu_fused_infer_attention_score_v2", create=True) as execute,
            ):
                quantize.side_effect = lambda t, **kw: (t, scales[quantize.call_count - 1])
                execute.side_effect = lambda t, *args, **kw: (t,)
                output = quant_attention(
                    q,
                    q,
                    q,
                    precision="mxfp8",
                    layout=layout,
                    k_rot=2 * torch.eye(64),
                    scale=0.5,
                    pre_tokens=7,
                    next_tokens=0,
                )
                torch.testing.assert_close(output, q)
                packed = (q.transpose(1, 2) if layout == "BNSD" else q).reshape(34, 2, 64)
                for call, factor, axis in zip(quantize.call_args_list, (1, 2, 1), (-1, -1, 0)):
                    torch.testing.assert_close(call.args[0], factor * packed)
                    self.assertEqual(call.kwargs, {"axis": axis, "dst_type": torch.float8_e4m3fn})
                options = execute.call_args.kwargs
                for tensor, mode, descale in zip(("query", "key", "value"), (6, 6, 8), scales):
                    self.assertEqual(options[tensor + "_quant_mode"], mode)
                    self.assertIs(options["dequant_scale_" + tensor], descale)
                    self.assertEqual(options["dequant_scale_" + tensor + "_dtype"], torch_npu.float8_e8m0fnu)
                    self.assertEqual(options[tensor + "_dtype"], torch.float8_e4m3fn)
                self.assertEqual(options["actual_seq_kvlen"], [17, 34])
                self.assertEqual(options["out_dtype"], q.dtype)
                self.assertEqual((options["softmax_scale"], options["pre_tokens"], options["next_tokens"]), (0.5, 7, 0))

    def test_invalid_rotation_fails_before_quantization(self):
        for options in (
            {"q_rot": torch.ones(64, 32)},
            {"k_rot": torch.eye(64, device="meta")},
        ):
            with (
                self.subTest(options=options),
                patch.object(torch_npu, "npu_dynamic_mx_quant", create=True) as quantize,
            ):
                with self.assertRaises(ValueError):
                    quant_attention(self.q, self.q, self.q, precision="mxfp8", **options)
                quantize.assert_not_called()

    def test_operator_exception_is_not_retried(self):
        with (
            patch.object(
                torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
            ),
            patch.object(
                torch_npu, "npu_fused_infer_attention_score_v2", create=True, side_effect=RuntimeError("native failure")
            ) as execute,
        ):
            with self.assertRaisesRegex(RuntimeError, "native failure"):
                quant_attention(self.q, self.q, self.q, precision="mxfp8")
            execute.assert_called_once()

    def test_legacy_class_delegates_and_preserves_historical_options(self):
        layer = importlib.import_module("mindiesd.quantization.layer")
        weights = SimpleNamespace(keys=lambda: ("a.q_rot", "a.k_rot"), get_tensor=lambda _: torch.eye(64))
        model = layer.MXFP8RotateQuantFA(prefix="a", weights=weights)
        with (
            patch.object(layer, "quant_attention", wraps=quant_attention) as entry,
            (
                patch.object(
                    torch_npu, "npu_dynamic_mx_quant", create=True, side_effect=lambda t, **kw: (t, torch.ones(1))
                )
            ),
            patch.object(
                torch_npu, "npu_fused_infer_attention_score_v2", create=True, side_effect=lambda t, *a, **kw: (t,)
            ) as execute,
        ):
            output = model(self.q, self.q, self.q, softmax_scale=0.75, metadata=object(), precision="unused")
            torch.testing.assert_close(output, self.q)
            self.assertEqual(entry.call_args.kwargs["precision"], "mxfp8")
            self.assertIs(entry.call_args.kwargs["q_rot"], model.q_rot)
            self.assertEqual(execute.call_args.kwargs["softmax_scale"], 0.125)


class TestQuantAttentionMxfp8Npu(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "npu"), "Requires TorchNPU on an A5 device.")
    def test_real_npu_unequal_lengths_and_gqa(self):
        if not torch.npu.is_available() or not is_a5_device():
            self.skipTest("Requires an available A5 device.")
        # Constant V per batch/head has a known attention result for any Q/K.
        # Distinct values detect batch/head mixing without a quantization-error baseline.
        head_values = torch.tensor([[1.0, -1.0], [0.5, -0.5]], device="npu", dtype=torch.bfloat16)
        for layout, (q_len, kv_len) in product(("BNSD", "BSND"), ((128, 256), (256, 128))):
            with self.subTest(layout=layout, lengths=(q_len, kv_len)):
                q = torch.randn(2, 4, q_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                k = torch.randn(2, 2, kv_len, 64, device="npu", dtype=torch.bfloat16) * 0.1
                v = head_values.reshape(2, 2, 1, 1).expand(2, 2, kv_len, 64).contiguous()
                expected = head_values.repeat_interleave(2, dim=1).reshape(2, 4, 1, 1).expand_as(q)
                if layout == "BSND":
                    q, k, v = (tensor.transpose(1, 2).contiguous() for tensor in (q, k, v))
                    expected = expected.transpose(1, 2)
                output = quant_attention(q, k, v, precision="mxfp8", layout=layout)
                actual = output.cpu()
                self.assertTrue(torch.isfinite(actual).all().item())
                torch.testing.assert_close(actual, expected.cpu(), rtol=0.02, atol=0.02)


if __name__ == "__main__":
    unittest.main()
