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

"""CPU tests for the quantized attention API, FP8 execution and legacy compatibility."""

import importlib
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import torch
import torch_npu
from mindiesd import quant_attention
from mindiesd.quantization.mode import FP8FAMode
from mindiesd.utils.exception import ParametersInvalid

fp8 = importlib.import_module("mindiesd.layers.flash_attn.fused_infer_attention_score")


class TestQuantAttentionForward(unittest.TestCase):
    def test_entry_import_without_fp8_dtype(self):
        modules = (
            "mindiesd.layers.quant.block_quant",
            "mindiesd.layers.flash_attn.fused_infer_attention_score",
            "mindiesd.layers.flash_attn.quant_flash_attn",
        )
        root = Path(__file__).resolve().parents[2]
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
        with self.assertRaisesRegex(ValueError, "q_rot.*shape"):
            quant_attention(self.q, self.q, self.q, q_rot=torch.ones(64, 32))

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


if __name__ == "__main__":
    unittest.main()
