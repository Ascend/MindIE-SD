#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib
import inspect
import os
import re
import unittest
from unittest import mock

import torch

from mindiesd import (
    split_qkv_norm_rope_concat as top_level_split_qkv_norm_rope_concat,
    triton_split_qkv_norm_rope_concat as top_level_triton_split_qkv_norm_rope_concat,
    validate_split_qkv_norm_rope_concat_inputs as top_level_validate_inputs,
)
from mindiesd.layers import (
    split_qkv_norm_rope_concat,
    validate_split_qkv_norm_rope_concat_inputs,
)
from mindiesd.layers.split_qkv_norm_rope_concat import (
    triton_split_qkv_norm_rope_concat,
    validate_inputs,
)
from mindiesd.utils import ParametersInvalid


operator_module = importlib.import_module("mindiesd.layers.split_qkv_norm_rope_concat")


def _make_inputs(
    tokens=3,
    num_heads=2,
    head_dim=128,
    rope_dim=96,
    dtype=torch.bfloat16,
    max_position=16,
):
    generator = torch.Generator(device="cpu").manual_seed(20261029)
    qkv = torch.randn(tokens, 3 * num_heads * head_dim, generator=generator, dtype=dtype)
    q_weight = torch.ones(head_dim, dtype=dtype)
    k_weight = torch.ones(head_dim, dtype=dtype)
    cache = torch.randn(max_position, rope_dim, generator=generator, dtype=dtype)
    positions = torch.arange(tokens, dtype=torch.int32)
    return qkv, q_weight, k_weight, cache, positions


def _reference(qkv, q_weight, k_weight, cache, positions, num_heads, head_dim, rope_dim, eps):
    tokens = qkv.shape[0]
    q, k, v = torch.split(qkv, num_heads * head_dim, dim=-1)
    q = q.reshape(tokens, num_heads, head_dim).float()
    k = k.reshape(tokens, num_heads, head_dim).float()
    q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + eps)
    k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + eps)
    q = (q * q_weight.float()).to(qkv.dtype)
    k = (k * k_weight.float()).to(qkv.dtype)

    selected = cache.index_select(0, positions.long()).float()
    half = rope_dim // 2
    cos = selected[:, :half].unsqueeze(1)
    sin = selected[:, half:].unsqueeze(1)
    q_rope, k_rope = q.float(), k.float()
    q1, q2 = q_rope[..., :half], q_rope[..., half:rope_dim]
    k1, k2 = k_rope[..., :half], k_rope[..., half:rope_dim]
    q_out = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin, q_rope[..., rope_dim:]), dim=-1)
    k_out = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin, k_rope[..., rope_dim:]), dim=-1)
    return q_out.to(qkv.dtype), k_out.to(qkv.dtype), v.reshape(tokens, num_heads, head_dim)


class _RecordingKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _record(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return _record


def _fake_launch(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    *,
    num_heads,
    head_dim,
    rope_dim,
    eps,
    v_out,
):
    del q_weight, k_weight, cos_sin_cache, positions, rope_dim, eps
    shape = (qkv.shape[0], num_heads, head_dim)
    q_out = torch.zeros(shape, dtype=qkv.dtype, device=qkv.device)
    k_out = torch.ones(shape, dtype=qkv.dtype, device=qkv.device)
    if v_out is not None:
        v_out.fill_(2)
    return q_out, k_out, v_out


class TestSplitQkvNormRopeConcatValidation(unittest.TestCase):
    def setUp(self):
        self.inputs = _make_inputs()
        self.args = {
            "num_heads": 2,
            "head_dim": 128,
            "rope_dim": 96,
        }

    def test_public_api_is_exported(self):
        self.assertTrue(callable(split_qkv_norm_rope_concat))
        self.assertIs(top_level_split_qkv_norm_rope_concat, split_qkv_norm_rope_concat)
        self.assertIs(top_level_triton_split_qkv_norm_rope_concat, split_qkv_norm_rope_concat)
        self.assertIs(top_level_validate_inputs, validate_inputs)
        self.assertIs(validate_split_qkv_norm_rope_concat_inputs, validate_inputs)
        self.assertIs(triton_split_qkv_norm_rope_concat, split_qkv_norm_rope_concat)

    def test_valid_contract(self):
        validate_inputs(*self.inputs, **self.args)

    def test_rejects_invalid_metadata(self):
        cases = (
            ({"num_heads": 0}, "num_heads"),
            ({"head_dim": 95}, "head_dim"),
            ({"rope_dim": 95}, "rope_dim"),
            ({"rope_dim": 130}, "rope_dim"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                args = {**self.args, **overrides}
                with self.assertRaisesRegex(ParametersInvalid, message):
                    validate_inputs(*self.inputs, **args)

    def test_rejects_noncontiguous_and_bad_dtype(self):
        qkv, q_weight, k_weight, cache, positions = self.inputs
        with self.assertRaisesRegex(ParametersInvalid, "contiguous"):
            validate_inputs(qkv.T, q_weight, k_weight, cache, positions, **self.args)
        with self.assertRaisesRegex(ParametersInvalid, "dtype"):
            validate_inputs(qkv.float(), q_weight, k_weight, cache, positions, **self.args)

    def test_rejects_invalid_positions_and_nonfinite_values(self):
        qkv, q_weight, k_weight, cache, _ = self.inputs
        bad_positions = torch.full((qkv.shape[0],), cache.shape[0], dtype=torch.int32)
        with self.assertRaisesRegex(ParametersInvalid, "positions"):
            validate_inputs(qkv, q_weight, k_weight, cache, bad_positions, **self.args)

        bad_qkv = qkv.clone()
        bad_qkv[0, 0] = float("nan")
        positions = torch.arange(qkv.shape[0], dtype=torch.int32)
        with self.assertRaisesRegex(ParametersInvalid, "NaN or Inf"):
            validate_inputs(bad_qkv, q_weight, k_weight, cache, positions, **self.args)

    def test_public_default_is_metadata_only(self):
        parameter = inspect.signature(split_qkv_norm_rope_concat).parameters["validate_values"]
        self.assertIs(parameter.default, False)

    def test_public_validate_values_rejects_invalid_positions(self):
        qkv, q_weight, k_weight, cache, _ = self.inputs
        bad_positions = torch.full((qkv.shape[0],), cache.shape[0], dtype=torch.int32)
        with self.assertRaisesRegex(ParametersInvalid, "positions"):
            split_qkv_norm_rope_concat(
                qkv,
                q_weight,
                k_weight,
                cache,
                bad_positions,
                **self.args,
                validate_values=True,
            )

    def test_metadata_error_paths(self):
        qkv, q_weight, k_weight, cache, positions = self.inputs
        cases = (
            ("qkv type", {"qkv": object()}, "qkv must be torch.Tensor"),
            ("qkv rank", {"qkv": qkv.unsqueeze(0)}, "qkv must be 2-D"),
            ("bool heads", {"num_heads": True}, "num_heads"),
            ("head_dim type", {"head_dim": 128.0}, "head_dim must be an integer"),
            ("head_dim alignment", {"head_dim": 100}, "divisible by 8"),
            ("rope_dim type", {"rope_dim": 96.0}, "rope_dim must be an integer"),
            ("eps", {"eps": 0.0}, "eps must be a finite positive number"),
            (
                "too many tokens",
                {"qkv": torch.empty((262145, 3 * 2 * 128), dtype=torch.bfloat16, device="meta")},
                "num_tokens",
            ),
            ("qkv width", {"qkv": torch.empty((3, 3 * 2 * 128 - 8), dtype=torch.bfloat16)}, "qkv width"),
            ("q_weight missing", {"q_weight": None}, "q_weight is required"),
            ("k_weight missing", {"k_weight": None}, "k_weight is required"),
            ("q_weight type", {"q_weight": 1}, "q_weight must be torch.Tensor"),
            ("q_weight shape", {"q_weight": torch.ones(127, dtype=torch.bfloat16)}, "shape [head_dim]"),
            ("q_weight dtype", {"q_weight": torch.ones(128, dtype=torch.float32)}, "weight dtype"),
            (
                "q_weight contiguous",
                {"q_weight": torch.ones((128, 2), dtype=torch.bfloat16)[:, 0]},
                "weights must be contiguous",
            ),
            (
                "q_weight device",
                {"q_weight": torch.ones(128, dtype=torch.bfloat16, device="meta")},
                "same device",
            ),
            ("cache type", {"cos_sin_cache": object()}, "cos_sin_cache must be torch.Tensor"),
            (
                "cache shape",
                {"cos_sin_cache": torch.empty((16, 94), dtype=torch.bfloat16)},
                "shape [maxPos, rope_dim]",
            ),
            (
                "cache dtype",
                {"cos_sin_cache": torch.empty((16, 96), dtype=torch.float32)},
                "cache dtype",
            ),
            (
                "cache contiguous",
                {"cos_sin_cache": torch.empty((16, 192), dtype=torch.bfloat16)[:, ::2]},
                "cache dtype",
            ),
            (
                "cache device",
                {"cos_sin_cache": torch.empty((16, 96), dtype=torch.bfloat16, device="meta")},
                "same device",
            ),
            ("positions type", {"positions": object()}, "positions must be torch.Tensor"),
            (
                "positions shape",
                {"positions": torch.empty(2, dtype=torch.int32)},
                "positions must have shape",
            ),
            (
                "positions dtype",
                {"positions": torch.empty(3, dtype=torch.int64)},
                "contiguous int32",
            ),
            (
                "positions contiguous",
                {"positions": torch.arange(6, dtype=torch.int32).reshape(3, 2)[:, 0]},
                "contiguous int32",
            ),
            (
                "positions device",
                {"positions": torch.empty(3, dtype=torch.int32, device="meta")},
                "same device",
            ),
            ("out dtype", {"out_dtype": torch.float32}, "out_dtype"),
            ("v_out type", {"v_out": object()}, "v_out must be torch.Tensor"),
            (
                "v_out shape",
                {"v_out": torch.empty((3, 2, 127), dtype=torch.bfloat16)},
                "shape [T, N, D]",
            ),
            (
                "v_out dtype",
                {"v_out": torch.empty((3, 2, 128), dtype=torch.float32)},
                "v_out dtype",
            ),
            (
                "v_out contiguous",
                {"v_out": torch.empty((3, 2, 256), dtype=torch.bfloat16)[..., ::2]},
                "v_out must be contiguous",
            ),
            (
                "v_out device",
                {"v_out": torch.empty((3, 2, 128), dtype=torch.bfloat16, device="meta")},
                "v_out device",
            ),
        )
        defaults = {
            "qkv": qkv,
            "q_weight": q_weight,
            "k_weight": k_weight,
            "cos_sin_cache": cache,
            "positions": positions,
            **self.args,
        }
        for name, overrides, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ParametersInvalid, re.escape(message)):
                    validate_inputs(**{**defaults, **overrides})

        valid_v_out = torch.empty((3, 2, 128), dtype=qkv.dtype)
        validate_inputs(**defaults, v_out=valid_v_out, out_dtype=qkv.dtype)

    def test_value_checks_cover_negative_and_empty_positions(self):
        qkv, q_weight, k_weight, cache, _ = self.inputs
        negative_positions = torch.tensor([-1, 0, 1], dtype=torch.int32)
        with self.assertRaisesRegex(ParametersInvalid, "0 <= p < maxPos"):
            validate_inputs(qkv, q_weight, k_weight, cache, negative_positions, **self.args)

        empty_qkv, empty_q_weight, empty_k_weight, empty_cache, _ = _make_inputs(tokens=0)
        empty_positions = torch.empty(0, dtype=torch.int32)
        validate_inputs(
            empty_qkv,
            empty_q_weight,
            empty_k_weight,
            empty_cache,
            empty_positions,
            **self.args,
        )


class TestSplitQkvNormRopeConcatDispatch(unittest.TestCase):
    def setUp(self):
        self.inputs = _make_inputs()
        self.args = {
            "num_heads": 2,
            "head_dim": 128,
            "rope_dim": 96,
        }

    def test_launch_schedules_kernel_chunks(self):
        qkv, q_weight, k_weight, cache, positions = self.inputs
        kernel = _RecordingKernel()
        with (
            mock.patch.object(operator_module, "_split_qkv_norm_rope_kernel", kernel),
            mock.patch.object(operator_module, "_MAX_GRID", 2),
        ):
            q_out, k_out, v_out = operator_module._launch(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                num_heads=17,
                head_dim=128,
                rope_dim=96,
                eps=1e-5,
                v_out=None,
            )

        self.assertEqual(tuple(q_out.shape), (3, 17, 128))
        self.assertEqual(tuple(k_out.shape), (3, 17, 128))
        self.assertIsNone(v_out)
        self.assertEqual([call[0] for call in kernel.calls], [(2,), (1,)])
        self.assertEqual([call[1][-1] for call in kernel.calls], [0, 2])
        self.assertFalse(kernel.calls[0][2]["WRITE_V"])
        self.assertEqual(kernel.calls[0][2]["BLOCK_HEADS"], 16)

        supplied_v = torch.empty((3, 4, 256), dtype=qkv.dtype)
        with mock.patch.object(operator_module, "_split_qkv_norm_rope_kernel", kernel):
            q_out, k_out, returned_v = operator_module._launch(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                num_heads=4,
                head_dim=256,
                rope_dim=256,
                eps=1e-5,
                v_out=supplied_v,
            )
        self.assertEqual(tuple(q_out.shape), (3, 4, 256))
        self.assertEqual(tuple(k_out.shape), (3, 4, 256))
        self.assertIs(returned_v, supplied_v)
        self.assertTrue(kernel.calls[-1][2]["WRITE_V"])
        self.assertEqual(kernel.calls[-1][2]["BLOCK_HEADS"], 8)
        self.assertFalse(kernel.calls[-1][2]["HAS_TAIL"])

    def test_launch_empty_input_does_not_launch_kernel(self):
        qkv, q_weight, k_weight, cache, positions = _make_inputs(tokens=0)
        kernel = _RecordingKernel()
        with mock.patch.object(operator_module, "_split_qkv_norm_rope_kernel", kernel):
            q_out, k_out, v_out = operator_module._launch(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                num_heads=2,
                head_dim=128,
                rope_dim=96,
                eps=1e-5,
                v_out=None,
            )
        self.assertEqual(tuple(q_out.shape), (0, 2, 128))
        self.assertEqual(tuple(k_out.shape), (0, 2, 128))
        self.assertIsNone(v_out)
        self.assertEqual(kernel.calls, [])

    def test_next_power_of_two(self):
        self.assertEqual(operator_module._next_power_of_two(0), 1)
        self.assertEqual(operator_module._next_power_of_two(1), 1)
        self.assertEqual(operator_module._next_power_of_two(2), 2)
        self.assertEqual(operator_module._next_power_of_two(3), 4)
        self.assertEqual(operator_module._next_power_of_two(5), 8)

    def test_runtime_guard(self):
        with mock.patch.object(operator_module, "_TRITON_ON_ASCEND", True):
            operator_module._require_triton_ascend()
        with (
            mock.patch.object(operator_module, "_TRITON_ON_ASCEND", False),
            self.assertRaisesRegex(RuntimeError, "Ascend NPU with Triton-Ascend"),
        ):
            operator_module._require_triton_ascend()

    def test_custom_op_implementations_delegate_to_launch(self):
        qkv, q_weight, k_weight, cache, positions = self.inputs
        with (
            mock.patch.object(operator_module, "_require_triton_ascend"),
            mock.patch.object(operator_module, "_launch", side_effect=_fake_launch),
        ):
            q_out, k_out, v_out = operator_module._split_qkv_norm_rope_concat_op._opoverload(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                2,
                128,
                96,
                1e-5,
                False,
            )
            self.assertEqual(tuple(q_out.shape), (3, 2, 128))
            self.assertEqual(tuple(k_out.shape), (3, 2, 128))
            self.assertEqual(tuple(v_out.shape), (3, 2, 128))

            no_v_q, no_v_k = operator_module._split_qkv_norm_rope_concat_no_v_op._opoverload(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                2,
                128,
                96,
                1e-5,
                False,
            )
            self.assertEqual(tuple(no_v_q.shape), (3, 2, 128))
            self.assertEqual(tuple(no_v_k.shape), (3, 2, 128))

            supplied_v = torch.empty((3, 2, 128), dtype=qkv.dtype)
            out_q, out_k = operator_module._split_qkv_norm_rope_concat_out_op._opoverload(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                2,
                128,
                96,
                1e-5,
                False,
                supplied_v,
            )
            self.assertEqual(tuple(out_q.shape), (3, 2, 128))
            self.assertEqual(tuple(out_k.shape), (3, 2, 128))
            self.assertTrue(torch.all(supplied_v == 2))

    def test_public_dispatch_paths(self):
        qkv, q_weight, k_weight, cache, positions = self.inputs
        with (
            mock.patch.object(operator_module, "_require_triton_ascend"),
            mock.patch.object(operator_module, "_launch", side_effect=_fake_launch),
        ):
            q_out, k_out, v_out = split_qkv_norm_rope_concat(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                **self.args,
            )
            self.assertEqual(tuple(q_out.shape), (3, 2, 128))
            self.assertEqual(tuple(k_out.shape), (3, 2, 128))
            self.assertIsNone(v_out)

            supplied_v = torch.empty((3, 2, 128), dtype=qkv.dtype)
            q_out, k_out, returned_v = split_qkv_norm_rope_concat(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                **self.args,
                v_out=supplied_v,
            )
            self.assertEqual(tuple(q_out.shape), (3, 2, 128))
            self.assertEqual(tuple(k_out.shape), (3, 2, 128))
            self.assertIs(returned_v, supplied_v)
            self.assertTrue(torch.all(supplied_v == 2))

    def test_fake_shape_functions(self):
        qkv, q_weight, k_weight, cache, positions = (tensor.to("meta") for tensor in self.inputs)
        v_out = torch.empty((3, 2, 128), dtype=qkv.dtype, device="meta")

        q_out, k_out, fake_v_out = operator_module._split_qkv_norm_rope_concat_fake(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            2,
            128,
            96,
            1e-5,
            False,
        )
        no_v_out = operator_module._split_qkv_norm_rope_concat_no_v_fake(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            2,
            128,
            96,
            1e-5,
            False,
        )
        out_q, out_k = operator_module._split_qkv_norm_rope_concat_out_fake(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            2,
            128,
            96,
            1e-5,
            False,
            v_out,
        )

        outputs = (q_out, k_out, fake_v_out, *no_v_out, out_q, out_k)
        self.assertEqual([tuple(output.shape) for output in outputs], [(3, 2, 128)] * 7)
        self.assertTrue(all(output.device.type == "meta" for output in outputs))


@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
    "Skip NPU-dependent tests when MINDIE_TEST_MODE is CPU.",
)
class TestSplitQkvNormRopeConcatNpu(unittest.TestCase):
    def setUp(self):
        if not torch.npu.is_available():  # pylint: disable=no-member
            self.skipTest("NPU is not available")
        torch.npu.set_device(0)  # pylint: disable=no-member

    def _run_case(self, dtype):
        host_inputs = _make_inputs(dtype=dtype)
        q_ref, k_ref, _ = _reference(*host_inputs, num_heads=2, head_dim=128, rope_dim=96, eps=1e-5)
        device_inputs = tuple(tensor.to("npu") for tensor in host_inputs)
        q_out, k_out, v_out = split_qkv_norm_rope_concat(
            *device_inputs,
            num_heads=2,
            head_dim=128,
            rope_dim=96,
        )
        torch.npu.synchronize()  # pylint: disable=no-member
        self.assertIsNone(v_out)
        self.assertEqual(q_out.shape, (3, 2, 128))
        self.assertTrue(torch.allclose(q_out.cpu(), q_ref, rtol=3e-2, atol=3e-2))
        self.assertTrue(torch.allclose(k_out.cpu(), k_ref, rtol=3e-2, atol=3e-2))

    def test_bf16(self):
        self._run_case(torch.bfloat16)

    def test_fp16(self):
        self._run_case(torch.float16)

    def test_explicit_v_out(self):
        inputs = tuple(tensor.to("npu") for tensor in _make_inputs())
        expected_v = inputs[0][:, 2 * 2 * 128 :].reshape(3, 2, 128)
        v_out = torch.empty_like(expected_v)
        _, _, returned_v = split_qkv_norm_rope_concat(
            *inputs,
            num_heads=2,
            head_dim=128,
            rope_dim=96,
            v_out=v_out,
        )
        torch.npu.synchronize()  # pylint: disable=no-member
        self.assertIs(returned_v, v_out)
        self.assertTrue(torch.equal(v_out, expected_v))

    def test_empty_tensor(self):
        qkv, q_weight, k_weight, cache, positions = _make_inputs(tokens=0)
        inputs = tuple(tensor.to("npu") for tensor in (qkv, q_weight, k_weight, cache, positions))
        q_out, k_out, v_out = split_qkv_norm_rope_concat(
            *inputs,
            num_heads=2,
            head_dim=128,
            rope_dim=96,
        )
        self.assertEqual(tuple(q_out.shape), (0, 2, 128))
        self.assertEqual(tuple(k_out.shape), (0, 2, 128))
        self.assertIsNone(v_out)

    def test_public_compile_fullgraph_no_v(self):
        inputs = tuple(tensor.to("npu") for tensor in _make_inputs())
        q_eager, k_eager, v_eager = split_qkv_norm_rope_concat(
            *inputs,
            num_heads=2,
            head_dim=128,
            rope_dim=96,
        )

        def _no_v(qkv, q_weight, k_weight, cache, positions):
            return split_qkv_norm_rope_concat(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                num_heads=2,
                head_dim=128,
                rope_dim=96,
            )

        compiled = torch.compile(_no_v, fullgraph=True, dynamic=False)
        q_compiled, k_compiled, v_compiled = compiled(*inputs)
        torch.npu.synchronize()  # pylint: disable=no-member

        self.assertIsNone(v_eager)
        self.assertIsNone(v_compiled)
        self.assertTrue(torch.equal(q_compiled, q_eager))
        self.assertTrue(torch.equal(k_compiled, k_eager))

    def test_public_compile_fullgraph_explicit_v_out(self):
        inputs = tuple(tensor.to("npu") for tensor in _make_inputs())
        eager_v_out = torch.empty((3, 2, 128), dtype=inputs[0].dtype, device=inputs[0].device)
        q_eager, k_eager, _ = split_qkv_norm_rope_concat(
            *inputs,
            num_heads=2,
            head_dim=128,
            rope_dim=96,
            v_out=eager_v_out,
        )

        def _with_v_out(qkv, q_weight, k_weight, cache, positions, v_out):
            q_out, k_out, returned_v = split_qkv_norm_rope_concat(
                qkv,
                q_weight,
                k_weight,
                cache,
                positions,
                num_heads=2,
                head_dim=128,
                rope_dim=96,
                v_out=v_out,
            )
            return q_out, k_out, returned_v

        compiled_v_out = torch.empty_like(eager_v_out)
        compiled = torch.compile(_with_v_out, fullgraph=True, dynamic=False)
        q_compiled, k_compiled, v_compiled = compiled(*inputs, compiled_v_out)
        torch.npu.synchronize()  # pylint: disable=no-member

        self.assertIs(v_compiled, compiled_v_out)
        self.assertTrue(torch.equal(q_compiled, q_eager))
        self.assertTrue(torch.equal(k_compiled, k_eager))
        self.assertTrue(torch.equal(compiled_v_out, eager_v_out))

    @unittest.skipUnless(
        os.environ.get("MINDIE_RUN_LARGE_CACHE_TEST") == "1",
        "Set MINDIE_RUN_LARGE_CACHE_TEST=1 to run the 4 GiB cache address regression.",
    )
    def test_rope_cache_offset_uses_int64(self):
        max_position = 2**23 + 1
        rope_dim = 256
        generator = torch.Generator(device="cpu").manual_seed(20261106)
        qkv = torch.randn(1, 3 * rope_dim, generator=generator, dtype=torch.bfloat16)
        q_weight = torch.ones(rope_dim, dtype=qkv.dtype)
        k_weight = torch.ones(rope_dim, dtype=qkv.dtype)
        cache = torch.empty((max_position, rope_dim), dtype=qkv.dtype, device="npu")
        cache[-1, : rope_dim // 2].fill_(1.0)
        cache[-1, rope_dim // 2 :].zero_()
        positions = torch.tensor([max_position - 1], dtype=torch.int32, device="npu")

        q_out, k_out, _ = split_qkv_norm_rope_concat(
            qkv.to("npu"),
            q_weight.to("npu"),
            k_weight.to("npu"),
            cache,
            positions,
            num_heads=1,
            head_dim=rope_dim,
            rope_dim=rope_dim,
            validate_values=False,
        )
        torch.npu.synchronize()  # pylint: disable=no-member

        q = qkv[:, :rope_dim].float().reshape(1, 1, rope_dim)
        k = qkv[:, rope_dim : 2 * rope_dim].float().reshape(1, 1, rope_dim)
        q_expected = (q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-5)).to(qkv.dtype)
        k_expected = (k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-5)).to(qkv.dtype)
        self.assertTrue(torch.allclose(q_out.cpu(), q_expected, rtol=1e-2, atol=1e-2))
        self.assertTrue(torch.allclose(k_out.cpu(), k_expected, rtol=1e-2, atol=1e-2))

    def test_registered_op_fake_shape(self):
        qkv = torch.empty((3, 3 * 2 * 128), dtype=torch.bfloat16, device="meta")
        q_weight = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        k_weight = torch.empty((128,), dtype=torch.bfloat16, device="meta")
        cache = torch.empty((16, 96), dtype=torch.bfloat16, device="meta")
        positions = torch.empty((3,), dtype=torch.int32, device="meta")
        outputs = torch.ops.mindiesd.split_qkv_norm_rope_concat(
            qkv,
            q_weight,
            k_weight,
            cache,
            positions,
            2,
            128,
            96,
            1e-5,
            True,
        )
        self.assertEqual([tuple(output.shape) for output in outputs], [(3, 2, 128)] * 3)
        self.assertTrue(all(output.device.type == "meta" for output in outputs))
