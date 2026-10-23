#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from itertools import product

import torch

from mindiesd import norm_rope_concat


def inputs(device='cpu'):
    q = torch.ones(1, 2, 2, 4, dtype=torch.bfloat16, device=device)
    return (q, q.clone(), q.clone())


@unittest.skipUnless(hasattr(torch, "npu") and torch.npu.is_available(), "NPU is required")
class TestNormRopeConcatAPI(unittest.TestCase):
    def test_npu_layout_combinations(self):
        for input_layout, output_layout, concat_order, with_encoder, dtype in product(
            ['BSND', 'BNSD'], ['BSND', 'BNSD'], [0, 1], [False, True], [torch.float32, torch.bfloat16]
        ):
            with self.subTest(
                input_layout=input_layout,
                output_layout=output_layout,
                concat_order=concat_order,
                with_encoder=with_encoder,
                dtype=dtype,
            ):
                x = torch.arange(2 * 5 * 3 * 128, dtype=torch.float32, device='npu').reshape(2, 5, 3, 128).to(dtype)
                main = (x, -x, x + 4)
                encoder = tuple((tensor[:, :2].clone() + 8 for tensor in main)) if with_encoder else (None,) * 3
                args = tuple(
                    (
                        tensor.transpose(1, 2).contiguous()
                        if input_layout == 'BNSD'
                        else tensor.transpose(1, 2).contiguous().transpose(1, 2)
                        for tensor in (*main, *(tensor for tensor in encoder if tensor is not None))
                    )
                )
                total_seq = 7 if with_encoder else 5
                sin = torch.zeros(total_seq, 128, dtype=dtype, device='npu')
                actual = norm_rope_concat(
                    *args[:3],
                    encoder_query=args[3] if with_encoder else None,
                    encoder_key=args[4] if with_encoder else None,
                    encoder_value=args[5] if with_encoder else None,
                    input_layout=input_layout,
                    output_layout=output_layout,
                    concat_order=concat_order,
                    rope_type=1,
                    rope_sin=sin,
                    rope_cos=torch.ones_like(sin),
                )
                assert len(actual) == 11
                for result, primary, added in zip(actual[:3], main, encoder):
                    expected = primary
                    if added is not None:
                        parts = (primary, added) if concat_order == 0 else (added, primary)
                        expected = torch.cat(parts, dim=1)
                    if output_layout == 'BNSD':
                        expected = expected.transpose(1, 2)
                        assert result.is_contiguous()
                    else:
                        assert result.transpose(1, 2).is_contiguous()
                    torch.testing.assert_close(result.cpu(), expected.cpu(), atol=0, rtol=0)

    def test_npu_default_layout_is_bsnd(self):
        q = torch.arange(128, dtype=torch.float32, device='npu').repeat(1, 5, 3, 1)
        actual = norm_rope_concat(q, q, q)
        for result in actual[:3]:
            torch.testing.assert_close(result.cpu(), q.cpu(), atol=0, rtol=0)

    def test_npu_entry_point_matches_registered_operator(self):
        q, k, v = inputs('npu')
        sin = torch.zeros(2, 4, dtype=q.dtype, device=q.device)
        cos = torch.ones_like(sin)
        actual = norm_rope_concat(q, k, v, rope_sin=sin, rope_cos=cos, rope_type=1, output_layout='BNSD')
        assert len(actual) == 11
        torch.testing.assert_close(actual[0].transpose(1, 2), q, atol=0, rtol=0)
        torch.testing.assert_close(actual[1].transpose(1, 2), k, atol=0, rtol=0)
        torch.testing.assert_close(actual[2].transpose(1, 2), v, atol=0, rtol=0)

    def test_npu_partial_rope_preserves_tail(self):
        for dtype, rope_type, concat_order in product([torch.float32, torch.bfloat16], [1, 2], [0, 1]):
            with self.subTest(dtype=dtype, rope_type=rope_type, concat_order=concat_order):
                x = torch.arange(128, dtype=dtype).repeat(1, 8, 2, 1)
                q, k, v = (x.to('npu'), (-x).to('npu'), (x + 4).to('npu'))
                encoder_q = (-x[:, :3]).to('npu')
                encoder_k = x[:, :3].to('npu')
                encoder_v = (x[:, :3] - 4).to('npu')
                sin = torch.ones(4, 128, dtype=dtype, device='npu')
                cos = torch.zeros_like(sin)
                actual = norm_rope_concat(
                    q,
                    k,
                    v,
                    encoder_query=encoder_q,
                    encoder_key=encoder_k,
                    encoder_value=encoder_v,
                    rope_sin=sin,
                    rope_cos=cos,
                    rope_type=rope_type,
                    concat_order=concat_order,
                    output_layout='BNSD',
                )
                for result, main, encoder, rotate in zip(
                    actual[:3], (q, k, v), (encoder_q, encoder_k, encoder_v), (True, True, False)
                ):
                    parts = (main.cpu(), encoder.cpu()) if concat_order == 0 else (encoder.cpu(), main.cpu())
                    expected = torch.cat(parts, dim=1).transpose(1, 2).contiguous()
                    if rotate:
                        prefix = expected[:, :, :4].clone()
                        if rope_type == 1:
                            expected[:, :, :4] = torch.stack((-prefix[..., 1::2], prefix[..., ::2]), dim=-1).flatten(-2)
                        else:
                            expected[:, :, :4] = torch.cat((-prefix[..., 64:], prefix[..., :64]), dim=-1)
                    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)

    def test_exact_quarter_turn_exposes_gather_offset_race(self):
        for dtype, heads, seq, rope_type in product([torch.float32, torch.bfloat16], [1, 4, 32], [1, 17], [1, 2]):
            with self.subTest(dtype=dtype, heads=heads, seq=seq, rope_type=rope_type):
                x = torch.arange(128, dtype=dtype).repeat(1, seq, heads, 1)
                sin = torch.ones(seq, 128, dtype=dtype, device='npu')
                cos = torch.zeros_like(sin)
                if rope_type == 1:
                    expected = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
                else:
                    expected = torch.cat((-x[..., 64:], x[..., :64]), dim=-1)
                q = x.to('npu')
                expected = expected.transpose(1, 2)
                for _ in range(8):
                    actual = norm_rope_concat(
                        q, q, q, rope_sin=sin, rope_cos=cos, rope_type=rope_type, output_layout='BNSD'
                    )
                    torch.testing.assert_close(actual[0].cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual[1].cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(actual[2].cpu(), x.transpose(1, 2), rtol=0, atol=0)

    def test_no_rope_tables(self):
        cases = product(
            ("BSND", "BNSD"),
            ("BSND", "BNSD"),
            (0, 1),
            (False, True),
            (torch.float32, torch.bfloat16),
            range(5),
            (4, 128),
        )
        for input_layout, output_layout, order, with_encoder, dtype, norm_type, dim in cases:
            with self.subTest(
                input_layout=input_layout,
                output_layout=output_layout,
                concat_order=order,
                with_encoder=with_encoder,
                dtype=dtype,
                norm_type=norm_type,
                head_dim=dim,
            ):
                x = (torch.arange(2 * 5 * 3 * dim).reshape(2, 5, 3, dim).float() % 37 - 18) / 17
                main = tuple(tensor.to(dtype) for tensor in (x, -x + 0.25, x + 2))
                encoder = tuple(tensor[:, :2].clone() + 0.5 for tensor in main) if with_encoder else ()
                tensors = tuple(tensor.to("npu") for tensor in (*main, *encoder))
                if input_layout == "BNSD":
                    tensors = tuple(tensor.transpose(1, 2).contiguous() for tensor in tensors)
                weight = torch.linspace(0.75, 1.25, dim).to(dtype)
                bias = torch.linspace(-0.2, 0.2, dim).to(dtype)
                added_type = 4 - norm_type
                kwargs = {}
                for prefix, mode in (("norm", norm_type), ("norm_added", added_type)):
                    if mode in (2, 4):
                        for name in ("query", "key"):
                            kwargs[f"{prefix}_{name}_weight"] = weight.to("npu")
                            if mode == 2:
                                kwargs[f"{prefix}_{name}_bias"] = bias.to("npu")
                actual = norm_rope_concat(
                    *tensors[:3],
                    encoder_query=tensors[3] if with_encoder else None,
                    encoder_key=tensors[4] if with_encoder else None,
                    encoder_value=tensors[5] if with_encoder else None,
                    input_layout=input_layout,
                    output_layout=output_layout,
                    concat_order=order,
                    norm_type=norm_type,
                    norm_added_type=added_type,
                    **kwargs,
                )
                for index, result in enumerate(actual[:3]):
                    expected = self._normalize(main[index], norm_type if index < 2 else 0, weight, bias)
                    if with_encoder:
                        added = self._normalize(encoder[index], added_type if index < 2 else 0, weight, bias)
                        parts = (expected, added) if order == 0 else (added, expected)
                        expected = torch.cat(parts, dim=1)
                    if output_layout == "BNSD":
                        expected = expected.transpose(1, 2)
                    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
                    torch.testing.assert_close(result.cpu(), expected, rtol=tolerance, atol=tolerance)

    @staticmethod
    def _normalize(tensor, mode, weight, bias):
        if mode == 0:
            return tensor
        x = tensor.float()
        if mode in (1, 2):
            x = torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=1e-5)
        else:
            x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5)
        if mode in (2, 4):
            x = x * weight.float()
        if mode == 2:
            x = x + bias.float()
        return x.to(tensor.dtype)
