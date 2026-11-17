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
from mindiesd.utils import ParametersInvalid


def inputs(device='cpu'):
    q = torch.ones(1, 2, 2, 4, dtype=torch.bfloat16, device=device)
    return (q, q.clone(), q.clone())


class TestNormRopeConcatValidation(unittest.TestCase):
    def test_rejects_invalid_arguments(self):
        for mutation, match in [
            (lambda a: a.update(query='bad'), 'query must'),
            (lambda a: a.update(key=torch.ones(1, 2, 2, dtype=torch.bfloat16)), 'nonempty BSND'),
            (lambda a: a.update(value=torch.ones(1, 3, 2, 4, dtype=torch.bfloat16)), 'sequence lengths'),
            (lambda a: a.update(encoder_key=a['key']), 'appear together'),
            (lambda a: a.update(encoder_query=torch.ones(2, 2, 2, 4, dtype=torch.bfloat16)), 'batch'),
            (lambda a: a.update(key=a['key'].float()), 'must use'),
            (lambda a: a.update(norm_type=9), 'norm_type'),
            (lambda a: a.update(norm_added_type=-1), 'norm_added_type'),
            (lambda a: a.update(rope_type=True), 'rope_type'),
            (lambda a: a.update(concat_order=2), 'concat_order'),
            (lambda a: a.update(is_training=1), 'is_training'),
            (lambda a: a.update(eps=float('nan')), 'eps'),
            (lambda a: a.update(eps=0), 'eps'),
            (lambda a: a.update(norm_type=4), 'weights'),
            (lambda a: a.update(rope_type=1), 'sin and cos'),
            (
                lambda a: a.update(
                    rope_sin=torch.ones(3, 4, dtype=torch.bfloat16), rope_cos=torch.ones(3, 4, dtype=torch.bfloat16)
                ),
                'fit output sequence',
            ),
            (lambda a: a.update(norm_query_weight=torch.ones(3, dtype=torch.bfloat16)), 'shape'),
            (
                lambda a: a.update(
                    rope_sin=torch.ones(2, 4, dtype=torch.bfloat16), rope_cos=torch.ones(2, 2, dtype=torch.bfloat16)
                ),
                'columns',
            ),
        ]:
            with self.subTest(match=match):
                q, k, v = inputs()
                args = {'query': q, 'key': k, 'value': v}
                mutation(args)
                with self.assertRaisesRegex(ParametersInvalid, match):
                    norm_rope_concat(**args)

    def test_rejects_cpu_even_with_valid_shapes(self):
        with self.assertRaisesRegex(ParametersInvalid, 'NPU'):
            norm_rope_concat(*inputs())

    def test_rejects_invalid_layout(self):
        for name, layout in product(['input_layout', 'output_layout'], [None, 'BSH', 'bsnd', 1]):
            with self.subTest(name=name, layout=layout):
                with self.assertRaisesRegex(ParametersInvalid, name):
                    norm_rope_concat(*inputs(), **{name: layout})

    def test_rejects_mismatched_kv_sequence_lengths(self):
        for input_layout, with_encoder in product(['BSND', 'BNSD'], [False, True]):
            with self.subTest(input_layout=input_layout, with_encoder=with_encoder):
                q = torch.ones(1, 5, 3, 4, dtype=torch.bfloat16)
                shorter = q[:, :4].clone()
                if input_layout == 'BNSD':
                    q = q.transpose(1, 2).contiguous()
                    shorter = shorter.transpose(1, 2).contiguous()
                kwargs = {'encoder_key': q, 'encoder_value': shorter} if with_encoder else {}
                value = q if with_encoder else shorter
                with self.assertRaisesRegex(ParametersInvalid, 'sequence lengths must match'):
                    norm_rope_concat(q, q, value, input_layout=input_layout, **kwargs)

    def test_sequence_lengths_are_independent_between_groups(self):
        for (input_layout,) in product(['BSND', 'BNSD']):
            with self.subTest(input_layout=input_layout):
                tensors = [torch.ones(1, length, 3, 4, dtype=torch.bfloat16) for length in (5, 4, 4, 3, 2, 2)]
                if input_layout == 'BNSD':
                    tensors = [tensor.transpose(1, 2).contiguous() for tensor in tensors]
                with self.assertRaisesRegex(ParametersInvalid, 'NPU'):
                    norm_rope_concat(
                        *tensors[:3],
                        encoder_query=tensors[3],
                        encoder_key=tensors[4],
                        encoder_value=tensors[5],
                        input_layout=input_layout,
                    )

    def test_bnsd_validation_uses_sequence_axis(self):
        q = torch.ones(1, 3, 5, 128)
        sin = torch.zeros(6, 128)
        with self.assertRaisesRegex(ParametersInvalid, 'fit output sequence'):
            norm_rope_concat(q, q, q, input_layout='BNSD', rope_type=1, rope_sin=sin, rope_cos=sin)
        with self.assertRaisesRegex(ParametersInvalid, 'sequence lengths'):
            norm_rope_concat(q, q, q[:, :, :4], input_layout='BNSD')

    def test_rejects_affine_layer_norm_without_bias(self):
        for (missing_bias,) in product(
            ['norm_query_bias', 'norm_key_bias', 'norm_added_query_bias', 'norm_added_key_bias']
        ):
            with self.subTest(missing_bias=missing_bias):
                q, k, v = inputs()
                param = torch.ones(q.shape[-1], dtype=q.dtype)
                args = {
                    'encoder_query': q.clone(),
                    'encoder_key': k.clone(),
                    'encoder_value': v.clone(),
                    'norm_type': 2,
                    'norm_added_type': 2,
                    'norm_query_weight': param,
                    'norm_query_bias': param,
                    'norm_key_weight': param,
                    'norm_key_bias': param,
                    'norm_added_query_weight': param,
                    'norm_added_query_bias': param,
                    'norm_added_key_weight': param,
                    'norm_added_key_bias': param,
                }
                args[missing_bias] = None
                with self.assertRaisesRegex(ParametersInvalid, missing_bias):
                    norm_rope_concat(q, k, v, **args)

    def test_affine_rms_norm_does_not_require_bias(self):
        q, k, v = inputs()
        weight = torch.ones(q.shape[-1], dtype=q.dtype)
        with self.assertRaisesRegex(ParametersInvalid, 'NPU'):
            norm_rope_concat(
                q,
                k,
                v,
                encoder_query=q.clone(),
                encoder_key=k.clone(),
                encoder_value=v.clone(),
                norm_query_weight=weight,
                norm_key_weight=weight,
                norm_added_query_weight=weight,
                norm_added_key_weight=weight,
                norm_type=4,
                norm_added_type=4,
            )

    def test_partial_rope_tables_pass_validation_before_cpu_rejection(self):
        for (rope_type,) in product([1, 2]):
            with self.subTest(rope_type=rope_type):
                q, k, v = inputs()
                sin = torch.ones(1, q.shape[-1], dtype=q.dtype)
                cos = torch.zeros_like(sin)
                with self.assertRaisesRegex(ParametersInvalid, 'NPU'):
                    norm_rope_concat(q, k, v, rope_sin=sin, rope_cos=cos, rope_type=rope_type)
