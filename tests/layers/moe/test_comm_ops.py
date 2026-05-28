#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest
from unittest.mock import MagicMock, patch

import torch

from mindiesd.layers.moe.comm_ops import all_gather, all_reduce, all_to_all_single, reduce_scatter
from mindiesd.utils import ParametersInvalid


class TestCommOps(unittest.TestCase):
    def setUp(self):
        self.group = MagicMock()

    def test_all_gather_world_size_one_returns_input(self):
        tensor = torch.randn(2, 3)

        with patch("mindiesd.layers.moe.comm_ops.dist.get_world_size", return_value=1):
            result = all_gather(tensor, self.group)

        self.assertIs(result, tensor)

    def test_all_gather_supports_nonzero_dim(self):
        tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])

        with patch("mindiesd.layers.moe.comm_ops.dist.get_world_size", return_value=2):
            with patch("mindiesd.layers.moe.comm_ops.dist.all_gather_into_tensor") as gather:
                gather.side_effect = lambda out, inp, group: out.copy_(torch.cat([inp, inp + 10], dim=0))
                result = all_gather(tensor, self.group, dim=2)

        self.assertTrue(torch.equal(result, torch.cat([tensor, tensor + 10], dim=2)))
        self.assertIs(gather.call_args.kwargs["group"], self.group)

    def test_reduce_scatter_supports_nonzero_dim(self):
        tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])

        with patch("mindiesd.layers.moe.comm_ops.dist.get_world_size", return_value=2):
            with patch("mindiesd.layers.moe.comm_ops.dist.reduce_scatter_tensor") as scatter:
                scatter.side_effect = lambda out, inp, group: out.copy_(inp[:2] + inp[2:])
                result = reduce_scatter(tensor, self.group, dim=-1)

        self.assertTrue(torch.equal(result, tensor[:, :, :2] + tensor[:, :, 2:]))
        self.assertIs(scatter.call_args.kwargs["group"], self.group)

    def test_reduce_scatter_rejects_non_divisible_dim(self):
        with patch("mindiesd.layers.moe.comm_ops.dist.get_world_size", return_value=2):
            with self.assertRaises(ParametersInvalid):
                reduce_scatter(torch.randn(2, 3), self.group, dim=1)

    def test_all_reduce_is_inplace(self):
        tensor = torch.tensor([1.0, 2.0])

        with patch("mindiesd.layers.moe.comm_ops.dist.all_reduce") as reduce:
            reduce.side_effect = lambda t, group: t.add_(10)
            result = all_reduce(tensor, self.group)

        self.assertIs(result, tensor)
        self.assertTrue(torch.equal(result, torch.tensor([11.0, 12.0])))
        self.assertIs(reduce.call_args.kwargs["group"], self.group)

    def test_all_to_all_allocates_output_and_passes_options(self):
        tensor = torch.randn(4, 3).t()
        output_sizes = [1, 2]
        input_sizes = [1, 2]

        with patch("mindiesd.layers.moe.comm_ops.dist.all_to_all_single") as a2a:
            a2a.side_effect = lambda out, inp, **kwargs: out.copy_(inp[: out.shape[0]])
            result = all_to_all_single(tensor, output_sizes, input_sizes, self.group)

        self.assertEqual(result.shape, (sum(output_sizes), tensor.shape[-1]))
        self.assertTrue(a2a.call_args.args[1].is_contiguous())
        self.assertEqual(a2a.call_args.kwargs["output_split_sizes"], output_sizes)
        self.assertEqual(a2a.call_args.kwargs["input_split_sizes"], input_sizes)
        self.assertIs(a2a.call_args.kwargs["group"], self.group)
        self.assertFalse(a2a.call_args.kwargs["async_op"])


if __name__ == "__main__":
    unittest.main()
