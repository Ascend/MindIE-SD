#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU contracts exercise real validation and numerical normalization paths."""

import importlib

import pytest
import torch
import torch.nn.functional as F

from mindiesd.layers import norm
from mindiesd.utils.exception import ParametersInvalid

usp = importlib.import_module("mindiesd.layers.usp")


@pytest.mark.parametrize(
    "invalid",
    [
        "input",
        "fused",
        "rank",
        "group_bool",
        "group_zero",
        "channels",
        "eps_bool",
        "eps_zero",
        "eps_nan",
        "weight_type",
        "weight_shape",
        "weight_dtype",
        "bias_type",
        "bias_shape",
        "bias_dtype",
    ],
)
def test_group_norm_rejects_invalid_contract(invalid):
    args = dict(input=torch.ones(2, 4, 3), weight=torch.ones(4), bias=torch.zeros(4), group=2, eps=1e-5, fused=False)
    edits = {
        "input": ("input", None),
        "fused": ("fused", 1),
        "rank": ("input", torch.ones(4)),
        "group_bool": ("group", True),
        "group_zero": ("group", 0),
        "channels": ("group", 3),
        "eps_bool": ("eps", True),
        "eps_zero": ("eps", 0),
        "eps_nan": ("eps", float("nan")),
        "weight_type": ("weight", []),
        "weight_shape": ("weight", torch.ones(3)),
        "weight_dtype": ("weight", torch.ones(4).double()),
        "bias_type": ("bias", []),
        "bias_shape": ("bias", torch.zeros(3)),
        "bias_dtype": ("bias", torch.zeros(4).double()),
    }
    key, value = edits[invalid]
    args[key] = value
    with pytest.raises(ParametersInvalid):
        norm.group_norm_silu(**args)


@pytest.mark.parametrize("affine", [False, True])
def test_group_norm_cpu_math(affine):
    x = torch.linspace(-3, 4, 48).reshape(2, 4, 6)
    w, b = (torch.arange(1, 5).float(), torch.arange(4).float()) if affine else (None, None)
    actual = norm.group_norm_silu(x, w, b, 2, fused=False)
    torch.testing.assert_close(actual, F.silu(F.group_norm(x, 2, w, b)))


@pytest.mark.parametrize(
    "invalid",
    [
        "layout",
        "comm_dtype",
        "backend",
        "comm_quant_scope",
        "scatter_dim",
        "gather_dim",
        "comm_tensors",
        "scale",
        "q_block_size",
        "kv_block_size",
        "chunk_size",
        "head_chunk_size",
        "mixed_chunks",
        "lse",
    ],
)
def test_usp_public_options_fail_before_dispatch(invalid):
    x = torch.ones(1, 4, 2, 8)
    options = {
        "layout": {"layout": "bad"},
        "comm_dtype": {"comm_dtype": "bad"},
        "backend": {"backend": "bad"},
        "comm_quant_scope": {"comm_quant_scope": "bad"},
        "scatter_dim": {"scatter_dim": 1},
        "gather_dim": {"gather_dim": 2},
        "comm_tensors": {"comm_tensors": ("invalid",)},
        "scale": {"q_scale": torch.ones(1)},
        "q_block_size": {"q_block_size": 0},
        "kv_block_size": {"kv_block_size": 0},
        "chunk_size": {"chunk_size": 0},
        "head_chunk_size": {"head_chunk_size": 0},
        "mixed_chunks": {"chunk_size": 2, "overlap": True},
        "lse": {"return_lse": True},
    }
    with pytest.raises(usp.USPError):
        usp.usp_attention(x, x, x, **options[invalid])


@pytest.mark.parametrize(
    "invalid", ["type", "rank", "device", "dtype", "batch", "kv_seq", "kv_heads", "head_dim", "q_heads"]
)
def test_usp_qkv_validation(invalid):
    q = torch.ones(1, 4, 4, 8)
    k = v = torch.ones(1, 4, 2, 8)
    if invalid == "type":
        q = []
    elif invalid == "rank":
        q = q[0]
    elif invalid == "device":
        k = torch.ones(k.shape, device="meta")
    elif invalid == "dtype":
        k = k.double()
    elif invalid == "batch":
        q = q.expand(2, -1, -1, -1)
    elif invalid == "kv_seq":
        v = v[:, :2]
    elif invalid == "kv_heads":
        v = v[:, :, :1]
    elif invalid == "head_dim":
        q = q[..., :4]
    else:
        q = q[:, :, :3]
    with pytest.raises(usp.USPShapeError):
        usp._validate_qkv(q, k, v, "BSND")


@pytest.mark.parametrize("invalid", ["pair", "shape", "batch"])
def test_joint_kv_validation(invalid):
    key = torch.ones(1, 4, 2, 8)
    left = torch.ones(1, 2, 2, 8)
    right = None if invalid == "pair" else left[:, :1] if invalid == "shape" else left.expand(2, -1, -1, -1)
    if invalid == "batch":
        left = right
    with pytest.raises(usp.USPShapeError):
        usp._validate_joint(left, right, key, "BSND")


@pytest.mark.parametrize(
    "invalid",
    [
        "workspace_type",
        "workspace_dtype",
        "workspace_device",
        "workspace_stride",
        "out_type",
        "out_device",
        "out_stride",
    ],
)
def test_workspace_contract(invalid):
    x = torch.ones(1, 4, 2, 8)
    workspace = output = None
    if invalid == "workspace_type":
        workspace = []
    elif invalid == "workspace_dtype":
        workspace = x
    elif invalid == "workspace_device":
        workspace = torch.ones(8, dtype=torch.uint8, device="meta")
    elif invalid == "workspace_stride":
        workspace = torch.ones(8, dtype=torch.uint8)[::2]
    elif invalid == "out_type":
        output = []
    elif invalid == "out_device":
        output = torch.ones(x.shape, device="meta")
    else:
        output = x.transpose(1, 2)
    with pytest.raises(usp.USPWorkspaceError):
        usp._validate_workspace(workspace, output, x)


@pytest.mark.parametrize(
    "lengths",
    [
        [],
        torch.ones(1),
        torch.ones(1, 1, dtype=torch.int64),
        torch.ones(2, dtype=torch.int32),
        torch.tensor([0]),
        torch.tensor([9]),
    ],
)
def test_lengths_validation(lengths):
    with pytest.raises(usp.USPShapeError):
        usp._validate_lengths(lengths, 1, 8)


def test_lengths_valid_and_out_copy():
    usp._validate_lengths(torch.tensor([3]), 1, 8)
    x = torch.arange(8).float()
    out = torch.empty_like(x)
    assert usp._copy_to_out(x, out) is out
    torch.testing.assert_close(x, out)
    with pytest.raises(usp.USPWorkspaceError):
        usp._copy_to_out(x, torch.empty(2))
