#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

if importlib.util.find_spec("torch_npu") is None:
    torch_npu = MagicMock()
    torch_npu.__spec__ = importlib.util.spec_from_loader("torch_npu", loader=None)
    torch_npu.npu.device_count.return_value = 0
    torch_npu.npu.is_available.return_value = False
    sys.modules["torch_npu"] = torch_npu
    torch.npu = torch_npu.npu

from mindiesd.layers.usp import (
    USPNotSupported,
    USPShapeError,
    USPTopologyError,
    _all_gather_sequence,
    _QuantizedFAInput,
    _ulysses_forward,
    _ulysses_reverse,
    usp_attention,
)


def _copy_collective(output, input_tensor, **kwargs):
    output.copy_(input_tensor)


def test_single_rank_calls_native_fa_with_explicit_backend():
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 4, 2, 8)
    value = torch.randn(1, 4, 2, 8)

    with patch("mindiesd.layers.usp.attention_forward", return_value=query + 1) as fa:
        output = usp_attention(query, key, value, backend="npu_fa")

    assert torch.equal(output, query + 1)
    assert fa.call_args.kwargs == {
        "attn_mask": None,
        "fused": True,
        "head_first": False,
        "opt_mode": "manual",
        "op_type": "fused_attn_score",
        "layout": "BSND",
    }


def test_bnsd_layout_is_restored_and_out_buffer_is_used():
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    value = torch.randn(1, 2, 4, 8)
    output_buffer = torch.empty_like(query)

    with patch("mindiesd.layers.usp.attention_forward", side_effect=lambda q, *_args, **_kwargs: q):
        output = usp_attention(query, key, value, layout="BNSD", out=output_buffer)

    assert output is output_buffer
    assert torch.equal(output, query)


def test_ulysses_forward_and_reverse_preserve_partition_mapping():
    group = MagicMock()
    tensor = torch.arange(1 * 3 * 4 * 2, dtype=torch.float32).reshape(1, 3, 4, 2)

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_to_all_single", side_effect=_copy_collective),
    ):
        transformed = _ulysses_forward(tensor, group, "none", 128, None)
        restored = _ulysses_reverse(transformed, group)

    assert transformed.shape == (1, 6, 2, 2)
    assert torch.equal(restored, tensor)


def test_kv_gather_concatenates_rank_sequences():
    group = MagicMock()
    tensor = torch.arange(1 * 2 * 1 * 2, dtype=torch.float32).reshape(1, 2, 1, 2)

    def gather(output, input_tensor, group):
        output.copy_(torch.cat((input_tensor, input_tensor + 10), dim=0))

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_gather_into_tensor", side_effect=gather),
    ):
        output = _all_gather_sequence(tensor, group, "none", 256, None)

    assert torch.equal(output, torch.cat((tensor, tensor + 10), dim=1))


def test_public_api_uses_kv_gather_group():
    group = MagicMock()
    query = torch.randn(1, 2, 1, 8)
    captured_key = None

    def gather(output, input_tensor, group):
        output.copy_(torch.cat((input_tensor, input_tensor + 10), dim=0))

    def attention(query, key, *_args, **_kwargs):
        nonlocal captured_key
        captured_key = key
        return query

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_gather_into_tensor", side_effect=gather),
        patch("mindiesd.layers.usp.attention_forward", side_effect=attention),
    ):
        usp_attention(query, query, query, kv_gather_group=group)

    assert captured_key is not None
    assert captured_key.shape[1] == 4


def test_fp8_ulysses_communicates_payload_and_scale():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build has no FP8 dtype")
    group = MagicMock()
    tensor = torch.randn(1, 2, 2, 8, dtype=torch.float16)

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_to_all_single", side_effect=_copy_collective) as a2a,
    ):
        output = _ulysses_forward(tensor, group, "fp8_e4m3", 8, None)

    assert output.shape == (1, 4, 1, 8)
    assert a2a.call_count == 2
    assert a2a.call_args_list[0].args[1].dtype == torch.float8_e4m3fn


def test_invalid_topology_fails_before_collective():
    group = MagicMock()
    query = torch.randn(1, 2, 3, 8)

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_to_all_single") as a2a,
        pytest.raises(USPTopologyError),
    ):
        usp_attention(query, query, query, ulysses_group=group)

    a2a.assert_not_called()


def test_unknown_keyword_arguments_are_rejected():
    query = torch.randn(1, 2, 1, 8)

    with pytest.raises(TypeError):
        # pylint: disable-next=unexpected-keyword-arg
        usp_attention(query, query, query, unknown_option=object())

    with pytest.raises(TypeError):
        # pylint: disable-next=unexpected-keyword-arg
        usp_attention(query, query, query, ring_group=object())


def test_unsupported_modes_raise_structured_errors():
    query = torch.randn(1, 2, 1, 8)

    with pytest.raises(USPNotSupported, match="return_lse"):
        usp_attention(query, query, query, return_lse=True)
    with pytest.raises(USPShapeError, match="provided together"):
        usp_attention(query, query, query, joint_k=query)
    with (
        patch("mindiesd.layers.usp._get_npu_runtime", return_value=None),
        pytest.raises(USPNotSupported, match="torch.npu"),
    ):
        usp_attention(query, query, query, overlap=True)
    with pytest.raises(USPNotSupported, match="cannot be combined"):
        usp_attention(query, query, query, chunk_size=1, head_chunk_size=1)


def test_communication_quantization_is_opt_in_and_exposed_by_default():
    query = torch.randn(1, 2, 1, 8)

    with (
        patch(
            "mindiesd.layers.usp._block_quantize",
            wraps=__import__("mindiesd.layers.usp", fromlist=["_block_quantize"])._block_quantize,
        ) as quantize,
        patch("mindiesd.layers.usp.attention_forward", side_effect=lambda q, *_args, **_kwargs: q),
    ):
        usp_attention(query, query, query)

    quantize.assert_not_called()


def test_head_cut_runs_one_attention_per_global_head_chunk():
    group = MagicMock()
    query = torch.randn(1, 2, 4, 8)

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.get_rank", return_value=0),
        patch("mindiesd.layers.usp.dist.all_to_all_single", side_effect=_copy_collective),
        patch("mindiesd.layers.usp.attention_forward", side_effect=lambda q, *_args, **_kwargs: q) as fa,
    ):
        output = usp_attention(query, query, query, ulysses_group=group, head_chunk_size=2)

    assert output.shape == query.shape
    assert fa.call_count == 2


def test_exposed_quant_scope_only_quantizes_first_forward_and_last_reverse_chunk():
    group = MagicMock()
    query = torch.randn(1, 2, 4, 8, dtype=torch.float16)
    quantized_calls = []

    def capture_a2a(packed, group, comm_dtype, block_size, scale):
        quantized_calls.append(comm_dtype)
        return packed

    class FakeStream:
        def wait_event(self, event):
            return None

    class FakeEvent:
        def record(self, stream=None):
            return None

    class FakeContext:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    runtime = MagicMock()
    runtime.current_stream.return_value = FakeStream()
    runtime.Stream.return_value = FakeStream()
    runtime.Event.side_effect = FakeEvent
    runtime.stream.side_effect = lambda _stream: FakeContext()

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.get_rank", return_value=0),
        patch("mindiesd.layers.usp._all_to_all", side_effect=capture_a2a),
        patch("mindiesd.layers.usp._all_gather_sequence", side_effect=lambda tensor, *_args: tensor),
        patch("mindiesd.layers.usp._get_npu_runtime", return_value=runtime),
        patch("mindiesd.layers.usp.attention_forward", side_effect=lambda q, *_args, **_kwargs: q),
    ):
        usp_attention(
            query,
            query,
            query,
            ulysses_group=group,
            overlap=True,
            comm_dtype="fp8_e4m3",
            comm_tensors=("k", "v", "out"),
        )

    # Two head chunks: Q/K/V forward then output reverse for each chunk.
    assert quantized_calls == [
        "none",
        "fp8_e4m3",
        "fp8_e4m3",
        "none",
        "none",
        "none",
        "none",
        "fp8_e4m3",
    ]


def test_quant_fa_reuses_kv_quantization_performed_before_gather():
    group = MagicMock()
    query = torch.randn(1, 4, 1, 8, dtype=torch.float16)
    quantized = query.to(torch.float16)
    scale = torch.ones(1, 1, 4, 1)

    def gather(output, input_tensor, group):
        output.copy_(torch.stack((input_tensor, input_tensor), dim=0))

    with (
        patch("mindiesd.layers.usp.dist.get_world_size", return_value=2),
        patch("mindiesd.layers.usp.dist.all_gather_into_tensor", side_effect=gather),
        patch(
            "mindiesd.layers.usp._fa_block_quantize",
            return_value=_QuantizedFAInput(quantized.transpose(1, 2), scale),
        ) as fa_quant,
        patch(
            "mindiesd.layers.usp.fused_infer_attention_score_v2",
            return_value=(quantized.transpose(1, 2), scale),
        ),
    ):
        usp_attention(
            query,
            query,
            query,
            kv_gather_group=group,
            backend="quant_fa",
            comm_dtype="fp8_e4m3",
            comm_tensors=("k", "v"),
            kv_block_size=4,
        )

    # K/V are quantized before their payload+scale AllGather; Q is quantized once at FA.
    assert fa_quant.call_count == 3
