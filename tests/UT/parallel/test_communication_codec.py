# SPDX-License-Identifier: Apache-2.0
"""The native USP and portable codecs use identical FP8 bytes."""

import sys
from types import ModuleType

import pytest
import torch

from cpu_support import REPO_ROOT, load_module, new_package, register_module


def codec_and_usp(monkeypatch):
    load_module("mindiesd.comm_quant", REPO_ROOT / "mindiesd/comm_quant.py", monkeypatch)
    root_name = "_usp_codec_contracts"
    new_package(root_name, monkeypatch, directory=REPO_ROOT / "mindiesd")
    new_package(root_name + ".layers", monkeypatch, directory=REPO_ROOT / "mindiesd/layers")
    new_package(root_name + ".layers.flash_attn", monkeypatch)
    for module_name, function_name in (
        ("attention_forward", "attention_forward"),
        ("fused_infer_attention_score", "fused_infer_attention_score_v2"),
    ):
        stub_name = root_name + ".layers.flash_attn." + module_name
        module = ModuleType(stub_name)
        setattr(module, function_name, lambda *args, **kwargs: None)
        register_module(stub_name, module, monkeypatch)
    usp = load_module(root_name + ".layers.usp", REPO_ROOT / "mindiesd/layers/usp.py", monkeypatch)
    return sys.modules["mindiesd.comm_quant"], usp


@pytest.mark.parametrize("comm_dtype", ["fp8_e4m3", "mxfp8"])
@pytest.mark.parametrize("zeros", [False, True])
def test_portable_and_usp_codecs_match_bytes_scales_and_restored_values(monkeypatch, comm_dtype, zeros):
    codec, usp = codec_and_usp(monkeypatch)
    block_size = 32
    torch.manual_seed(57)
    value = torch.zeros(4, 3, 17) if zeros else torch.randn(4, 3, 17)
    native, scales, width = usp._block_quantize(value, block_size, comm_dtype, None)
    portable, portable_scales, portable_width = codec.block_quantize(value, block_size, comm_dtype, portable=True)
    assert torch.equal(native.view(torch.uint8), portable)
    assert torch.equal(scales, portable_scales) and width == portable_width
    decoded = codec.block_dequantize(portable, scales, width, value.shape, torch.bfloat16)
    expected = usp._block_dequantize(native, scales, width, value.shape, torch.bfloat16)
    assert torch.equal(decoded, expected)
    assert torch.isfinite(decoded).all()


def test_usp_scale_override_errors_keep_the_public_exception(monkeypatch):
    _, usp = codec_and_usp(monkeypatch)
    value = torch.randn(2, 37)
    payload, scales, width = usp._block_quantize(value, 32, "fp8_e4m3", torch.tensor(0.25))
    assert width == 37 and payload.shape == (2, 64)
    assert torch.equal(scales, torch.full((2, 2), 0.25))
    with pytest.raises(usp.USPShapeError, match="scale override"):
        usp._block_quantize(value, 32, "fp8_e4m3", torch.ones(3))


def test_portable_encoder_matches_native_rounding_midpoints(monkeypatch):
    codec, _ = codec_and_usp(monkeypatch)
    positive = torch.arange(127).to(torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (positive[:-1] + positive[1:]) * 0.5
    values = torch.cat((midpoints, -midpoints, torch.linspace(-448, 448, 8193)))
    assert torch.equal(codec.encode_fp8_e4m3(values), values.to(torch.float8_e4m3fn).view(torch.uint8))


def test_portable_encoder_saturates_large_finite_inputs(monkeypatch):
    codec, _ = codec_and_usp(monkeypatch)
    values = torch.tensor([-10000.0, -496.0, -448.0, 448.0, 496.0, 10000.0])
    expected = values.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(codec.encode_fp8_e4m3(values), expected)
