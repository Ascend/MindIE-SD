"""CPU regression for device-specific native RoPE reference rounding."""

from unittest.mock import patch

import pytest
import torch

from tests.ops.layers.test_apply_rotary_pos_emb import cpu_rotary_reference, rotate
from mindiesd.utils.get_platform import NPUDevice


@pytest.mark.parametrize("family", [NPUDevice.A2, NPUDevice.A3, NPUDevice.A5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_reference_selects_only_documented_fp16_stage_rounding(family, dtype):
    generator = torch.Generator().manual_seed(17)
    x, cos, sin = (torch.randn(2, 3, 8, generator=generator).to(dtype) for _ in range(3))
    original = x.clone()
    first = x.float() * cos.float()
    second = rotate(x.float(), "half") * sin.float()
    expected = (first + second).to(dtype)
    if family in (NPUDevice.A2, NPUDevice.A3) and dtype == torch.float16:
        expected = (first.to(dtype).float() + second.to(dtype).float()).to(dtype)
    actual = cpu_rotary_reference(x, cos, sin, "half", device_family=family)
    assert torch.equal(actual, expected)
    assert actual.device.type == "cpu"
    assert torch.equal(x, original)


def test_reference_infers_platform_and_preserves_a5_behavior():
    x = torch.tensor([1.111, 2.333, 3.777, 4.111], dtype=torch.float16)
    cos, sin = torch.full_like(x, 0.7), torch.full_like(x, 0.3)
    with patch("tests.ops.layers.test_apply_rotary_pos_emb.get_npu_device", return_value=NPUDevice.A5):
        actual = cpu_rotary_reference(x, cos, sin, "half")
    expected = (x.float() * cos.float() + rotate(x.float(), "half") * sin.float()).half()
    assert torch.equal(actual, expected)
