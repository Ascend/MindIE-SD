# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
import pytest
import torch

from tests.ops.device_support import init_test_device, require_support
from tests.ops.sigmoid_mul_dynamic_mx_quant.workload import compare, cpu_reference, fused, make_inputs, metadata


@pytest.fixture(scope="module", name="device")
def device_fixture():
    selected = init_test_device()
    require_support("sigmoid_mul_dynamic_mx_quant")
    import mindiesd  # noqa: F401

    return selected


@pytest.mark.parametrize(
    "shape,mode,distribution",
    [
        ((1, 1), 0, "normal"),
        ((2, 31), 1, "uniform"),
        ((3, 32), 0, "uniform"),
        ((4, 33), 1, "normal"),
        ((5, 64), 1, "uniform"),
        ((17, 65), 0, "normal"),
        ((3, 257), 1, "normal"),
        ((17, 1536), 0, "uniform"),
        ((10, 2048), 1, "normal"),
        ((2, 18433), 0, "uniform"),
        ((2, 18433), 1, "uniform"),
    ],
)
@pytest.mark.parametrize("scale_alg", [0, 1])
def test_accuracy(device, shape, mode, distribution, scale_alg):
    x, gate = make_inputs(shape, distribution)
    expected = cpu_reference(x, gate, mode, scale_alg)
    actual = fused(x.to(device), gate.to(device), mode, scale_alg)
    metadata(actual, shape, actual[0].device)
    compare(tuple(t.cpu() for t in actual), expected, small=True)


def test_distinct_operands_and_default(device):
    x = torch.tensor([[0, 1, -2, 3] * 16], dtype=torch.bfloat16)
    gate = torch.tensor([[-8, 0, 1, 8] * 16], dtype=torch.bfloat16)
    result = torch.ops.mindiesd.sigmoid_mul_dynamic_mx_quant(gate=gate.to(device), x=x.to(device))
    compare(tuple(t.cpu() for t in result), cpu_reference(x, gate, 0), small=True)


@pytest.mark.parametrize("mode", [0, 1])
def test_scale_ceiling_boundary(device, mode):
    from mindiesd.layers._custom_ops import sigmoid_mul_dynamic_mx_quant

    # Sigmoid(0) is exactly 0.5. Each consecutive group has a different maximum.
    x = torch.tensor([0, 896, 960], dtype=torch.bfloat16, device=device).repeat_interleave(32).reshape(1, 96)
    gate = torch.zeros_like(x)
    for alg, scales, values in [(0, [0, 127, 127, 0], [0, 448, 448]), (1, [0, 127, 128, 0], [0, 448, 240])]:
        y, scale = sigmoid_mul_dynamic_mx_quant(x=x, gate=gate, precision_mode=mode, scale_alg=alg)
        assert scale.cpu().flatten().tolist() == scales
        assert y.cpu().float().flatten().tolist() == [v for v in values for _ in range(32)]
    default_y, default_scale = torch.ops.mindiesd.sigmoid_mul_dynamic_mx_quant(x, gate, mode)
    assert default_y.cpu().float()[0, -1].item() == 448
    assert default_scale.cpu().flatten().tolist() == [0, 127, 127, 0]


def test_fp32_scale_keeps_product_mantissa(device):
    x = torch.full((1, 32), 896, dtype=torch.bfloat16, device=device)
    gate = torch.full_like(x, 2.0**-10)
    # The FP32 product exceeds 448 while the BF16 gate rounds to 0.5.
    for mode, expected in [(0, 128), (1, 127)]:
        _, scale = fused(x, gate, mode, scale_alg=1)
        assert scale.cpu().flatten().tolist() == [expected, 0]


@pytest.mark.parametrize("shape", [(0, 32), (2, 0)])
def test_empty(device, shape):
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    metadata(fused(x, x, 0), shape, x.device)


@pytest.mark.parametrize("invalid", ["precision", "scale_alg", "shape", "dtype", "rank"])
def test_invalid(device, invalid):
    x = torch.ones(4, 64, dtype=torch.bfloat16, device=device)
    gate = {"shape": x[:1], "dtype": x.float(), "rank": x.flatten()}.get(invalid, x)
    with pytest.raises(RuntimeError):
        fused(x, gate, 2 if invalid == "precision" else 0, 2 if invalid == "scale_alg" else 0)


def test_noncontiguous_and_fake(device):
    from torch._subclasses.fake_tensor import FakeTensorMode

    x, gate = make_inputs((2, 128), "normal")
    xd, gd = x.to(device)[:, ::2], gate.to(device)[:, ::2]
    compare(tuple(t.cpu() for t in fused(xd, gd, 1)), cpu_reference(x[:, ::2], gate[:, ::2], 1), small=True)
    with FakeTensorMode() as fake_mode:
        fx, fg = fake_mode.from_tensor(xd), fake_mode.from_tensor(gd)
        for mode in (0, 1):
            for alg in (0, 1):
                metadata(fused(fx, fg, mode, alg), tuple(xd.shape), fx.device)
        with pytest.raises(RuntimeError):
            fused(fx, fg, 1, 2)
