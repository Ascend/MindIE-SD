"""Numerical CPU execution of the unchanged Triton vector kernel bodies."""

import importlib

import pytest
import torch

from UT.kernel_unit.triton_cpu import install


@pytest.mark.parametrize("size,cores", [(7, 1), (1031, 2), (4099, 2)])
@pytest.mark.parametrize("scale", [-1.25, 0.0, 2.0])
def test_muls_add_body(monkeypatch, size, cores, scale):
    module = importlib.import_module("mindiesd.layers.muls_add")
    install(monkeypatch, module)
    x = torch.linspace(-2, 2, size)
    y = x.flip(0).clone() + 0.25
    out = torch.full_like(x, torch.nan)
    before = x.clone(), y.clone()
    module.muls_add_kernel[(cores,)](x, y, out, scale, size, (size + 31) // 32, BLOCK_SIZE=32)
    torch.testing.assert_close(out, x * scale + y)
    assert torch.equal(x, before[0]) and torch.equal(y, before[1])
    with pytest.raises(ValueError, match="same shape"):
        module._muls_add_triton(x, y[:-1], scale)


@pytest.mark.parametrize("rows,width", [(1, 7), (4, 17), (7, 32)])
def test_scale_shift_and_gathers(monkeypatch, rows, width):
    module = importlib.import_module("mindiesd.layers.scale_shift")
    install(monkeypatch, module)
    x = torch.linspace(-2, 2, rows * width).reshape(1, rows, width)
    table = torch.linspace(-0.8, 0.7, 3 * width).reshape(3, width)
    shift = table.flip(0).clone()
    indices = torch.arange(rows) % 3
    scale = table[indices]
    bias = shift[indices]
    torch.testing.assert_close(module._scale_shift_triton(x, scale, bias), x * (1 + scale) + bias)
    torch.testing.assert_close(module._gather_scale_shift_triton(x, table, shift, indices), x * (1 + scale) + bias)
    value = x.flip(-1).clone()
    torch.testing.assert_close(module._gather_residual_gate_triton(x, value, table, indices), x + value * scale)
    proj = torch.cat((x, value), dim=-1)
    torch.testing.assert_close(module._swiglu_triton(proj), x * torch.nn.functional.silu(value))


@pytest.mark.parametrize("batch,seq,width", [(1, 1, 7), (2, 3, 33), (3, 2, 128)])
def test_residual_gate_body(monkeypatch, batch, seq, width):
    module = importlib.import_module("mindiesd.layers.residual_gate_add")
    install(monkeypatch, module)
    x = torch.linspace(-2, 3, batch * seq * width).reshape(batch, seq, width)
    value = x.flip(-1).clone()
    gate = torch.linspace(-0.9, 0.7, batch * width).reshape(batch, 1, width)
    for left, right in [(value, gate), (gate, value)]:
        torch.testing.assert_close(module._residual_gate_add_triton(x, left, right), x + value * gate)


@pytest.mark.parametrize("rows,width,cores", [(1, 1, 1), (7, 17, 2), (3, 128, 1)])
def test_indexed_rms_body_with_storage_gaps(monkeypatch, rows, width, cores):
    module = importlib.import_module("mindiesd.layers.index_modulation")
    install(monkeypatch, module)
    storage = torch.full((rows, width + 7), torch.nan, dtype=torch.bfloat16)
    x = storage[:, 3 : width + 3]
    x.copy_(torch.linspace(-2, 2, rows * width).reshape(rows, width))
    weight = torch.linspace(0.8, 1.2, width).to(torch.bfloat16)
    table = torch.linspace(-0.4, 0.6, 3 * width).reshape(3, width).to(torch.bfloat16)
    indices_storage = torch.full((rows * 2,), 99, dtype=torch.int64)
    indices = indices_storage[::2]
    indices.copy_(torch.arange(rows) % 3)
    out = torch.full_like(x, torch.nan)
    module._rms_norm_indexed_scale_shift_kernel[(cores,)](
        rows,
        width,
        1e-5,
        width,
        x,
        x.stride(0),
        weight,
        table,
        table.stride(0),
        table,
        table.stride(0),
        indices,
        indices.stride(0),
        out,
    )
    xf = x.float()
    expected = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5) * weight.float()
    expected = expected * (1 + table[indices].float()) + table[indices].float()
    torch.testing.assert_close(out, expected.to(out.dtype), rtol=0, atol=0)
    assert torch.isnan(storage[:, :3]).all() and torch.isnan(storage[:, width + 3 :]).all()
