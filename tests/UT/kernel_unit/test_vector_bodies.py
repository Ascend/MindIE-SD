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


def test_rms_norm_indexed_mx_e4m3_full_row_body(monkeypatch):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    rows, hidden = 2, 32
    x = torch.ones((rows, hidden), dtype=torch.bfloat16)
    weight = torch.ones(hidden, dtype=torch.bfloat16)
    shift = torch.ones((3, hidden), dtype=torch.bfloat16)
    scale = torch.zeros((3, hidden), dtype=torch.bfloat16)
    indices = torch.tensor([0, 2], dtype=torch.int64)
    y = torch.empty((rows, hidden), dtype=torch.float8_e4m3fn)
    mxscale = torch.empty((rows, 1, 2), dtype=torch.uint8)
    module._rms_norm_indexed_scale_shift_dynamic_mx_quant_kernel[(rows,)](
        y,
        mxscale,
        x,
        weight,
        shift,
        scale,
        indices,
        0,
        hidden,
        1e-6,
        hidden,
        2,
        3,
        multibuffer=False,
    )
    assert torch.equal(y.float(), torch.full_like(y, 256.0, dtype=torch.float32))
    assert torch.equal(mxscale[:, 0, 0], torch.full((rows,), 120, dtype=torch.uint8))


def _mx_inputs(rows, hidden, dtype, conditions=3):
    x = torch.ones((rows, hidden), dtype=dtype)
    weight = torch.ones(hidden, dtype=dtype)
    shift = torch.ones((conditions, hidden), dtype=dtype)
    scale = torch.full((conditions, hidden), 0.125, dtype=dtype)
    indices = torch.arange(rows, dtype=torch.int64) % conditions
    return x, weight, shift, scale, indices


def _assert_uniform_mx(outputs, rows, hidden, dst_type):
    expected_y = {
        "float8_e4m3fn": 0x78,
        "float8_e5m2": 0x78,
        "float4_e2m1fn_x2": 0x66,
        "float4_e1m2fn_x2": 0x44,
    }[dst_type]
    expected_scale = {
        "float8_e4m3fn": 120,
        "float8_e5m2": 113,
        "float4_e2m1fn_x2": 126,
        "float4_e1m2fn_x2": 128,
    }[dst_type]
    width = hidden // 2 if dst_type.startswith("float4") else hidden
    assert outputs[0].shape == (rows, width)
    assert torch.equal(
        outputs[0].reshape(-1).view(torch.uint8),
        torch.full((rows * width,), expected_y, dtype=torch.uint8),
    )

    n_groups = (hidden + 31) // 32
    n_pad = (n_groups + 1) // 2 * 2
    scale = torch.zeros((rows, n_pad), dtype=torch.uint8)
    for row in range(rows):
        scale[row, :n_groups] = expected_scale
    assert outputs[1].shape == (rows, n_pad // 2, 2)
    assert torch.equal(outputs[1].reshape(-1).view(torch.uint8), scale.reshape(-1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dst_type",
    ["float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2", "float4_e1m2fn_x2"],
)
def test_rms_norm_indexed_mx_general_multirow_body(monkeypatch, dtype, dst_type):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    outputs = module._fused_general_mx(*_mx_inputs(32, 32, dtype), 1e-6, dst_type)
    _assert_uniform_mx(outputs, 32, 32, dst_type)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dst_type",
    ["float8_e4m3fn", "float8_e5m2", "float4_e2m1fn_x2", "float4_e1m2fn_x2"],
)
def test_rms_norm_indexed_mx_general_full_row_body(monkeypatch, dtype, dst_type):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    outputs = module._fused_general_mx(*_mx_inputs(2, 1000, dtype), 1e-6, dst_type)
    _assert_uniform_mx(outputs, 2, 1000, dst_type)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "dst_type,hidden",
    [
        ("float8_e4m3fn", 8192),
        ("float8_e5m2", 8192),
        ("float4_e2m1fn_x2", 4098),
        ("float4_e1m2fn_x2", 4098),
    ],
)
def test_rms_norm_indexed_mx_general_tiled_body(monkeypatch, dtype, dst_type, hidden):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    outputs = module._fused_general_mx(*_mx_inputs(1, hidden, dtype), 1e-6, dst_type)
    _assert_uniform_mx(outputs, 1, hidden, dst_type)


@pytest.mark.parametrize(
    "rows,hidden",
    [(1, 1), (128, 32), (1, 8194), (1, 32769)],
)
def test_rms_norm_indexed_mx_e4m3_branch_bodies(monkeypatch, rows, hidden):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    outputs = module._fused_e4m3(
        *_mx_inputs(rows, hidden, torch.bfloat16),
        1e-6,
        "float8_e4m3fn",
    )
    _assert_uniform_mx(outputs, rows, hidden, "float8_e4m3fn")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("hidden", [1000, 1024])
def test_rms_norm_indexed_mx_hybrid_modulation_body(monkeypatch, dtype, hidden):
    module = importlib.import_module("mindiesd.layers.rms_norm_indexed_scale_shift_dynamic_mx_quant")
    install(monkeypatch, module)
    modulated = []

    def quantize(value, **kwargs):
        del kwargs
        modulated.append(value.clone())
        rows, width = value.shape
        scales = torch.empty(module._mxscale_shape(rows, width), dtype=torch.uint8)
        return torch.empty((rows, width), dtype=torch.float8_e4m3fn), scales

    monkeypatch.setattr(module.torch_npu, "npu_dynamic_mx_quant", quantize)
    outputs = module._hybrid_mx(*_mx_inputs(2, hidden, dtype), 1e-6, "float8_e4m3fn")
    assert outputs[0].shape == (2, hidden)
    assert len(modulated) == 1
    torch.testing.assert_close(modulated[0], torch.full_like(modulated[0], 2.125), rtol=0, atol=0)
