"""CPU launch contracts, numerical dispatch, and hardware-property boundaries."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from UT.kernel_unit.triton_cpu import install


@pytest.mark.parametrize("enabled", [False, True])
def test_public_vector_dispatch_math(monkeypatch, enabled):
    x = torch.linspace(-2, 3, 28).reshape(1, 4, 7)
    y = x.flip(-1).clone()
    table = torch.linspace(-0.5, 0.5, 21).reshape(3, 7)
    indices = torch.tensor([2, 0, 1, 2])
    for name in ["muls_add", "scale_shift", "residual_gate_add"]:
        module = importlib.import_module("mindiesd.layers." + name)
        install(monkeypatch, module)
        monkeypatch.setattr(module, "_TRITON_ON_ASCEND", enabled)
        if name == "muls_add":
            torch.testing.assert_close(module.muls_add(x, y, 0.25), x * 0.25 + y)
        elif name == "scale_shift":
            scale = table[indices]
            torch.testing.assert_close(module.scale_shift(x, scale, -scale), x * (1 + scale) - scale)
            torch.testing.assert_close(module.gather_scale_shift(x, table, -table, indices), x * (1 + scale) - scale)
            torch.testing.assert_close(module.gather_residual_gate(x, y, table, indices), x + y * scale)
            torch.testing.assert_close(module.swiglu(torch.cat((x, y), -1)), x * torch.nn.functional.silu(y))
        else:
            gate = table[:1].unsqueeze(0)
            torch.testing.assert_close(module.residual_gate_add(x, y, gate), x + y * gate)


def test_gather_large_width_fallback_and_swiglu_chunks(monkeypatch):
    module = importlib.import_module("mindiesd.layers.scale_shift")
    install(monkeypatch, module)
    x = torch.linspace(-1, 1, 16385).reshape(1, 1, -1)
    scale = x.reshape(1, -1) * 0.2
    indices = torch.tensor([0])
    torch.testing.assert_close(module._gather_scale_shift_triton(x, scale, -scale, indices), x * (1 + scale) - scale)
    torch.testing.assert_close(module._gather_residual_gate_triton(x, -x, scale, indices), x - x * scale)
    proj = torch.linspace(-1, 1, 2 * 7168).reshape(1, -1)
    hidden, gate = proj.chunk(2, -1)
    torch.testing.assert_close(module._swiglu_triton(proj), hidden * torch.nn.functional.silu(gate))


def test_hardware_property_cache_and_failures(monkeypatch):
    module = importlib.import_module("mindiesd.layers.triton_utils")
    monkeypatch.setattr(module, "_HAS_TRITON", True)
    properties = {"num_aicore": 20, "num_vectorcore": 40}
    calls = []

    def query(device):
        calls.append(device)
        return properties

    driver = SimpleNamespace(active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=query)))
    monkeypatch.setattr(module, "triton", SimpleNamespace(runtime=SimpleNamespace(driver=driver)))
    monkeypatch.setattr(torch.npu, "current_device", lambda: 3)
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)
    assert module._check_triton_ascend()
    monkeypatch.setattr(module, "_NUM_AICORE", -1)
    monkeypatch.setattr(module, "_NUM_VECTORCORE", -1)
    assert module.get_aicore_num() == 20
    assert module.get_vectorcore_num() == 40
    count = len(calls)
    assert module.get_aicore_num() == 20 and module.get_vectorcore_num() == 40
    module.init_device_properties_triton()
    assert len(calls) == count
    monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    assert not module._check_triton_ascend()

    def failed_query(device):
        raise RuntimeError("mock driver failure")

    monkeypatch.setattr(driver.active.utils, "get_device_properties", failed_query)
    monkeypatch.setattr(torch.npu, "is_available", lambda: True)
    assert not module._check_triton_ascend()
    monkeypatch.setattr(module, "_NUM_AICORE", -1)
    monkeypatch.setattr(module, "_NUM_VECTORCORE", -1)
    assert module.get_aicore_num() == 1 and module.get_vectorcore_num() == 1
    monkeypatch.setattr(module, "_HAS_TRITON", False)
    assert not module._check_triton_ascend()
    assert module.get_aicore_num() == 1 and module.get_vectorcore_num() == 1


def _indexed_inputs():
    return [
        torch.ones(3, 7, dtype=torch.bfloat16),
        torch.ones(7, dtype=torch.bfloat16),
        torch.ones(2, 7, dtype=torch.bfloat16),
        torch.ones(2, 7, dtype=torch.bfloat16),
        torch.tensor([0, 1, 0]),
        1e-5,
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        "rank",
        "zero_width",
        "wide",
        "weight",
        "shift",
        "scale",
        "indices",
        "index_dtype",
        "dtype",
        "stride",
        "negative_eps",
        "overflow",
        "nan",
    ],
)
def test_indexed_contract_rejects_invalid_inputs(invalid):
    module = importlib.import_module("mindiesd.layers.index_modulation")
    args = _indexed_inputs()
    if invalid == "rank":
        args[0] = args[0].unsqueeze(0)
    elif invalid == "zero_width":
        args[0] = args[0][:, :0]
    elif invalid == "wide":
        args[0] = torch.empty(1, 8193)
    elif invalid == "weight":
        args[1] = args[1][:-1]
    elif invalid == "shift":
        args[2] = args[2][:0]
    elif invalid == "scale":
        args[3] = args[3][:1]
    elif invalid == "indices":
        args[4] = args[4][:-1]
    elif invalid == "index_dtype":
        args[4] = args[4].float()
    elif invalid == "dtype":
        args[1] = args[1].float()
    elif invalid == "stride":
        args[0] = torch.ones(3, 14, dtype=torch.bfloat16)[:, ::2]
    elif invalid == "negative_eps":
        args[5] = -1e-5
    elif invalid == "overflow":
        args[5] = 1e100
    else:
        args[5] = float("nan")
    with pytest.raises(ValueError):
        module._check_inputs(*args)


def test_indexed_driver_checks(monkeypatch):
    module = importlib.import_module("mindiesd.layers.index_modulation")
    monkeypatch.setattr(module, "_HAS_TRITON", False)
    with pytest.raises(RuntimeError, match="triton"):
        module._get_num_vector_cores(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="triton"):
        module.rms_norm_indexed_scale_shift(*_indexed_inputs())
    monkeypatch.setattr(module, "_HAS_TRITON", True)
    with pytest.raises(ValueError, match="NPU"):
        module.rms_norm_indexed_scale_shift(*_indexed_inputs())
    driver = SimpleNamespace(
        active=SimpleNamespace(
            get_current_device=lambda: 2,
            utils=SimpleNamespace(get_device_properties=lambda device: {"num_vectorcore": 4}),
        )
    )
    monkeypatch.setattr(module, "driver", driver)
    assert module._get_num_vector_cores(torch.device("cpu")) == 4
    monkeypatch.setattr(driver.active.utils, "get_device_properties", lambda device: {"num_vectorcore": 0})
    with pytest.raises(RuntimeError, match="Invalid"):
        module._get_num_vector_cores(torch.device("cpu", 0))


@pytest.mark.parametrize("empty", [False, True])
def test_indexed_public_launch_math_with_device_interface_mock(monkeypatch, empty):
    module = importlib.import_module("mindiesd.layers.index_modulation")
    install(monkeypatch, module)
    device = SimpleNamespace(type="npu", index=0)

    class DeviceTensor:
        def __init__(self, value):
            self.cpu_tensor = value
            self.device = device

        def __getattr__(self, name):
            return getattr(self.cpu_tensor, name)

    args = _indexed_inputs()
    if empty:
        args[0] = args[0][:0]
        args[4] = args[4][:0]
    args[0].copy_(torch.linspace(-1, 2, args[0].numel()).reshape(args[0].shape))
    proxy_args = [DeviceTensor(value) for value in args[:-1]] + args[-1:]
    original_empty = torch.empty

    def empty_ignoring_device(*args, device=None, **kwargs):
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_ignoring_device)
    monkeypatch.setattr(torch.npu, "device", lambda selected: nullcontext())
    monkeypatch.setattr(
        module,
        "driver",
        SimpleNamespace(
            active=SimpleNamespace(utils=SimpleNamespace(get_device_properties=lambda selected: {"num_vectorcore": 2}))
        ),
    )
    actual = module.rms_norm_indexed_scale_shift(*proxy_args)
    x, weight, shift, scale, indices, eps = args
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight.float()
    expected = expected * (1 + scale[indices].float()) + shift[indices].float()
    torch.testing.assert_close(actual, expected.to(x.dtype), rtol=0, atol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("invalid", ["rank", "branch", "gate", "empty", "device", "dtype", "stride"])
def test_residual_launch_input_contract(monkeypatch, invalid):
    module = importlib.import_module("mindiesd.layers.residual_gate_add")
    install(monkeypatch, module)
    x = torch.ones(2, 3, 7)
    value = x.clone()
    gate = torch.ones(2, 1, 7)
    if invalid == "rank":
        x, value, gate = x[0], value[0], gate[0]
        torch.testing.assert_close(module._residual_gate_add_triton(x, value, gate), x + value * gate)
        return
    if invalid == "branch":
        value = value[:, :2]
    elif invalid == "gate":
        gate = x.clone()
    elif invalid == "empty":
        x, value = x[:, :0], value[:, :0]
    elif invalid == "device":
        gate = torch.empty(gate.shape, device="meta")
    elif invalid == "dtype":
        gate = gate.double()
    elif invalid == "stride":
        x, value = x.transpose(0, 1), value.transpose(0, 1)
        gate = torch.ones(3, 1, 7)
    with pytest.raises(ValueError):
        module._residual_gate_add_triton(x, value, gate)


@pytest.mark.parametrize("invalid", [None, "rank", "branch", "gate", "dtype"])
@pytest.mark.parametrize("enabled", [False, True])
def test_qwen_public_contract_and_math(monkeypatch, invalid, enabled):
    module = importlib.import_module("mindiesd.layers.residual_gate_add")
    install(monkeypatch, module)
    monkeypatch.setattr(module, "_TRITON_ON_ASCEND", enabled)
    x = torch.linspace(-1, 2, 42).reshape(2, 3, 7)
    branch = x.flip(-1).clone()
    gate = torch.full((2, 1, 7), 0.3)
    if invalid == "rank":
        x = x[0]
    elif invalid == "branch":
        branch = branch[:, :2]
    elif invalid == "gate":
        gate = gate[:, 0]
    elif invalid == "dtype":
        gate = gate.double()
    if invalid:
        with pytest.raises(ValueError):
            module.qwen_residual_gate_add(x, branch, gate)
    else:
        torch.testing.assert_close(module.qwen_residual_gate_add(x, branch, gate), x + branch * gate)
