# SPDX-License-Identifier: Apache-2.0
"""Original payload bodies under the existing CPU facade; device proof is separate."""

from types import SimpleNamespace
import sys
import pytest
import torch
from cpu_support import load_parallel_module, new_package
from UT.kernel_unit.triton_cpu import install


@pytest.fixture(name="kernels")
def payload_kernels(monkeypatch):
    monkeypatch.setattr(
        sys.modules["triton.language.extra.cann"], "libdevice", SimpleNamespace(rint=torch.round), raising=False
    )
    module = load_parallel_module(
        new_package("_payload_body_test", monkeypatch), "_quantized_payload_kernels", monkeypatch
    )
    language = install(monkeypatch, module)
    language.cdiv = lambda n, d: (n + d - 1) // d
    store = language.store
    language.store = lambda pointer, value, *args, **kw: store(pointer, torch.as_tensor(value), *args, **kw)
    language.num_programs = lambda axis=0: torch.tensor(language.grid[axis], dtype=torch.int32)
    language.trans = lambda value, axes: value.permute(axes)
    language.fma = lambda a, b, c: (
        torch.as_tensor(a).double() * torch.as_tensor(b).double() + torch.as_tensor(c).double()
    ).float()
    monkeypatch.setattr(module, "libdevice", SimpleNamespace(rint=torch.round))
    power = module.triton.next_power_of_2
    properties = SimpleNamespace(get_device_properties=lambda _: {"num_vectorcore": 2})
    module.triton = SimpleNamespace(
        next_power_of_2=power, runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(utils=properties)))
    )
    return module


def test_int8_scatter_rounding_and_padding(kernels, monkeypatch):
    rows = torch.tensor([0, 1, 3, 4, 6])
    heads = torch.tensor([0, 2, 1, 3])
    source = ((torch.arange(64 * 4 * 128).float() % 127 - 63) / 2).reshape(64, 4, 128).bfloat16()
    scales = torch.ones(4, 1)
    scales[0] = 0
    state = SimpleNamespace(
        payload=torch.full((4, 7, 128), 99, dtype=torch.int8),
        payload_padded=source,
        scales=scales,
        maps=SimpleNamespace(n_owned=5, src=rows, padded_row=rows, gblk=torch.tensor([0])),
        head_row=heads,
        hc=4,
        hs=128,
        nblk=1,
    )
    result = kernels.quantize(state)
    expected = torch.zeros_like(result)
    for head in range(4):
        expected[heads[head], rows] = source[rows, head].float().round().to(torch.int8)
    assert torch.equal(result, expected)
    from importlib import import_module

    config = import_module(kernels.__package__ + ".config")
    monkeypatch.setattr(config, "QUANT_BLOCK", 128)
    with pytest.raises(ValueError, match="QUANT_BLOCK=64"):
        kernels.quantize(state)


@pytest.mark.parametrize("reuse,strided", [(False, False), (True, False), (True, True)])
def test_fp8_send_preserves_source_strides_and_bytes(kernels, reuse, strided):
    value = (torch.arange(4 * 7 * 128).reshape(1, 4, 7, 128).float() % 1024 - 512).transpose(1, 2).bfloat16()
    divisor = torch.full((1, 4, 256 if strided else 128), 2.0)
    divisor = divisor[..., ::2] if strided else divisor
    divisor[:, 0] = 0
    output = torch.empty(2, 2, 7, 1, 1, 128, dtype=torch.int8) if reuse else None
    result = kernels.v_send(value, divisor, 2, 2, out=output)
    expected = (
        value.float()
        .div(torch.where(divisor == 0, 1, divisor))
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
        .reshape(1, 7, 2, 2, 1, 128)
        .permute(3, 2, 1, 0, 4, 5)
        .contiguous()
    )
    assert torch.equal(result.contiguous().view(torch.uint8), expected.view(torch.uint8))
    with pytest.raises(ValueError, match="fixed one-head"):
        kernels.v_send(value, divisor, 2, 2, out=torch.empty(1, dtype=torch.int8))
