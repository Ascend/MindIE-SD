# pylint: disable=redefined-outer-name

"""CPU contracts for quantized linear preprocessing and device API dispatch."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch_npu

from mindiesd.quantization import layer as linear
from mindiesd.quantization.config import QuantConfig, TimestepPolicyConfig
from mindiesd.quantization.utils import OnlineFARotWeights


@pytest.fixture
def device_boundaries():
    """Replace only device transfers, quantizers, format conversion and matmuls."""

    def dynamic(x, **kwargs):
        assert x.device.type == "cpu"
        return x.to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)

    def mx(x, **kwargs):
        assert x.device.type == "cpu"
        return x.to(torch.uint8), torch.ones(x.shape[0], 2)

    def dual(x, **kwargs):
        quantized, scale = mx(x)
        return quantized, torch.ones(x.shape[0], 1), scale

    def matmul(x, *args, **kwargs):
        assert x.device.type == "cpu"
        # A distinctive return value makes flatten/unflatten and dispatch observable.
        return torch.full((*x.shape[:-1], 8), 3, dtype=kwargs.get("output_dtype", x.dtype))

    replacements = {
        "npu_dynamic_quant": dynamic,
        "npu_dynamic_mx_quant": mx,
        "npu_dynamic_dual_level_mx_quant": dual,
        "npu_quantize": lambda x, **kwargs: x.to(torch.int8),
        "npu_format_cast": lambda x, *args, **kwargs: x.clone(),
        "npu_convert_weight_to_int4pack": lambda x: x.clone(),
        "npu_dtype_cast": lambda x, dtype: x.to(torch.uint8),
        "npu_quant_matmul": matmul,
        "npu_dual_level_quant_matmul": matmul,
        "npu_weight_quant_batchmatmul": matmul,
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.Tensor, "npu", lambda tensor, *args, **kwargs: tensor))
        mocks = {
            name: stack.enter_context(patch.object(torch_npu, name, side_effect=implementation, create=True))
            for name, implementation in replacements.items()
        }
        yield mocks


def weights(prefix="fc2", scale_width=2, timestep=False):
    values = {
        "weight": torch.ones(8, 64, dtype=torch.uint8),
        "weight_scale": torch.ones(8, scale_width),
        "weight_dual_scale": torch.ones(8, 1, 1),
        "bias": torch.ones(8),
        "quant_bias": torch.ones(2, 8) if timestep else torch.ones(8),
        "deq_scale": torch.ones(2, 8) if timestep else torch.ones(8),
        "input_scale": torch.ones(2, 1) if timestep else torch.ones(1),
        "input_offset": torch.zeros(2, 1) if timestep else torch.zeros(1),
    }
    return OnlineFARotWeights({f"{prefix}.{key}": value for key, value in values.items()})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 64), (2, 3, 64)])
@pytest.mark.parametrize(
    "nz,w4,bias", [(False, False, False), (True, False, True), (False, True, True), (True, True, False)]
)
def test_weight_only_transfer_pack_and_shape(device_boundaries, dtype, shape, nz, w4, bias):
    model = linear.WeightQuantLinear(
        64, 8, bias=bias, weights=weights(), prefix="fc2", dtype=dtype, use_nz=nz, is_w4=w4
    )
    result = model(torch.ones(shape))
    call = device_boundaries["npu_weight_quant_batchmatmul"].call_args
    assert call.args[0].dtype == dtype
    assert call.args[0].shape == (torch.tensor(shape[:-1]).prod().item(), 64)
    assert result.shape == (*shape[:-1], 8)
    torch.testing.assert_close(result, torch.full_like(result, 3))
    assert (call.kwargs["bias"] is not None) == bias
    if bias:
        assert model.bias.dtype == torch.float32
    assert device_boundaries["npu_convert_weight_to_int4pack"].call_count == int(w4)
    assert device_boundaries["npu_format_cast"].call_count == int(nz)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dynamic,nz,mul", [(False, False, False), (False, False, True), (True, True, False), (True, False, True)]
)
@pytest.mark.parametrize("shape", [(3, 64), (2, 3, 64)])
def test_w8a8_static_and_dynamic_contract(device_boundaries, dtype, dynamic, nz, mul, shape):
    model = linear.W8A8QuantLinear(
        64,
        8,
        weights=weights(),
        prefix="fc2",
        dtype=dtype,
        is_dynamic=dynamic,
        use_nz=nz,
        mul_scale=torch.full((64,), 2.0) if mul else None,
    )
    result = model(torch.ones(shape))
    assert result.shape == (*shape[:-1], 8)
    quantizer = device_boundaries["npu_dynamic_quant" if dynamic else "npu_quantize"]
    torch.testing.assert_close(
        quantizer.call_args.args[0], torch.full((result.numel() // 8, 64), 2 if mul else 1, dtype=dtype)
    )
    kwargs = device_boundaries["npu_quant_matmul"].call_args.kwargs
    assert kwargs["output_dtype"] == dtype
    assert ("pertoken_scale" in kwargs) == dynamic
    if not dynamic:
        assert model.input_scale.shape == (64,)
        assert model.input_offset.dtype == (torch.int8 if dtype == torch.float16 else torch.bfloat16)


@pytest.mark.parametrize(
    "cls",
    [linear.W4A4QuantLinear, linear.W8A8MXFP8QuantLinear, linear.W4A4MXFP4QuantLinear, linear.W4A4MXFP4DualQuantLinear],
)
@pytest.mark.parametrize("nz,mul", [(False, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_offline_dynamic_preprocessing(device_boundaries, cls, nz, mul, dtype):
    model = cls(
        64,
        8,
        weights=weights(scale_width=3 if cls in (linear.W8A8MXFP8QuantLinear, linear.W4A4MXFP4QuantLinear) else 2),
        prefix="fc2",
        use_nz=nz,
        dtype=dtype,
        mul_scale=torch.ones(64) if mul else None,
    )
    result = model(torch.ones(2, 3, 64))
    assert result.shape == (2, 3, 8)
    torch.testing.assert_close(result, torch.full_like(result, 3))
    api = "npu_dual_level_quant_matmul" if cls is linear.W4A4MXFP4DualQuantLinear else "npu_quant_matmul"
    call = device_boundaries[api].call_args
    assert call.args[0].shape[0] == 6
    assert call.kwargs["output_dtype"] == dtype
    if cls in (linear.W8A8MXFP8QuantLinear, linear.W4A4MXFP4QuantLinear):
        assert model.weight_scale.shape == (8, 2, 2)
        assert torch.count_nonzero(model.weight_scale[:, -1, -1]) == 0


@pytest.mark.parametrize(
    "cls",
    [
        linear.W8A8OnlineQuantLinear,
        linear.W8A8MXFP8OnlineQuantLinear,
        linear.W4A4MXFP4OnlineQuantLinear,
        linear.W4A4MXFP4DualOnlineQuantLinear,
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 64), (2, 3, 64)])
def test_online_weights_and_forward(device_boundaries, cls, bias, dtype, shape):
    original = torch.nn.Linear(64, 8, bias=bias)
    model = cls(original, dtype=dtype)
    result = model(torch.ones(shape))
    assert result.shape == (*shape[:-1], 8)
    assert result.dtype == dtype
    torch.testing.assert_close(result, torch.full_like(result, 3))
    api = "npu_dual_level_quant_matmul" if cls is linear.W4A4MXFP4DualOnlineQuantLinear else "npu_quant_matmul"
    call = device_boundaries[api].call_args
    assert (call.kwargs["bias"] is not None) == bias
    assert call.args[0].shape == (result.numel() // 8, 64)
    assert all(buffer.device.type == "cpu" for buffer in model.buffers())


@pytest.mark.parametrize(
    "cls", [linear.W4A4MXFP4QuantLinear, linear.W4A4MXFP4OnlineQuantLinear, linear.W4A4MXFP4DualOnlineQuantLinear]
)
@pytest.mark.parametrize("bias", [False, True])
def test_w4a8_policy_selects_native_contract(device_boundaries, cls, bias):
    policy = QuantConfig(timestep_config=TimestepPolicyConfig(w4a4_default_strategy="W4A8"))
    if cls is linear.W4A4MXFP4QuantLinear:
        model = cls(64, 8, bias=bias, weights=weights(), prefix="fc2", quant_config=policy)
    else:
        model = cls(torch.nn.Linear(64, 8, bias=bias), quant_config=policy)
    result = model(torch.ones(3, 64))
    call = device_boundaries["npu_quant_matmul"].call_args
    assert result.shape == (3, 8)
    assert call.kwargs["group_sizes"] == [0, 0, 32]
    assert call.kwargs["x2_dtype"] == torch_npu.float4_e2m1fn_x2
    if bias:
        assert call.kwargs["bias"].shape == (1, 8)
        assert call.kwargs["bias"].dtype == torch.bfloat16
    else:
        assert call.kwargs["bias"] is None
