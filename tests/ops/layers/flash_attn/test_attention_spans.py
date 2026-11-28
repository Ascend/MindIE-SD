import pytest
import torch

from mindiesd import block_causal_attention
from mindiesd.layers.flash_attn._prebuilt import PrebuiltUnavailable
from mindiesd.utils.get_platform import is_a5_device


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("op", [None, "auto", "piecewise", "flex"])
@torch.inference_mode()
def test_structured_attention_against_dense(op, head_first):
    if not is_a5_device():
        pytest.skip("950 device test")
    torch.manual_seed(10103)
    device = torch.device("npu", torch.npu.current_device())
    length = 391
    spans = ((7, 19), (120, 272), (300, 389))
    query, key, value = (torch.randn(1, length, 2, 128, device=device, dtype=torch.bfloat16) * 0.2 for _ in range(3))
    positions = torch.arange(length, device=device)
    keep = positions[:, None] >= positions
    for start, end in spans:
        keep[start:end, start:end] = True
    q_heads, k_heads, v_heads = (tensor.transpose(1, 2).float() for tensor in (query, key, value))
    scores = (q_heads @ k_heads.transpose(-1, -2)) * 0.125
    expected = scores.masked_fill(~keep, -torch.inf).softmax(-1) @ v_heads
    if head_first:
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
    else:
        expected = expected.transpose(1, 2)
    options = {"backend": op} if op else {}
    try:
        actual = block_causal_attention(query, key, value, spans=spans, head_first=head_first, scale=0.125, **options)
    except PrebuiltUnavailable as exc:
        if op != "flex":
            raise
        pytest.skip(f"This wheel has no compatible executor: {exc}")
    repeated = block_causal_attention(query, key, value, spans=spans, head_first=head_first, scale=0.125, **options)
    assert torch.equal(actual, repeated)
    torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=0.003)
