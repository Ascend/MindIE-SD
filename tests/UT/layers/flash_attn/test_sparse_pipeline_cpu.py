# pylint: disable=redefined-outer-name

"""CPU sparse pipeline permutation/mask contracts; only native launches are doubled."""

import importlib
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from mindiesd.utils import ParametersInvalid

v2 = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2")
v3 = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3")
public = importlib.import_module("mindiesd.layers.flash_attn.sparse_flash_attn")


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("shape", [(1, 9, 8), (1, 8, 9), (1, 9, 9), (1, 6, 6)])
def test_single_frame_unaligned_rearrange_preserves_first_frame(layout, shape):
    sequence = shape[1] * shape[2]
    tensor = torch.arange(sequence * 8).reshape(1, sequence, 2, 4).float()
    if layout == "BNSD":
        tensor = tensor.transpose(1, 2)
    actual = v2.rearrange_with_remaining(tensor, shape, shape, layout)
    torch.testing.assert_close(actual, tensor)
    restored = v2.inv_rearrange_with_remaining(actual, shape, shape, layout)
    torch.testing.assert_close(restored, tensor)


@pytest.fixture
def native_sparse():
    def dynamic(x, **kwargs):
        return x.to(torch.int8), torch.ones(x.shape[:-1])

    def block(x, **kwargs):
        return x.clone(), torch.ones(x.shape[:-1])

    def sparse(*args, **kwargs):
        q = kwargs["query"]
        return torch.full(q.shape, 3.0, dtype=torch.float32), torch.ones(1)

    with ExitStack() as stack:
        calls = {}
        for name, impl in (("npu_dynamic_quant", dynamic), ("npu_dynamic_block_quant", block)):
            calls[name] = stack.enter_context(patch.object(v3.torch_npu, name, side_effect=impl))
        for name in ("block_sparse_attention", "eagle_quant_block_sparse_attention"):
            calls[name] = stack.enter_context(patch.object(torch.ops.mindiesd, name, side_effect=sparse, create=True))
        yield calls


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("shape", [(2, 8, 8), (2, 9, 8), (2, 8, 9), (2, 9, 9)])
@pytest.mark.parametrize("precision", ["bf16", "mix", "fp8"])
@pytest.mark.parametrize("text", [0, 8])
def test_rf_v3_actual_rearrange_mask_quant_and_inverse(native_sparse, layout, shape, precision, text):
    sequence = torch.tensor(shape).prod().item() + text
    q = torch.arange(sequence * 2 * 64).reshape(1, sequence, 2, 64).float() / 1000
    if layout == "BNSD":
        q = q.transpose(1, 2).contiguous()
    output, mask = v3.bsa_sparse_attention_v3(
        q,
        q,
        q,
        shape,
        txt_len=text,
        precision=precision,
        input_layout=layout,
        block_size=64,
        block_size_kv=256,
        sparsity=0.5,
    )
    assert output.shape == q.shape and mask.dtype == torch.int8
    torch.testing.assert_close(output, torch.full_like(output, 3))
    # Cached masks follow the actual rearrange-only path on the second step.
    cached_output, cached = v3.bsa_sparse_attention_v3(
        q,
        q,
        q,
        shape,
        txt_len=text,
        precision=precision,
        input_layout=layout,
        block_size=64,
        block_size_kv=256,
        cached_mask=mask,
    )
    assert cached_output.shape == q.shape
    torch.testing.assert_close(cached, mask)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_multispan_permutation_inverts_and_protects_dense_context(layout):
    q = torch.arange(272 * 8).reshape(1, 272, 1, 8).float()
    if layout == "BNSD":
        q = q.transpose(1, 2)
    spans = [dict(start=8, latent_shape=[2, 8, 8]), dict(start=144, latent_shape=[2, 8, 8])]
    qr, kr, vr, pool, inverse, dense, first, reference_plan = v2.do_multi_span_tensor_rearrange_pooling(
        q, q, q, spans, 64, layout, return_reference_plan=True
    )
    assert reference_plan == ((), ())
    torch.testing.assert_close(qr.index_select(1 if layout == "BSND" else 2, inverse), q)
    mask = v2.get_multi_span_blockwise_mask(pool, 1.0, 0.3, dense, first, layout, return_binary=True)
    for index in dense | first:
        assert torch.all(mask[:, :, index]) and torch.all(mask[:, :, :, index])
    indices, counts = v2.get_multi_span_blockwise_mask(pool, 0.5, 0.3, dense, first, layout)
    assert indices.shape[:-1] == counts.shape


@pytest.mark.parametrize(
    "spans",
    [
        None,
        "bad",
        [],
        [1],
        [{}],
        [dict(start=0, latent_shape=[0, 8, 8])],
        [dict(start=0, latent_shape=[1, 8])],
        [dict(start=-1, latent_shape=[1, 8, 8])],
        [dict(start=200, latent_shape=[1, 8, 8])],
        [dict(start=0, latent_shape=[1, 8, 8]), dict(start=32, latent_shape=[1, 8, 8])],
    ],
)
def test_multispan_invalid_descriptors(spans):
    with pytest.raises(ParametersInvalid):
        v2._normalize_video_spans(spans, 256)


@pytest.mark.parametrize("qblock,kblock", [(64, 64), (128, 64), (64, 128), (128, 128)])
def test_mask_any_merge_retains_selected_cells(qblock, kblock):
    mask = torch.eye(3, dtype=torch.int8).reshape(1, 1, 3, 3)
    actual = v3._adapt_mask_for_block_sizes(mask, qblock, kblock, 64)
    expected = torch.zeros(1, 1, (3 * 64 + qblock - 1) // qblock, (3 * 64 + kblock - 1) // kblock, dtype=torch.int8)
    for index in range(3):
        expected[0, 0, index * 64 // qblock, index * 64 // kblock] = 1
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError):
        v3._adapt_mask_for_block_sizes(mask, 65, 64, 64)
    with pytest.raises(ValueError):
        v3._adapt_mask_for_block_sizes(mask, 64, 65, 64)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("kind", ["rf_v2", "rf_v3", None, "ada_bsa"])
def test_public_sparse_routing_to_actual_native_boundary(native_sparse, layout, multi, kind):
    shape = (2, 8, 8)
    sequence = 272 if multi else 136
    q = torch.arange(sequence * 2 * 8).reshape(1, sequence, 2, 8).float()
    if layout == "BNSD":
        q = q.transpose(1, 2).contiguous()
    spans = [dict(start=8, latent_shape=shape), dict(start=144, latent_shape=shape)]
    kwargs = dict(head_num=2, input_layout=layout, sparse_type=kind, block_size=64, sparsity=0.5, inner_precise=4)
    if kind in ("rf_v2", "rf_v3"):
        if multi:
            kwargs.update(video_spans=spans)
        else:
            kwargs.update(txt_len=8, latent_shape_q=shape, latent_shape_k=shape)
    with (
        patch.object(public, "is_a5_device", return_value=False),
        patch.object(v2, "is_a5_device", return_value=False),
        patch("mindiesd.layers.flash_attn.sparse_flash_attn_ada_bsa.is_a5_device", return_value=False),
        patch.object(
            torch.ops.mindiesd, "rainfusionattention", side_effect=lambda **kw: (kw["query"].clone(), None), create=True
        ) as rain,
        patch.object(
            torch.ops.mindiesd, "sparse_block_estimate", return_value=(torch.ones(1), torch.ones(1)), create=True
        ),
        patch.object(
            torch.ops.mindiesd, "ada_block_sparse_attention", side_effect=lambda **kw: kw["query"].clone(), create=True
        ),
        patch.object(public.torch_npu, "npu_fusion_attention", side_effect=lambda x, *a, **kw: (x.clone(),)),
    ):
        actual = public.sparse_attention(q, q, q, **kwargs)
    assert actual.shape == q.shape
    if kind != "rf_v3":
        torch.testing.assert_close(actual, q)
    if kind == "rf_v2":
        assert rain.call_args.kwargs["q_input_layout"] == ("TND" if layout == "BSND" else "BNSD")


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(input_layout="bad"),
        dict(sparse_type="bad"),
        dict(video_spans=[], sparse_type=None),
        dict(video_spans=[], sparse_type="rf_v2", txt_len=1),
    ],
)
def test_public_rejects_invalid_sparse_options(kwargs):
    q = torch.ones(1, 2, 128, 8)
    with pytest.raises(ParametersInvalid):
        public.sparse_attention(q, q, q, **kwargs)


def test_all_video_queries_select_reference_images_independently():
    query = torch.ones(1, 1728, 1, 1)
    key = torch.arange(1728).reshape(1, 1728, 1, 1).float() / 1728
    videos = [{"start": 1280, "latent_shape": [3, 8, 8]}, {"start": 1472, "latent_shape": [3, 8, 8]}]
    images = [{"start": 0, "latent_shape": [8, 80]}, {"start": 640, "latent_shape": [8, 80]}]
    qr, kr, _, pooled, inverse, dense, first, reference_plan = v2.do_multi_span_tensor_rearrange_pooling(
        query,
        key,
        key,
        videos,
        64,
        "BSND",
        image_spans=images,
        return_reference_plan=True,
    )
    torch.testing.assert_close(qr.index_select(1, inverse), query)
    torch.testing.assert_close(kr.index_select(1, inverse), key)
    assert reference_plan == ((tuple(range(10)), tuple(range(10, 20))), (20, 21, 22, 23, 24, 25))
    mask = v2.get_multi_span_blockwise_mask(
        pooled,
        0.2,
        1.0,
        dense,
        first,
        "BSND",
        return_binary=True,
        reference_plan=reference_plan,
        image_sparsity=0.9,
    ).bool()
    # Each target query independently retains one of each image's ten KV blocks.
    assert torch.all(mask[..., 24:26, :10].sum(-1) == 1)
    assert torch.all(mask[..., 24:26, 10:20].sum(-1) == 1)
    assert mask[..., 24:26, 9].all() and mask[..., 24:26, 19].all()
    # Both video spans select image blocks; first frames and text remain protected.
    assert torch.all(mask[..., 21:23, :10].sum(-1) == 1)
    assert torch.all(mask[..., 21:23, 10:20].sum(-1) == 1)
    for index in first | {26}:
        assert mask[..., index, :].all() and mask[..., :, index].all()


def test_sparse_interfaces_do_not_expose_target_ranges():
    import inspect

    for function in (
        public.sparse_attention,
        v3.bsa_sparse_attention_v3,
        v2.do_multi_span_tensor_rearrange_pooling,
        v2._multi_span_permutation,
    ):
        assert "target_ranges" not in inspect.signature(function).parameters


@pytest.mark.parametrize("value", ["0.5", True, -0.1, 1.1, float("nan"), float("inf")])
def test_public_rejects_invalid_image_sparsity(value):
    q = torch.ones(1, 2, 128, 8)
    with pytest.raises(ParametersInvalid):
        public.sparse_attention(q, q, q, sparse_type="rf_v2", video_spans=[], image_spans=[], image_sparsity=value)


@pytest.mark.parametrize("function", [public.sparse_attention, v3.bsa_sparse_attention_v3])
@pytest.mark.parametrize("kwargs", [{"image_spans": []}, {"image_sparsity": 0.0}, {"image_sparsity": 0.9}])
def test_image_options_must_be_provided_together(function, kwargs):
    with pytest.raises(ParametersInvalid, match="provided together"):
        extra = {"latent_shape_q": None} if function is v3.bsa_sparse_attention_v3 else {}
        function(None, None, None, **extra, **kwargs)


@pytest.mark.parametrize("function", [public.sparse_attention, v3.bsa_sparse_attention_v3])
def test_image_options_default_to_none(function):
    import inspect

    signature = inspect.signature(function)
    assert signature.parameters["image_spans"].default is None
    assert signature.parameters["image_sparsity"].default is None


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_reference_images_preserve_upstream_pool_and_pre_rearranged_contracts(native_sparse, layout):
    q = torch.arange(448 * 8).reshape(1, 448, 1, 8).float() / 1000
    k, v = q.flip(1), q + 1
    if layout == "BNSD":
        q, k, v = (x.transpose(1, 2).contiguous() for x in (q, k, v))
    videos = [dict(start=128, latent_shape=[3, 8, 8])]
    images = [dict(start=0, latent_shape=[8, 16])]
    args = (q, k, v, videos, 64, layout)
    full = v2.do_multi_span_tensor_rearrange_pooling(*args, image_spans=images)
    assert len(full) == 7
    qk = v2.do_multi_span_tensor_rearrange_pooling(*args, image_spans=images, pool_parts=2, return_reference_plan=True)
    assert len(qk) == 8 and qk[-1][0]
    torch.testing.assert_close(qk[3], full[3][:2])
    skipped = v2.do_multi_span_tensor_rearrange_pooling(*args, image_spans=images, pool_parts=0)
    assert skipped[3] is None
    for actual, expected in zip(skipped[:3], full[:3]):
        torch.testing.assert_close(actual, expected)
    options = dict(
        latent_shape_q=None,
        video_spans=videos,
        image_spans=images,
        image_sparsity=0.9,
        precision="bf16",
        input_layout=layout,
        block_size=64,
        block_size_kv=64,
        sparsity=0.8,
    )
    normal, mask = v3.bsa_sparse_attention_v3(q, k, v, **options)
    reordered, reordered_mask = v3.bsa_sparse_attention_v3(*qk[:3], pre_rearranged=True, **options)
    torch.testing.assert_close(mask, reordered_mask)
    torch.testing.assert_close(reordered.index_select(1 if layout == "BSND" else 2, qk[4]), normal)
    cached, cached_mask = v3.bsa_sparse_attention_v3(
        *qk[:3], pre_rearranged=True, cached_mask=reordered_mask, **options
    )
    torch.testing.assert_close(cached, reordered)
    torch.testing.assert_close(cached_mask, reordered_mask)
