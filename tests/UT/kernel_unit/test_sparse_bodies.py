# pylint: disable=redefined-outer-name,attribute-defined-outside-init

"""Sparse attention subkernel math against independent PyTorch references."""

import importlib
import inspect

import pytest
import torch

from UT.kernel_unit.triton_cpu import Ascend, BlockPointer, Pointer, install


@pytest.fixture
def sparse(monkeypatch):
    module = importlib.import_module("mindiesd.layers.flash_attn.sparse_linear_attn_triton")
    install(monkeypatch, module)
    return module


@pytest.mark.parametrize("length,block", [(7, 4), (8, 4), (13, 8)])
def test_pool_and_block_selection(sparse, length, block):
    x = torch.linspace(-2, 2, 2 * length * 4).reshape(1, 2, length, 4)
    expected = torch.stack([part.mean(-2) for part in x.split(block, dim=-2)], dim=-2)
    torch.testing.assert_close(sparse.mean_pool(x, block), expected)
    key = x.flip(-1).clone()
    result, lut, topk = sparse.get_block_map(x, key, 0.75, block, block)
    centered = key - key.mean(-2, keepdim=True)
    pooled_k = torch.stack([part.mean(-2) for part in centered.split(block, dim=-2)], dim=-2)
    scores = expected @ pooled_k.transpose(-1, -2)
    expected_lut = scores.topk(int(0.75 * scores.shape[-1]), sorted=False).indices
    assert topk == expected_lut.shape[-1]
    assert torch.equal(lut.sort(-1).values, expected_lut.sort(-1).values)
    assert torch.equal(result, torch.zeros_like(result).scatter_(-1, expected_lut, 1))


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("stage", [1, 2])
def test_softmax_statistics_and_nz_storage(sparse, masked, update, stage):
    rows, cols = 16, 32
    qk = torch.linspace(-3, 4, rows * cols).reshape(rows, cols)
    mask = ((torch.arange(rows * cols).reshape(rows, cols) % 7) == 0).to(torch.int8)
    mask_pointer = BlockPointer(Pointer(mask), mask.shape, mask.stride(), (0, 0), mask.shape, (1, 0))
    m = torch.linspace(-0.4, 0.8, rows)
    normalizers = torch.linspace(0.5, 1.5, rows)
    old_m, old_l = m.clone(), normalizers.clone()
    alpha = torch.full((rows,), torch.nan)
    p_nz = torch.full((cols // 16, rows, 16), torch.nan)
    scratch = torch.full_like(qk, torch.nan)
    sparse.softmax_vf_select(
        torch.tensor(1),
        torch.tensor(masked),
        torch.tensor(update),
        qk,
        0.7,
        mask_pointer,
        m,
        normalizers,
        alpha,
        p_nz,
        scratch,
        torch.float32,
        rows * 2,
        cols,
        stage,
    )
    scores = qk * 0.7
    if masked and stage == 1:
        scores = scores + torch.where(mask.bool(), -1e4, 0)
    expected_m = scores.max(-1).values
    if update:
        expected_m = torch.maximum(expected_m, old_m)
    probabilities = (scores - expected_m[:, None]).exp()
    expected_l = probabilities.sum(-1)
    if update:
        expected_alpha = (old_m - expected_m).exp()
        expected_l = expected_l + old_l * expected_alpha
        torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(m, expected_m)
    torch.testing.assert_close(normalizers, expected_l)
    unpacked = p_nz.permute(1, 0, 2).reshape(rows, cols)
    torch.testing.assert_close(unpacked, probabilities)
    assert sparse.al.events


@pytest.mark.parametrize("length", [4, 7, 9])
def test_a2_forward_and_backward_against_autograd(sparse, length):
    block, width = 4, 4
    torch.manual_seed(70 + length)
    q, k, v = [torch.randn(1, 1, length, width, requires_grad=True) for _ in range(3)]
    blocks = (length + block - 1) // block
    lut = torch.arange(blocks, dtype=torch.int32).repeat(1, 1, blocks, 1)
    output = torch.full_like(q, torch.nan)
    lse = torch.full((1, 1, length), torch.nan)
    sparse._attn_fwd_A2[(2,)](
        q, k, v, 0.5, blocks, lut, lse, output, 1, 1, length, width, block, block, block, blocks, 2
    )
    logits = q @ k.transpose(-1, -2) * 0.5
    expected = logits.softmax(-1) @ v
    torch.testing.assert_close(output, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lse, torch.logsumexp(logits, -1) / torch.log(torch.tensor(2.0)))
    grad = torch.randn_like(output)
    delta = torch.full_like(lse, torch.nan)
    sparse._attn_bwd_preprocess[(2,)](output, grad, delta, length, width)
    torch.testing.assert_close(delta, (output * grad).sum(-1))
    dq, dk, dv = [torch.full_like(q, torch.nan) for _ in range(3)]
    sparse._attn_bwd_dq[(blocks,)](
        Q=q,
        K=k,
        V=v,
        LSE=lse,
        DELTAS=delta,
        DOS=grad,
        DQ=dq,
        LUT=lut,
        qk_scale=0.5,
        topk=blocks,
        B=1,
        H=1,
        L=length,
        M_BLOCKS=blocks,
        D=width,
        BLOCK_M=block,
        BLOCK_N=block,
        sub_BN=2,
    )
    kbid = torch.ones(1, 1, blocks, blocks, dtype=torch.int32)
    sparse._attn_bwd_dkdv[(2,)](
        q, k, v, grad, dk, dv, kbid, lse, delta, 0.5, 1, 1, length, width, block, block, block, blocks, blocks, 2
    )
    expected.backward(grad)
    for actual, reference in [(dq, q.grad), (dk, k.grad), (dv, v.grad)]:
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-6)


@pytest.mark.parametrize("slot", range(4))
def test_task_ring_preserves_other_slots(sparse, slot):
    ring = tuple(range(20))
    replacement = (11, 12, 13, 14, 15)
    updated = sparse.create_task(slot, *replacement, *ring)
    expected = list(ring)
    expected[slot * 5 : slot * 5 + 5] = replacement
    assert tuple(updated) == tuple(expected)
    assert tuple(sparse.get_cur_task(slot, *updated)) == replacement
    source = (2, 0, 4, 0)
    changed = sparse.update_s2_loop(slot, 9, *source)
    expected_s2 = list(source)
    expected_s2[slot] = 9
    assert tuple(changed) == tuple(expected_s2)
    assert bool(sparse.is_need_update(slot, *source)) == (source[slot] != 0)
    assert bool(sparse.is_first_skv_loop(slot, *source)) == (source[slot] == 0)
    sizes = tuple(v + 1 for v in source)
    pairs = tuple(value for pair in zip(source, sizes) for value in pair)
    assert sparse.is_last_skv(slot, *pairs)
    slots = (11, 12, 13, 14)
    mapped = sparse.update_task(slot, 8, *slots)
    expected_slots = list(slots)
    expected_slots[slot] = 2
    assert tuple(mapped) == tuple(expected_slots)
    assert sparse.get_s_task(slot, *mapped) == 2


@pytest.mark.parametrize("stage", [1, 2])
def test_task_position_matches_flattened_batch_heads(sparse, stage):
    for pos in [0, 2, 3, 7, 11]:
        actual = sparse.create_and_get_basic_pos(pos, 1, 2, 2, 0, 3, stage, 17, 8)
        expected = (pos // 6, (pos // 3) % 2, pos % 3, pos % 3 + 1 if stage == 1 else 3)
        assert tuple(actual) == expected
        b, h, row, _ = actual
        offset = sparse.get_s_offset(0, 2, 17, 8, torch.tensor(b), torch.tensor(h), torch.tensor(row))
        assert int(offset) == (b * 2 + h) * 17 + row * 8


@pytest.mark.parametrize("parity", [0, 1])
def test_cube_qk_matmul_writes_selected_buffer(sparse, parity):
    q = torch.linspace(-1, 2, 32 * 16).reshape(32, 16)
    k = torch.linspace(0.2, 1.3, 32 * 16).reshape(32, 16)
    kptr = BlockPointer(Pointer(k), k.shape, k.stride(), (0, 0), k.shape, (1, 0))
    ping, pong = [torch.full((16, 32), torch.nan) for _ in range(2)]
    cube = torch.full((32, 32), torch.nan)
    sparse._qk_matmul(q, kptr, ping, pong, cube, 16, 32, parity)
    expected = q @ k.T
    torch.testing.assert_close(cube, expected)
    torch.testing.assert_close([ping, pong][parity], expected[:16])
    assert torch.isnan([ping, pong][1 - parity]).all()
    assert [event[0] for event in sparse.al.events] == ["wait", "set"]


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("update", [False, True])
@pytest.mark.parametrize("slot", range(3))
def test_flash_update_ping_pong_and_rescaling(sparse, parity, update, slot):
    pvid = next(i for i in range(3, 10) if i % 2 == parity and (i - 3) % 3 == slot)
    ping = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    pong = ping + 100
    alphas = [torch.full((4,), v) for v in [0.25, 0.5, 0.75]]
    accumulator = torch.full_like(ping, 2)
    sparse._flash_update(ping, pong, *alphas, accumulator, slot, 8, 8, pvid, update)
    pv = ping if parity else pong
    expected = 2 * alphas[slot][:, None] + pv if update else pv
    torch.testing.assert_close(accumulator, expected)


@pytest.mark.parametrize("slot", range(3))
@pytest.mark.parametrize("previous", range(3))
@pytest.mark.parametrize("update,masked", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("vector", [0, 1])
def test_vector_pipeline_stage_keeps_banks_and_layout(sparse, slot, previous, update, masked, vector):
    bm, bn = 32, 32
    rows = bm // 2
    sparse.al.vector_id = vector
    task = previous + 1
    ping = torch.linspace(-1, 2, rows * bn).reshape(rows, bn)
    pong = ping + 0.25
    qk = ping if task % 2 else pong
    p_ping, p_pong = [torch.full((bn // 16, bm // 16, 16, 16), torch.nan) for _ in range(2)]
    mask = (torch.arange(bm * bn).reshape(bm, bn) % 11 == 0).to(torch.int8)
    m = [torch.full((rows,), float(i)) for i in range(3)]
    normalizers = [torch.full((rows,), float(i + 1)) for i in range(3)]
    alphas = [torch.full((rows,), torch.nan) for _ in range(3)]
    sparse.process_v1(
        ping,
        pong,
        p_ping,
        p_pong,
        BlockPointer(Pointer(mask), mask.shape, mask.stride(), (0, 0), (rows, bn), (1, 0)),
        *m,
        *normalizers,
        *alphas,
        0.8,
        torch.tensor(task),
        slot,
        torch.float32,
        torch.tensor(masked),
        torch.tensor(update),
        bm,
        bn,
        1,
    )
    scores = qk * 0.8
    if masked:
        scores += torch.where(mask[vector * rows : (vector + 1) * rows].bool(), -1e4, 0)
    maxima = scores.max(-1).values
    if update:
        maxima = torch.maximum(maxima, torch.tensor(float(slot)))
    probabilities = (scores - maxima[:, None]).exp()
    totals = probabilities.sum(-1)
    if update:
        alpha = (slot - maxima).exp()
        totals += (slot + 1) * alpha
        torch.testing.assert_close(alphas[previous], alpha)
    torch.testing.assert_close(m[slot], maxima)
    torch.testing.assert_close(normalizers[slot], totals)
    for bank in set(range(3)) - {slot}:
        assert torch.equal(m[bank], torch.full((rows,), float(bank)))
        assert torch.equal(normalizers[bank], torch.full((rows,), float(bank + 1)))
    selected = p_ping if task % 2 else p_pong
    unpacked = selected.permute(1, 2, 0, 3).reshape(bm, bn)
    torch.testing.assert_close(unpacked[vector * rows : (vector + 1) * rows], probabilities)
    assert torch.isnan(unpacked[(1 - vector) * rows : (2 - vector) * rows]).all()


@pytest.mark.parametrize("parity", [0, 1])
def test_pv_matmul_selected_ping_pong(sparse, parity):
    bm, bn, width = 32, 32, 16
    values = torch.linspace(-1, 1, bn * width).reshape(bn, width)
    probabilities = torch.linspace(0.01, 0.5, bm * bn).reshape(bm, bn)
    # Explicit dense view isolates matmul/copy semantics from the NZ packing test above.
    ping = probabilities.clone()
    pong = probabilities + 0.1
    out_ping, out_pong = [torch.full((bm // 2, width), torch.nan) for _ in range(2)]
    cube = torch.full((bm, width), torch.nan)
    vptr = BlockPointer(Pointer(values), values.shape, values.stride(), (0, 0), values.shape, (1, 0))
    sparse._pv_matmul(ping, pong, out_ping, out_pong, cube, vptr, width, bm, bn, parity)
    expected = [ping, pong][parity] @ values
    torch.testing.assert_close(cube, expected)
    torch.testing.assert_close([out_ping, out_pong][parity], expected[: bm // 2])
    assert torch.isnan([out_ping, out_pong][1 - parity]).all()


def test_attention_wrapper_forward_backward_math(sparse, monkeypatch):
    # Device properties are the hardware boundary; all launched kernels execute.
    monkeypatch.setattr(sparse, "get_npu_aicore_num", lambda: 20)
    torch.manual_seed(811)
    length, width, block = 35, 16, 32
    q, k, v = [torch.randn(1, 1, length, width) for _ in range(3)]
    blocks = 2
    lut = torch.arange(blocks, dtype=torch.int32).repeat(1, 1, blocks, 1)
    sparse_map = torch.ones(1, 1, blocks, blocks, dtype=torch.int32)

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    ctx = Context()
    output, lse = sparse._attention.forward(ctx, q, k, v, sparse_map, lut, blocks, None, False, 0.25, block, block)
    qr, kr, vr = [t.clone().requires_grad_() for t in (q, k, v)]
    scores = qr @ kr.transpose(-1, -2) * 0.25
    reference = scores.softmax(-1) @ vr
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lse, scores.logsumexp(-1) / torch.log(torch.tensor(2.0)))
    grad = torch.randn_like(q)
    derivatives = sparse._attention.backward(ctx, grad)
    reference.backward(grad)
    for actual, expected in zip(derivatives[:3], [qr.grad, kr.grad, vr.grad]):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    assert all(value is None for value in derivatives[3:])


def test_mixed_cube_vector_one_block_numerical_pipeline(sparse, monkeypatch):
    """Replay the peer vector at its hardware wait; never substitute an oracle.

    Python executes the cube scope before the vector scope. With one Q/KV block,
    the only data dependency crossing the scopes is P: the hardware-boundary
    callback computes P using the original process_v1 body for both vector halves.
    The later original vector scope recomputes its own P and consumes original PV.
    This tests data semantics, not the device scheduler's synchronization safety.
    """
    bm, bn, width = 32, 32, 16

    class OneBlockRuntime(Ascend):
        def fixpipe(self, source, destination, *args):
            if source.shape == (bm, bn):
                self.qk = source.clone()
            super().fixpipe(source, destination, *args)

        def sync_block_wait(self, *args):
            super().sync_block_wait(*args)
            if args[:3] != ("vector", "cube", 4):
                return
            frame = inspect.currentframe().f_back
            while frame is not None and frame.f_code.co_name != "_attn_fwd":
                frame = frame.f_back
            assert frame is not None
            state = frame.f_locals
            active = self.vector_id
            for half in (0, 1):
                self.vector_id = half
                state["qk_ub_ping"].copy_(self.qk[half * (bm // 2) : (half + 1) * (bm // 2)])
                maxima = [torch.full((bm // 2,), torch.nan) for _ in range(3)]
                totals = [torch.full((bm // 2,), torch.nan) for _ in range(3)]
                alphas = [torch.full((bm // 2,), torch.nan) for _ in range(3)]
                sparse.process_v1(
                    state["qk_ub_ping"],
                    state["qk_ub_pong"],
                    state["p_l1_ping"],
                    state["p_l1_pong"],
                    None,
                    *maxima,
                    *totals,
                    *alphas,
                    state["sm_scale"],
                    torch.tensor(1),
                    1,
                    torch.float32,
                    torch.tensor(False),
                    torch.tensor(False),
                    bm,
                    bn,
                    2,
                )
                assert torch.isfinite(maxima[1]).all() and (totals[1] > 0).all()
            self.vector_id = active
            state["qk_ub_ping"].copy_(self.qk[active * (bm // 2) : (active + 1) * (bm // 2)])

    torch.manual_seed(972)
    q, k, v = [torch.randn(1, 1, bm, width) for _ in range(3)]
    output = torch.full_like(q, torch.nan)
    lse = torch.full((1, 1, bm), torch.nan)
    lut = torch.zeros(1, 1, 1, 1, dtype=torch.int32)
    for vector in (0, 1):
        runtime = OneBlockRuntime()
        runtime.vector_id = vector
        monkeypatch.setattr(sparse, "al", runtime)
        sparse._attn_fwd[(1,)](
            Q=q,
            K=k,
            V=v,
            ATTEN_MASK=None,
            M=lse,
            Out=output,
            sparse_start_idx=None,
            sm_scale=0.25,
            stride_qz=q.stride(0),
            stride_qh=q.stride(1),
            stride_qm=q.stride(2),
            stride_qk=q.stride(3),
            stride_kz=k.stride(0),
            stride_kh=k.stride(1),
            stride_kn=k.stride(2),
            stride_kk=k.stride(3),
            stride_vz=v.stride(0),
            stride_vh=v.stride(1),
            stride_vn=v.stride(2),
            stride_vk=v.stride(3),
            stride_oz=output.stride(0),
            stride_oh=output.stride(1),
            stride_om=output.stride(2),
            stride_on=output.stride(3),
            stride_am=bm,
            Z=1,
            H=1,
            N_CTX=bm,
            HEAD_DIM=width,
            BLOCK_M=bm,
            BLOCK_N=bn,
            STAGE=2,
            NUM_BLOCKS_PER_CORE=1,
            NUM_BLOCKS=1,
            NUM_BLOCKS_M=1,
            CORE_NUM=1,
            TOPK=1,
            LUT=lut,
        )
    logits = q @ k.transpose(-1, -2) * 0.25
    torch.testing.assert_close(output, logits.softmax(-1) @ v, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(lse, logits.logsumexp(-1), atol=2e-6, rtol=2e-6)
