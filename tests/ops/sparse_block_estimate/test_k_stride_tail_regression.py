"""Aligned and partial K packs use the same independent full-token golden."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_a5_correctness import Case, PRESETS, run  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def native_operator():
    from tests.ops.device_support import init_test_device, require_support
    from mindiesd.layers.register_ops import _load_mindie_ops_library

    init_test_device()
    require_support("sparse_block_estimate")
    _load_mindie_ops_library()


@pytest.mark.parametrize("q_len", [256, 257])
@pytest.mark.parametrize("k_len", [1, 7, 8, 9, 2048, 2049])
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
def test_k_stride_tail(q_len, k_len, dtype_name, monkeypatch):
    import torch_npu

    monkeypatch.setitem(PRESETS, "k_stride_regression", Case(1, 8, 1, q_len, k_len))
    args = argparse.Namespace(
        preset="k_stride_regression",
        device=torch_npu.npu.current_device(),
        seed=2026,
        input_profile=None,
        case_mode="realistic",
        enhance_mode=1.0,
        sample_rows=0,
        threshold=0.85,
        row_sparse=1.0,
        keep_sink=True,
        keep_recent=True,
        max_diff_ratio=0.02,
        max_omitted_ratio=0.01,
        reference_only=False,
    )
    result = run(args, dtype_name)
    assert result["diff_ratio"] < 0.02
    assert result["omitted_ratio"] < 0.01
    assert result["count_mismatch_rows"] == 0
    assert result["padding_nonzero"] == 0


@pytest.mark.parametrize("layout", ["BSH", "BSND"])
@pytest.mark.parametrize("k_len", [2048, 2049, 4097])
@pytest.mark.parametrize("dtype_name", ["fp16", "bf16"])
def test_interleaved_kv_heads_layout(layout, k_len, dtype_name, monkeypatch):
    import torch
    import torch_npu
    from tests.ops.device_support import current_device_family

    monkeypatch.setitem(PRESETS, "k_layout_regression", Case(1, 8, 2, 257, k_len))
    native = torch.ops.mindiesd.sparse_block_estimate

    def layout_adapter(**kwargs):
        # Change input storage layout, not the native calculation or output.
        for key in ("query", "key"):
            value = kwargs[key].transpose(1, 2).contiguous()
            if layout == "BSH":
                value = value.flatten(-2)
            kwargs[key] = value
        kwargs["input_layout"] = layout
        return native(**kwargs)

    monkeypatch.setattr(torch.ops.mindiesd, "sparse_block_estimate", layout_adapter)
    args = argparse.Namespace(
        preset="k_layout_regression",
        device=torch_npu.npu.current_device(),
        seed=2026,
        input_profile=None,
        case_mode="realistic",
        enhance_mode=1.0,
        sample_rows=0,
        threshold=0.85,
        row_sparse=1.0,
        keep_sink=True,
        keep_recent=True,
        max_diff_ratio=0.02,
        max_omitted_ratio=0.01,
        reference_only=False,
    )
    if current_device_family() == "A5":
        # Arch35 tiling accepts only 4-D BNSD, so the interleaved BSH/BSND
        # contract is A2/A3-only; A5 must fail closed rather than misread it.
        with pytest.raises(RuntimeError, match="aclnnSparseBlockEstimate|BNSD|tiling"):
            run(args, dtype_name)
        return
    result = run(args, dtype_name)
    assert result["diff_ratio"] < 0.02
    assert result["omitted_ratio"] < 0.01
    assert result["count_mismatch_rows"] == 0 and result["padding_nonzero"] == 0
