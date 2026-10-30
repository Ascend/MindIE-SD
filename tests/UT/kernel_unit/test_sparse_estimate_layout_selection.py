"""The existing BSH case is positive on A2/A3 and rejection-only on A5."""

import importlib

import pytest
import torch


@pytest.mark.parametrize("family", ["A2", "A3", "A5"])
def test_bsh_case_follows_native_layout_contract(monkeypatch, family):
    module = importlib.import_module("tests.ops.plugin.test_sparseblockestimate")
    monkeypatch.setattr(module, "current_device_family", lambda: family, raising=False)
    calls = []

    def native_boundary(**kwargs):
        assert kwargs["input_layout"] == "BSH"
        assert kwargs["query"].dtype == kwargs["key"].dtype == torch.float16
        calls.append(kwargs)
        if family == "A5":
            raise RuntimeError("aclnnSparseBlockEstimateGetWorkspaceSize: tiling requires BNSD on A5")
        return torch.empty(1, 1, 1, 32, dtype=torch.int8), torch.empty(1, 1, 1, dtype=torch.int32)

    monkeypatch.setattr(torch.ops.mindiesd, "sparse_block_estimate", native_boundary, raising=False)
    case = module.TestSparseBlockEstimate("test_bsh_layout_output_shape")
    for key, value in dict(
        batch=1,
        qseqlen=128,
        head_num=1,
        head_dim=128,
        device="cpu",
        stride=8,
        sparse_size=128,
        scale_value=0.01,
        threshold=0.85,
        causal=False,
        keep_sink=True,
        keep_recent=True,
        row_sparse=1.0,
    ).items():
        setattr(case, key, value)
    case.test_bsh_layout_output_shape()
    assert len(calls) == 1
