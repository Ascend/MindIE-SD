# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch_npu

from tests.ops import device_support as support


@pytest.mark.parametrize(
    "soc,family", [(200, "Duo"), (205, "Duo"), (220, "A2"), (225, "A2"), (250, "A3"), (256, "A3"), (260, "A5")]
)
def test_known_soc(soc, family):
    assert support.identify_device(soc) == family


@pytest.mark.parametrize("soc", [0, 206, 226, 249, 257, 261, True, None, "260"])
def test_unknown_soc_fails_closed(soc):
    with pytest.raises(RuntimeError):
        support.identify_device(soc)


@pytest.mark.parametrize("operator", support.REGISTRATIONS)
def test_matrix_exactly_matches_opdef(operator):
    root = Path(__file__).resolve().parents[2]
    sources = list((root / "csrc/ops" / operator / "op_host").glob("*def.cpp"))
    assert sources
    registrations = set()
    for source in sources:
        registrations.update(re.findall(r'AddConfig\("([^"]+)"', source.read_text(encoding="utf-8")))
    assert set(support.REGISTRATIONS[operator]) == registrations
    for family in support.FAMILIES:
        assert support.supports(operator, family) == (family in {support.SOC_FAMILY.get(s) for s in registrations})
        for kind in support.CONTRACT_KINDS:
            assert support.supports(operator, family, kind)


def test_indexed_split_not_generalized_to_a3_or_a5():
    for operator in ["eagle_indexed_gate_norm_modulate", "eagle_split_norm_rope_concat"]:
        assert support.supports(operator, "A2")
        assert not support.supports(operator, "A3")
        assert not support.supports(operator, "A5")
        assert not support.supports(operator, "Duo")


def test_real_query_requires_available_device(monkeypatch):
    with patch.object(torch_npu.npu, "is_available", return_value=False):
        with pytest.raises(RuntimeError):
            support.current_device_family()
    with (
        patch.object(torch_npu.npu, "is_available", return_value=True),
        patch.object(torch_npu.npu, "get_soc_version", return_value=250),
    ):
        assert support.current_device_family() == "A3"
        with pytest.raises(Exception, match="not registered"):
            support.require_support("eagle_indexed_gate_norm_modulate")
        assert support.require_support("eagle_indexed_gate_norm_modulate", "meta") == "A3"
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", "3")
    assert support.selected_device_index() == 3
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", "-1")
    with pytest.raises(ValueError):
        support.selected_device_index()
    monkeypatch.delenv("MINDIE_TEST_DEVICE_ID")
    with patch.object(torch_npu.npu, "current_device", return_value=2):
        assert support.selected_device_index() == 2


def test_unknown_operator_family_or_kind_is_not_a_skip():
    with pytest.raises(ValueError):
        support.supports("missing", "A2")
    with pytest.raises(RuntimeError):
        support.supports("mul_add", "unknown")
    with pytest.raises(ValueError):
        support.supports("mul_add", "A2", "unknown")


def test_init_uses_visible_logical_ordinal_without_changing_mask(monkeypatch):
    import os
    import torch

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2")
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", "1")
    with (
        patch.object(torch_npu.npu, "device_count", return_value=2),
        patch.object(torch_npu.npu, "is_available", return_value=True),
        patch.object(torch_npu.npu, "set_device") as select,
        patch.object(torch, "device", return_value="logical-device"),
    ):
        assert support.init_test_device() == "logical-device"
        select.assert_called_once_with(1)
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "1,2"
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", "2")
    with (
        patch.object(torch_npu.npu, "device_count", return_value=2),
        patch.object(torch_npu.npu, "is_available", return_value=True),
        patch.object(torch_npu.npu, "set_device") as select,
    ):
        with pytest.raises(RuntimeError, match="resource unavailable"):
            support.init_test_device()
        select.assert_not_called()


def test_legacy_910_registration_does_not_enable_duo():
    assert "ascend910" in support.REGISTRATIONS["laser_attention"]
    assert not support.supports("laser_attention", "Duo")
    assert "ascend910" not in support.SOC_FAMILY


@pytest.mark.parametrize("family", sorted(support.FAMILIES))
def test_qfa_pytest_gates_before_logical_device_initialization(monkeypatch, family):
    import importlib
    import unittest

    module = importlib.import_module("tests.ops.quant_flash_attn.test_qfa_accuracy")
    with (
        patch.object(support, "current_device_family", return_value=family),
        patch.object(module, "init_test_device", side_effect=RuntimeError("logical-init-reached")) as initialize,
    ):
        if family == "A5":
            with pytest.raises(RuntimeError, match="logical-init-reached"):
                module.qfa_device.__wrapped__()
            initialize.assert_called_once_with()
        else:
            with pytest.raises(unittest.SkipTest, match="not registered"):
                module.qfa_device.__wrapped__()
            initialize.assert_not_called()


@pytest.mark.parametrize(
    "path,target,operator",
    [
        ("plugin/test_la.py", "TestLaMindieSd", "laser_attention"),
        ("plugin/test_adablocksparseattention.py", "TestBsaMindieSd", "ada_block_sparse_attention"),
        ("layers/flash_attn/test_attn_forward.py", "test_attn_forward_manual_la_bsnd", "laser_attention"),
        ("layers/flash_attn/test_attn_forward.py", "test_attn_forward_manual_la_bnsd", "laser_attention"),
        ("layers/flash_attn/test_sparse_attention.py", "test_ada_bsa", "ada_block_sparse_attention"),
        ("layers/flash_attn/test_sparse_attention.py", "test_ada_bsa_result", "ada_block_sparse_attention"),
    ],
)
def test_native_positive_cases_use_audited_registration_guard(path, target, operator):
    import ast

    source = Path(__file__).resolve().parents[1] / "ops" / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        item for item in ast.walk(tree) if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == target
    )
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Name)
        and decorator.func.id == "native_test"
        and decorator.args
        and isinstance(decorator.args[0], ast.Constant)
        and decorator.args[0].value == operator
        for decorator in node.decorator_list
    )
