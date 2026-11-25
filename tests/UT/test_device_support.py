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


def test_indexed_and_split_soc_support():
    for operator in ["eagle_indexed_gate_norm_modulate", "eagle_split_norm_rope_concat"]:
        assert support.supports(operator, "A2")
        assert support.supports(operator, "A3") == (operator == "eagle_split_norm_rope_concat")
        assert support.supports(operator, "A5") == (operator == "eagle_split_norm_rope_concat")
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


@pytest.mark.parametrize("requested,expected", [(None, 1), ("0", 0)], ids=["current", "explicit"])
def test_split_fixture_selects_visible_logical_device(monkeypatch, requested, expected):
    import os
    import sys
    from types import ModuleType

    import torch
    from tests.ops.eagle_split_norm_rope_concat.test_eagle_split_norm_rope_concat import device_fixture

    monkeypatch.setenv("MINDIE_TEST_MODE", "NPU")
    # Exercise the real fixture and selector without loading the NPU operator library.
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3,7")
    if requested is None:
        monkeypatch.delenv("MINDIE_TEST_DEVICE_ID", raising=False)
    else:
        monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", requested)
    with (
        patch.object(torch_npu.npu, "current_device", return_value=1) as current,
        patch.object(torch_npu.npu, "get_soc_version", return_value=220),
        patch.object(torch_npu.npu, "device_count", return_value=2),
        patch.object(torch_npu.npu, "is_available", return_value=True),
        patch.object(torch_npu.npu, "set_device") as select,
        patch.object(torch, "device", side_effect=lambda name: name),
    ):
        assert device_fixture.__wrapped__() == f"npu:{expected}"
        select.assert_called_once_with(expected)
        if requested is None:
            current.assert_called_once_with()
        else:
            current.assert_not_called()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3,7"


@pytest.mark.parametrize("requested", ["", "npu:1", "-1"])
def test_invalid_logical_device_index(monkeypatch, requested):
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", requested)
    with pytest.raises(ValueError, match="MINDIE_TEST_DEVICE_ID must be a nonnegative logical device index"):
        support.selected_device_index()


def test_init_rejects_out_of_range_device(monkeypatch):
    monkeypatch.setenv("MINDIE_TEST_DEVICE_ID", "2")
    with (
        patch.object(torch_npu.npu, "device_count", return_value=2),
        patch.object(torch_npu.npu, "is_available", return_value=True),
        patch.object(torch_npu.npu, "set_device") as select,
    ):
        with pytest.raises(RuntimeError, match="requested logical device 2, visible count=2"):
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


_A3_MAP = """NPU ID  Chip ID  Chip Logic ID  Chip Name
3   0    8    Ascend910
3   1   12    Ascend910
3   2    -    MCU
"""


def test_split_a3_masked_driver_id_and_capture_pin(monkeypatch):
    import json
    from tests.ops import a3_device

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "12,8")
    monkeypatch.setenv("ASCEND_COMPUTE_UNIT", "ascend910_93")
    monkeypatch.delenv("EAGLE_TEST_DEVICE_RECORD", raising=False)
    with (
        patch.object(
            a3_device, "query_smi", side_effect=[_A3_MAP, "Chip Name: Ascend910\nSerial Number: board-1"] * 3
        ) as query,
        patch.object(torch_npu.npu, "device_count", return_value=2),
        patch.object(torch_npu.npu, "set_device") as select,
        patch.object(torch_npu.npu, "get_soc_version", return_value=250),
        patch.object(torch_npu._C, "_npu_get_soc_version", return_value=250),
    ):
        first = a3_device.bind_production_device(12)
        select.assert_called_once_with(0)
        assert (first["chip_logic_id"], first["npu_id"], first["chip_id"]) == (12, 3, 1)
        assert query.call_args.args == ("-t", "board", "-i", "3", "-c", "1")
        assert first["identity"]["serial number"] == "board-1"
        monkeypatch.setenv("EAGLE_TEST_DEVICE_RECORD", json.dumps(first))
        assert a3_device.bind_production_device(12) == first
        with pytest.raises(RuntimeError, match="identity changed"):
            a3_device.bind_production_device(8)
        with pytest.raises(ValueError, match="outside ASCEND_RT_VISIBLE_DEVICES"):
            a3_device.bind_production_device(3)


@pytest.fixture(name="unmasked_a3")
def unmasked_a3_fixture(monkeypatch):
    from tests.ops import a3_device

    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("EAGLE_TEST_DEVICE_RECORD", raising=False)
    monkeypatch.setenv("ASCEND_COMPUTE_UNIT", "ascend910_93")
    # Runtime ordinals 0/1 map to physical 6/4. SMI lists their dies in the opposite order.
    replies = {("-m",): _A3_MAP}
    for physical, logic, chip in [(6, 12, 1), (4, 8, 0)]:
        replies[("-t", "phyid-remap", "-p", str(physical))] = (
            f"Chip Physical ID: {physical}\nChip Logic ID: {logic}\nNPU ID: 3\nChip ID: {chip}"
        )
        replies[("-t", "board", "-i", "3", "-c", str(chip))] = f"Chip Name: Ascend910\nSerial Number: die-{chip}"
    with (
        patch.object(torch_npu, "npu") as npu,
        patch.object(torch_npu._C, "_npu_get_soc_version", return_value=250),
        patch.object(a3_device, "physical_device_id", side_effect=(6, 4).__getitem__),
        patch.object(a3_device, "query_smi", side_effect=lambda *args: replies[args]),
    ):
        npu.device_count.return_value = 2
        npu.get_soc_version.return_value = 250
        yield npu, replies


@pytest.mark.parametrize("ordinal,logic,chip", [(0, 12, 1), (1, 8, 0)])
def test_unmasked_a3_fixture_and_production_agree_on_die(unmasked_a3, ordinal, logic, chip, monkeypatch, capsys):
    import json
    import sys
    from types import ModuleType

    import torch
    from tests.ops import a3_device
    from tests.ops.eagle_split_norm_rope_concat.test_eagle_split_norm_rope_concat import device_fixture

    npu, _ = unmasked_a3
    monkeypatch.setenv("MINDIE_TEST_MODE", "NPU")
    monkeypatch.delenv("MINDIE_TEST_DEVICE_ID", raising=False)
    monkeypatch.setitem(sys.modules, "mindiesd", ModuleType("mindiesd"))
    npu.current_device.return_value = ordinal
    # The CPU test double has no registered npu device type; preserve torch.device's index semantics.
    device_type = torch.device
    with patch.object(torch, "device", side_effect=lambda name: device_type(name.replace("npu:", "privateuseone:"))):
        assert device_fixture.__wrapped__().index == ordinal
    npu.set_device.assert_called_once_with(ordinal)
    npu.set_device.reset_mock()
    selected = json.loads(capsys.readouterr().err)
    bound = a3_device.bind_production_device(logic)
    for record in (selected, bound):
        assert (record["runtime_ordinal"], record["chip_logic_id"], record["npu_id"], record["chip_id"]) == (
            ordinal,
            logic,
            3,
            chip,
        )
        assert record["identity"]["serial number"] == f"die-{chip}"
    npu.set_device.assert_called_once_with(ordinal)


@pytest.mark.parametrize(
    "remap",
    [
        "Chip Physical ID: 6\nChip Logic ID: 12\nChip ID: 1",
        "Chip Physical ID: 6\nChip Logic ID: 12\nNPU ID: 3\nChip ID: 0",
        "Chip Physical ID: 4\nChip Logic ID: 12\nNPU ID: 3\nChip ID: 1",
    ],
    ids=["missing-npu-id", "wrong-die", "wrong-physical-id"],
)
def test_unmasked_a3_unverifiable_mapping_fails_before_select(unmasked_a3, remap):
    from tests.ops import a3_device

    npu, replies = unmasked_a3
    replies[("-t", "phyid-remap", "-p", "6")] = remap
    with pytest.raises(RuntimeError, match="ASCEND_RT_VISIBLE_DEVICES"):
        a3_device.bind_production_device(12)
    npu.set_device.assert_not_called()


def test_unmasked_a3_duplicate_runtime_mapping_fails_before_select(unmasked_a3):
    from tests.ops import a3_device

    npu, _ = unmasked_a3
    with patch.object(a3_device, "physical_device_id", return_value=6):
        with pytest.raises(RuntimeError, match="ASCEND_RT_VISIBLE_DEVICES"):
            a3_device.bind_production_device(12)
    npu.set_device.assert_not_called()


def test_unmasked_a3_production_discovers_target_before_select(unmasked_a3, monkeypatch):
    from tests.ops import a3_device

    npu, _ = unmasked_a3
    monkeypatch.delenv("ASCEND_COMPUTE_UNIT")
    record = a3_device.bind_production_device(8)
    torch_npu._C._npu_get_soc_version.assert_called_once_with()
    assert record["runtime_ordinal"] == 1
    npu.set_device.assert_called_once_with(1)


def test_a3_rejects_stale_build_target_before_select(unmasked_a3, monkeypatch):
    from tests.ops import a3_device

    npu, _ = unmasked_a3
    monkeypatch.setenv("ASCEND_COMPUTE_UNIT", "ascend910b")
    with pytest.raises(RuntimeError, match="ASCEND_COMPUTE_UNIT"):
        a3_device.bind_production_device(1)
    npu.set_device.assert_not_called()


def test_unmasked_a2_production_uses_visible_ordinal(monkeypatch):
    from tests.ops import a3_device

    monkeypatch.setenv("ASCEND_COMPUTE_UNIT", "ascend910b")
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("EAGLE_TEST_DEVICE_RECORD", raising=False)
    with (
        patch.object(torch_npu, "npu") as npu,
        patch.object(torch_npu._C, "_npu_get_soc_version", return_value=220),
        patch.object(a3_device, "query_smi") as smi,
        patch.object(a3_device, "physical_device_id") as physical,
    ):
        npu.device_count.return_value = 2
        npu.get_soc_version.return_value = 220
        assert a3_device.bind_production_device(1)["runtime_ordinal"] == 1
        npu.set_device.assert_called_once_with(1)
        smi.assert_not_called()
        physical.assert_not_called()


@pytest.mark.parametrize(
    "table,logic,reason",
    [
        (_A3_MAP, 2, "absent"),
        (_A3_MAP + "4 1 12 Ascend910", 12, "Duplicate"),
        (_A3_MAP + "3 1 13 Ascend910", 12, "Duplicate"),
        ("garbled", 8, "columns"),
    ],
)
def test_a3_mapping_rejects_ambiguous_or_missing_die(table, logic, reason):
    from tests.ops.a3_device import find_die

    with pytest.raises(RuntimeError, match=reason):
        find_die(table, logic)


def test_a3_paired_store_fixture():
    from tests.ops.eagle_split_norm_rope_concat import test_eagle_split_norm_rope_concat as suite

    with patch.object(suite, "require_support", return_value="A3"):
        assert suite.device_910b_fixture.__wrapped__("npu:1") == "npu:1"
        with pytest.raises(pytest.skip.Exception):
            suite.device_950_fixture.__wrapped__("npu:1")
