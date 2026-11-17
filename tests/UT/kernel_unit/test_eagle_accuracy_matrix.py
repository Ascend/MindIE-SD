"""CPU assertions prevent reduced device matrices from dropping coverage axes."""

import ast
import importlib
import os
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tests.ops.eagle_indexed_gate_norm_modulate.accuracy_cases import (
    ACCURACY_CASES,
    BOUNDARY_CASES,
    DEFAULT_EPS_CASES,
    DEFAULT_OUTPUT_CASES,
    DTYPE_PAIRS,
)


def test_all_original_boundaries_and_dtype_distribution_triples():
    assert len(BOUNDARY_CASES) == 23
    assert {case[1] for case in BOUNDARY_CASES} == {
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        3968,
        3969,
        4095,
        4096,
        4097,
        5376,
        6144,
        6145,
        6655,
        6656,
        6657,
        7167,
        7168,
    }
    assert {case[0] for case in BOUNDARY_CASES} == {0, 1, 3, 81}
    assert len(ACCURACY_CASES) == 28
    assert {tuple(case[3:]) for case in ACCURACY_CASES} == set(BOUNDARY_CASES)
    assert {tuple(case[:3]) for case in ACCURACY_CASES} == {
        (distribution, *pair) for distribution, pair in product(("uniform", "normal"), DTYPE_PAIRS)
    }
    assert {tuple(case[1:3]) for case in ACCURACY_CASES if case[3] == 0} == set(DTYPE_PAIRS)
    for width in (1, 7168):
        assert {case[1] for case in ACCURACY_CASES if case[4] == width} == {torch.float16, torch.bfloat16}
    assert {case[5] for case in ACCURACY_CASES if case[6]} == {torch.int32, torch.int64}


def test_defaults_preserve_pairwise_axes():
    assert {tuple(case[:2]) for case in DEFAULT_OUTPUT_CASES} == set(
        product(("uniform", "normal"), (torch.float16, torch.bfloat16))
    )
    assert {case[2] for case in DEFAULT_OUTPUT_CASES} == {1e-6, 1e-3}
    for first, second in ((0, 1), (0, 2), (1, 2)):
        assert len({(case[first], case[second]) for case in DEFAULT_EPS_CASES}) == 4


def test_actual_pytest_matrix_uses_representative_rows():
    path = (
        Path(__file__).parents[2]
        / "ops/eagle_indexed_gate_norm_modulate/test_eagle_indexed_gate_norm_modulate_accuracy.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "test_accuracy")
    count = 1
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            if decorator.func.attr == "parametrize":
                values = decorator.args[1]
                count *= (
                    len(ACCURACY_CASES)
                    if isinstance(values, ast.Name) and values.id == "ACCURACY_CASES"
                    else len(values.elts)
                )
    assert count == 28


def test_split_retains_unique_regression_matrix():
    path = Path(__file__).parents[2] / "ops/eagle_split_norm_rope_concat/test_eagle_split_norm_rope_concat.py"
    functions = {
        n.name: n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    }
    total = 0
    native = 0
    for function in functions.values():
        count = 1
        positive = False
        for decorator in function.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                if decorator.func.attr == "parametrize":
                    count *= len(decorator.args[1].elts)
                elif decorator.func.attr == "usefixtures":
                    positive |= any(
                        isinstance(arg, ast.Constant) and arg.value == "supported_soc" for arg in decorator.args
                    )
        total += count
        native += count if positive else 0
    assert total == 100
    assert native == 38
    assert {
        "test_interleave_r96_gather_boundaries",
        "test_fp32_norm_flows_directly_into_rope",
        "test_resident_rope_reused_across_head_tiles",
        "test_pipeline_reuse",
        "test_value_signed_zero_bits",
        "test_tail_is_normalized_but_not_rotated",
    } <= functions.keys()


@pytest.mark.parametrize("name", ["eagle_indexed_gate_norm_modulate", "eagle_split_norm_rope_concat"])
@pytest.mark.parametrize("family", ["A2", "A3", "A5", "Duo"])
def test_fixture_selects_logical_device_and_actual_registered_family(monkeypatch, name, family):
    from tests.ops import device_support

    filename = (
        "test_eagle_indexed_gate_norm_modulate_accuracy" if "indexed" in name else "test_eagle_split_norm_rope_concat"
    )
    module = importlib.import_module(f"tests.ops.{name}.{filename}")
    monkeypatch.setenv("MINDIE_TEST_MODE", "NPU")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2")
    select = Mock(return_value="npu:1")
    monkeypatch.setattr(module, "init_test_device", select)
    monkeypatch.setattr(device_support, "current_device_family", lambda: family)
    fixture = module.npu_device if "indexed" in name else module.device_fixture
    assert fixture.__wrapped__() == "npu:1"
    select.assert_called_once_with()
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "1,2"
    if family == "A2":
        module.supported_soc_fixture.__wrapped__("npu:1")
    else:
        with pytest.raises(unittest.SkipTest):
            module.supported_soc_fixture.__wrapped__("npu:1")
