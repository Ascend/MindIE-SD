# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Test selection from repository OpDef registrations, never a hardware emulator.

A registered family is necessary, not sufficient: a missing native API/build on
that family is a test failure, not a reason to skip. Unknown hardware fails closed.
"""

import os
import unittest

FAMILIES = frozenset(("A2", "A3", "A5", "Duo"))
# Exact source evidence is checked by CPU tests against each operator's OpDef.
REGISTRATIONS = {
    "eagle_indexed_gate_norm_modulate": ("ascend910b",),
    "eagle_split_norm_rope_concat": ("ascend910b",),
    "eagle_block_sparse_attention": ("ascend910b", "ascend910_93"),
    "ada_block_sparse_attention": ("ascend910b", "ascend910_93"),
    "la_preprocess": ("ascend910b", "ascend910_93"),
    "laser_attention": ("ascend910", "ascend910b", "ascend910_93"),
    "mul_add": ("ascend910b", "ascend910_93", "ascend950"),
    "norm_rope_concat": ("ascend910b", "ascend910_93", "ascend950"),
    "sparse_block_estimate": ("ascend910b", "ascend910_93", "ascend950"),
    "eagle_ffn": ("ascend950",),
    "quant_four_over_six_a5": ("ascend950",),
    "quant_flash_attn": ("ascend950",),
    "quant_flash_attn_metadata": ("ascend950",),
    "eagle_quant_block_sparse_attention": ("ascend950",),
    "fused_infer_attention_score": ("ascend950",),
}
# Legacy ascend910 has no proven equivalence to this repository's Duo SoC
# range. Preserve its registration as evidence, but do not assign it to Duo.
# Duo requires its own supported registration/API evidence (e.g. ascend310p).
SOC_FAMILY = {"ascend910b": "A2", "ascend910_93": "A3", "ascend950": "A5"}
CONTRACT_KINDS = frozenset(("meta", "dtype_contract", "cpp_invalid"))


def identify_device(soc_version):
    if isinstance(soc_version, bool) or not isinstance(soc_version, int):
        raise RuntimeError(f"Invalid device SoC version: {soc_version!r}")
    for low, high, family in ((200, 205, "Duo"), (220, 225, "A2"), (250, 256, "A3"), (260, 260, "A5")):
        if low <= soc_version <= high:
            return family
    raise RuntimeError(f"Unknown device SoC version: {soc_version}; refusing to select a test family")


def current_device_family():
    import torch_npu

    if not torch_npu.npu.is_available():
        raise RuntimeError("Operator selection requires a real available NPU")
    return identify_device(torch_npu.npu.get_soc_version())


def selected_device_index():
    """Logical device index inside the caller's existing visibility allocation."""
    requested = os.environ.get("MINDIE_TEST_DEVICE_ID")
    if requested is not None:
        index = int(requested)
        if index < 0:
            raise ValueError("MINDIE_TEST_DEVICE_ID must be a nonnegative logical index")
        return index
    import torch_npu

    return int(torch_npu.npu.current_device())


def supports(operator, family, kind="positive"):
    if operator not in REGISTRATIONS:
        raise ValueError(f"No audited OpDef registration for {operator}")
    if family not in FAMILIES:
        raise RuntimeError(f"Unknown device family: {family}")
    if kind in CONTRACT_KINDS:
        # dtype_contract means parameter validation, NOT numerical precision.
        # All numerical operator execution uses kind='positive'.
        return True
    if kind != "positive":
        raise ValueError(f"Unknown test kind: {kind}")
    return family in {SOC_FAMILY.get(soc) for soc in REGISTRATIONS[operator]}


def require_support(operator, kind="positive"):
    family = current_device_family()
    if not supports(operator, family, kind):
        raise unittest.SkipTest(
            f"{operator} {kind} is not registered for {family}; OpDef AddConfig={REGISTRATIONS[operator]}"
        )
    return family


def native_test(operator):
    if os.environ.get("MINDIE_TEST_MODE") == "CPU":
        return unittest.skip("Real operator test requires the NPU suite")
    family = current_device_family()
    return unittest.skipUnless(supports(operator, family), f"{operator} has no native OpDef registration for {family}")


def init_test_device():
    """Select a visible logical ordinal; never unset or reinterpret visibility masks.

    Operator registration remains the caller's import responsibility. Device
    availability and visible resource shortages are failures, not SoC skips.
    """
    import torch
    import torch_npu

    index = selected_device_index()
    count = torch_npu.npu.device_count()
    if not torch_npu.npu.is_available() or index >= count:
        raise RuntimeError(f"NPU test resource unavailable: requested logical device {index}, visible count={count}")
    torch_npu.npu.set_device(index)
    return torch.device(f"npu:{index}")
