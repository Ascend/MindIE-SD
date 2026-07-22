#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
from pathlib import Path
from functools import wraps
import os
from typing import Callable
import torch
from torch.library import Library
from ..utils import file_utils, ParametersInvalid, is_npu_available
from ..utils.logs.logging import logger


MINDIE_NS = "mindiesd"  # 固定命名空间，与 torch.ops.mindiesd 对应
PLUGIN_LIBRARY_NAME = "libPTAExtensionOPS.so"
PLUGIN_VARIANT_ENV = "MINDIESD_PLUGIN_VARIANT"
SUPPORTED_TORCH_PLUGIN_VARIANTS = {
    "2.6": "torch26",
    "2.7": "torch27",
    "2.8": "torch28",
    "2.9": "torch29",
    "2.10": "torch210",
}


def _get_torch_major_minor(version: str) -> str:
    version_core = version.split("+", maxsplit=1)[0]
    parts = version_core.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"Cannot parse torch version: {version}")
    return ".".join(parts[:2])


def _has_plugin_variants(ops_path: str) -> bool:
    return any((Path(ops_path) / variant).is_dir() for variant in SUPPORTED_TORCH_PLUGIN_VARIANTS.values())


def _select_mindie_ops_file(ops_path: str) -> str:
    forced_variant = os.environ.get(PLUGIN_VARIANT_ENV, "").strip()
    if forced_variant:
        if forced_variant not in SUPPORTED_TORCH_PLUGIN_VARIANTS.values():
            raise RuntimeError(
                f"Unsupported {PLUGIN_VARIANT_ENV}={forced_variant}. "
                f"Expected one of: {', '.join(SUPPORTED_TORCH_PLUGIN_VARIANTS.values())}."
            )
        return os.path.join(ops_path, forced_variant, PLUGIN_LIBRARY_NAME)

    torch_version_key = _get_torch_major_minor(torch.__version__)
    variant = SUPPORTED_TORCH_PLUGIN_VARIANTS.get(torch_version_key)
    variants_exist = _has_plugin_variants(ops_path)
    if variant:
        variant_ops_file = os.path.join(ops_path, variant, PLUGIN_LIBRARY_NAME)
        if os.path.isfile(variant_ops_file) or variants_exist:
            return variant_ops_file

    if variants_exist:
        supported_versions = ", ".join(SUPPORTED_TORCH_PLUGIN_VARIANTS)
        raise RuntimeError(
            f"Unsupported torch version {torch.__version__} for MindIE-SD plugin variants. "
            f"Supported torch major.minor versions: {supported_versions}."
        )

    return os.path.join(ops_path, PLUGIN_LIBRARY_NAME)


def _load_mindie_ops_library() -> None:
    """Load the MindIE custom operator shared library.

    Raises:
        ParametersInvalid: If the parent directory level is insufficient.
        FileNotFoundError: If the operator SO file is not found.
        PermissionError: If the SO file has invalid permissions.
    """
    current_path = Path(__file__).resolve()
    if len(current_path.parents) < 2:
        raise ParametersInvalid("Insufficient parent directory levels to locate plugin folder.")

    ops_path = current_path.parents[1] / "plugin"
    ops_path = file_utils.standardize_path(str(ops_path))
    ops_file = _select_mindie_ops_file(ops_path)

    file_utils.check_file_safety(ops_file, permission_mode=file_utils.BINARY_FILE_PERMISSION)
    torch.ops.load_library(ops_file)


if is_npu_available():
    _load_mindie_ops_library()


def check_mindie_operator_exists(op_name: str) -> bool:
    """Check if a MindIE operator is registered in PyTorch.

    Args:
        op_name: Full name of the operator (e.g. "rope", "la")

    Returns:
        True if the operator exists, False otherwise.
    """
    try:
        getattr(torch.ops.mindiesd, op_name)
        return True
    except AttributeError:
        return False


if _get_torch_major_minor(torch.__version__) == "2.1":
    # PyTorch 2.1 使用 Library.impl
    _lib = Library(MINDIE_NS, "IMPL")

    def _compatible_register_fake(op_name: str):
        """Compatibility wrapper for PyTorch 2.1 fake registration."""

        def decorator(fake_func: Callable):
            @wraps(fake_func)
            def wrapper(*args, **kwargs):
                # Ensure all tensor inputs are on Meta device (required for PyTorch 2.1)
                args = [a.to(device="meta") if isinstance(a, torch.Tensor) else a for a in args]
                kwargs = {k: v.to(device="meta") if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
                return fake_func(*args, **kwargs)

            _lib.impl(op_name, wrapper, "Meta")
            return fake_func

        return decorator
else:
    # PyTorch 2.2+ 使用 register_fake 或 impl_abstract
    try:
        from torch.library import register_fake as _native_register_fake
    except ImportError:
        from torch.library import impl_abstract as _native_register_fake

    def _compatible_register_fake(op_name: str):
        """Compatibility wrapper for PyTorch 2.2+ fake registration."""
        return _native_register_fake(op_name)


def register_mindie_fake_op(op_name: str):
    """Decorator to register a fake implementation for a MindIE operator.

    Usage:
        @register_mindie_fake_op("rope")
        def rope_fake(x, cos, sin, mode):
            ...

    Args:
        op_name: Full name of the operator (e.g. "rope", "la")

    Returns:
        Decorator function that registers the fake implementation.
    """
    if not is_npu_available():

        def dummy_decorator(func):
            return func

        return dummy_decorator

    if not check_mindie_operator_exists(op_name):
        logger.error(
            "[MindIE-SD/layers] MindIE custom operator registration failed. "
            "issue=operator is not found in torch.ops.%s, op_name=%s, expected=%s::%s exists. "
            "possible_cause=custom operator shared library was not loaded or TORCH_LIBRARY registration is missing. "
            "Troubleshooting: check libPTAExtensionOPS.so path, ASCEND_CUSTOM_OPP_PATH, operator build output, "
            "and torch.ops.%s registry.",
            MINDIE_NS,
            op_name,
            MINDIE_NS,
            op_name,
            MINDIE_NS,
        )
        raise RuntimeError(
            f"MindIE operator {MINDIE_NS}::{op_name} not found! "
            "Ensure the SO library is loaded and the operator is registered with TORCH_LIBRARY."
        )

    return _compatible_register_fake(f"{MINDIE_NS}::{op_name}")
