"""Optional Triton import fallback must fail clearly without touching a device."""

import importlib
from unittest.mock import patch

import pytest


def test_missing_triton_exposes_actionable_public_fallback():
    package = importlib.import_module("mindiesd.layers.flash_attn")
    triton_utils = importlib.import_module("mindiesd.layers.triton_utils")
    try:
        with patch.object(triton_utils, "_HAS_TRITON", False):
            with pytest.warns(UserWarning, match="Triton-Ascend"):
                importlib.reload(package)
            with pytest.raises(RuntimeError, match="requires Triton-Ascend"):
                package.SparseLinearAttention()
    finally:
        importlib.reload(package)
