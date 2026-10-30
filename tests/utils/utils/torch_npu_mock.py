# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/Mulan PSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# pylint: disable=attribute-defined-outside-init

"""Explicit device boundary doubles for CPU unit tests.

Install before importing mindiesd, even on machines with torch_npu installed.
Unmocked device operations fail rather than silently returning MagicMock values.
"""

import os
import sys
from importlib.util import spec_from_loader
from types import ModuleType
from unittest.mock import MagicMock


def _unexpected_device_call(*args, **kwargs):
    raise AssertionError("CPU unit test called an NPU interface without a local mock")


class DeviceModule(ModuleType):
    """Expose patchable device APIs while rejecting their unmocked execution."""

    def __init__(self, name):
        super().__init__(name)
        self.__file__ = __file__
        self.__path__ = []
        self.__spec__ = spec_from_loader(name, loader=None, is_package=True)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        method = MagicMock(name=f"{self.__name__}.{name}", side_effect=_unexpected_device_call)
        setattr(self, name, method)
        return method


_CPU_OPERATOR_LIBRARIES = []


def mock_torch_npu():
    """Install CPU doubles without importing the real device backend or driver."""
    if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
        return
    existing = sys.modules.get("torch_npu")
    if isinstance(existing, DeviceModule):
        return
    if existing is not None:
        raise RuntimeError("CPU bootstrap must run before the real torch_npu is imported")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    backend = DeviceModule("torch_npu")
    device = DeviceModule("torch_npu.npu")
    backend.npu = device
    sys.modules["torch_npu"] = backend
    sys.modules["torch_npu.npu"] = device

    # Initialize CPU compiler capability detection before exposing the import-only
    # JIT facade. Inductor must observe an absent GPU compiler, not a partial SDK.
    sys.modules["triton"] = None
    import torch
    import torch.utils._triton

    # The import facade below contains no compiler. CPU Inductor must use its
    # native CPU backend rather than treating the facade as an installed GPU SDK.
    torch.utils._triton.has_triton_package = lambda: False
    torch.utils._triton.has_triton = lambda: False
    import torch._dynamo
    import torch._inductor
    import torch._inductor.runtime.triton_compat

    del sys.modules["triton"]
    from .triton_mock import mock_triton

    mock_triton()

    device.device_count = MagicMock(return_value=0)
    device.is_available = MagicMock(return_value=False)
    device.get_soc_version = MagicMock(return_value=0)
    device.current_device = MagicMock(return_value=0)
    device._is_in_bad_fork = lambda: False
    device.manual_seed_all = MagicMock()
    device._is_compiled = lambda: False
    backend.__version__ = "cpu-test-double"
    backend.float8_e4m3fn = torch.float8_e4m3fn
    backend.float8_e5m2 = torch.float8_e5m2
    backend.float8_e8m0fnu = getattr(torch, "float8_e8m0fnu", torch.uint8)
    backend.float4_e2m1fn_x2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    torch.npu = device
    sys.modules["torch.npu"] = device
    torch.Tensor.npu = _unexpected_device_call
    torch.nn.Module.npu = _unexpected_device_call

    # FX pattern matching needs stable operator schemas, but never a device kernel.
    library = torch.library.Library("npu", "FRAGMENT")
    for name in ("npu_dtype_cast", "_npu_dtype_cast"):
        library.define(f"{name}(Tensor self, ScalarType dtype) -> Tensor")
        library.impl(name, lambda value, dtype: value.to(dtype), "CPU")
        library.impl(name, lambda value, dtype: value.to(dtype), "Meta")
    _CPU_OPERATOR_LIBRARIES.append(library)
