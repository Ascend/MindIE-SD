# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/Mulan PSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Mock torch_npu for CPU-only test mode."""

import os
import sys
from importlib.util import spec_from_loader
from unittest.mock import MagicMock


def mock_torch_npu():
    """In CPU mode, mock torch_npu so that tests can import mindiesd without NPU hardware.

    This replaces ``torch_npu`` in ``sys.modules`` with a MagicMock that reports
    zero available devices.  It also patches ``torch.npu`` so that
    ``torch.npu.is_available()`` returns ``False``.
    """
    if os.environ.get("MINDIE_TEST_MODE", "ALL") != "CPU":
        return
    try:
        import torch_npu  # noqa: F401  # noqa: F811
    except Exception:
        import torch

        torch_npu = MagicMock()
        torch_npu.__spec__ = spec_from_loader("torch_npu", loader=None)
        torch_npu.npu.device_count = MagicMock(return_value=0)
        torch_npu.npu.is_available = MagicMock(return_value=False)
        sys.modules["torch_npu"] = torch_npu
        torch.npu = torch_npu.npu
