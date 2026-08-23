#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""NPU vendor provider for Ascend core ops.

Discovered by xpu-perf via PROVIDER_NAME; ops/ modules are imported
recursively by parse_vendor_ops and register through ProviderRegistry.
"""

import runpy
from pathlib import Path

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "NPU"


def _load_version() -> str:
    version_file = Path(__file__).resolve().parents[4] / "version.py"
    return runpy.run_path(str(version_file))["__version__"]


__version__ = _load_version()

ProviderRegistry.register_provider_info(
    PROVIDER_NAME,
    {"version": __version__, "description": "Ascend NPU vendor implementation for Ascend core ops"},
)

__all__ = ["PROVIDER_NAME", "__version__"]
