#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import os
import re
from packaging.version import Version


def normalize_version(value):
    """Convert legacy release overrides before both wheel and tar packaging."""
    value = re.sub(r"\.RC(\d+)\.(\d+)", r"rc\1.post\2", value)
    value = re.sub(r"\.T([0-9]*)$", r"+t\1", value)
    value = re.sub(r"T([0-9]+)$", r".post\1", value)
    return str(Version(value))


__version__ = normalize_version(os.environ.get("MINDIE_SD_VERSION_OVERRIDE", "3.1.0"))

if __name__ == "__main__":
    print(__version__)
