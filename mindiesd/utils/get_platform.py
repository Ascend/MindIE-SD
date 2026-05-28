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

from enum import Enum, auto
import torch_npu
from .logs import logger

PLATFORM = None


class NPUDevice(Enum):
    UNDEFINED = auto()
    A2 = auto()
    A3 = auto()
    A5 = auto()
    Duo = auto()


def get_npu_device() -> NPUDevice:
    global PLATFORM
    if PLATFORM is None:
        try:
            if torch_npu.npu.device_count() == 0:
                PLATFORM = NPUDevice.UNDEFINED
                return PLATFORM
            soc_version = torch_npu.npu.get_soc_version()
            if 200 <= soc_version <= 205:
                PLATFORM = NPUDevice.Duo
            elif 220 <= soc_version <= 225:
                PLATFORM = NPUDevice.A2
            elif 250 <= soc_version <= 255:
                PLATFORM = NPUDevice.A3
            elif soc_version == 260:
                PLATFORM = NPUDevice.A5
            else:
                PLATFORM = NPUDevice.UNDEFINED
        except RuntimeError as exc:
            logger.warning("Failed to get NPU SoC version: %s", exc)
            PLATFORM = NPUDevice.UNDEFINED
    return PLATFORM
