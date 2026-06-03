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

from .logs.logging import logger


def _log_exception(component, title, message, cause, troubleshooting):
    logger.error(
        "%s %s. issue=%s. possible_cause=%s. Troubleshooting: %s",
        component,
        title,
        message,
        cause,
        troubleshooting,
    )


class ParametersInvalid(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000001] Parameters invalid. {message}"
        _log_exception(
            "[MindIE-SD/utils]",
            "Parameter validation failed",
            self.message,
            "an input parameter does not match the required type, range, shape, or supported value list",
            "compare the actual parameter in the message with the expected value and fix the caller input",
        )
        super().__init__(self.message)


class ConfigError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000002] Config parameter err. {message}"
        _log_exception(
            "[MindIE-SD/utils]",
            "Configuration validation failed",
            self.message,
            "a configuration item is missing, invalid, or inconsistent with the runtime requirement",
            "check the configuration file, environment variables, and the expected value shown in the message",
        )
        super().__init__(self.message)


class TorchError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000003] Torch exec err. {message}"
        _log_exception(
            "[MindIE-SD/utils]",
            "Torch execution failed",
            self.message,
            "the torch or torch_npu operator failed during execution",
            "check the operator input shape, dtype, device placement, and the CANN/torch_npu error stack",
        )
        super().__init__(self.message)


class ModelInitError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000004] Model init err. {message}"
        _log_exception(
            "[MindIE-SD/utils]",
            "Model initialization failed",
            self.message,
            "model weights, configuration, or runtime resources are not ready",
            "check model path, weight files, configuration values, NPU memory, and initialization stack",
        )
        super().__init__(self.message)


class ModelExecError(Exception):
    def __init__(self, message):
        self.message = f"[MIE06E000005] Model exec err. {message}"
        _log_exception(
            "[MindIE-SD/utils]",
            "Model execution failed",
            self.message,
            "the model forward, scheduler, or custom operator path failed during execution",
            "check the request parameters, tensor shape and dtype, scheduler state, and CANN operator error stack",
        )
        super().__init__(self.message)
