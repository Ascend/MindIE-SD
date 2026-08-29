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

"""BackendNPU: Ascend NPU backend shim for xpu-perf micro_perf.

Registered via a local shim (npu_launch.py injects this module into the
backend module list) since xpu-perf only scans its own backends package.

Timing is single-path wall-clock: warmup iterations are issued and synced,
then `prefer_iterations` are run back-to-back with one sync at the end; the
average latency covers the whole measured op (quantization included when a
quantized vendor path is timed).
"""

import os

import torch
import torch.distributed as dist
from common.env_util import load_peaks
from xpu_perf.micro_perf.core.backend import Backend


class BackendNPU(Backend):
    """NPU backend implementing the xpu-perf Backend ABC."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        device_name = self.backend_info.get("device_name", "")
        self.peak_flops, self.peak_bw = load_peaks(kwargs.get("env_file") or "", device_name)

    @staticmethod
    def _npu():
        return torch.npu

    def process_envs(self):
        """Set default envs from env.json; numeric peak values become strings.

        The base Backend assumes env values are strings; env.json holds
        numeric peak_flops/peak_bw for MFU/MBU accounting, so coerce to str.
        """
        override_envs = {}
        for env, val in self.default_envs.items():
            env_val = str(val) if not isinstance(val, str) else val
            if env in os.environ:
                override_envs[env] = os.environ[env]
            else:
                os.environ[env] = env_val
        return override_envs

    def get_backend_info(self):
        npu = self._npu()
        info = {}
        info["device_name"] = npu.get_device_name(0)
        info["device_count"] = npu.device_count()
        try:
            props = npu.get_device_properties(0)
            info["device_memory_mb"] = props.total_memory / (1024**2)
        except Exception:
            info["device_memory_mb"] = 0
        info["torch_version"] = torch.__version__
        try:
            import torch_npu

            info["torch_npu_version"] = torch_npu.__version__
        except Exception:
            info["torch_npu_version"] = "unknown"
        return info

    def get_torch_device_name(self):
        return "npu"

    def get_device_name(self, index=0):
        return self._npu().get_device_name(index)

    def get_device_properties(self, index=0):
        return self._npu().get_device_properties(index)

    def get_mem_info(self, index=0):
        return self._npu().mem_get_info(index)

    def get_device_count(self):
        device_count = self._npu().device_count()
        return device_count, list(range(device_count))

    def set_device(self, index):
        self._npu().set_device(index)

    def get_device(self):
        return self._npu().current_device()

    def device_synchronize(self):
        self._npu().synchronize()

    def empty_cache(self):
        self._npu().empty_cache()

    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "hccl"

    def core_perf(self, op_instance, warmup_iterations, prefer_iterations, tensor_list, profiling=True):
        """Wall-clock kernel timing.

        warmup 2 + >=5 iters per the repo benchmark convention; latency is the
        average over all measured iterations with a single sync after warmup
        and after the measured loop. The `profiling` argument is accepted for
        Backend ABC compatibility and ignored (single timing methodology).
        """
        op_group = op_instance.op_group
        group_size = op_instance.group_size
        self.op_group_barrier(op_group=op_group, group_size=group_size)
        self.device_synchronize()

        for _ in range(warmup_iterations):
            op_instance.core_run(tensor_list[_ % len(tensor_list)])
        self.device_synchronize()

        start_time = _perf_counter_ns()
        for _ in range(prefer_iterations):
            op_instance.core_run(tensor_list[_ % len(tensor_list)])
        self.device_synchronize()
        end_time = _perf_counter_ns()
        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        return latency_us, []


def _perf_counter_ns():
    import time

    return time.perf_counter_ns()
