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
from xpu_perf.micro_perf.core.backend import Backend


class BackendNPU(Backend):
    """NPU backend implementing the xpu-perf Backend ABC.

    CUBE peaks (peak_flops / peak_bw) are provided by the user per run via
    --config and carried by each case; MFU/MBU accounting reads them from the
    op's args (op_defs/_common.py MfuMbuSummaryMixin).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _npu():
        return torch.npu

    def process_envs(self):
        """Set default envs from self.default_envs (base Backend convention).

        The base Backend assumes env values are strings; coerce any non-string
        value (e.g. numeric peaks) to str.
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

    def core_perf(
        self, op_instance, warmup_iterations, prefer_iterations, tensor_list, profiling=True
    ):
        """Wall-clock kernel timing with a per-case timeout.

        warmup 2 + >=5 iters per the repo benchmark convention; latency is the
        average over all measured iterations with a single sync after warmup
        and after the measured loop. The `profiling` argument is accepted for
        Backend ABC compatibility and ignored (single timing methodology).

        A single case must finish within `CASE_TIMEOUT_S` (default 5s) or it is
        skipped: the worker is abandoned and RuntimeError is raised so the
        xpu-perf `perf` wrapper records the case as a skip and moves on. The
        queued kernel is NOT drained with device_synchronize() on timeout —
        a kernel that is merely slow would block the drain and freeze the run;
        individual kernels complete on their own, so the next case proceeds
        while the abandoned worker finishes asynchronously. Unsupported ops
        likewise raise RuntimeError from the vendor impl and are skipped.
        Workload dtype/quant content is never modified.
        """
        import threading

        result = {}

        def _run():
            try:
                result["value"] = self._core_perf_impl(
                    op_instance, warmup_iterations, prefer_iterations, tensor_list
                )
            except Exception as exc:  # noqa: BLE001 - surface to caller
                result["error"] = exc

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        case_timeout = _op_timeout(op_instance)
        worker.join(timeout=case_timeout)
        if worker.is_alive():
            # Timeout: skip this case. Do NOT drain with device_synchronize()
            # here — a queued kernel that is merely slow would block the drain
            # and freeze the worker. Individual kernels have been verified to
            # complete on their own; the queued kernel finishes asynchronously
            # while the next case proceeds. The xpu-perf `perf` wrapper catches
            # the RuntimeError and records the case as a skip.
            raise RuntimeError(f"case timeout after {case_timeout}s")
        if "error" in result:
            raise result["error"]
        return result["value"]

    def _core_perf_impl(self, op_instance, warmup_iterations, prefer_iterations, tensor_list):
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


# Per-case wall-clock timeout (seconds). Default 5s fits mid-size seqlen
# scans; long-sequence sweeps (up to 1M tokens) take much longer per case, so
# a case may carry its own timeout via the workload config
# (--config {"timeout": 300}) — see _op_timeout.
CASE_TIMEOUT_S = 5.0


def _op_timeout(op_instance, default=CASE_TIMEOUT_S):
    """Per-case timeout from the case args ({"timeout": 300}), else default."""
    try:
        return float(op_instance.args_dict.get("timeout", default))
    except (AttributeError, TypeError, ValueError):
        return default


def _perf_counter_ns():
    import time

    return time.perf_counter_ns()
