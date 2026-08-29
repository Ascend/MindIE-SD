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

"""NPU benchmark entry for MindIE-SD core ops (FA/BSA/GMM/MM).

Based on xpu-perf launch.py but injects the local BackendNPU shim since
xpu-perf only scans its own backends package for --backend choices.

Usage:
    python npu_launch.py --task_dir ../workloads --task all
"""

import argparse
import pathlib
import sys

import torch.multiprocessing as mp
from backend_npu import BackendNPU
from xpu_perf.micro_perf.core.common_utils import (
    existing_dir_path,
    export_reports,
    logger,
    parse_tasks,
    setup_logger,
)
from xpu_perf.micro_perf.core.perf_engine import XpuPerfServer

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BENCHMARKS_DIR = FILE_DIR.parent
OP_DEFS_DIR = FILE_DIR.joinpath("op_defs")
VENDOR_NPU_DIR = FILE_DIR.joinpath("vendor_ops/NPU")

# benchmarks/ must be importable so the runtime plugin can use the shared
# `common` package (env_util/metrics/schema); `spawn` workers re-execute this
# module top-level, so the path is also present in child processes.
sys.path.insert(0, str(BENCHMARKS_DIR))

mp.set_start_method("spawn", force=True)


def parse_args():
    setup_logger("INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, default="NPU", choices=["NPU"])
    parser.add_argument("--op_defs", type=existing_dir_path, default=OP_DEFS_DIR)
    parser.add_argument(
        "--vendor_ops",
        type=existing_dir_path,
        default=[VENDOR_NPU_DIR],
        action="append",
    )
    # Forwarded to BackendNPU via XpuPerfServer as its env_file kwarg (the base
    # launcher contract); it is not consumed directly by this module.
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--numa", type=str, default="-1")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--node_world_size", type=int, default=1)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=49371)
    parser.add_argument("--host_port", type=int, default=49372)
    parser.add_argument("--device_port", type=int, default=49373)
    parser.add_argument("--task_dir", type=str, default=str(FILE_DIR.parent.joinpath("workloads")))
    parser.add_argument("--task", type=str, default="all")
    parser.add_argument("--workload", type=str)
    parser.add_argument("--report_dir", type=str, default=str(FILE_DIR.parent.joinpath("reports")))

    args = parser.parse_args()
    args.script_dir = FILE_DIR
    args.backend_name_list = ["NPU"]
    args.backend_mod_list = {"NPU": _npu_backend_module()}
    return args


def _npu_backend_module():
    """Return a module-like object exposing BackendNPU for the server.

    XpuPerfServer does `getattr(module, "Backend" + backend_type)`.
    """
    import types

    module = types.ModuleType("npu_backend_shim")
    module.BackendNPU = BackendNPU
    sys.modules["npu_backend_shim"] = module
    return module


def load_test_cases(args):
    return parse_tasks(args.task_dir, args.task)


def run_bench(args):
    test_cases = load_test_cases(args)
    if not test_cases:
        logger.error("No valid test cases found. Exiting.")
        raise SystemExit(1)
    logger.info("test cases: %s", {k: len(v) for k, v in test_cases.items()})

    with XpuPerfServer(args) as server_instance:
        info_dict = server_instance.get_info()
        bench_results = server_instance.normal_bench(test_cases)

    export_reports(args.report_dir, info_dict, test_cases, bench_results)


def main():
    args = parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
