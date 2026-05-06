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


import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

_CONFIG_ALLOW = ["*.json", "*.txt", "*.model", "tokenizer*"]
_CONFIG_IGNORE = ["*.safetensors", "*.bin", "*.msgpack", "*.ckpt", "*.pth"]


def check_npu():
    if not _get_npu_available():
        raise RuntimeError("NPU is not available")
    npu_smi = shutil.which("npu-smi")
    if npu_smi is None:
        raise RuntimeError("npu-smi not found in PATH")
    result = subprocess.run(
        [npu_smi, "info", "-l"], capture_output=True, text=True
    )
    total = sum(1 for line in result.stdout.splitlines() if "NPU ID" in line)
    logger.warning("NPU: %d card(s) available", total)


def _get_npu_available():
    try:
        import torch_npu

        return torch_npu.npu.is_available()
    except Exception:
        return False


def resolve_config_path(config_cache, model_id):
    if config_cache and os.path.isdir(config_cache):
        return config_cache

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            model_id,
            local_files_only=True,
            allow_patterns=_CONFIG_ALLOW,
            ignore_patterns=_CONFIG_IGNORE,
            max_workers=1,
        )
    except Exception:
        logger.debug("Local cache not found, trying remote download")

    mirror = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.warning("Downloading config files from %s : %s", mirror, model_id)
    try:
        return snapshot_download(
            model_id,
            endpoint=mirror,
            allow_patterns=_CONFIG_ALLOW,
            ignore_patterns=_CONFIG_IGNORE,
            max_workers=1,
        )
    except Exception as exc:
        logger.error("Failed to download config: %s", exc)
        logger.error(
            "Set HF_ENDPOINT=https://hf-mirror.com for mirrors, "
            "or pass --config_cache /path/to/config for offline mode."
        )
        raise RuntimeError("Failed to download config: %s" % exc) from exc


class _PhaseTimer:
    _SEP70 = "=" * 70
    _DIVIDER_58 = "  " + "-" * 58
    _DIVIDER_68 = "  " + "-" * 68

    def __init__(self, device_id=None):
        self._device_id = device_id
        self._build_records = []
        self._prev_mem = 0
        self._peak = 0
        self._infer_warmup = []
        self._infer_timed = []
        self._hook_records = []
        self._handles = []
        self._starts = {}

    def start_build(self, device_id=None):
        if device_id is not None:
            self._device_id = device_id
        self._prev_mem = self._allocated()

    def record_build(self, name, elapsed):
        mem = self._allocated()
        delta = max(0, mem - self._prev_mem)
        self._peak = max(self._peak, mem)
        self._build_records.append((name, elapsed, mem))
        if delta > 0:
            logger.warning(
                "  [%s] %.2f ms | +%.2fGB | alloc=%.2fGB peak=%.2fGB",
                name,
                elapsed * 1000,
                delta / (1024**3),
                mem / (1024**3),
                self._peak / (1024**3),
            )
        else:
            logger.warning(
                "  [%s] %.2f ms | alloc=%.2fGB peak=%.2fGB",
                name,
                elapsed * 1000,
                mem / (1024**3),
                self._peak / (1024**3),
            )
        self._prev_mem = mem

    # ---- Inference hooks ----
    def install(self, pipe, extra_attrs=None):
        for attr in ("text_encoder", "transformer", "transformer_2", "vae"):
            module = getattr(pipe, attr, None)
            if module is None:
                continue
            name = attr
            h_pre = module.register_forward_pre_hook(lambda _m, _in, _n=name: self._on_pre(_n))
            h_post = module.register_forward_hook(lambda _m, _in, _out, _n=name: self._on_post(_n))
            self._handles.append(h_pre)
            self._handles.append(h_post)
        if extra_attrs:
            for attr in extra_attrs:
                module = getattr(pipe, attr, None)
                if module is None:
                    continue
                name = attr
                h_pre = module.register_forward_pre_hook(lambda _m, _in, _n=name: self._on_pre(_n))
                h_post = module.register_forward_hook(
                    lambda _m, _in, _out, _n=name: self._on_post(_n)
                )
                self._handles.append(h_pre)
                self._handles.append(h_post)

    def capture_warmup(self):
        self._infer_warmup = list(self._hook_records)
        self._hook_records.clear()

    def capture_timed(self):
        self._infer_timed = list(self._hook_records)
        self._hook_records.clear()

    # ---- Summary ----
    def summary(self):
        logger.warning(self._SEP70)
        logger.warning("  %-40s %10s %10s", "BUILD", "Time(ms)", "Mem(GB)")
        logger.warning(self._DIVIDER_58)
        total_build = 0.0
        for name, elapsed, mem in self._build_records:
            logger.warning("  %-40s %10.2f %10.2f", name, elapsed * 1000, mem / (1024**3))
            total_build += elapsed * 1000
        logger.warning(self._DIVIDER_58)
        logger.warning("  %-40s %10.2f", "BUILD TOTAL", total_build)

        logger.warning("")
        logger.warning("  %-40s %10s %10s %10s", "INFERENCE", "Time(ms)", "Mem(GB)", "Peak(GB)")
        logger.warning(self._DIVIDER_68)
        logger.warning("  -- Warmup --")
        for name, elapsed, mem in self._infer_warmup:
            logger.warning(
                "  %-40s %10.2f %10.2f %10.2f",
                name,
                elapsed * 1000,
                mem / (1024**3),
                self._peak / (1024**3),
            )
        logger.warning("")
        logger.warning("  -- Timed --")
        total_timed = 0.0
        for name, elapsed, mem in self._infer_timed:
            logger.warning(
                "  %-40s %10.2f %10.2f %10.2f",
                name,
                elapsed * 1000,
                mem / (1024**3),
                self._peak / (1024**3),
            )
            total_timed += elapsed * 1000
        logger.warning(self._DIVIDER_68)
        logger.warning(
            "  %-40s %10.2f %10s %10.2f",
            "OVERALL TOTAL",
            total_build + total_timed,
            "",
            self._peak / (1024**3),
        )
        logger.warning(self._SEP70)

    def _on_pre(self, name):
        import torch_npu

        torch_npu.npu.synchronize()
        self._starts[name] = time.time()

    def _on_post(self, name):
        import torch_npu

        torch_npu.npu.synchronize()
        t_end = time.time()
        t_start = self._starts.get(name, t_end)
        elapsed = t_end - t_start
        mem = self._allocated()
        self._peak = max(self._peak, mem)
        self._hook_records.append((name, elapsed, mem))
        logger.warning(
            "  [%s] %.2f ms | alloc=%.2fGB peak=%.2fGB",
            name,
            elapsed * 1000,
            mem / (1024**3),
            self._peak / (1024**3),
        )

    def _allocated(self):
        try:
            import torch_npu

            if self._device_id is not None:
                return torch_npu.npu.memory_allocated(self._device_id)
            return torch_npu.npu.memory_allocated()
        except Exception:
            return 0
