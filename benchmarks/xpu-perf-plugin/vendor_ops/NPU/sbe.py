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

"""NPU sparse_block_estimate for Ascend 950."""

from xpu_perf.micro_perf.core.op import ProviderRegistry

# torch_npu.npu.get_soc_version() value for Ascend 950.
_A5_SOC_VERSION = 260


@ProviderRegistry.register_vendor_impl("sbe", "NPU")
class NPUSparseBlockEstimateOp:
    """Call mindiesd.sparse_block_estimate. Arguments match the A5 perf test."""

    # ProviderRegistry combines this vendor implementation with SparseBlockEstimateOp.
    head_dim: int
    stride: int
    sparse_size: int
    num_heads: int
    num_kv_heads: int
    threshold: float
    _mask_shape: tuple[int, ...]
    _count_shape: tuple[int, ...]

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._soc_checked = False
        self._lib_loaded = False
        self._output_checked = False

    def _ensure_op_loaded(self):
        """Load libPTAExtensionOPS.so once. torch.ops.mindiesd exists before that load and is empty."""
        if self._lib_loaded:
            return
        import torch

        try:
            getattr(torch.ops.mindiesd, "sparse_block_estimate")
        except AttributeError:
            from mindiesd.layers.register_ops import _load_mindie_ops_library

            _load_mindie_ops_library()
        self._lib_loaded = True

    def vendor_impl_run(self, tensor_mapping):
        import torch

        self._ensure_op_loaded()
        if not self._soc_checked:
            import torch_npu

            soc = torch_npu.npu.get_soc_version()
            if soc != _A5_SOC_VERSION:
                raise RuntimeError(f"sbe requires Ascend 950 (SoC {_A5_SOC_VERSION}), got {soc}")
            self._soc_checked = True

        query = tensor_mapping["q"]
        key = tensor_mapping["k"]
        scale_value = self.head_dim**-0.5 / self.stride
        mask, count = torch.ops.mindiesd.sparse_block_estimate(
            query=query,
            key=key,
            input_layout="BNSD",
            stride=self.stride,
            sparse_size=self.sparse_size,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale_value=scale_value,
            threshold=self.threshold,
            causal=False,
            keep_sink=True,
            keep_recent=True,
            row_sparse=1.0,
        )
        if not self._output_checked:
            self._output_checked = True
            if tuple(mask.shape) != self._mask_shape or mask.dtype != torch.int8:
                raise RuntimeError(f"unexpected sbe mask {tuple(mask.shape)} {mask.dtype}")
            if tuple(count.shape) != self._count_shape or count.dtype != torch.int32:
                raise RuntimeError(f"unexpected sbe count {tuple(count.shape)} {count.dtype}")
        return mask
