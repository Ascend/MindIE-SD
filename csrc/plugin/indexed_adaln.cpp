/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <string_view>
#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "indexed_adaln.h"

using namespace at;

constexpr std::string_view INDEXED_ADALN_OP_NAME = "aclnnIndexedAdaLayerNorm";

at::Tensor indexed_adaln_mindie_sd_impl_npu(const at::Tensor &x, const at::Tensor &scale_table,
    const at::Tensor &shift_table, const at::Tensor &mod_index, const c10::optional<double> &epsilon) {
    TORCH_CHECK(x.dim() == 3, "indexed_adaln: x must be 3D [B, T, H], got ", x.dim(), "D");
    TORCH_CHECK(scale_table.dim() == 3, "indexed_adaln: scale_table must be 3D [B, K, H]");
    TORCH_CHECK(shift_table.dim() == 3, "indexed_adaln: shift_table must be 3D [B, K, H]");
    TORCH_CHECK(mod_index.dim() == 1, "indexed_adaln: mod_index must be 1D [T]");
    const auto dtype = x.scalar_type();
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat,
        "indexed_adaln: x must be BF16, FP16 or FP32");
    TORCH_CHECK(scale_table.scalar_type() == dtype, "indexed_adaln: scale_table dtype must match x");
    TORCH_CHECK(shift_table.scalar_type() == dtype, "indexed_adaln: shift_table dtype must match x");
    TORCH_CHECK(mod_index.scalar_type() == at::kLong, "indexed_adaln: mod_index must be INT64");
    TORCH_CHECK(
        x.size(0) == scale_table.size(0) && x.size(0) == shift_table.size(0), "indexed_adaln: batch size mismatch");
    TORCH_CHECK(
        x.size(2) == scale_table.size(2) && x.size(2) == shift_table.size(2), "indexed_adaln: hidden size mismatch");
    TORCH_CHECK(scale_table.size(1) == shift_table.size(1), "indexed_adaln: K mismatch");
    TORCH_CHECK(mod_index.size(0) == x.size(1), "indexed_adaln: mod_index length must equal T");

    if (x.numel() == 0) {
        return at::empty_like(x);
    }

    at::Tensor out = at_npu::native::empty_with_format(x.sizes(), x.options(), at_npu::native::get_npu_format(x));
    const double eps = epsilon.value_or(1e-6);
    EXEC_NPU_CMD<INDEXED_ADALN_OP_NAME>(x, scale_table, shift_table, mod_index, eps, out);
    return out;
}
