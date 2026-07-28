/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
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
#include "mul_add.h"

using namespace at;

constexpr std::string_view MUL_ADD_OP_NAME = "aclnnMulAdd";

at::Tensor mul_add_mindie_sd_impl_npu(const at::Tensor &a, const at::Tensor &b, const at::Tensor &c)
{
    TORCH_CHECK(a.dim() == 3, "mul_add: a must be 3D [batch_size, seq_len, hidden_size], got ", a.dim(), "D");
    TORCH_CHECK(b.dim() == 3, "mul_add: b must be 3D [batch_size, seq_len, hidden_size], got ", b.dim(), "D");
    TORCH_CHECK(c.dim() == 3, "mul_add: c must be 3D [batch_size, 1, hidden_size], got ", c.dim(), "D");

    TORCH_CHECK(a.size(0) == b.size(0) && a.size(0) == c.size(0),
                "mul_add: batch_size must match across inputs, got ",
                a.size(0), " vs ", b.size(0), " vs ", c.size(0));
    TORCH_CHECK(a.size(2) == b.size(2) && a.size(2) == c.size(2),
                "mul_add: hidden_size must match across inputs, got ",
                a.size(2), " vs ", b.size(2), " vs ", c.size(2));
    TORCH_CHECK(a.size(1) == b.size(1),
                "mul_add: seq_len of a and b must match, got ",
                a.size(1), " vs ", b.size(1));
    TORCH_CHECK(c.size(1) == 1,
                "mul_add: c.size(1) must be 1 (broadcast dim), got ", c.size(1));
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(),
                "mul_add: all inputs must have the same dtype");

    if (a.numel() == 0) {
        return at::empty_like(a);
    }

    at::Tensor out = at_npu::native::empty_with_format(a.sizes(), a.options(), at_npu::native::get_npu_format(a));

    EXEC_NPU_CMD<MUL_ADD_OP_NAME>(a, b, c, out);

    return out;
}
