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

#include <torch/library.h>
#include <iostream>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "la_preprocess.h"

using namespace at;

namespace {
constexpr int EXPECTED_TENSOR_DIMENSION = 4;
constexpr std::string_view LAPREPROCESS_NAME = "aclnnLaPreprocess";

}

std::tuple<at::Tensor, at::Tensor, at::Tensor>la_preprocess_mindie_sd_impl_npu(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    int64_t align_len)
{
    TORCH_CHECK(align_len > 0, "align_len must be positive, but got ", align_len);
    TORCH_CHECK(query.dim() == EXPECTED_TENSOR_DIMENSION, "Query must be 4D tensor");
    TORCH_CHECK(key.dim() == EXPECTED_TENSOR_DIMENSION, "Key must be 4D tensor");
    TORCH_CHECK(value.dim() == EXPECTED_TENSOR_DIMENSION, "Value must be 4D tensor");

    auto batch_size = query.sizes()[0];
    auto q_seq_len = query.sizes()[1];
    auto k_seq_len = key.sizes()[1];
    auto v_seq_len = value.sizes()[1];
    auto head_num = query.sizes()[2];
    auto head_dim = query.sizes()[3];

    auto q_padded_seq_len = (q_seq_len + align_len - 1) / align_len * align_len;
    auto k_padded_seq_len = (k_seq_len + align_len - 1) / align_len * align_len;
    auto v_padded_seq_len = (v_seq_len + align_len - 1) / align_len * align_len;
    auto options = query.options().dtype(at::kHalf);
    auto format = at_npu::native::get_npu_format(query);

    at::Tensor out_query = at_npu::native::empty_with_format(
        {batch_size, head_num, q_padded_seq_len, head_dim}, options, format);
    at::Tensor out_key = at_npu::native::empty_with_format(
        {batch_size, head_num, k_padded_seq_len, head_dim}, options, format);
    at::Tensor out_value = at_npu::native::empty_with_format(
        {batch_size, head_num, v_padded_seq_len, head_dim}, options, format);

    EXEC_NPU_CMD<LAPREPROCESS_NAME>(query, key, value, align_len,
        out_query, out_key, out_value);
    return std::make_tuple(out_query, out_key, out_value);
}