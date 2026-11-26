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

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

// Block sparse attention via aclnnEagleQuantBlockSparseAttention (BF16/FP16/INT8/FP8/MXFP4).
// qmode1 routes query/key/value_scale to ACLNN q/k/vScale.
// MXFP4 routes query/key/value_dtype_scale to q/k/vScale and query/key/value_scale to q/k/vLevel0Scale.
// MXFP4 inputs are packed in uint8 tensors and exposed to ACLNN through dtype wrappers.
// MXFP4 output supports FP16/BF16, selected by output_dtype (BF16 by default).
// Takes block_sparse_mask (int8). Supports TND and BNSD layouts.
// Returns (attention_out, softmax_lse).
std::tuple<at::Tensor, at::Tensor> eagle_quant_block_sparse_attention_impl_npu(const at::Tensor &query,
    const at::Tensor &key, const at::Tensor &value, const c10::optional<at::Tensor> &block_sparse_mask,
    at::IntArrayRef block_shape, std::string q_input_layout, std::string kv_input_layout, int64_t num_key_value_heads,
    double scale_value, int64_t inner_precise, c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv, int64_t softmax_lse_flag,
    const c10::optional<at::Tensor> &query_scale = c10::nullopt,
    const c10::optional<at::Tensor> &key_scale = c10::nullopt,
    const c10::optional<at::Tensor> &value_scale = c10::nullopt,
    const c10::optional<int64_t> &query_dtype = c10::nullopt, const c10::optional<int64_t> &key_dtype = c10::nullopt,
    const c10::optional<int64_t> &value_dtype = c10::nullopt,
    const c10::optional<at::ScalarType> &output_dtype = c10::nullopt,
    const c10::optional<at::Tensor> &query_dtype_scale = c10::nullopt,
    const c10::optional<at::Tensor> &key_dtype_scale = c10::nullopt,
    const c10::optional<at::Tensor> &value_dtype_scale = c10::nullopt);

#endif // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H
