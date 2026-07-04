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

#ifndef FUSED_INFER_ATTENTION_SCORE_MINDIE_SD_IMPL_H
#define FUSED_INFER_ATTENTION_SCORE_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> fused_infer_attention_score_v2_impl_npu(const at::Tensor &query,
    const at::Tensor &key, const at::Tensor &value, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask, c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key, const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value, const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope, const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out, const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    std::string input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode,
    int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const c10::optional<int64_t> &query_dtype,
    const c10::optional<int64_t> &key_dtype, const c10::optional<int64_t> &value_dtype,
    const c10::optional<int64_t> &query_rope_dtype, const c10::optional<int64_t> &key_rope_dtype,
    const c10::optional<int64_t> &key_shared_prefix_dtype, const c10::optional<int64_t> &value_shared_prefix_dtype,
    const c10::optional<int64_t> &dequant_scale_query_dtype, const c10::optional<int64_t> &dequant_scale_key_dtype,
    const c10::optional<int64_t> &dequant_scale_value_dtype, const c10::optional<int64_t> &dequant_scale_key_rope_dtype,
    const c10::optional<at::ScalarType> &out_dtype);

#endif // FUSED_INFER_ATTENTION_SCORE_MINDIE_SD_IMPL_H
