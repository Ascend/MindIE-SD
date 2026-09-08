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

#ifndef BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H
#define BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

// Block sparse attention via aclnnBlockSparseAttentionV3/V2/V1 (BF16/FP16/FP8/MXFP4).
// V3 (preferred when present) adds quantMode/dstTypeMax for MXFP4:
//   quantMode 0 = none, 1 = FP8, 2 = MXFP4 OCP (floor truncation), 3 = MXFP4 CX (custom range).
// quant_mode < 0 (default) auto-derives so existing BF16/FP8 callers stay unchanged:
// no scales -> 0, FP8 dtype -> 1, packed FP4 storage -> 2.
// MXFP4 Q/K/V arrive as UINT8 storage; pass CANN dtype codes via q_dtype/k_dtype/v_dtype
// (296 = FLOAT4_E2M1) and *_scale_dtype (293 = FLOAT8_E8M0) to override the acl dtype.
// MXFP4 output head-dim doubles back (query packs 2 elements/byte) and is BF16.
// When dequant scales are not provided (BF16/FP16), nullptr is passed to the kernel.
// FP8 scales are FLOAT32; MXFP4 scales are E8M0 (UINT8 storage + dtype code).
// Takes block_sparse_mask (int8). Supports TND and BNSD layouts.
// Returns (attention_out, softmax_lse).
std::tuple<at::Tensor, at::Tensor> block_sparse_attention_impl_npu(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const c10::optional<at::Tensor> &block_sparse_mask, at::IntArrayRef block_shape,
    std::string q_input_layout, std::string kv_input_layout, int64_t num_key_value_heads, double scale_value,
    int64_t inner_precise, c10::OptionalIntArrayRef actual_seq_lengths, c10::OptionalIntArrayRef actual_seq_lengths_kv,
    int64_t softmax_lse_flag, const c10::optional<at::Tensor> &q_dequant_scale = c10::nullopt,
    const c10::optional<at::Tensor> &k_dequant_scale = c10::nullopt,
    const c10::optional<at::Tensor> &v_dequant_scale = c10::nullopt, int64_t quant_mode = -1,
    double dst_type_max = 0.0, const c10::optional<int64_t> &q_dtype = c10::nullopt,
    const c10::optional<int64_t> &k_dtype = c10::nullopt, const c10::optional<int64_t> &v_dtype = c10::nullopt,
    const c10::optional<int64_t> &q_scale_dtype = c10::nullopt, const c10::optional<int64_t> &k_scale_dtype = c10::nullopt,
    const c10::optional<int64_t> &v_scale_dtype = c10::nullopt);

#endif // BLOCK_SPARSE_ATTENTION_MINDIE_SD_IMPL_H
