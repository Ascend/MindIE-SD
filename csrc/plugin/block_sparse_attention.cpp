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

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "block_sparse_attention.h"

using namespace at;

namespace {
constexpr std::string_view BLOCK_SPARSE_ATTENTION_NAME = "aclnnBlockSparseAttention";

// Fixed op constraints per spec.
constexpr int64_t MASK_TYPE    = 0;           // no attention mask
constexpr int64_t PRE_TOKENS   = 2147483647;  // full context window
constexpr int64_t NEXT_TOKENS  = 2147483647;
} // namespace

std::tuple<at::Tensor, at::Tensor> block_sparse_attention_impl_npu(
    const at::Tensor                 &query,
    const at::Tensor                 &key,
    const at::Tensor                 &value,
    const c10::optional<at::Tensor>  &block_sparse_mask,
    at::IntArrayRef                   block_shape,
    std::string                       q_input_layout,
    std::string                       kv_input_layout,
    int64_t                           num_key_value_heads,
    double                            scale_value,
    int64_t                           inner_precise,
    c10::OptionalIntArrayRef          actual_seq_lengths,
    c10::OptionalIntArrayRef          actual_seq_lengths_kv,
    int64_t                           softmax_lse_flag)
{
    TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
        "block_sparse_attention: q_input_layout only supports 'TND' and 'BNSD', got ", q_input_layout);
    TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
        "block_sparse_attention: kv_input_layout only supports 'TND' and 'BNSD', got ", kv_input_layout);
    TORCH_CHECK(q_input_layout == kv_input_layout,
        "block_sparse_attention: q_input_layout and kv_input_layout must be consistent.");
    TORCH_CHECK(q_input_layout != "TND" ||
                (actual_seq_lengths.has_value() && actual_seq_lengths_kv.has_value()),
        "block_sparse_attention: actual_seq_lengths and actual_seq_lengths_kv are required for TND layout.");

    const char *qLayoutPtr  = q_input_layout.c_str();
    const char *kvLayoutPtr = kv_input_layout.c_str();

    // attenMaskOptional and blockTableOptional must be nullptr.
    c10::optional<at::Tensor> nulltensor = c10::nullopt;

    /* EXEC_NPU_CMD has ConvertType for c10::optional<at::IntArrayRef> only, not
        c10::OptionalIntArrayRef. Convert explicitly: nullopt -> nullptr (op tiling
        skips batch check), has_value() -> AclIntArray*. Do not use .value_or({})
        — empty array is interpreted as batch=0, conflicting with query batch dim. */
    c10::optional<at::IntArrayRef> optSeqLen = actual_seq_lengths.has_value()
        ? c10::optional<at::IntArrayRef>(actual_seq_lengths.value())
        : c10::nullopt;
    c10::optional<at::IntArrayRef> optSeqLenKv = actual_seq_lengths_kv.has_value()
        ? c10::optional<at::IntArrayRef>(actual_seq_lengths_kv.value())
        : c10::nullopt;

    // blockSize=0: PagedAttention not supported.
    constexpr int64_t blockSize = 0;

    at::Tensor attentionOut = at_npu::native::empty_with_format(
        query.sizes(), query.options(), at_npu::native::get_npu_format(query));

    // TND: [T, N, 1], BNSD: [B, N, S, 1]
    at::Tensor softmaxLse;
    if (q_input_layout == "TND") {
        softmaxLse = at_npu::native::empty_with_format(
            {query.size(0), query.size(1), 1},
            query.options().dtype(at::kFloat),
            at_npu::native::get_npu_format(query));
    } else {
        softmaxLse = at_npu::native::empty_with_format(
            {query.size(0), query.size(1), query.size(2), 1},
            query.options().dtype(at::kFloat),
            at_npu::native::get_npu_format(query));
    }
    // Pass nullptr when flag=0 (op skips lse write).
    c10::optional<at::Tensor> softmaxLseOpt = (softmax_lse_flag != 0)
        ? c10::optional<at::Tensor>(softmaxLse)
        : c10::nullopt;

    EXEC_NPU_CMD<BLOCK_SPARSE_ATTENTION_NAME>(
        query,
        key,
        value,
        block_sparse_mask,
        nulltensor,              // attenMaskOptional (nullptr)
        block_shape,
        optSeqLen,               // nullptr when not set
        optSeqLenKv,             // nullptr when not set
        nulltensor,              // blockTableOptional (nullptr)
        qLayoutPtr,
        kvLayoutPtr,
        num_key_value_heads,
        MASK_TYPE,
        scale_value,
        inner_precise,
        blockSize,
        PRE_TOKENS,
        NEXT_TOKENS,
        softmax_lse_flag,
        attentionOut,
        softmaxLseOpt);          // nullptr when flag=0

    return std::make_tuple(attentionOut, softmaxLse);
}
