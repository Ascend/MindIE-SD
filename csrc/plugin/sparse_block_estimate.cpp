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
#include <vector>
#include <cmath>
#include <algorithm>
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "sparse_block_estimate.h"

using namespace at;
namespace {
constexpr int DIM_INDEX_0 = 0;
constexpr int DIM_INDEX_1 = 1;
constexpr int DIM_INDEX_2 = 2;
constexpr int DIM_INDEX_3 = 3;
constexpr std::string_view SPARSEBLOCKESTIMATE_NAME = "aclnnSparseBlockEstimate";
}
std::tuple<at::Tensor, at::Tensor> sparse_block_estimate_mindie_sd_impl_npu(const at::Tensor &query,
    const at::Tensor &key, c10::OptionalIntArrayRef actual_seq_lengths, c10::OptionalIntArrayRef actual_seq_lengths_kv,
    std::string input_layout, int64_t stride, int64_t sparse_size, int64_t num_heads, int64_t num_key_value_heads,
    double scale_value, double threshold, bool causal, bool keep_sink, bool keep_recent, double row_sparse) {
    TORCH_CHECK(num_heads > 0, "num_heads must be positive.");
    TORCH_CHECK(num_key_value_heads > 0, "num_key_value_heads must be positive.");
    TORCH_CHECK(num_heads % num_key_value_heads == 0, "num_heads must be divisible by num_key_value_heads.");
    TORCH_CHECK(query.device() == key.device(), "query and key must be on the same device.");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key must have the same dtype.");
    TORCH_CHECK(stride > 0, "stride must be positive.");
    TORCH_CHECK(sparse_size > 0, "sparse_size must be positive.");

    auto actSeqLen = actual_seq_lengths.value_or(at::IntArrayRef{});
    auto actSeqLenKv = actual_seq_lengths_kv.value_or(at::IntArrayRef{});
    const char *inputLayoutPtr = input_layout.c_str();

    int64_t b = 0;
    int64_t nq = 0;
    int64_t s = 0;
    int64_t d = 0;

    if (input_layout == "BNSD") {
        TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "BNSD query and key must both be rank 4.");
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_1);
        s = query.size(DIM_INDEX_2);
        d = query.size(DIM_INDEX_3);
        TORCH_CHECK(nq == num_heads && key.size(DIM_INDEX_1) == num_key_value_heads,
            "BNSD tensor head dimensions must match num_heads and num_key_value_heads.");
        TORCH_CHECK(key.size(DIM_INDEX_0) == b && key.size(DIM_INDEX_3) == d,
            "BNSD query and key must have equal batch size and head dimension.");
    } else if (input_layout == "BSND") {
        TORCH_CHECK(query.dim() == 4 && key.dim() == 4, "BSND query and key must both be rank 4.");
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_2);
        s = query.size(DIM_INDEX_1);
        d = query.size(DIM_INDEX_3);
        TORCH_CHECK(nq == num_heads && key.size(DIM_INDEX_2) == num_key_value_heads,
            "BSND tensor head dimensions must match num_heads and num_key_value_heads.");
        TORCH_CHECK(key.size(DIM_INDEX_0) == b && key.size(DIM_INDEX_3) == d,
            "BSND query and key must have equal batch size and head dimension.");
    } else if (input_layout == "BSH") {
        TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "BSH query and key must both be rank 3.");
        b = query.size(DIM_INDEX_0);
        s = query.size(DIM_INDEX_1);
        TORCH_CHECK(query.size(DIM_INDEX_2) % num_heads == 0 && key.size(DIM_INDEX_2) % num_key_value_heads == 0,
            "BSH hidden dimensions must be divisible by their head counts.");
        d = query.size(DIM_INDEX_2) / num_heads;
        nq = num_heads;
        TORCH_CHECK(key.size(DIM_INDEX_0) == b && key.size(DIM_INDEX_2) / num_key_value_heads == d,
            "BSH query and key must have equal batch size and head dimension.");
    } else {
        TORCH_CHECK(false, "input_layout only supports BNSD, BSND, BSH, but got: ", input_layout);
    }
    const int64_t kvs = input_layout == "BNSD" ? key.size(DIM_INDEX_2) : key.size(DIM_INDEX_1);
    TORCH_CHECK(s > 0 && kvs > 0, "query and key sequence lengths must be positive.");
    int64_t seqlenSparse = (s + sparse_size - 1) / sparse_size;
    int64_t seqlenSparseKV = (kvs + sparse_size - 1) / sparse_size;
    int64_t seqlenSparseAlign32 = (seqlenSparseKV + 31) / 32 * 32;

    at::Tensor sparse_mask = at_npu::native::empty_with_format({b, nq, seqlenSparse, seqlenSparseAlign32},
        query.options().dtype(c10::ScalarType::Char), at_npu::native::get_npu_format(query));

    at::Tensor sparse_count_table = at_npu::native::empty_with_format(
        {b, nq, seqlenSparse}, query.options().dtype(c10::ScalarType::Int), at_npu::native::get_npu_format(query));

    EXEC_NPU_CMD<SPARSEBLOCKESTIMATE_NAME>(query, key, actSeqLen, actSeqLenKv, inputLayoutPtr, stride, sparse_size,
        num_heads, num_key_value_heads, scale_value, threshold, causal, keep_sink, keep_recent, row_sparse, sparse_mask,
        sparse_count_table);

    return std::tuple<at::Tensor, at::Tensor>(sparse_mask, sparse_count_table);
}
