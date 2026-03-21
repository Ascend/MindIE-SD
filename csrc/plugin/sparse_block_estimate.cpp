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
#include "sparse_block_estimate.h"

using namespace at;
namespace {
constexpr int DIM_INDEX_0 = 0;
constexpr int DIM_INDEX_1 = 1;
constexpr int DIM_INDEX_2 = 2;
constexpr int DIM_INDEX_3 = 3;
}
std::tuple<at::Tensor, at::Tensor> sparse_block_estimate_mindie_sd_impl_npu(
    const at::Tensor &query, const at::Tensor &key,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv,
    std::string input_layout, int64_t stride, int64_t sparse_size,
    int64_t num_heads, int64_t num_key_value_heads, double scale_value,
    double threshold, bool causal, bool keep_sink, bool keep_recent, double row_sparse)
{
    TORCH_CHECK(num_heads != 0, "num_heads must be nonzero.");
    TORCH_CHECK(sparse_size != 0, "sparse_size must be nonzero.");
    int64_t b;
    int64_t nq;
    int64_t s;
    int64_t d;

    if (input_layout == "BNSD") {
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_1);
        s = query.size(DIM_INDEX_2);
        d = query.size(DIM_INDEX_3);
    } else if (input_layout == "BSND") {
        b = query.size(DIM_INDEX_0);
        nq = query.size(DIM_INDEX_2);
        s = query.size(DIM_INDEX_1);
        d = query.size(DIM_INDEX_3);
    } else if (input_layout == "BSH") {
        b = query.size(DIM_INDEX_0);
        s = query.size(DIM_INDEX_1);
        d = query.size(DIM_INDEX_2) / num_heads;
        nq = num_heads;
    } else {
        std::cerr << "Error: input_layout only support BNSD, BSND, BSH!!!" << std::endl;
    }
    int64_t seqlenSparse = (s + sparse_size - 1) / sparse_size;
    int64_t seqlenSparseAlign32 = (seqlenSparse + 31) / 32 * 32;

    at::Tensor sparse_mask =
        at_npu::native::empty_with_format({b, nq, seqlenSparse, seqlenSparseAlign32},
        query.options().dtype(c10::ScalarType::Char), at_npu::native::get_npu_format(query));

    at::Tensor sparse_count_table =
        at_npu::native::empty_with_format({b, nq, seqlenSparse}, query.options().dtype(c10::ScalarType::Int),
        at_npu::native::get_npu_format(query));

    at_npu::native::OpCommand cmd;

    cmd.Name("SparseBlockEstimate")
            .Input(query, "query")
            .Input(key, "key")
            .Input().Input()
            .Output(sparse_mask, "sparse_mask")
            .Output(sparse_count_table, "sparse_count_table")
            .Attr("input_layout", input_layout)
            .Attr("stride", stride)
            .Attr("sparse_size", sparse_size)
            .Attr("num_heads", num_heads)
            .Attr("num_key_value_heads", num_key_value_heads)
            .Attr("scale_value", static_cast<float>(scale_value))
            .Attr("threshold", static_cast<float>(threshold))
            .Attr("causal", causal)
            .Attr("keep_sink", keep_sink)
            .Attr("keep_recent", keep_recent)
            .Attr("row_sparse", static_cast<float>(row_sparse))
            .Run();

    return std::tuple<at::Tensor, at::Tensor>(sparse_mask, sparse_count_table);
}