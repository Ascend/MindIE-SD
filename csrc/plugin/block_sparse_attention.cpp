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
#include "block_sparse_attention.h"

using namespace at;

at::Tensor block_sparse_attention_impl_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const at::Tensor &sparse_mask,
    const at::Tensor &sparse_count_table,
    std::string input_layout, int64_t sparse_size, int64_t num_heads,
    int64_t num_key_value_heads, double scale_value, bool causal,
    int64_t inner_precise, int64_t pre_tokens, int64_t next_tokens,
    c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv)
{
    TORCH_CHECK(input_layout != "TND", "input_layout currently does not support 'TND'.");
    at::Tensor attention_out =
        at_npu::native::empty_with_format(query.sizes(), query.options(),
        at_npu::native::get_npu_format(query));

    int64_t sparseMode = 0;

    at_npu::native::OpCommand cmd;
    cmd.Name("BlockSparseAttention")
            .Input(query, "query")
            .Input(key, "key")
            .Input(value, "value")
            .Input().Input().Input()
            .Input().Input().Input()
            .Input().Input().Input()
            .Input(sparse_mask, "sparse_mask")
            .Input(sparse_count_table, "sparse_count_table")
            .Output(attention_out, "attention_out")
            .Attr("num_heads", num_heads)
            .Attr("scale_value", static_cast<float>(scale_value))
            .Attr("pre_tokens", pre_tokens)
            .Attr("next_tokens", next_tokens)
            .Attr("input_layout", input_layout)
            .Attr("num_key_value_heads", num_key_value_heads)
            .Attr("sparse_mode", sparseMode)
            .Attr("inner_precise", inner_precise)
            .Attr("sparse_size", sparse_size)
            .Attr("causal", causal)
            .Run();

    return attention_out;
}