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
#include "la.h"

using namespace at;

std::tuple<at::Tensor, at::Tensor> la_mindie_sd_impl_npu(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask_opt,
    const c10::optional<at::Tensor> &alibi_mask_opt,
    const c10::optional<at::Tensor> &drop_mask_opt,
    double scale_value, int64_t head_num, std::string input_layout,
    double keep_prob, int64_t pre_tokens, int64_t next_tokens, bool is_highPrecision)
{
    size_t query_dim = query.sizes().size();
    if (query_dim != 4) { // 4 is the first input dimension
        throw std::invalid_argument("The first input dimension of la must be 4 but got " + str(query_dim));
    }
    const at::Tensor& atten_mask = c10::value_or_else(atten_mask_opt, [] {return at::Tensor();});
    const at::Tensor& alibi_mask = c10::value_or_else(alibi_mask_opt, [] {return at::Tensor();});
    const at::Tensor& drop_mask = c10::value_or_else(drop_mask_opt, [] {return at::Tensor();});

    at::Tensor softmax_log_max_sum =
        at_npu::native::empty_with_format({query.sizes()[0], query.sizes()[1], query.sizes()[2]},
        query.options(), at_npu::native::get_npu_format(query));

    at::Tensor attention_out =
        at_npu::native::empty_with_format(query.sizes(), query.options(),
        at_npu::native::get_npu_format(query));

    at_npu::native::OpCommand cmd;

    cmd.Name("AscendLaserAttention")
            .Input(query, "query")
            .Input(key, "key")
            .Input(value, "value");

    if (atten_mask.defined()) {
        cmd.Input(atten_mask, "atten_mask");
    }

    if (alibi_mask.defined()) {
        cmd.Input(alibi_mask, "alibi_mask");
    }

    if (drop_mask.defined()) {
        cmd.Input(drop_mask, "drop_mask");
    }

    cmd.Output(softmax_log_max_sum, "softmax_log_max_sum")
    .Output(attention_out, "attention_out")
    .Attr("scale_value", static_cast<float>(scale_value))
    .Attr("head_num", head_num)
    .Attr("input_layout", input_layout)
    .Attr("keep_prob", static_cast<float>(keep_prob))
    .Attr("pre_tokens", pre_tokens)
    .Attr("next_tokens", next_tokens)
    .Attr("is_highPrecision", is_highPrecision)
    .Run();

    return std::tuple<at::Tensor, at::Tensor>(softmax_log_max_sum, attention_out);
}
