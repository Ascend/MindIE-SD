/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
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
#include "norm_rope_concat.h"

using namespace at;

constexpr std::string_view NORM_ROPE_CONCAT_NAME = "aclnnNormRopeConcat";

std::tuple<at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor,
           at::Tensor, at::Tensor, at::Tensor, at::Tensor>
norm_rope_concat_mindie_sd_impl_npu(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &encoder_query,
    const c10::optional<at::Tensor> &encoder_key,
    const c10::optional<at::Tensor> &encoder_value,
    const c10::optional<at::Tensor> &norm_query_weight,
    const c10::optional<at::Tensor> &norm_query_bias,
    const c10::optional<at::Tensor> &norm_key_weight,
    const c10::optional<at::Tensor> &norm_key_bias,
    const c10::optional<at::Tensor> &norm_added_query_weight,
    const c10::optional<at::Tensor> &norm_added_query_bias,
    const c10::optional<at::Tensor> &norm_added_key_weight,
    const c10::optional<at::Tensor> &norm_added_key_bias,
    const c10::optional<at::Tensor> &rope_sin,
    const c10::optional<at::Tensor> &rope_cos,
    int64_t norm_type,
    int64_t norm_added_type,
    int64_t rope_type,
    int64_t concat_order,
    double eps,
    bool is_training)
{
    TORCH_CHECK(query.dim() == 4, "query must be 4D [B,S,N,D], got ", query.dim(), "D");
    TORCH_CHECK(key.dim() == 4, "key must be 4D [B,S,N,D], got ", key.dim(), "D");
    TORCH_CHECK(value.dim() == 4, "value must be 4D [B,S,N,D], got ", value.dim(), "D");

    int64_t B = query.size(0);
    int64_t Sq = query.size(1);
    int64_t N = query.size(2);
    int64_t D = query.size(3);
    int64_t Sk = key.size(1);
    int64_t Sv = value.size(1);

    int64_t Seq = (encoder_query.has_value() && encoder_query.value().defined())
                      ? encoder_query.value().size(1)
                      : 0;
    int64_t Sek = (encoder_key.has_value() && encoder_key.value().defined())
                      ? encoder_key.value().size(1)
                      : 0;
    int64_t Sev = (encoder_value.has_value() && encoder_value.value().defined())
                      ? encoder_value.value().size(1)
                      : 0;

    c10::SmallVector<int64_t, 4> query_output_size = {B, N, Sq + Seq, D};
    c10::SmallVector<int64_t, 4> key_output_size = {B, N, Sk + Sek, D};
    c10::SmallVector<int64_t, 4> value_output_size = {B, N, Sv + Sev, D};

    at::Tensor query_output = at_npu::native::empty_with_format(
        query_output_size, query.options(), at_npu::native::get_npu_format(query));
    at::Tensor key_output = at_npu::native::empty_with_format(
        key_output_size, key.options(), at_npu::native::get_npu_format(key));
    at::Tensor value_output = at_npu::native::empty_with_format(
        value_output_size, value.options(), at_npu::native::get_npu_format(value));

    auto float_options = query.options().dtype(at::kFloat);

    c10::SmallVector<int64_t, 4> q_mean_size = {B, Sq, N, 1};
    c10::SmallVector<int64_t, 4> k_mean_size = {B, Sk, N, 1};
    c10::SmallVector<int64_t, 4> aq_mean_size =
        Seq > 0 ? c10::SmallVector<int64_t, 4>{B, Seq, N, 1} : c10::SmallVector<int64_t, 4>{1};
    c10::SmallVector<int64_t, 4> ak_mean_size =
        Sek > 0 ? c10::SmallVector<int64_t, 4>{B, Sek, N, 1} : c10::SmallVector<int64_t, 4>{1};

    at::Tensor norm_query_mean =
        at_npu::native::empty_with_format(q_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_query_rstd =
        at_npu::native::empty_with_format(q_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_key_mean =
        at_npu::native::empty_with_format(k_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_key_rstd =
        at_npu::native::empty_with_format(k_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_added_query_mean =
        at_npu::native::empty_with_format(aq_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_added_query_rstd =
        at_npu::native::empty_with_format(aq_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_added_key_mean =
        at_npu::native::empty_with_format(ak_mean_size, float_options, ACL_FORMAT_ND);
    at::Tensor norm_added_key_rstd =
        at_npu::native::empty_with_format(ak_mean_size, float_options, ACL_FORMAT_ND);

    EXEC_NPU_CMD<NORM_ROPE_CONCAT_NAME>(query, key, value,
        encoder_query, encoder_key, encoder_value,
        norm_query_weight, norm_query_bias,
        norm_key_weight, norm_key_bias,
        norm_added_query_weight, norm_added_query_bias,
        norm_added_key_weight, norm_added_key_bias,
        rope_sin, rope_cos,
        norm_type, norm_added_type, rope_type,
        concat_order, eps, is_training,
        query_output, key_output, value_output,
        norm_query_mean, norm_query_rstd,
        norm_key_mean, norm_key_rstd,
        norm_added_query_mean, norm_added_query_rstd,
        norm_added_key_mean, norm_added_key_rstd);

    return std::make_tuple(query_output, key_output, value_output,
                           norm_query_mean, norm_query_rstd,
                           norm_key_mean, norm_key_rstd,
                           norm_added_query_mean, norm_added_query_rstd,
                           norm_added_key_mean, norm_added_key_rstd);
}

