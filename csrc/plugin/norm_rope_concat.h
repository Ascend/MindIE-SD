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

#ifndef NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H
#define NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <tuple>

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
    bool is_training);

#endif // NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H

