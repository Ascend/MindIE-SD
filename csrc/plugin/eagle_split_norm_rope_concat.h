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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Optional.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor> eagle_split_norm_rope_concat_impl_npu(const at::Tensor &qkv,
    const c10::optional<at::Tensor> &norm_query_weight, const c10::optional<at::Tensor> &norm_key_weight,
    const at::Tensor &rope_cos, const at::Tensor &rope_sin, at::IntArrayRef head_counts, int64_t norm_type,
    int64_t rope_type, c10::optional<int64_t> rotary_dim, double eps, bool output_bnsd);

#endif // EAGLE_SPLIT_NORM_ROPE_CONCAT_MINDIE_SD_IMPL_H
