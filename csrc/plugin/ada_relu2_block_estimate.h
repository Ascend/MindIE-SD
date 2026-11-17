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

#ifndef ADA_RELU2_BLOCK_ESTIMATE_MINDIE_SD_IMPL_H
#define ADA_RELU2_BLOCK_ESTIMATE_MINDIE_SD_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> ada_relu2_block_estimate_mindie_sd_impl_npu(const at::Tensor &query,
    const at::Tensor &key, c10::OptionalIntArrayRef actualSeqLengths, c10::OptionalIntArrayRef actualSeqLengthsKV,
    std::string inputLayout, int64_t stride, int64_t sparseSize, int64_t numHeads, int64_t numKeyValueHeads,
    double scaleValue, double threshold, const c10::optional<at::Tensor> &qScale,
    const c10::optional<at::Tensor> &kScale);

#endif // ADA_RELU2_BLOCK_ESTIMATE_MINDIE_SD_IMPL_H
