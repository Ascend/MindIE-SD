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

#pragma once

#include <ATen/Tensor.h>
#include <cstdint>
#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> eagle_hadamard_quant_impl_npu(
    const at::Tensor &x, const at::Tensor &sign_h, int64_t block_size, const std::string &dst_type, bool is_sign_h_nz);

std::tuple<at::Tensor, at::Tensor> eagle_hadamard_quant_impl_meta(
    const at::Tensor &x, const at::Tensor &sign_h, int64_t block_size, const std::string &dst_type, bool is_sign_h_nz);
