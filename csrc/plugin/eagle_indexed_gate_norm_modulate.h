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
#ifndef EAGLE_INDEXED_GATE_NORM_MODULATE_IMPL_H
#define EAGLE_INDEXED_GATE_NORM_MODULATE_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor> eagle_indexed_gate_norm_modulate_impl_npu(const at::Tensor &residual,
    const at::Tensor &gate, const at::Tensor &branch, const at::Tensor &shift, const at::Tensor &scale,
    const at::Tensor &indices, const c10::optional<at::Tensor> &weight, const c10::optional<at::Tensor> &bias,
    int64_t norm_type, double eps, const c10::optional<at::ScalarType> &output_dtype);
#endif
