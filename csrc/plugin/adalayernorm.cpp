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
#include "adalayernorm.h"

using namespace at;

constexpr std::string_view ADALAYER_NORM_OP_NAME = "aclnnAdaLayerNorm";
at::Tensor adaln_mindie_sd_impl_npu(
    const at::Tensor &x,
    const at::Tensor &scale,
    const at::Tensor &shift,
    const c10::optional<at::Tensor> &weight_opt,
    const c10::optional<at::Tensor> &bias_opt,
    const c10::optional<double> &epsilon_opt)
{
    double epsilon = epsilon_opt.value_or(1e-5);
    const at::Tensor& weight = c10::value_or_else(weight_opt, [] {return at::Tensor();});
    const at::Tensor& bias = c10::value_or_else(bias_opt, [] {return at::Tensor();});

    at::Tensor output = at_npu::native::empty_with_format(
        x.sizes(),
        x.options(),
        at_npu::native::get_npu_format(x)
    );

    EXEC_NPU_CMD<ADALAYER_NORM_OP_NAME>(x, scale, shift, weight, bias, epsilon, output);
    return output;
}