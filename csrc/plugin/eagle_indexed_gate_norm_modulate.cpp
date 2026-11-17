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
#include "eagle_indexed_gate_norm_modulate.h"

#include <ATen/ATen.h>
#include <cmath>
#include <string_view>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"

namespace {
constexpr std::string_view ACLNN_OP = "aclnnEagleIndexedGateNormModulate";
constexpr const char *OP = "eagle_indexed_gate_norm_modulate: ";
} // namespace

std::tuple<at::Tensor, at::Tensor> eagle_indexed_gate_norm_modulate_impl_npu(const at::Tensor &residual,
    const at::Tensor &gate, const at::Tensor &branch, const at::Tensor &shift, const at::Tensor &scale,
    const at::Tensor &indices, const c10::optional<at::Tensor> &weightOptional, const c10::optional<at::Tensor> &bias,
    int64_t norm_type, double eps, const c10::optional<at::ScalarType> &output_dtype) {
    TORCH_CHECK(norm_type == 0, OP, "only norm_type=0 (weighted RMSNorm) is supported");
    TORCH_CHECK(!bias.has_value(), OP, "bias must be None for norm_type=0 (weighted RMSNorm)");
    TORCH_CHECK(weightOptional.has_value() && weightOptional->defined(), OP,
        "weight is required for norm_type=0 (weighted RMSNorm)");
    const auto &weight = weightOptional.value();
    TORCH_CHECK(residual.dim() == 2, OP, "residual must be [N,D]");
    TORCH_CHECK(residual.device().type() == c10::DeviceType::PrivateUse1, OP, "inputs must be on NPU");
    const auto inputType = residual.scalar_type();
    TORCH_CHECK(inputType == at::kHalf || inputType == at::kBFloat16, OP, "input dtype must be FP16 or BF16");
    const auto outputType = output_dtype.value_or(inputType);
    TORCH_CHECK(outputType == at::kHalf || outputType == at::kBFloat16, OP, "output dtype must be FP16 or BF16");
    const auto hidden = residual.size(1);
    TORCH_CHECK(hidden >= 1 && hidden <= 7168, OP, "hidden_size must be in [1,7168]");
    TORCH_CHECK(std::isfinite(eps) && std::isfinite(static_cast<float>(eps)) && static_cast<float>(eps) > 0, OP,
        "eps must be finite and positive in FP32");
    const at::Tensor *floatInputs[] = {&residual, &gate, &branch, &weight, &shift, &scale};
    for (const auto *input : floatInputs) {
        TORCH_CHECK(input->device() == residual.device(), OP, "all inputs must be on the same NPU device");
        TORCH_CHECK(input->scalar_type() == inputType, OP, "all six floating inputs must have the same dtype");
        TORCH_CHECK(input->layout() == c10::kStrided, OP, "floating inputs must have strided layout");
    }
    TORCH_CHECK(branch.sizes() == residual.sizes(), OP, "branch must match residual [N,D]");
    TORCH_CHECK(gate.dim() == 2 && gate.size(0) >= 1 && gate.size(1) == hidden, OP, "gate must be [K,D], K >= 1");
    TORCH_CHECK(shift.sizes() == gate.sizes() && scale.sizes() == gate.sizes(), OP, "shift and scale must match gate");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden, OP, "weight must be [D]");
    TORCH_CHECK(indices.dim() == 1 && indices.size(0) == residual.size(0), OP, "indices must be [N]");
    TORCH_CHECK(indices.device() == residual.device() && indices.layout() == c10::kStrided, OP,
        "indices must be strided and on the same NPU device");
    TORCH_CHECK(
        indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong, OP, "indices must be int32 or int64");

    auto options = residual.options().dtype(outputType);
    auto residualOut = at_npu::native::empty_with_format(residual.sizes(), options, ACL_FORMAT_ND);
    auto modulatedOut = at_npu::native::empty_with_format(residual.sizes(), options, ACL_FORMAT_ND);
    if (residual.size(0) == 0) {
        return {residualOut, modulatedOut};
    }
    // The public API accepts strided tensors; the AscendC kernel consumes contiguous ND.
    auto residualContiguous = residual.contiguous();
    auto gateContiguous = gate.contiguous();
    auto branchContiguous = branch.contiguous();
    auto weightContiguous = weight.contiguous();
    auto shiftContiguous = shift.contiguous();
    auto scaleContiguous = scale.contiguous();
    auto indicesContiguous = indices.contiguous();
    const int64_t outputDtype = outputType == at::kHalf ? 0 : 1;
    EXEC_NPU_CMD<ACLNN_OP>(residualContiguous, gateContiguous, branchContiguous, weightContiguous, shiftContiguous,
        scaleContiguous, indicesContiguous, eps, outputDtype, residualOut, modulatedOut);
    return {residualOut, modulatedOut};
}
