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

#include <c10/core/DeviceGuard.h>
#include "pytorch_npu_helper.h"
#include "sigmoid_mul_dynamic_mx_quant.h"

namespace {
constexpr std::string_view OP_NAME = "aclnnSigmoidMulDynamicMxQuant";
}

std::tuple<at::Tensor, at::Tensor> sigmoid_mul_dynamic_mx_quant_impl_npu(
    const at::Tensor &x, const at::Tensor &gate, int64_t precisionMode, int64_t scaleAlg) {
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1 && gate.device() == x.device(),
        "x and gate must be on the same NPU");
    TORCH_CHECK(x.dim() == 2 && x.sizes() == gate.sizes(), "expected equal [rows, cols] shapes");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && gate.scalar_type() == at::kBFloat16, "x and gate must be BF16");
    TORCH_CHECK(precisionMode == 0 || precisionMode == 1, "precision_mode must be 0 or 1");
    TORCH_CHECK(scaleAlg == 0 || scaleAlg == 1, "scale_alg must be 0 or 1");
    const c10::DeviceGuard guard(x.device());
    const int64_t cols = x.size(1);
    auto y = at::empty(x.sizes(), x.options().dtype(at::ScalarType::Float8_e4m3fn));
    // torch_npu exposes E8M0 scales as one-byte unsigned storage.
    auto scale = at::empty({x.size(0), cols / 64 + (cols % 64 != 0), 2}, x.options().dtype(at::kByte));
    if (x.numel() > 0) {
        auto xContiguous = x.contiguous();
        auto gateContiguous = gate.contiguous();
        TensorWrapper dataOutput{y, ACL_DTYPE_FLOAT8_E4M3FN};
        TensorWrapper scaleOutput{scale, ACL_DTYPE_FLOAT8_E8M0};
        EXEC_NPU_CMD<OP_NAME>(xContiguous, gateContiguous, precisionMode, scaleAlg, dataOutput, scaleOutput);
    }
    return {y, scale};
}
