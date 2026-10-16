/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <limits>
#include <string_view>
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "pytorch_npu_helper.h"
#include "frequency_regulator.h"

constexpr std::string_view FREQUENCY_REGULATOR_OP_NAME = "aclnnFrequencyRegulator";

at::Tensor frequency_regulator_impl_npu(int64_t freq)
{
    TORCH_CHECK(freq >= 0 && freq <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
        "freq must be in range [0, UINT32_MAX], but got ", freq);

    auto options = at::TensorOptions(torch_npu::utils::get_npu_device_type())
                       .dtype(c10::ScalarType::Int);
    at::Tensor output =
        at_npu::native::empty_with_format({1}, options, ACL_FORMAT_ND);
    auto output_wrapper = TensorWrapper{output, ACL_UINT32};
    uint32_t frequency = static_cast<uint32_t>(freq);

    EXEC_NPU_CMD<FREQUENCY_REGULATOR_OP_NAME>(frequency, output_wrapper);
    return output.to(c10::ScalarType::Long);
}
