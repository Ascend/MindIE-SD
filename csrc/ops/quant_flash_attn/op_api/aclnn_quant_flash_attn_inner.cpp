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

#include "aclnn_quant_flash_attn_inner.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

void QuantFlashAttnProcessSoftmaxLse(int32_t returnSoftmaxLse, const aclTensor *softmaxLse,
    const aclTensor *&tempTensor, const aclTensor *&placeHolder) {}

// sinks shape为{0}时置nullptr
void QuantFlashAttnProcessSinks(const aclTensor *&sinksOptional) {
    if (sinksOptional != nullptr) {
        const auto &shape = sinksOptional->GetViewShape();
        if (shape.GetDimNum() == 1U && shape[0] == 0) {
            OP_LOGD("sinks shape is {0}, treat as nullptr.");
            sinksOptional = nullptr;
        }
    }
}

} // namespace

#ifdef __cplusplus
}
#endif
