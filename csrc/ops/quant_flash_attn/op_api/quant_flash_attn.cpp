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

#include "quant_flash_attn.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(QuantFlashAttn);

const std::array<const aclTensor *, 2> QuantFlashAttn(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int32_t qQuantMode, int32_t kQuantMode,
    int32_t vQuantMode, int32_t quantBlockSizeQs, int32_t quantBlockSizeKs, int32_t quantBlockSizeVs,
    double softmaxScale, int32_t maskMode, int32_t winLeft, int32_t winRight, int32_t maxSeqlenQ, int32_t maxSeqlenKV,
    const char *layoutQ, const char *layoutKv, const char *layoutOut, int32_t softmaxPrecision,
    int32_t returnSoftmaxLse, aclOpExecutor *executor) {
    L0_DFX(QuantFlashAttn, q, k, v, qDescale, kDescale, vDescale, blockTableOptional, cuSeqlensQOptional,
        cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional, attnMaskOptional, metadataOptional,
        qQuantMode, kQuantMode, vQuantMode, quantBlockSizeQs, quantBlockSizeKs, quantBlockSizeVs, softmaxScale,
        maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut, softmaxPrecision,
        returnSoftmaxLse);

    if (blockTableOptional == nullptr) {
        blockTableOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensQOptional == nullptr) {
        cuSeqlensQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (cuSeqlensKvOptional == nullptr) {
        cuSeqlensKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedQOptional == nullptr) {
        sequsedQOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sequsedKvOptional == nullptr) {
        sequsedKvOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (sinksOptional == nullptr) {
        sinksOptional = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (attnMaskOptional == nullptr) {
        attnMaskOptional = executor->AllocTensor(DataType::DT_INT8, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (metadataOptional == nullptr) {
        metadataOptional = executor->AllocTensor(DataType::DT_INT32, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto attentionOutAlloc = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseAlloc = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    auto ret = INFER_SHAPE(QuantFlashAttn,
        OP_INPUT(q, k, v, qDescale, kDescale, vDescale, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional,
            sequsedQOptional, sequsedKvOptional, sinksOptional, attnMaskOptional, metadataOptional),
        OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
        OP_ATTR(qQuantMode, kQuantMode, vQuantMode, quantBlockSizeQs, quantBlockSizeKs, quantBlockSizeVs, softmaxScale,
            maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut, softmaxPrecision,
            returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantFlashAttn InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(QuantFlashAttn,
        OP_INPUT(q, k, v, qDescale, kDescale, vDescale, blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional,
            sequsedQOptional, sequsedKvOptional, sinksOptional, attnMaskOptional, metadataOptional),
        OP_OUTPUT(attentionOutAlloc, softmaxLseAlloc),
        OP_ATTR(qQuantMode, kQuantMode, vQuantMode, quantBlockSizeQs, quantBlockSizeKs, quantBlockSizeVs, softmaxScale,
            maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv, layoutOut, softmaxPrecision,
            returnSoftmaxLse));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantFlashAttn launch kernel failed.");
        return {nullptr, nullptr};
    }

    return {attentionOutAlloc, softmaxLseAlloc};
}

} // namespace l0op
