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

#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/platform.h"
#include "aclnn_fused_infer_attention_score.h"
#include "fused_infer_attention_score_inner.h"
#include "aclnnInner_fused_infer_attention_score.h" // 该文件为自动生成，在build/autogen/inner路径下

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(const aclTensor *query, const aclTensorList *key,
    const aclTensorList *value, const aclTensor *pseShift, const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
            "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnFusedInferAttentionScoreGetWorkspaceSize is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }

    aclnnStatus ret = CheckKVContiguous(key, value);
    if (ret != ACLNN_SUCCESS && NnopbaseSupportTensorV2 == nullptr) {
        return ret;
    }

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);

    ret = aclnnInnerFusedInferAttentionScoreGetWorkspaceSize(query, key, value, pseShift, attenMask, actualSeqLengths,
        actualSeqLengthsKv, deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale,
        antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue,
        preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode,
        softmaxLseFlag, 0, 0, 0, 0, 0, attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
            "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW(
            "aclnnFusedInferAttentionScore is scheduled to be deprecated in December 2026, "
            "and will be replaced by the aclnnFusedInferAttentionScoreV5. "
            "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ");
        isFirstCall = false;
    }
    return aclnnInnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
