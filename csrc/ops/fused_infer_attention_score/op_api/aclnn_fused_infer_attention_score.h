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

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_H_
#warning \
    "aclnn_fused_infer_attention_score.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_fused_infer_attention_score_v5.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFusedInferAttentionScore calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((deprecated(
    "aclnnFusedInferAttentionScoreGetWorkspaceSize is scheduled to be deprecated in December 2026, and will be "
    "replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus
aclnnFusedInferAttentionScoreGetWorkspaceSize(const aclTensor *query, const aclTensorList *key,
    const aclTensorList *value, const aclTensor *pseShift, const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths, const aclIntArray *actualSeqLengthsKv, const aclTensor *deqScale1,
    const aclTensor *quantScale1, const aclTensor *deqScale2, const aclTensor *quantScale2,
    const aclTensor *quantOffset2, const aclTensor *antiquantScale, const aclTensor *antiquantOffset,
    const aclTensor *blockTable, const aclTensor *queryPaddingSize, const aclTensor *kvPaddingSize, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFusedInferAttentionScore is used to perform calculations.
 */
__attribute__((deprecated(
    "aclnnFusedInferAttentionScore is scheduled to be deprecated in December 2026, and will be replaced by the "
    "aclnnFusedInferAttentionScoreV5. "
    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus
aclnnFusedInferAttentionScore(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
