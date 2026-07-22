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

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_V3_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_V3_H_
#warning \
    "aclnn_fused_infer_attention_score_v3.h is scheduled to be deprecated in December 2026, and will be replaced by the aclnn_fused_infer_attention_score_v5.h. We apologize for any inconvenience caused and appreciate your timely migration to the new interface. "
#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief The first interface of aclnnFusedInferAttentionScoreV3 calculates the workspace size based on the specific calculation process.
 * @domain aclnn_ops_infer
 */
__attribute__((deprecated(
    "aclnnFusedInferAttentionScoreV3GetWorkspaceSize is scheduled to be deprecated in December 2026, and will be "
    "replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus
aclnnFusedInferAttentionScoreV3GetWorkspaceSize(const aclTensor *query, const aclTensorList *key,
    const aclTensorList *value, const aclTensor *pseShiftOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional, const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional, const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional, const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional, const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional, const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional, const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional, const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional, const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional, const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional, const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional, int64_t numHeads, double scaleValue, int64_t preTokens,
    int64_t nextTokens, char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode, int64_t innerPrecise,
    int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag, int64_t keyAntiquantMode, int64_t valueAntiquantMode,
    const aclTensor *attentionOut, const aclTensor *softmaxLse, uint64_t *workspaceSize, aclOpExecutor **executor);

/**
 * @brief The second interface of aclnnFusedInferAttentionScoreV3 is used to perform calculations.
 */
__attribute__((deprecated(
    "aclnnFusedInferAttentionScoreV3 is scheduled to be deprecated in December 2026, and will be replaced by the "
    "aclnnFusedInferAttentionScoreV5. "
    "We apologize for any inconvenience caused and appreciate your timely migration to the new interface. ")))
__attribute__((visibility("default"))) aclnnStatus
aclnnFusedInferAttentionScoreV3(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
