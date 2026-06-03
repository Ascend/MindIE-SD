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

#ifndef ACLNN_QUANT_FLASH_ATTN_INNER_H_
#define ACLNN_QUANT_FLASH_ATTN_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

extern aclnnStatus aclnnInnerQuantFlashAttnGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int32_t qQuantMode, int32_t kQuantMode,
    int32_t vQuantMode, int32_t quantBlockSizeQs, int32_t quantBlockSizeKs, int32_t quantBlockSizeVs,
    double softmaxScale, int32_t maskMode, int32_t winLeft, int32_t winRight, int32_t maxSeqlenQ, int32_t maxSeqlenKV,
    const char *layoutQ, const char *layoutKv, const char *layoutOut, int32_t softmaxPrecision,
    int32_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLse, uint64_t *workspaceSize,
    aclOpExecutor **executor);

extern aclnnStatus aclnnInnerQuantFlashAttn(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream);

void QuantFlashAttnProcessSoftmaxLse(
    int32_t returnSoftmaxLse, const aclTensor *softmaxLse, const aclTensor *&tempTensor, const aclTensor *&placeHolder);

void QuantFlashAttnProcessSinks(const aclTensor *&sinksOptional);

#ifdef __cplusplus
}
#endif

#endif
