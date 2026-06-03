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

#include "aclnn_quant_flash_attn.h"

#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "aclnn_quant_flash_attn_inner.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

// 第一段接口：计算workspace大小
aclnnStatus aclnnQuantFlashAttnGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *qDescale, const aclTensor *kDescale, const aclTensor *vDescale,
    const aclTensor *blockTableOptional, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, const aclTensor *sinksOptional,
    const aclTensor *attnMaskOptional, const aclTensor *metadataOptional, int32_t qQuantMode, int32_t kQuantMode,
    int32_t vQuantMode, int32_t quantBlockSizeQs, int32_t quantBlockSizeKs, int32_t quantBlockSizeVs,
    double softmaxScale, int32_t maskMode, int32_t winLeft, int32_t winRight, int32_t maxSeqlenQ, int32_t maxSeqlenKV,
    const char *layoutQ, const char *layoutKv, const char *layoutOut, int32_t softmaxPrecision,
    int32_t returnSoftmaxLse, const aclTensor *attnOut, const aclTensor *softmaxLseOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor) {
    OP_LOGI("start aclnnQuantFlashAttnGetWorkspaceSize");
    OP_LOGI("q_quant_mode = %d", qQuantMode);
    OP_LOGI("k_quant_mode = %d", kQuantMode);
    OP_LOGI("v_quant_mode = %d", vQuantMode);
    OP_LOGI("quant_block_size_qs = %d", quantBlockSizeQs);
    OP_LOGI("quant_block_size_ks = %d", quantBlockSizeKs);
    OP_LOGI("quant_block_size_vs = %d", quantBlockSizeVs);

    // sinks shape为{0}时置nullptr
    QuantFlashAttnProcessSinks(sinksOptional);

    const aclTensor *placeHolder = nullptr;
    const aclTensor *tempTensor = nullptr;
    //todo:check and set  预留
    QuantFlashAttnProcessSoftmaxLse(returnSoftmaxLse, softmaxLseOptional, tempTensor, placeHolder);

    aclnnStatus ret = aclnnInnerQuantFlashAttnGetWorkspaceSize(q, k, v, qDescale, kDescale, vDescale,
        blockTableOptional, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, sinksOptional,
        attnMaskOptional, metadataOptional, qQuantMode, kQuantMode, vQuantMode, quantBlockSizeQs, quantBlockSizeKs,
        quantBlockSizeVs, softmaxScale, maskMode, winLeft, winRight, maxSeqlenQ, maxSeqlenKV, layoutQ, layoutKv,
        layoutOut, softmaxPrecision, returnSoftmaxLse, attnOut, placeHolder, workspaceSize, executor);

    // 销毁占位符
    if (returnSoftmaxLse == 0) {
        aclDestroyTensor(tempTensor);
    }

    return ret;
}

// 第二段接口：执行计算
aclnnStatus aclnnQuantFlashAttn(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
    return aclnnInnerQuantFlashAttn(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
