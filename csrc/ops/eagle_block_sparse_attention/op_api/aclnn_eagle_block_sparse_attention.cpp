/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <acl/acl.h>
#include "opdev/common_types.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "eagle_block_sparse_attention.h"
#include "aclnn_eagle_block_sparse_attention.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static constexpr uint64_t LSE_OUT = 1;

static bool CheckDataType(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    const DataType qDtype = query->GetDataType();
    const DataType kDtype = key->GetDataType();
    const DataType vDtype = value->GetDataType();

    static const std::unordered_map<DataType, std::vector<DataType>> validKvType = {
        {DataType::DT_FLOAT16, {DataType::DT_FLOAT16}},
        {DataType::DT_BF16, {DataType::DT_BF16}},
    };

    auto iter = validKvType.find(qDtype);
    if (iter == validKvType.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Unsupported query datatype %d.", static_cast<int>(qDtype));
        return false;
    }

    if (std::find(iter->second.begin(), iter->second.end(), kDtype) == iter->second.end() ||
        std::find(iter->second.begin(), iter->second.end(), vDtype) == iter->second.end()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Key/Value datatype mismatch with query.");
        return false;
    }

    return true;
}

static aclnnStatus CheckMandatoryTensors(const aclTensor *query, const aclTensor *key, const aclTensor *value)
{
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParseblockShapeOptional(const aclIntArray *blockShapeOptional)
{
    if (blockShapeOptional != nullptr) {
        uint64_t size = blockShapeOptional->Size();
        if (size != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional must contain two elements [x, y].");
            return ACLNN_ERR_PARAM_INVALID;
        }

        const int64_t *data = blockShapeOptional->GetData();
        if (data == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional data is null.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (data[0] <= 0 || data[1] <= 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockShapeOptional values must be positive, got [%ld, %ld].", data[0],
                    data[1]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateParams(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                  char *qInputLayout, char *kvInputLayout, const aclIntArray *blockShapeOptional)
{
    CHECK_RET(CheckMandatoryTensors(query, key, value) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    if (!CheckDataType(query, key, value)) {
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (qInputLayout == nullptr || kvInputLayout == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Input layout strings are null.");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    std::string qLayout(qInputLayout);
    std::string kvLayout(kvInputLayout);

    // 验证Q layout
    if (qLayout != "TND" && qLayout != "BNSD" && qLayout != "BSND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qInputLayout only supports TND, BNSD or BSND, got %s.", qLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证KV layout
    if (kvLayout != "TND" && kvLayout != "BNSD" && qLayout != "BSND") {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kvInputLayout only supports TND, BNSD or BSND, got %s.", kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 验证Q和KV格式一致性：两者必须相同
    if (qLayout != kvLayout) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The parameters qInputLayout and kvInputLayout must be consistent, but currently qInputLayout is %s "
                "and kvInputLayout is %s.",
                qLayout.c_str(), kvLayout.c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ParseblockShapeOptional(blockShapeOptional);
}

static aclnnStatus MakeContiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                  const aclTensor *&blockSparseMaskOptional, const aclTensor *&attenMaskOptional,
                                  const aclTensor *&blockTableOptional, aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // 新增blockSparseMaskOptional非空校验且必须为四维
    if (blockSparseMaskOptional != nullptr) {
        blockSparseMaskOptional = l0op::Contiguous(blockSparseMaskOptional, executor);
        CHECK_RET(blockSparseMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (blockSparseMaskOptional->GetStorageShape().GetDimNum() != 4) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blockSparseMask must be 4D tensor");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    if (attenMaskOptional != nullptr) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    if (blockTableOptional != nullptr) {
        blockTableOptional = l0op::Contiguous(blockTableOptional, executor);
        CHECK_RET(blockTableOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus ValidateAdditionalParams(int64_t innerPrecise, const aclTensor *attentionOut,
                                            uint64_t *workspaceSize, aclOpExecutor **executor)
{
    if (innerPrecise != 0 && innerPrecise != 1 && innerPrecise != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "innerPrecise must be 0 or 1 or 4, got %ld.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    return ACLNN_SUCCESS;
}

static string ConvertLayoutString(char *layoutStr) { return op::ToString(layoutStr).GetString(); }

} // namespace

__attribute__((visibility("default"))) aclnnStatus aclnnEagleBlockSparseAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *blockSparseMask,
    const aclTensor *attenMaskOptional, const aclIntArray *blockShape, const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional, const aclTensor *blockTableOptional, char *qInputLayout,
    char *kvInputLayout, int64_t numKeyValueHeads, int64_t maskType, double scaleValue, int64_t innerPrecise,
    int64_t blockSize, int64_t preTokens, int64_t nextTokens, int64_t softmaxLseFlag, aclTensor *attentionOut,
    aclTensor *softmaxLseOptional, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    aclnnStatus ret = ValidateParams(query, key, value, qInputLayout, kvInputLayout, blockShape);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = ValidateAdditionalParams(innerPrecise, attentionOut, workspaceSize, executor);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    // 去掉了idx ,idxnums
    L2_DFX_PHASE_1(aclnnEagleBlockSparseAttention,
                   DFX_IN(query, key, value, blockSparseMask, attenMaskOptional, blockShape, actualSeqLengthsOptional,
                          actualSeqLengthsKvOptional, blockTableOptional, qInputLayout, kvInputLayout, numKeyValueHeads,
                          maskType, scaleValue, innerPrecise, blockSize, preTokens, nextTokens, softmaxLseFlag),
                   DFX_OUT(attentionOut, softmaxLseOptional));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto *executorImpl = uniqueExecutor.get();
    // 新增blockSparseMaskOptional参数
    ret = MakeContiguous(query, key, value, blockSparseMask, attenMaskOptional, blockTableOptional, executorImpl);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    string qInputLayoutStr = ConvertLayoutString(qInputLayout);
    string kvInputLayoutStr = ConvertLayoutString(kvInputLayout);
    // 新增blockSparseMaskOptional参数
    auto outputs = l0op::EagleBlockSparseAttention(
        query, key, value, blockSparseMask, attenMaskOptional, blockShape, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, blockTableOptional, nullptr, nullptr, nullptr, nullptr, qInputLayoutStr.c_str(),
        kvInputLayoutStr.c_str(), numKeyValueHeads, maskType, scaleValue, innerPrecise, blockSize, preTokens,
        nextTokens, softmaxLseFlag, 0, 0.0, attentionOut, executorImpl);
    if (outputs[0] == nullptr || outputs[1] == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "EagleBlockSparseAttention returned nullptr outputs.");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewCopyResult = l0op::ViewCopy(outputs[0], attentionOut, executorImpl);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (softmaxLseFlag == LSE_OUT) {
        auto viewCopyLseResult = l0op::ViewCopy(outputs[1], softmaxLseOptional, executorImpl);
        CHECK_RET(viewCopyLseResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = executorImpl->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnEagleBlockSparseAttention(void *workspace, uint64_t workspaceSize,
                                                                             aclOpExecutor *executor,
                                                                             aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnEagleBlockSparseAttention);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
