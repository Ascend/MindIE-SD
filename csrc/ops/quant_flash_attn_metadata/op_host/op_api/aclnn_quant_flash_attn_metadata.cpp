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

/*!
 * \file aclnn_quant_flash_attn_metadata.cpp
 * \brief
 */

#include "aclnn_quant_flash_attn_metadata.h"
#include "l0_quant_flash_attn_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t qQuantMode, int64_t kQuantMode,
    int64_t vQuantMode, int64_t qDtype, int64_t kDtype, int64_t vDtype, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char *layoutQ, const char *layoutKv, const char *layoutOut, const aclTensor *metaData) {
    return ACLNN_SUCCESS;
    if (maskMode != 1 && maskMode != 3 && maskMode != 4) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maskMode only supports 1, 3, 4, but got %ld", maskMode);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (winLeft < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "winLeft must be >= -1, but got %ld", winLeft);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (winRight < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "winRight must be >= -1, but got %ld", winRight);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxSeqlenQ < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenQ must be >= -1, but got %ld", maxSeqlenQ);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (maxSeqlenKv < -1) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "maxSeqlenKv must be >= -1, but got %ld", maxSeqlenKv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutQValid = (strcmp(layoutQ, "BSND") == 0 || strcmp(layoutQ, "TND") == 0 || strcmp(layoutQ, "BNSD") == 0);
    if (!layoutQValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutQ only supports BSND, TND, BNSD, but got %s", layoutQ);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutOutValid =
        (strcmp(layoutOut, "BSND") == 0 || strcmp(layoutOut, "TND") == 0 || strcmp(layoutOut, "BNSD") == 0);
    if (!layoutOutValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutOut only supports BSND, TND, BNSD, but got %s", layoutOut);
        return ACLNN_ERR_PARAM_INVALID;
    }

    bool layoutKvValid =
        (strcmp(layoutKv, "BSND") == 0 || strcmp(layoutKv, "TND") == 0 || strcmp(layoutKv, "BNSD") == 0 ||
            strcmp(layoutKv, "PA_BNBD") == 0 || strcmp(layoutKv, "PA_BBND") == 0 || strcmp(layoutKv, "PA_NZ") == 0);
    if (!layoutKvValid) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "layoutKv only supports BSND, TND, BNSD, PA_BNBD, PA_BBND, PA_NZ, but got %s",
            layoutKv);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (batchSize != 0) {
        if (cuSeqlensQOptional != nullptr) {
            if (cuSeqlensQOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional must be 1D tensor, but got %ld dims",
                    cuSeqlensQOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (cuSeqlensQOptional->GetViewShape().GetDim(0) != batchSize + 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensQOptional shape must be (batchSize+1,), but got %ld",
                    cuSeqlensQOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (cuSeqlensKvOptional != nullptr) {
            if (cuSeqlensKvOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensKvOptional must be 1D tensor, but got %ld dims",
                    cuSeqlensKvOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (cuSeqlensKvOptional->GetViewShape().GetDim(0) != batchSize + 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "cuSeqlensKvOptional shape must be (batchSize+1,), but got %ld",
                    cuSeqlensKvOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (sequsedQOptional != nullptr) {
            if (sequsedQOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional must be 1D tensor, but got %ld dims",
                    sequsedQOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (sequsedQOptional->GetViewShape().GetDim(0) != batchSize) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedQOptional shape must be (batchSize,), but got %ld",
                    sequsedQOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }

        if (sequsedKvOptional != nullptr) {
            if (sequsedKvOptional->GetViewShape().GetDimNum() != 1) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional must be 1D tensor, but got %ld dims",
                    sequsedKvOptional->GetViewShape().GetDimNum());
                return ACLNN_ERR_PARAM_INVALID;
            }
            if (sequsedKvOptional->GetViewShape().GetDim(0) != batchSize) {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "sequsedKvOptional shape must be (batchSize,), but got %ld",
                    sequsedKvOptional->GetViewShape().GetDim(0));
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }

    if (numHeadsKv != 0) {
        if (numHeadsQ % numHeadsKv != 0) {
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "numHeadsQ must be divisible by numHeadsKv, but got numHeadsQ=%ld, numHeadsKv=%ld", numHeadsQ,
                numHeadsKv);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantFlashAttnMetadataGetWorkspaceSize(const aclTensor *cuSeqlensQOptional,
    const aclTensor *cuSeqlensKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional,
    int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim,
    int64_t qQuantMode, int64_t kQuantMode, int64_t vQuantMode, int64_t qDtype, int64_t kDtype, int64_t vDtype,
    int64_t maskMode, int64_t winLeft, int64_t winRight, const char *layoutQ, const char *layoutKv,
    const char *layoutOut, const aclTensor *metaData, uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnQuantFlashAttnMetadata,
        DFX_IN(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize, maxSeqlenQ,
            maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, qQuantMode, kQuantMode, vQuantMode, qDtype, kDtype, vDtype,
            maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut),
        DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional, batchSize,
        maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, qQuantMode, kQuantMode, vQuantMode, qDtype, kDtype,
        vDtype, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto output = l0op::QuantFlashAttnMetadata(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional,
        sequsedKvOptional, batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, qQuantMode, kQuantMode,
        vQuantMode, qDtype, kDtype, vDtype, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, socVersion,
        aicCoreNum, aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnQuantFlashAttnMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnQuantFlashAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
