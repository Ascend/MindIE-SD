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
 * \file l0_quant_flash_attn_metadata.cpp
 * \brief
 */

#include "l0_quant_flash_attn_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantFlashAttnMetadata);

const aclTensor *QuantFlashAttnMetadata(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t qQuantMode, int64_t kQuantMode,
    int64_t vQuantMode, int64_t qDtype, int64_t kDtype, int64_t vDtype, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char *layoutQ, const char *layoutKv, const char *layoutOut, const char *socVersion,
    int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor) {
    L0_DFX(QuantFlashAttnMetadata, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional,
        batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, qQuantMode, kQuantMode, vQuantMode, qDtype,
        kDtype, vDtype, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, socVersion, aicCoreNum, aivCoreNum,
        metaData);

    static internal::AicpuTaskSpace space("QuantFlashAttnMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(QuantFlashAttnMetadata,
        OP_ATTR_NAMES({"batch_size", "max_seqlen_q", "max_seqlen_kv", "num_heads_q", "num_heads_kv", "head_dim",
            "q_quant_mode", "k_quant_mode", "v_quant_mode", "q_dtype", "k_dtype", "v_dtype", "mask_mode", "win_left",
            "win_right", "layout_q", "layout_kv", "layout_out", "custom_soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional), OP_OUTPUT(metaData),
        OP_ATTR(batchSize, maxSeqlenQ, maxSeqlenKv, numHeadsQ, numHeadsKv, headDim, qQuantMode, kQuantMode, vQuantMode,
            qDtype, kDtype, vDtype, maskMode, winLeft, winRight, layoutQ, layoutKv, layoutOut, socVersion, aicCoreNum,
            aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
            "QuantFlashAttnMetadata"
            " ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return metaData;
}

} // namespace l0op
