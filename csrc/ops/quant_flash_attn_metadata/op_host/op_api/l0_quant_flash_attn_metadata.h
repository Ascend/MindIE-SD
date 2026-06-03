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

#ifndef L0_QUANT_FLASH_ATTN_METADATA_H
#define L0_QUANT_FLASH_ATTN_METADATA_H

#include "opdev/op_executor.h"

namespace l0op {
const aclTensor *QuantFlashAttnMetadata(const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenKv, int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t qQuantMode, int64_t kQuantMode,
    int64_t vQuantMode, int64_t qDtype, int64_t kDtype, int64_t vDtype, int64_t maskMode, int64_t winLeft,
    int64_t winRight, const char *layoutQ, const char *layoutKv, const char *layoutOut, const char *socVersion,
    int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metaData, aclOpExecutor *executor);
} // namespace l0op

#endif
