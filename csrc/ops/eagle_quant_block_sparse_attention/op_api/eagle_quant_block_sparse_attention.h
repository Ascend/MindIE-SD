/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_H_
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_H_

#include <array>
#include "opdev/op_executor.h"
#include "opdev/common_types.h"

namespace l0op {

const std::array<const aclTensor *, 2> EagleQuantBlockSparseAttention(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryScaleOptional,
    const aclTensor *keyScaleOptional,
    const aclTensor *valueScaleOptional,
    const char *qInputLayout,
    const char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t softmaxLseFlag,
    aclOpExecutor *executor,
    op::DataType outDType);

} // namespace l0op

#endif  // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_H_

