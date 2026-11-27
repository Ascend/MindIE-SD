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

#include "eagle_hadamard_quant.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(EagleHadamardQuant);

aclnnStatus EagleHadamardQuant(const aclTensor *x, const aclTensor *signH, int64_t blockSize, const char *dstType,
    bool isSignHNz, aclTensor *y, aclTensor *scale, aclOpExecutor *executor) {
    L0_DFX(EagleHadamardQuant, x, signH, blockSize, dstType, isSignHNz, y, scale);
    return ADD_TO_LAUNCHER_LIST_AICORE(
        EagleHadamardQuant, OP_INPUT(x, signH), OP_OUTPUT(y, scale), OP_ATTR(blockSize, dstType, isSignHNz));
}
} // namespace l0op
