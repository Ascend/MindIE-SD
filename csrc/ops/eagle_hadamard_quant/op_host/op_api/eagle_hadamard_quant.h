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

#pragma once

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {
aclnnStatus EagleHadamardQuant(const aclTensor *x, const aclTensor *signH, int64_t blockSize, const char *dstType,
    bool isSignHNz, aclTensor *y, aclTensor *scale, aclOpExecutor *executor);
} // namespace l0op
