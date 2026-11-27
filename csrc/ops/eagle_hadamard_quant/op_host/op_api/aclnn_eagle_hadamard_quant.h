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

#ifdef __cplusplus
extern "C" {
#endif

// BF16 x [M,N]; BF16 signH is ND [K,K]/[B,K,K] or explicit NZ [B,K/16,K/16,16,16].
// NZ axes are [B,columnTile,rowTile,rowInTile,columnInTile], with 16x16 tiles.
// blockSize is 32/64/128/256; dstType is "mxfp4" or "mxfp8" (E4M3).
// y: UINT8 [M,N/2] for packed MXFP4, or FLOAT8_E4M3FN [M,N] for MXFP8.
// scale: FLOAT8_E8M0 [M,ceil(N/64),2]. All tensors use non-private formats.
// For nonempty tensors each output must occupy a disjoint byte range from both inputs and the other output.
// Ranges include gaps in strided views. Output axes must describe nonoverlapping slices in increasing
// stride order. Nonoverlapping slices of one allocation are supported. Inputs may alias one another.
// Compatible per-block rotations are grouped internally; each rotation is computed independently.
__attribute__((visibility("default"))) aclnnStatus aclnnEagleHadamardQuantGetWorkspaceSize(const aclTensor *x,
    const aclTensor *signH, int64_t blockSize, const char *dstType, bool isSignHNz, const aclTensor *y,
    const aclTensor *scale, uint64_t *workspaceSize, aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnEagleHadamardQuant(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
