/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

/* !
 * \file eagle_quant_batch_matmul_swiglu_kernel_tiling_data.h
 * \brief
 */

#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_KERNEL_TILING_DATA_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_KERNEL_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) EagleQuantBatchMatmulSwigluParams {
    uint32_t batchA;
    uint32_t batchB;
    uint32_t batchC;
    uint32_t batchA1;
    uint32_t batchA2;
    uint32_t batchA3;
    uint32_t batchA4;
    uint32_t batchB1;
    uint32_t batchB2;
    uint32_t batchB3;
    uint32_t batchB4;
    uint32_t batchC1;
    uint32_t batchC2;
    uint32_t batchC3;
    uint32_t batchC4;
    uint32_t singleCoreBatch;
    uint32_t isPerTensor;
    uint32_t isPertoken;
    uint32_t isDoubleScale;
    uint32_t biasThreeDim;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint32_t needUbBuffer;
    uint32_t realSingleCoreM;
    uint32_t realSingleCoreN;
    uint32_t biasDtype;
    uint32_t ubSize;
    uint32_t isMClash;
    uint32_t isNClash;
    uint32_t groupSizeM;
    uint32_t groupSizeN;
    uint32_t groupSizeK;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) L2cacheTileParam {
    uint32_t mTileCntL2;
    uint32_t nTileCntL2;
    uint32_t mTileBlock;
    uint32_t nTileBlock;
    uint32_t calOrder;
    uint32_t isBasicTiling;
    uint32_t nPairSize; // 0:普通调度；非0:前后两半的列间距，L2的N计数按单半边计算
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) SlidingWindowParam {
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
};
#pragma pack(pop)

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) EagleQuantBatchMatmulSwigluTilingData {
    EagleQuantBatchMatmulSwigluParams params;
    TCubeTiling matmulTiling;
    L2cacheTileParam tileL2cacheTiling;
    SlidingWindowParam adaptiveSlidingWin;
};
#pragma pack(pop)

#pragma pack(push, 8)
struct QuantMatmulPertokenTilingDataArch20 {
    uint32_t batchSize;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t m0;
    uint32_t k0;
    uint32_t n0;
    uint32_t mLoop;
    uint32_t kLoop;
    uint32_t nLoop;
    uint32_t coreLoop;
    uint32_t blockDim;
    uint32_t swizzleDirect;
    uint32_t swizzleCount;
    uint32_t biasDtype;
    bool withBias;
    bool biasWithBatch;
};
#pragma pack(pop)

#endif
