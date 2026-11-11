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

#ifndef ADA_RELU2_BLOCK_ESTIMATE_TILING_H
#define ADA_RELU2_BLOCK_ESTIMATE_TILING_H

#include "register/tilingdata_base.h"

#define ADA_RELU2_QBNSD_KVBNSD_KEY 0
#define ADA_RELU2_QBNSD_KVBNSD_INT8_KEY 1

namespace optiling {
struct AdaRelu2BlockEstimateCompileInfo {};

BEGIN_TILING_DATA_DEF(AdaRelu2BlockEstimateTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batchSize);
TILING_DATA_FIELD_DEF(uint32_t, seqLenQ);
TILING_DATA_FIELD_DEF(uint32_t, seqLenKV);
TILING_DATA_FIELD_DEF(uint32_t, headNumQ);
TILING_DATA_FIELD_DEF(uint32_t, headNumKV);
TILING_DATA_FIELD_DEF(uint32_t, headDim);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsSize);
TILING_DATA_FIELD_DEF(uint32_t, actualSeqLengthsKVSize);
TILING_DATA_FIELD_DEF(uint32_t, stride);
TILING_DATA_FIELD_DEF(uint32_t, sparseSize);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(float, threshold);
TILING_DATA_FIELD_DEF(uint32_t, inputLayout);
TILING_DATA_FIELD_DEF(uint32_t, qTileSize);
TILING_DATA_FIELD_DEF(uint32_t, qLoadChunk);
TILING_DATA_FIELD_DEF(uint32_t, kvTileSize);
TILING_DATA_FIELD_DEF(uint32_t, mL0TileSize);
TILING_DATA_FIELD_DEF(uint32_t, nL0TileSize);
TILING_DATA_FIELD_DEF(uint32_t, kL0TileSize);
TILING_DATA_FIELD_DEF(uint32_t, kL0LoopNum);
TILING_DATA_FIELD_DEF(uint32_t, mL1SliceNum);
TILING_DATA_FIELD_DEF(uint32_t, qBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, qTileNum);
TILING_DATA_FIELD_DEF(uint32_t, kvBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, scoredKvBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, totalTaskNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdaRelu2BlockEstimate, AdaRelu2BlockEstimateTilingData)
} // namespace optiling

#endif // ADA_RELU2_BLOCK_ESTIMATE_TILING_H
