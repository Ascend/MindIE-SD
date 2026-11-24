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

#ifndef INDEXED_ADALN_TILING_H_
#define INDEXED_ADALN_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

struct IndexedAdaLayerNormCompileInfo {
    uint64_t ubSize;
    uint32_t maxAivCoresNum;
};

BEGIN_TILING_DATA_DEF(IndexedAdaLayerNormTilingData)
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, seqLen);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, tableK);
TILING_DATA_FIELD_DEF(int64_t, totalRows);
TILING_DATA_FIELD_DEF(int64_t, rowsPerCore);
TILING_DATA_FIELD_DEF(int64_t, tailCoreNum);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(float, aveFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexedAdaLayerNorm, IndexedAdaLayerNormTilingData)

} // namespace optiling

#endif // INDEXED_ADALN_TILING_H_
