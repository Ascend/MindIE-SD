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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_TILING_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EagleSplitNormRopeConcatTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seq);
TILING_DATA_FIELD_DEF(int64_t, qHeads);
TILING_DATA_FIELD_DEF(int64_t, kHeads);
TILING_DATA_FIELD_DEF(int64_t, vHeads);
TILING_DATA_FIELD_DEF(uint32_t, dim);
TILING_DATA_FIELD_DEF(uint32_t, rotaryDim);
TILING_DATA_FIELD_DEF(uint32_t, ropeWidth);
TILING_DATA_FIELD_DEF(uint32_t, ropeType);
TILING_DATA_FIELD_DEF(uint32_t, normType);
TILING_DATA_FIELD_DEF(uint32_t, ropeBatch);
TILING_DATA_FIELD_DEF(uint32_t, outputBnsd);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(uint32_t, headTile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EagleSplitNormRopeConcat, EagleSplitNormRopeConcatTilingData)
} // namespace optiling
#endif
