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
#ifndef EAGLE_INDEXED_GATE_NORM_MODULATE_TILING_H
#define EAGLE_INDEXED_GATE_NORM_MODULATE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EagleIndexedGateNormModulateTilingData)
TILING_DATA_FIELD_DEF(int64_t, rows);
TILING_DATA_FIELD_DEF(int64_t, indexedRows);
TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
TILING_DATA_FIELD_DEF(uint32_t, alignedSize);
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(float, eps);
TILING_DATA_FIELD_DEF(float, inverseHidden);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EagleIndexedGateNormModulate, EagleIndexedGateNormModulateTilingData)
} // namespace optiling
#endif
