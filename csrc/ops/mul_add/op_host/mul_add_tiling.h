/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef MUL_ADD_TILING_H_
#define MUL_ADD_TILING_H_

#include "register/tilingdata_base.h"

namespace optiling {

struct MulAddCompileInfo {
    uint64_t ubSize;
    uint32_t maxAivCoresNum;
};

BEGIN_TILING_DATA_DEF(MulAddTilingData)
    TILING_DATA_FIELD_DEF(int64_t, batchSize);
    TILING_DATA_FIELD_DEF(int64_t, seqLen);
    TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
    TILING_DATA_FIELD_DEF(int64_t, hiddenSizeAlign);
    TILING_DATA_FIELD_DEF(int64_t, formerNum);
    TILING_DATA_FIELD_DEF(int64_t, formerLength);
    TILING_DATA_FIELD_DEF(int64_t, tailLength);
    TILING_DATA_FIELD_DEF(int64_t, rowsPerTile);
    TILING_DATA_FIELD_DEF(int64_t, dtypeFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MulAdd, MulAddTilingData)

} // namespace optiling

#endif // MUL_ADD_TILING_H_
