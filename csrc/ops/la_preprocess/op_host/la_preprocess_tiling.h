/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LA_PREPROCESS_H_
#define LA_PREPROCESS_H_

#include "register/tilingdata_base.h"
namespace optiling {

struct LaPreprocessCompileInfo {
    uint64_t l0ASize;
    uint64_t l0BSize;
    uint64_t l0CSize;
    uint64_t l1Size;
    uint64_t ubSize;
    uint32_t maxAicCoresNum;
    uint32_t maxAivCoresNum;
    size_t defaultSysWorkspaceSize;
};

BEGIN_TILING_DATA_DEF(LaPreprocessTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, qSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, kSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, vSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, headNum);
    TILING_DATA_FIELD_DEF(uint32_t, alignLen);
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LaPreprocess, LaPreprocessTilingData)


} // namespace optiling

#endif // __SRC_OPS_HOST_LA_PREPROCESS_H__
