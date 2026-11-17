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

/*!
 * \file eagle_quant_batch_matmul_swiglu_tiling_cache.h
 * \brief
 */
#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_TILING_CACHE_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_TILING_CACHE_H

#include "eagle_quant_batch_matmul_swiglu_basic_tiling.h"
#include "../deps/matmul/common/op_host/op_tiling/tiling_cache.h"

namespace optiling {
struct EagleQuantBatchMatmulSwigluBitField {
    // 这里要保证是32bit
    uint16_t transA : 1;
    uint16_t transB : 1;
    uint16_t hasBias : 1;
    uint16_t aFormatNd : 1;
    uint16_t bFormatNd : 1;
    uint16_t cFormatNd : 1;
    uint16_t isPerTensor : 1;
    uint16_t isPertoken : 1;
    uint32_t reserved : 24;
};

class EagleQuantBatchMatmulSwigluHashInput {
  public:
    explicit EagleQuantBatchMatmulSwigluHashInput(
        const EagleQbmmInfo &params, const optiling::AiCoreParams &aicoreParams);
    ~EagleQuantBatchMatmulSwigluHashInput() = default;
    bool operator==(const EagleQuantBatchMatmulSwigluHashInput &params) const {
        return memcmp(this, &params, sizeof(params)) == 0;
    }
    std::string ToString() const;

  private:
    uint64_t mSize = 0;
    uint64_t mSizePerNpu = 0;
    uint64_t kSize = 0;
    uint64_t nSize = 0;
    uint64_t batchA1 = 0;
    uint64_t batchA2 = 0;
    uint64_t batchA3 = 0;
    uint64_t batchA4 = 0;
    uint64_t batchB1 = 0;
    uint64_t batchB2 = 0;
    uint64_t batchB3 = 0;
    uint64_t batchB4 = 0;
    uint64_t batchBias = 0;
    int32_t aDtype = 0;
    int32_t bDtype = 0;
    int32_t cDtype = 0;
    int32_t biasDtype = 0;
    int32_t scaleDtype = 0;
    int32_t outDtype = 0;
    int32_t aicNum = 0;
    int32_t reserved = 0;
    EagleQuantBatchMatmulSwigluBitField bitField = {0, 0, 0, 0, 0, 0, 0, 0, 0};
};

class EagleQuantBatchMatmulSwigluHashItem {
  public:
    explicit EagleQuantBatchMatmulSwigluHashItem(
        const EagleQbmmInfo &params, const optiling::AiCoreParams &aicoreParams)
        : hashKey_(params, aicoreParams) {}
    const EagleQuantBatchMatmulSwigluHashInput &input() const { return hashKey_; }
    const BasicTiling &GetTiling() const { return tiling_; }
    void SetTiling(const BasicTiling &tiling) { tiling_ = tiling; }

  private:
    EagleQuantBatchMatmulSwigluHashInput hashKey_;
    BasicTiling tiling_;
};

using MMBasicTilingHash =
    Ops::EagleQbmm::TilingCache<EagleQuantBatchMatmulSwigluHashInput, EagleQuantBatchMatmulSwigluHashItem>;
} // namespace optiling
#endif // EAGLE_QUANT_BATCH_MATMUL_SWIGLU_TILING_CACHE_H
