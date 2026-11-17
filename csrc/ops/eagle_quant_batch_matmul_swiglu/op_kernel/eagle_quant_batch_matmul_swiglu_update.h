/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
 * \file eagle_quant_batch_matmul_swiglu_update.h
 * \brief
 */
#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_UPDATE_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_UPDATE_H

#include "eagle_quant_batch_matmul_swiglu_base.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace AscendC {

struct QBmmUpdateInfo {
    uint64_t mBaseTail; // 只量化mm用
    uint64_t nBaseTail; // 只量化mm用
    uint64_t alignedKaSize;
    uint64_t alignedKbSize;
};

class EagleQuantBatchMatmulSwigluUpdate {
  public:
    __aicore__ inline EagleQuantBatchMatmulSwigluUpdate() {}
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void Init(const TCubeTiling *mmTiling, const QBmmBaseBlockArgs &params, uint64_t logicalN = 0);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset,
        uint64_t mTileIndex, uint64_t nTileIndex, uint64_t nBranchOffset = 0);
    __aicore__ inline void UpdateBlockParams(QBmmBaseBlockArgs &params, uint64_t mTileIndex, uint64_t nTileIndex);
    template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type = int8_t, typename x2Type = int8_t>
    __aicore__ inline void CalcGMOffset(QBmmBaseBlockArgs &params, QBmmBlockOffset &offset, uint64_t nBranchOffset = 0);

  private:
    QBmmUpdateInfo info_;
    const TCubeTiling *mmTiling_;
};

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void EagleQuantBatchMatmulSwigluUpdate::Init(
    const TCubeTiling *mmTiling, const QBmmBaseBlockArgs &params, uint64_t logicalN) {
    mmTiling_ = mmTiling;
    info_.nBaseTail = (logicalN == 0 ? static_cast<uint64_t>(mmTiling_->N) : logicalN) -
        (params.nTotalCnt - 1) * mmTiling_->singleCoreN; // n方向上的base尾块
    info_.mBaseTail =
        static_cast<uint64_t>(mmTiling_->M) - (params.mTotalCnt - 1) * mmTiling_->singleCoreM; // m方向上的base尾块
    if constexpr (aTrans) { // (k, m)
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, BMM_BLOCK_NUM);
    } else { // (m, k)
        info_.alignedKaSize = DequantBmm::Align(mmTiling_->Ka, DequantBmm::GetC0Size<x1Type>());
    }
    if constexpr (bTrans) { // (n, k)
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, DequantBmm::GetC0Size<x2Type>());
    } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) { // (n1,k1,k0,n0), k0=16
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, BMM_BLOCK_NUM);
    } else { // (k, n) ND
        info_.alignedKbSize = DequantBmm::Align(mmTiling_->Kb, BMM_BLOCK_NUM);
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void EagleQuantBatchMatmulSwigluUpdate::UpdateBlockParamsAndCalcGmOffset(QBmmBaseBlockArgs &params,
    QBmmBlockOffset &offset, uint64_t mTileIndex, uint64_t nTileIndex, uint64_t nBranchOffset) {
    UpdateBlockParams(params, mTileIndex, nTileIndex);
    CalcGMOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(params, offset, nBranchOffset);
}

__aicore__ inline void EagleQuantBatchMatmulSwigluUpdate::UpdateBlockParams(
    QBmmBaseBlockArgs &params, uint64_t mTileIndex, uint64_t nTileIndex) {
    if ((mTileIndex == (params.mTileCntL2 - 1)) && (nTileIndex == (params.nTileCntL2 - 1)) &&
        (params.index == (params.totalTileCnt - 1))) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = info_.nBaseTail;
    } else if ((mTileIndex == (params.mTileCntL2 - 1)) && (params.index >= (params.mCntUse - 1) * params.nCntUse)) {
        params.singleCoreM = info_.mBaseTail;
        params.singleCoreN = mmTiling_->baseN;
    } else if ((nTileIndex == (params.nTileCntL2 - 1)) && ((params.index + 1) % params.nCntUse == 0)) {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = info_.nBaseTail;
    } else {
        params.singleCoreM = mmTiling_->baseM;
        params.singleCoreN = mmTiling_->baseN;
    }
}

template <int x1Format, int x2Format, bool aTrans, bool bTrans, typename x1Type, typename x2Type>
__aicore__ inline void EagleQuantBatchMatmulSwigluUpdate::CalcGMOffset(
    QBmmBaseBlockArgs &params, QBmmBlockOffset &offset, uint64_t nBranchOffset) {
    uint64_t mCntIndex = params.index / params.nCntUse;
    uint64_t nCntIndex = params.index - mCntIndex * params.nCntUse;
    // tiling已保证baseM/K/N低轴16/32对齐
    // 前面的m都能保证是singleCoreM(baseM)的倍数，m尾块只会出现在m轴最后
    uint64_t mOffset = mCntIndex * mmTiling_->singleCoreM + params.mTileAddrOffset;
    if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::ND) {
        if constexpr (aTrans) { // (k, m)
            offset.offsetA = mOffset;
        } else { // (m, k)
            offset.offsetA = mOffset * mmTiling_->Ka;
        }
    } else if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ) {
        if constexpr (aTrans) { // (m1, k1, k0, m0)
            offset.offsetA = mOffset * info_.alignedKaSize;
        } else { // (k1, m1, m0, k0)
            offset.offsetA = mOffset * DequantBmm::GetC0Size<x1Type>();
        }
    }
    // 前面的n都能保证是singleCoreN(baseN)的倍数，n尾块只会出现在n轴最后
    uint64_t nOffset = nCntIndex * mmTiling_->singleCoreN + params.nTileAddrOffset + nBranchOffset;
    if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::ND) {
        if constexpr (bTrans) { // (n, k)
            offset.offsetB = nOffset * mmTiling_->Kb;
        } else { // (k, n)
            offset.offsetB = nOffset;
        }
    } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
        if constexpr (bTrans) { // (k1, n1, n0, k0)
            offset.offsetB = nOffset * DequantBmm::GetC0Size<x2Type>();
        } else { // (n1, k1, k0, n0)
            offset.offsetB = nOffset * info_.alignedKbSize;
        }
    }
    // 输出只支持ND
    offset.offsetC = mOffset * mmTiling_->N + nOffset;
    // scale的计算当perchannel统一处理，实际参与mm/ub计算时会区分perchannel/pertensor
    offset.offsetScale = nOffset;
    // 当前基本块算法无batch轴，所以只考虑单维度，可选输入当存在时计算
    offset.offsetBias = nOffset;
    // 即使是纯cube场景，也可以计算pertoken偏移，这样mm输入输出的update接口可以共用
    offset.offsetPertoken = mOffset;
}
} // namespace AscendC
#endif // EAGLE_QUANT_BATCH_MATMUL_SWIGLU_UPDATE_H
