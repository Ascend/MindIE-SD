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
#ifndef VF_MASK2IDX_H_
#define VF_MASK2IDX_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

namespace Mask2IdxApi {
using namespace AscendC;
using namespace MicroAPI;

constexpr uint16_t VL_U32 = 64U;

template <typename IdxT = int32_t>
__simd_vf__ inline void Mask2IdxRowVF(__ubuf__ int32_t *idxOutI32, __ubuf__ int32_t *countOut, __ubuf__ uint8_t *maskIn,
    __ubuf__ int32_t *countAccumCurrent, __ubuf__ int32_t *countAccumPrev, uint32_t colNum0, uint32_t segStart,
    uint32_t actRowCur, uint32_t segRowStrideI32Elems, uint32_t segRowStrideU8, uint16_t segmentLoop = 1,
    uint16_t seg = 0) {
    using namespace MicroAPI;
    RegTensor<uint32_t> maskU32_0, maskU32_1;
    RegTensor<int32_t> idxI32_0, dstI32_0;
    RegTensor<int32_t> idxI32_1, dstI32_1;
    RegTensor<int32_t> zerosI32, onesI32, baseI32, accReg, vecAccum, cntVec_0, cntVec_1;
    RegTensor<int32_t> currentAcc, prevAcc;
    uint32_t actRowCur0 = actRowCur;
    MaskReg tailMask = UpdateMask<int32_t>(actRowCur0);
    MaskReg tailMask_0, predMask_0;
    MaskReg tailMask_1, predMask_1;
    MaskReg pregAllU32 = CreateMask<uint32_t, MaskPattern::ALL>();

    Duplicate<int32_t>(zerosI32, static_cast<int32_t>(0));
    Duplicate<int32_t>(onesI32, static_cast<int32_t>(1));
    Arange<int32_t, IndexOrder::INCREASE_ORDER>(baseI32, static_cast<int32_t>(0));

    uint16_t chunkNum = CeilDiv(colNum0, VL_U32);
    uint16_t chunkLoop = chunkNum / 2;
    uint16_t chunkRemain = chunkNum % 2;

    for (uint16_t k = 0; k < (uint16_t)actRowCur; k++) {
        ClearSpr<SpecialPurposeReg::AR>();
        UnalignRegForStore ureg;
        __ubuf__ int32_t *idxWritePtr = idxOutI32 + k * segRowStrideI32Elems;
        __ubuf__ uint8_t *maskInPtr = maskIn + k * segRowStrideU8;
        __ubuf__ int32_t *countOutPtr = countOut + k;

        Duplicate<int32_t>(vecAccum, static_cast<int32_t>(0));
        int32_t curChunkStart = static_cast<int32_t>(segStart);
        uint32_t remainingCols = colNum0;

        for (uint16_t chunk = 0; chunk < chunkLoop; chunk++) {
            Adds<int32_t>(idxI32_0, baseI32, curChunkStart, pregAllU32);
            Adds<int32_t>(idxI32_1, baseI32, curChunkStart + static_cast<int32_t>(VL_U32), pregAllU32);
            curChunkStart += static_cast<int32_t>(VL_U32 << 1);

            LoadAlign<uint8_t, LoadDist::DIST_UNPACK4_B8>((RegTensor<uint8_t> &)maskU32_0, maskInPtr);
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK4_B8>((RegTensor<uint8_t> &)maskU32_1, maskInPtr + VL_U32);
            maskInPtr += (VL_U32 << 1);

            tailMask_0 = UpdateMask<uint32_t>(remainingCols);
            tailMask_1 = UpdateMask<uint32_t>(remainingCols);
            Compares<uint32_t, CMPMODE::NE>(predMask_0, maskU32_0, static_cast<uint32_t>(0), tailMask_0);
            Compares<uint32_t, CMPMODE::NE>(predMask_1, maskU32_1, static_cast<uint32_t>(0), tailMask_1);

            // Count selected indices, not the numeric sum of mask bytes.
            Select<int32_t>(cntVec_0, onesI32, zerosI32, predMask_0);
            Select<int32_t>(cntVec_1, onesI32, zerosI32, predMask_1);
            Add<int32_t>(vecAccum, vecAccum, cntVec_0, pregAllU32);
            Add<int32_t>(vecAccum, vecAccum, cntVec_1, pregAllU32);

            Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(dstI32_0, idxI32_0, predMask_0);
            StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(idxWritePtr, dstI32_0, ureg);

            Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(dstI32_1, idxI32_1, predMask_1);
            StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(idxWritePtr, dstI32_1, ureg);
        }

        for (uint16_t i = 0; i < chunkRemain; i++) {
            tailMask_0 = UpdateMask<uint32_t>(remainingCols);
            Adds<int32_t>(idxI32_0, baseI32, curChunkStart, pregAllU32);
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK4_B8>((RegTensor<uint8_t> &)maskU32_0, maskInPtr);
            Compares<uint32_t, CMPMODE::NE>(predMask_0, maskU32_0, static_cast<uint32_t>(0), tailMask_0);
            Select<int32_t>(cntVec_0, onesI32, zerosI32, predMask_0);
            Add<int32_t>(vecAccum, vecAccum, cntVec_0, pregAllU32);
            Squeeze<int32_t, AscendC::Reg::GatherMaskMode::STORE_REG>(dstI32_0, idxI32_0, predMask_0);
            StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(idxWritePtr, dstI32_0, ureg);
        }

        StoreUnAlignPost<int32_t>(idxWritePtr, ureg);
        Reduce<ReduceType::SUM, int32_t, int32_t>(accReg, vecAccum, pregAllU32);
        StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(countOutPtr, accReg, pregAllU32);
    }

    if (segmentLoop > 1) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        LoadAlign<int32_t, LoadDist::DIST_NORM>(currentAcc, countOut);

        if (seg > 0) {
            LoadAlign<int32_t, LoadDist::DIST_NORM>(prevAcc, countAccumPrev);
            Add<int32_t>(currentAcc, currentAcc, prevAcc, tailMask);
        }
        StoreAlign<int32_t, StoreDist::DIST_NORM_B32>(countAccumCurrent, currentAcc, tailMask);
    }
}

} // namespace Mask2IdxApi
#endif // VF_MASK2IDX_H_
