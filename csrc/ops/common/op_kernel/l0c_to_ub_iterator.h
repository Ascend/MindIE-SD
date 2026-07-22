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
 * \file l0c_to_ub_iterator.h
 * \brief
 */
#ifndef L0C_TO_UB_ITERATOR_H
#define L0C_TO_UB_ITERATOR_H

#include "iterator.h"

/////////////////////////////////////////////////////
// l0c_to_ub
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag, typename ElementIn, typename ElementOut, bool MatrixMode = true> struct l0c_to_ub {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<ElementOut> ubTensor, AscendC::LocalTensor<ElementIn> l0cTensor,
        uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride) {
        constexpr auto mode =
            MatrixMode ? AscendC::BlockMode::BLOCK_MODE_MATRIX : AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(ubTensor, l0cTensor,
            AscendC::DataCopyParams(nBurst, // count
                lenBurst, // len
                srcStride, // srcStrideIn
                dstStride), // dstStrideIn
            AscendC::DataCopyEnhancedParams(mode, // blockModeIn
                AscendC::DeqScale::DEQ_NONE, // deqScaleIn
                0, // deqValueIn
                0, // sidStoreModeIn
                false, // isReluIn
                pad_t::PAD_NONE, // padModeIn
                0) // padValueIn
        );
    };
};

template <ArchType ArchTag> struct l0c_to_ub<ArchTag, int32_t, half> {
    __aicore__ l0c_to_ub(AscendC::LocalTensor<half> ubTensor, AscendC::LocalTensor<int32_t> l0cTensor, uint16_t nBurst,
        uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride) {
        AscendC::DataCopy(ubTensor, l0cTensor,
            AscendC::DataCopyParams(nBurst, // count
                lenBurst, // len
                srcStride, // srcStrideIn
                dstStride), // dstStrideIn
            AscendC::DataCopyEnhancedParams(AscendC::BlockMode::BLOCK_MODE_MATRIX, // blockModeIn
                AscendC::DeqScale::VDEQ16, // deqScaleIn
                0, // deqValueIn
                0, // sidStoreModeIn
                false, // isReluIn
                pad_t::PAD_NONE, // padModeIn
                0) // padValueIn
        );
    };
};

#endif // L0C_TO_UB_ITERATOR_H
