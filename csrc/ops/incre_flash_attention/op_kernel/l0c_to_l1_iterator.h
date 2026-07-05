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
 * \file l0c_to_l1_iterator.h
 * \brief
 */

#ifndef L0C_TO_L1_ITERATOR_H
#define L0C_TO_L1_ITERATOR_H

#include "iterator.h"
/////////////////////////////////////////////////////
// l0c_to_l1
/////////////////////////////////////////////////////

// Partial specialization ZN, half, int32_t
template <ArchType ArchTag> struct l0c_to_l1<ArchTag, DataFormatT::ZN, half, int32_t> {
    using ElementOut = half;
    using ElementIn = int32_t;
    __aicore__ l0c_to_l1(AscendC::LocalTensor<ElementOut> l1Tensor, AscendC::LocalTensor<ElementIn> l0cTensor,
        AscendC::LocalTensor<uint64_t> deqTensor, uint32_t mTileActual, uint32_t nTileActual, uint32_t mTileCeil,
        uint32_t nActual) {
        constexpr uint32_t BLOCK_NUM = 16;
        constexpr uint32_t BLOCK_SIZE = 32;
        AscendC::FixpipeParams<ElementIn> intriParams((nTileActual + BLOCK_NUM - 1) / AscendC::BLOCK_CUBE,
            static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE), 0,
            mTileCeil -
                static_cast<uint16_t>(mTileActual * BLOCK_NUM * sizeof(float) / BLOCK_SIZE) * sizeof(ElementOut) /
                    sizeof(ElementIn));
        intriParams.nz2ndParams = {false, 1, 0, 0, static_cast<uint16_t>(nTileActual)};
        intriParams.quantParams = {QuantMode_t::VDEQF16};
        AscendC::Fixpipe(l1Tensor, l0cTensor, deqTensor, intriParams);
    };
};

#endif // L0C_TO_L1_ITERATOR_H
