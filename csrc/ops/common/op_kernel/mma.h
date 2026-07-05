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
 * \file mma.h
 * \brief
 */

#ifndef INCLUDE_MMA_H
#define INCLUDE_MMA_H

#include "hardware.h"
#include "kernel_tensor.h"

template <ArchType ArchTag, typename ElementA, typename ElementB, typename AccDTypeC, bool IsTransposeA> struct mmad {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor, AscendC::LocalTensor<ElementA> l0aTensor,
        AscendC::LocalTensor<ElementB> l0bTensor, uint32_t mTileActual, uint32_t nTileActual, uint32_t kPartActual,
        bool initC) {};

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor, AscendC::LocalTensor<ElementA> l0aTensor,
        AscendC::LocalTensor<ElementB> l0bTensor, uint64_t biasBt, uint32_t mTileActual, uint32_t nTileActual,
        uint32_t kPartActual, bool initC) {};
};

// Partial specialization for V220, int8_t, not_vector_A, not TransposeA
template <ArchType ArchTag, typename AccDTypeC, typename ElementA, typename ElementB>
struct mmad<ArchTag, ElementA, ElementB, AccDTypeC, false> {
    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor, AscendC::LocalTensor<ElementA> l0aTensor,
        AscendC::LocalTensor<ElementB> l0bTensor, uint32_t mTileActual, uint32_t nTileActual, uint32_t kPartActual,
        bool initC) {
        AscendC::Mmad(l0cTensor, l0aTensor, l0bTensor,
            AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
    };

    __aicore__ mmad(AscendC::LocalTensor<AccDTypeC> l0cTensor, AscendC::LocalTensor<ElementA> l0aTensor,
        AscendC::LocalTensor<ElementB> l0bTensor, uint64_t biasBt, uint32_t mTileActual, uint32_t nTileActual,
        uint32_t kPartActual, bool initC) {
        AscendC::LocalTensor<ElementA> biasTensor;
        biasTensor.InitBuffer(biasBt, mTileActual);
        biasTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::C2);
        AscendC::Mmad(l0cTensor, l0aTensor, l0bTensor, biasTensor,
            AscendC::MmadParams(mTileActual, nTileActual, kPartActual, 0, false, initC));
    };
};

#endif
