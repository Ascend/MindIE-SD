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
 * \file vf_row_invalid.h
 * \brief
 */
#ifndef VF_ROW_INVALID_INTERFACE_H
#define VF_ROW_INVALID_INTERFACE_H

#include "kernel_tensor.h"

namespace AscendC {
/* **************************************************************************************************
 * RowInvalid
 * ************************************************************************************************* */

/*
 * @ingroup RowInvalidUpdate
 * @Update finalRes invalid row to 0
 * @param [in and out] finalTensor, tempBmm2Ub
 * @param [in] maxTensor, softmaxMaxUb
 * @param [in] m, input rows
 * @param [in] dSize, alignColums
 * @param [in] d, input colums
 * @param [in] reduceSize, 多少个相同值表示当前行最大值
 */
template <typename T>
__aicore__ inline void RowInvalidUpdate(const LocalTensor<T> &finalTensor, const LocalTensor<float> &maxTensor,
    const uint16_t m, const uint32_t dSize, const uint16_t d, const uint16_t reduceSize) {
    __ubuf__ T *finalUb = (__ubuf__ T *)finalTensor.GetPhyAddr();
    __ubuf__ float *maxUb = (__ubuf__ float *)maxTensor.GetPhyAddr();

    constexpr uint16_t floatRepSize = 64; // 64: 一个寄存器可以存储64个float类型数据
    const uint16_t dLoops = d / floatRepSize;
    const uint16_t tailD = d % floatRepSize;
    uint32_t pltTailD = static_cast<uint32_t>(tailD);
    uint16_t hasTail = 0;
    if (tailD > 0) {
        hasTail = 1;
    }

    constexpr uint32_t tmpZero = 0x00000000; // zero value of fp16 and fp32
    const T zeroValue = *((T *)&tmpZero);
    constexpr uint32_t tmpMin = 0xFF7FFFFF; // min value of float
    const float minValue = *((float *)&tmpMin);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> vregMinValue;
        MicroAPI::RegTensor<T> vregZeroValue;
        MicroAPI::RegTensor<float> vregMax;
        MicroAPI::RegTensor<T> vregFinal;
        MicroAPI::RegTensor<T> vregFinalNew;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregTailD = MicroAPI::UpdateMask<T>(pltTailD);
        MicroAPI::MaskReg pregCompare;

        MicroAPI::Duplicate<float, float>(vregMinValue, minValue);
        MicroAPI::Duplicate<T, T>(vregZeroValue, zeroValue);
        for (uint16_t i = 0; i < m; ++i) {
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(vregMax, maxUb + i * reduceSize);
            MicroAPI::Compare<float, CMPMODE::EQ>(pregCompare, vregMax, vregMinValue, pregAll);
            for (uint16_t j = 0; j < dLoops; ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(vregFinal, finalUb + i * dSize + j * floatRepSize);
                MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    finalUb + i * dSize + j * floatRepSize, vregFinalNew, pregAll);
            }
            for (uint16_t t = 0; t < hasTail; ++t) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(
                    vregFinal, finalUb + i * dSize + dLoops * floatRepSize);
                MicroAPI::Select<T>(vregFinalNew, vregZeroValue, vregFinal, pregCompare);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                    finalUb + i * dSize + dLoops * floatRepSize, vregFinalNew, pregTailD);
            }
        }
    }
}

} // namespace

#endif // VF_ROW_INVALID_H
