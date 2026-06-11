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
 * \file vf_computeScale_dn_mxfp4.h
 * \brief
 */

#ifndef COMPUTESCALE_DN_MXFP4_H_
#define COMPUTESCALE_DN_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

// T: half
template <bool clear_gmax, typename T, uint16_t m = 128>
__simd_vf__ inline void computePscaleVF(__ubuf__ uint8_t *mxscale1, __ubuf__ uint8_t *mxscale2, __ubuf__ T *ulmax1,
    __ubuf__ T *ulmax2, __ubuf__ T *umax1, __ubuf__ T *umax2, __ubuf__ T *ugmax_old, __ubuf__ float *urs,
    uint16_t firstLoop, uint16_t secondLoop) {
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all8 = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_max2, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

    constexpr uint16_t ulmaxLoopRow = m * 8;

    if constexpr (clear_gmax) {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_all16);
        StoreAlign(ugmax_old, vreg_gmax, preg_all16);
    } else {
        LoadAlign(vreg_max1, umax1);
        LoadAlign(vreg_max2, umax2);
        LoadAlign(vreg_gmax_old, ugmax_old);

        Max(vreg_gmax, vreg_max1, vreg_max2, preg_all16);
        Max(vreg_gmax, vreg_gmax, vreg_gmax_old, preg_all16);
        StoreAlign(ugmax_old, vreg_gmax, preg_all16);

        Sub(vreg_rs16, vreg_gmax_old, vreg_gmax, preg_all16);
        Adds(vreg_rs16, vreg_rs16, NUM_127, preg_all16);
        Maxs(vreg_rs16, vreg_rs16, ZERO_VALUE, preg_all16);
        Cast<int32_t, T, h2iZero>((RegTensor<int32_t> &)vreg_rs1, vreg_rs16, preg_all16);
        Cast<int32_t, T, h2iOne>((RegTensor<int32_t> &)vreg_rs2, vreg_rs16, preg_all16);
        ShiftLefts((RegTensor<int32_t> &)vreg_rs1, (RegTensor<int32_t> &)vreg_rs1, SHIFT_VALUE, preg_all16);
        ShiftLefts((RegTensor<int32_t> &)vreg_rs2, (RegTensor<int32_t> &)vreg_rs2, SHIFT_VALUE, preg_all16);

        Interleave((RegTensor<int32_t> &)vreg_rs1, (RegTensor<int32_t> &)vreg_rs2, (RegTensor<int32_t> &)vreg_rs1,
            (RegTensor<int32_t> &)vreg_rs2);
        StoreAlign(urs, vreg_rs1, preg_all16);
        StoreAlign(urs + m / 2, vreg_rs2, preg_all16);
    }

    Duplicate(vreg_max1, MIN_VALUE);
    StoreAlign(umax1, vreg_max1, preg_all16);
    Adds(vreg_gmax, vreg_gmax, NUM_NEG_125, preg_all16);

    for (uint16_t i = 0; i < firstLoop; i++) {
        RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4, vreg_scale5, vreg_scale6, vreg_scale7,
            vreg_scale8;
        LoadAlign(vreg_scale1, ulmax1 + (i * ulmaxLoopRow));
        LoadAlign(vreg_scale2, ulmax1 + (i * ulmaxLoopRow + 1 * m));
        LoadAlign(vreg_scale3, ulmax1 + (i * ulmaxLoopRow + 2 * m));
        LoadAlign(vreg_scale4, ulmax1 + (i * ulmaxLoopRow + 3 * m));
        LoadAlign(vreg_scale5, ulmax1 + (i * ulmaxLoopRow + 4 * m));
        LoadAlign(vreg_scale6, ulmax1 + (i * ulmaxLoopRow + 5 * m));
        LoadAlign(vreg_scale7, ulmax1 + (i * ulmaxLoopRow + 6 * m));
        LoadAlign(vreg_scale8, ulmax1 + (i * ulmaxLoopRow + 7 * m));

        Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_all16);
        Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_all16);
        Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_all16);
        Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_all16);
        Sub(vreg_scale5, vreg_scale5, vreg_gmax, preg_all16);
        Sub(vreg_scale6, vreg_scale6, vreg_gmax, preg_all16);
        Sub(vreg_scale7, vreg_scale7, vreg_gmax, preg_all16);
        Sub(vreg_scale8, vreg_scale8, vreg_gmax, preg_all16);

        // SAT Mode, if < 0, then = 0
        RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4, vreg_mxscale5, vreg_mxscale6,
            vreg_mxscale7, vreg_mxscale8;
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_all16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_all16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_all16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_all16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale5, vreg_scale5, preg_all16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale6, vreg_scale6, preg_all16);
        Cast<uint8_t, T, castTraitZero>(vreg_mxscale7, vreg_scale7, preg_all16);
        Cast<uint8_t, T, castTraitOne>(vreg_mxscale8, vreg_scale8, preg_all16);

        Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_all8);
        Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_all8);
        Or(vreg_mxscale5, vreg_mxscale5, vreg_mxscale6, preg_all8);
        Or(vreg_mxscale7, vreg_mxscale7, vreg_mxscale8, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 0, vreg_mxscale1, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 1, vreg_mxscale3, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 2, vreg_mxscale5, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale1 + i * 32 * 5 * 8 + 32 * 3, vreg_mxscale7, 5, preg_all8);
    }

    for (uint16_t i = 0; i < secondLoop; i++) {
        RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4, vreg_scale5, vreg_scale6, vreg_scale7,
            vreg_scale8;
        LoadAlign(vreg_scale1, ulmax2 + (i * ulmaxLoopRow));
        LoadAlign(vreg_scale2, ulmax2 + (i * ulmaxLoopRow + 1 * m));
        LoadAlign(vreg_scale3, ulmax2 + (i * ulmaxLoopRow + m * 2));
        LoadAlign(vreg_scale4, ulmax2 + (i * ulmaxLoopRow + m * 3));
        LoadAlign(vreg_scale5, ulmax2 + (i * ulmaxLoopRow + m * 4));
        LoadAlign(vreg_scale6, ulmax2 + (i * ulmaxLoopRow + m * 5));
        LoadAlign(vreg_scale7, ulmax2 + (i * ulmaxLoopRow + m * 6));
        LoadAlign(vreg_scale8, ulmax2 + (i * ulmaxLoopRow + m * 7));

        Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_all16);
        Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_all16);
        Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_all16);
        Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_all16);
        Sub(vreg_scale5, vreg_scale5, vreg_gmax, preg_all16);
        Sub(vreg_scale6, vreg_scale6, vreg_gmax, preg_all16);
        Sub(vreg_scale7, vreg_scale7, vreg_gmax, preg_all16);
        Sub(vreg_scale8, vreg_scale8, vreg_gmax, preg_all16);

        // SAT Mode, if < 0, then = 0
        RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4, vreg_mxscale5, vreg_mxscale6,
            vreg_mxscale7, vreg_mxscale8;
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_all16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_all16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_all16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_all16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale5, vreg_scale5, preg_all16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale6, vreg_scale6, preg_all16);
        Cast<uint8_t, half, castTraitZero>(vreg_mxscale7, vreg_scale7, preg_all16);
        Cast<uint8_t, half, castTraitOne>(vreg_mxscale8, vreg_scale8, preg_all16);

        Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_all8);
        Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_all8);
        Or(vreg_mxscale5, vreg_mxscale5, vreg_mxscale6, preg_all8);
        Or(vreg_mxscale7, vreg_mxscale7, vreg_mxscale8, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 0, vreg_mxscale1, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 1, vreg_mxscale3, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 2, vreg_mxscale5, 5, preg_all8);

        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(
            mxscale2 + i * 32 * 5 * 8 + 32 * 3, vreg_mxscale7, 5, preg_all8);
    }
}

template <bool clear_gmax, typename T, uint16_t S1Base = 128>
__aicore__ inline void computePscale(const LocalTensor<uint8_t> &mxscale1, const LocalTensor<uint8_t> &mxscale2,
    const LocalTensor<T> &ulmax1, const LocalTensor<T> &ulmax2, const LocalTensor<T> &umax1,
    const LocalTensor<T> &umax2, const LocalTensor<T> &ugmaxOld, const LocalTensor<float> &urs, uint16_t firstLoop,
    uint16_t secondLoop) {
    __ubuf__ uint8_t *mxscale1VF = (__ubuf__ uint8_t *)mxscale1.GetPhyAddr();
    __ubuf__ uint8_t *mxscale2VF = (__ubuf__ uint8_t *)mxscale2.GetPhyAddr();
    __ubuf__ T *ulmax1VF = (__ubuf__ T *)ulmax1.GetPhyAddr();
    __ubuf__ T *ulmax2VF = (__ubuf__ T *)ulmax2.GetPhyAddr();
    __ubuf__ T *umax1VF = (__ubuf__ T *)umax1.GetPhyAddr();
    __ubuf__ T *umax2VF = (__ubuf__ T *)umax2.GetPhyAddr();
    __ubuf__ T *ugmaxOldVF = (__ubuf__ T *)ugmaxOld.GetPhyAddr();
    __ubuf__ float *ursVF = (__ubuf__ float *)urs.GetPhyAddr();

    computePscaleVF<clear_gmax, T, S1Base>(
        mxscale1VF, mxscale2VF, ulmax1VF, ulmax2VF, umax1VF, umax2VF, ugmaxOldVF, ursVF, firstLoop, secondLoop);
}
}
#endif // COMPUTESCALE_DN_MXFP4_H_
