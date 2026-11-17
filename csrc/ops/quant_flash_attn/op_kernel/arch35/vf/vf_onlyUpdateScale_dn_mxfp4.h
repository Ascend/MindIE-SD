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
 * \file vf_onlyUpdateScale_dn_mxfp4.h
 * \brief
 */

#ifndef ONLYUPDATESCALE_DN_MXFP4_H_
#define ONLYUPDATESCALE_DN_MXFP4_H_
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
__simd_vf__ inline void computeOnlyScaleVF(
    __ubuf__ T *umax1, __ubuf__ T *umax2, __ubuf__ T *ugmax_old, __ubuf__ float *urs, uint16_t coreIndex) {
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_max2, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

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
        for (uint16_t i = 0; i < 1 - coreIndex; ++i) {
            StoreAlign(urs, vreg_rs1, preg_all16);
        }

        for (uint16_t i = 0; i < coreIndex; ++i) {
            StoreAlign(urs, vreg_rs2, preg_all16);
        }
    }
}

template <bool clear_gmax, typename T, uint16_t S1Base = 128>
__aicore__ inline void computeOnlyScale(const LocalTensor<T> &umax1, const LocalTensor<T> &umax2,
    const LocalTensor<T> &ugmaxOld, const LocalTensor<float> &urs, uint16_t coreIndex) {
    __ubuf__ T *umax1VF = (__ubuf__ T *)umax1.GetPhyAddr();
    __ubuf__ T *umax2VF = (__ubuf__ T *)umax2.GetPhyAddr();
    __ubuf__ T *ugmaxOldVF = (__ubuf__ T *)ugmaxOld.GetPhyAddr();
    __ubuf__ float *ursVF = (__ubuf__ float *)urs.GetPhyAddr();

    computeOnlyScaleVF<clear_gmax, T, S1Base>(umax1VF, umax2VF, ugmaxOldVF, ursVF, coreIndex);
}
}
#endif // ONLYUPDATESCALE_DN_MXFP4_H_
