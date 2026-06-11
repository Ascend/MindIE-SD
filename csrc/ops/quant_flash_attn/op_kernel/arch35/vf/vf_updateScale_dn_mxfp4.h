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
 * \file vf_mxfp4_attenout_dn.h
 * \brief
 */

#ifndef UPDATESCALE_DN_MXFP4_H_
#define UPDATESCALE_DN_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

template <bool isUpdate, uint16_t S2Base = 256>
__simd_vf__ void processUpdateVF(__ubuf__ float *updateResUb, __ubuf__ float *mm2ResUb, __ubuf__ float *urs,
    __ubuf__ float *rsum, __ubuf__ float *ursum) {
    // 更新 mm2
    MaskReg preg_all32 = CreateMask<uint32_t, MaskPattern::ALL>();
    if constexpr (isUpdate) {
        RegTensor<float> vreg_rs, vreg_ursum, vreg_rsum;
        LoadAlign(vreg_rs, (__ubuf__ float *)urs);
        LoadAlign(vreg_rsum, (__ubuf__ float *)rsum);
        LoadAlign(vreg_ursum, (__ubuf__ float *)ursum);
        Mul(vreg_rsum, vreg_rsum, vreg_rs, preg_all32);
        Add(vreg_rsum, vreg_rsum, vreg_ursum, preg_all32);

        StoreAlign((__ubuf__ float *)rsum, vreg_rsum, preg_all32);

        for (uint16_t i = 0; i < S2Base / 2; i += 4) {
            RegTensor<float> vreg_pv1, vreg_pv2, vreg_pv3, vreg_pv4, vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float *)updateResUb + (i + 0) * 64);
            LoadAlign(vreg_out2, (__ubuf__ float *)updateResUb + (i + 1) * 64);
            LoadAlign(vreg_out3, (__ubuf__ float *)updateResUb + (i + 2) * 64);
            LoadAlign(vreg_out4, (__ubuf__ float *)updateResUb + (i + 3) * 64);

            Mul(vreg_out1, vreg_out1, vreg_rs, preg_all32);
            Mul(vreg_out2, vreg_out2, vreg_rs, preg_all32);
            Mul(vreg_out3, vreg_out3, vreg_rs, preg_all32);
            Mul(vreg_out4, vreg_out4, vreg_rs, preg_all32);

            LoadAlign(vreg_pv1, (__ubuf__ float *)mm2ResUb + (i + 0) * 64);
            LoadAlign(vreg_pv2, (__ubuf__ float *)mm2ResUb + (i + 1) * 64);
            LoadAlign(vreg_pv3, (__ubuf__ float *)mm2ResUb + (i + 2) * 64);
            LoadAlign(vreg_pv4, (__ubuf__ float *)mm2ResUb + (i + 3) * 64);

            Add(vreg_out1, vreg_out1, vreg_pv1, preg_all32);
            Add(vreg_out2, vreg_out2, vreg_pv2, preg_all32);
            Add(vreg_out3, vreg_out3, vreg_pv3, preg_all32);
            Add(vreg_out4, vreg_out4, vreg_pv4, preg_all32);

            StoreAlign((__ubuf__ float *)updateResUb + (i + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 3) * 64, vreg_out4, preg_all32);
        }
    } else {
        RegTensor<float> vreg_grsum;
        LoadAlign(vreg_grsum, (__ubuf__ float *)ursum);
        StoreAlign((__ubuf__ float *)rsum, vreg_grsum, preg_all32);

        for (uint16_t i = 0; i < S2Base / 2; i += 4) {
            RegTensor<float> vreg_out1, vreg_out2, vreg_out3, vreg_out4;
            LoadAlign(vreg_out1, (__ubuf__ float *)mm2ResUb + (i + 0) * 64);
            LoadAlign(vreg_out2, (__ubuf__ float *)mm2ResUb + (i + 1) * 64);
            LoadAlign(vreg_out3, (__ubuf__ float *)mm2ResUb + (i + 2) * 64);
            LoadAlign(vreg_out4, (__ubuf__ float *)mm2ResUb + (i + 3) * 64);

            StoreAlign((__ubuf__ float *)updateResUb + (i + 0) * 64, vreg_out1, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 1) * 64, vreg_out2, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 2) * 64, vreg_out3, preg_all32);
            StoreAlign((__ubuf__ float *)updateResUb + (i + 3) * 64, vreg_out4, preg_all32);
        }
    }
}

template <bool isUpdate, uint16_t S2Base = 256>
__aicore__ inline void processUpdate(const LocalTensor<float> &updateResUb, const LocalTensor<float> &mm2ResUb,
    const LocalTensor<float> &urs, const LocalTensor<float> &rsum, const LocalTensor<float> &ursum) {
    __ubuf__ float *updateResUbVF = (__ubuf__ float *)updateResUb.GetPhyAddr();
    __ubuf__ float *mm2ResUbVF = (__ubuf__ float *)mm2ResUb.GetPhyAddr();
    __ubuf__ float *ursVF = (__ubuf__ float *)urs.GetPhyAddr();
    __ubuf__ float *rsumVF = (__ubuf__ float *)rsum.GetPhyAddr();
    __ubuf__ float *ursumVF = (__ubuf__ float *)ursum.GetPhyAddr();

    processUpdateVF<isUpdate, S2Base>(updateResUbVF, mm2ResUbVF, ursVF, rsumVF, ursumVF);
}

} // namespace Mxfp4Api
#endif // UPDATESCALE_DN_MXFP4_H_
