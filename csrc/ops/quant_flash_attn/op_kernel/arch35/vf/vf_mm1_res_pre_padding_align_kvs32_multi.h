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
 * \file vf_softmax_dn_cast_nz_mxfp4.h
 * \brief
 */

#ifndef VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_H
#define VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_H
// #include "kernel_tensor.h"

#include "vf_common_def.h"

namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <typename T, uint16_t S1Base = 128>
__simd_vf__ inline void mm1_res_pre_padding_align_kvs32_nulti_vf(
    __ubuf__ T *s, uint16_t actSingleLoopS2Size, uint16_t actSingleLoopS2SizeAlign32) {
    // ====================== 寄存器定义 ======================
    MaskReg mask_reg;
    if constexpr (S1Base == 128) {
        mask_reg = CreateMask<uint16_t, MaskPattern::ALL>();
    } else {
        mask_reg = CreateMask<uint16_t, MaskPattern::VL128>();
    }
    uint16_t s2Idx = 0;
    // uint16_t idx2 = 0;
    RegTensor<T> padding_tensor1;
    RegTensor<T> padding_tensor2;
    Duplicate(padding_tensor1, MIN_VALUE, mask_reg);
    // Duplicate(padding_tensor2, MIN_VALUE, mask_reg);
    for (s2Idx = actSingleLoopS2Size; s2Idx < actSingleLoopS2SizeAlign32 - 1; s2Idx += 2) {
        StoreAlign(s + (s2Idx * S1Base) * 2, padding_tensor1, mask_reg);
        StoreAlign(s + (s2Idx * S1Base + 1 * S1Base) * 2, padding_tensor2, mask_reg);
    }

    for (uint16_t idx = s2Idx; idx < actSingleLoopS2SizeAlign32; ++idx) {
        StoreAlign(s + (idx * S1Base) * 2, padding_tensor1, mask_reg);
    }
}

template <typename T>
__aicore__ inline void Mm1ResPrePaddingAlignKvs32MultiCallVF(
    const LocalTensor<T> &srcTensor, uint16_t actSingleLoopS2Size, uint16_t actSingleLoopS2SizeAlign32) {
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();

    mm1_res_pre_padding_align_kvs32_nulti_vf<T>(input_x_local_UB, actSingleLoopS2Size, actSingleLoopS2SizeAlign32);
}

}
#endif // VF_MM1_RES_PRE_PADDING_ALIGN_KVS32_MULTI_H
