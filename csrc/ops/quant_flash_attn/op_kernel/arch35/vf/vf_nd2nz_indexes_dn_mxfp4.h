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
 * \file vf_nd2nz_indexes_dn_mxfp4.h
 * \brief
 */

#ifndef ND2NZ_INDEXES_DN_MXFP4_H_
#define ND2NZ_INDEXES_DN_MXFP4_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <typename T>
__simd_vf__ void inline vf_init_indexs_and_duplicate(
    __ubuf__ uint8_t *index_nd2xz, __ubuf__ T *dupDest1, __ubuf__ T *dupDest2) {
    // Index
    RegTensor<uint8_t> index_reg, index_reg_1, index_reg_2;
    MaskReg mask_32 = CreateMask<int8_t, MaskPattern::VL32>();

    Arange((RegTensor<int8_t> &)index_reg, 0);

    ShiftLefts(index_reg, index_reg, NUM_2, mask_32);
    for (uint16_t i = 0; i < 4; ++i) {
        Adds(index_reg_1, index_reg, i, mask_32);
        Adds(index_reg_2, index_reg_1, NUM_128, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i, (RegTensor<uint8_t> &)index_reg_1, mask_32);
        StoreAlign(index_nd2xz + indexSubLength * i + NUM_128, (RegTensor<uint8_t> &)index_reg_2, mask_32);
    }

    // Duplicate
    RegTensor<T> src;
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    Duplicate(src, MIN_VALUE);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(dupDest1, src, preg_all_16bit);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(dupDest2, src, preg_all_16bit);
}

template <typename T, uint16_t S1Base = 128>
__aicore__ inline void InitIndexesAndDuplicateCallVF(
    LocalTensor<uint8_t> &nd2nzIndexes, const LocalTensor<T> &localGlobalMaxUB) {
    __ubuf__ uint8_t *index_nd2xz = (__ubuf__ uint8_t *)nd2nzIndexes.GetPhyAddr();

    __ubuf__ T *localGlobalMax1Buf = (__ubuf__ T *)localGlobalMaxUB.GetPhyAddr();
    __ubuf__ T *localGlobalMax2Buf = (__ubuf__ T *)localGlobalMaxUB[S1Base].GetPhyAddr();
    vf_init_indexs_and_duplicate<T>(index_nd2xz, localGlobalMax1Buf, localGlobalMax2Buf);
}

}
#endif // ND2NZ_INDEXES_DN_MXFP4_H_
