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
 * \file vf_basic_block_aligned128_no_update_sfa.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_SFA_H
#define VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_SFA_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_128_common_sfa.h"

using namespace regbaseutil;

namespace FaVectorApi {
// no update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__simd_vf__ void ProcessVec1NoUpdateImpl128VF(__ubuf__ T2 *expUb, __ubuf__ T *expSumUb, __ubuf__ T *maxUb,
    __ubuf__ T *maxUbStart, __ubuf__ T *srcUb, const uint32_t blockStride, const uint32_t repeatStride,
    const uint16_t m, const T scale, const T minValue) {
    AscendC::MicroAPI::RegTensor<float> vreg_input_x;
    AscendC::MicroAPI::RegTensor<float> vreg_input_x_unroll;
    AscendC::MicroAPI::RegTensor<float> vreg_max_tmp;
    AscendC::MicroAPI::RegTensor<float> vreg_input_max;
    AscendC::MicroAPI::RegTensor<float> vreg_max_brc;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_sum;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_even;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_odd;

    AscendC::MicroAPI::UnalignRegForStore ureg_max;
    AscendC::MicroAPI::UnalignRegForStore ureg_exp_sum;

    AscendC::MicroAPI::MaskReg preg_all = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg_all_b16 =
        AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

    for (uint16_t i = 0; i < m; ++i) {
        AlignedScaleStoreMax128<T>(
            vreg_input_x, vreg_input_x_unroll, vreg_max_tmp, srcUb, i, s2BaseSize, scale, preg_all);

        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_input_max, vreg_max_tmp, preg_all);
        AscendC::MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }

    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)maxUb), ureg_max, 0);
    AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        // maxUb is [S1, 1], BRC_B32 is reading one fp32 element and broadcast it to all 64 vreg element
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_DINTLV_B32>(
            vreg_input_x, vreg_input_x_unroll, srcUb + i * s2BaseSize);

        AscendC::MicroAPI::ExpSub(vreg_exp_even, vreg_input_x, vreg_max_brc, preg_all);
        AscendC::MicroAPI::ExpSub(vreg_exp_odd, vreg_input_x_unroll, vreg_max_brc, preg_all);

        // x_sum = sum(x_exp, axis=-1, keepdims=True)
        ExpSumReduceStore128<T>(vreg_exp_sum, vreg_exp_even, vreg_exp_odd, ureg_exp_sum, expSumUb, preg_all);

        CastStoreExp128<T, T2>(vreg_exp_even, vreg_exp_odd, expUb, blockStride, repeatStride, preg_all, preg_all_b16);
    }
    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

// no update, originN == 128
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__aicore__ inline void ProcessVec1NoUpdateImpl128(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor, const LocalTensor<T> &inMaxTensor,
    const LocalTensor<T> &sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue) {
    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();

    ProcessVec1NoUpdateImpl128VF<T, T2, s1BaseSize, s2BaseSize>(
        expUb, expSumUb, maxUb, maxUbStart, srcUb, blockStride, repeatStride, m, scale, minValue);
}
} // namespace

#endif // VF_BASIC_BLOCK_ALIGNED128_NO_UPDATE_SFA_H
