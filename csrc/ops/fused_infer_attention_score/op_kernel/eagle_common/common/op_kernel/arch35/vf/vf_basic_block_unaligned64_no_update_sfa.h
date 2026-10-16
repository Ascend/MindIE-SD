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
 * \file vf_basic_block_unaligned64_no_update_sfa.h
 * \brief
 */
#ifndef VF_BASIC_BLOCK_UNALIGNED64_NO_UPDATE_SFA_H
#define VF_BASIC_BLOCK_UNALIGNED64_NO_UPDATE_SFA_H

#include "vf_basic_block_utils.h"
#include "vf_basic_block_unaligned64_common_sfa.h"

using namespace regbaseutil;

namespace FaVectorApi {
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__simd_vf__ void ProcessVec1NoUpdateImpl64VF(__ubuf__ T2 *expUb, __ubuf__ T *expSumUb, __ubuf__ T *maxUb,
    __ubuf__ T *maxUbStart, __ubuf__ T *srcUb, const uint32_t blockStride, const uint32_t repeatStride,
    const uint16_t m, const T scale, const T minValue, uint32_t pltOriginalN, uint32_t pltSrcN) {
    AscendC::MicroAPI::RegTensor<float> vreg_min;
    AscendC::MicroAPI::RegTensor<float> vreg_input_x;
    AscendC::MicroAPI::RegTensor<float> vreg_input_max;
    AscendC::MicroAPI::RegTensor<float> vreg_max_brc;
    AscendC::MicroAPI::RegTensor<float> vreg_exp;
    AscendC::MicroAPI::RegTensor<float> vreg_exp_sum;

    AscendC::MicroAPI::UnalignRegForStore ureg_max;
    AscendC::MicroAPI::UnalignRegForStore ureg_exp_sum;

    AscendC::MicroAPI::MaskReg preg_all = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg_all_b16 =
        AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
    AscendC::MicroAPI::MaskReg preg_src_n = AscendC::MicroAPI::UpdateMask<float>(pltSrcN);
    AscendC::MicroAPI::MaskReg preg_src_n_b16 =
        AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::H>();
    AscendC::MicroAPI::MaskReg preg_ori_src_n = AscendC::MicroAPI::UpdateMask<T>(pltOriginalN);

    // x_max = max(src, axis=-1, keepdims=True)
    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        AscendC::MicroAPI::Muls(vreg_input_x, vreg_input_x, scale, preg_ori_src_n); // Muls(scale)
        AscendC::MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
            (__ubuf__ T *&)srcUb + i * s2BaseSize, vreg_input_x, preg_src_n);
        AscendC::MicroAPI::Reduce<MicroAPI::ReduceType::MAX, float, float, MicroAPI::MaskMergeMode::ZEROING>(
            vreg_input_max, vreg_input_x, preg_ori_src_n);
        AscendC::MicroAPI::StoreUnAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T *&)maxUb), vreg_input_max, ureg_max, 1);
    }
    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)maxUb), ureg_max, 0);
    AscendC::MicroAPI::LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < m; ++i) {
        AscendC::MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vreg_max_brc, maxUbStart + i);
        AscendC::MicroAPI::LoadAlign(vreg_input_x, srcUb + i * s2BaseSize);
        AscendC::MicroAPI::ExpSub(vreg_exp, vreg_input_x, vreg_max_brc, preg_ori_src_n);

        ExpSumReduceStore64<T>(vreg_exp_sum, vreg_exp, ureg_exp_sum, expSumUb, preg_ori_src_n);

        CastStoreExp64<T, T2>(vreg_exp, expUb, blockStride, repeatStride, preg_all_b16, preg_src_n_b16);
    }
    AscendC::MicroAPI::StoreUnAlignPost<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        ((__ubuf__ T *&)expSumUb), ureg_exp_sum, 0);
}

// no update, originN <= 64
template <typename T, typename T2, uint32_t s1BaseSize = 64, uint32_t s2BaseSize = 128>
__aicore__ inline void ProcessVec1NoUpdateImpl64(const LocalTensor<T2> &dstTensor, const LocalTensor<T> &srcTensor,
    const LocalTensor<T> &expSumTensor, const LocalTensor<T> &maxTensor, const LocalTensor<T> &inMaxTensor,
    const LocalTensor<T> &sharedTmpBuffer, const uint16_t m, const uint32_t originN, const T scale, const T minValue) {
    __ubuf__ T2 *expUb = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *expSumUb = (__ubuf__ T *)expSumTensor.GetPhyAddr();
    __ubuf__ T *maxUb = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *maxUbStart = (__ubuf__ T *)maxTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();

    // 写的时候固定用65或者33的stride去写，因为正向目前使能settail之后mm2的s1方向必须算满128或者64行
    // stride, high 16bits: blockStride (m*16*2/32), low 16bits: repeatStride (1)
    const uint32_t blockStride = s1BaseSize >> 1 | 0x1;
    const uint32_t repeatStride = 1;
    uint32_t pltOriginalN = originN;
    uint32_t pltSrcN = s2BaseSize;

    ProcessVec1NoUpdateImpl64VF<T, T2, s1BaseSize, s2BaseSize>(expUb, expSumUb, maxUb, maxUbStart, srcUb, blockStride,
        repeatStride, m, scale, minValue, pltOriginalN, pltSrcN);
}
} // namespace

#endif // VF_BASIC_BLOCK_UNALIGNED64_NO_UPDATE_SFA_H
