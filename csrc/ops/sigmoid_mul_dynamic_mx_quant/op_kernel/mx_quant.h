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

/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Upstream: cann/ops-nn, quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis_fp8.h
// https://gitcode.com/cann/ops-nn/blob/a5a8fbf3e02d7632ffa2c56b29e8755f1aa62720/LICENSE
#pragma once
#include "kernel_operator.h"

namespace SigmoidMulMx {
using namespace AscendC;

constexpr uint32_t VF_LEN_16 = VECTOR_REG_WIDTH / sizeof(uint16_t);
constexpr uint32_t VF_LEN_32 = VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t VF_LEN_16_DOUBLE = 2 * VF_LEN_16;
constexpr uint32_t VF_LEN_32_DOUBLE = 2 * VF_LEN_32;
constexpr uint16_t ELEMENT_AFTER_REDUCE_ = VF_LEN_16_DOUBLE / 32;
constexpr uint16_t BF16_NAN_CUSTOM = 0x7f81;
constexpr uint16_t BF16_MAX_EXP = 0x7f80;
constexpr uint32_t FP32_MX_MAX_EXP = 0x7f800000;
constexpr uint16_t FP8_DEFAULT_MAX_EXP = 0x00ff;
constexpr uint16_t BF16_SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr int16_t BF16_SHR_NUM = 7;
constexpr int16_t FP32_SHR_NUM = 23;
constexpr int16_t FP32_PACK_SHR_NUM = 16;
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)

__aicore__ inline void CastBf16ScaleToFloat(
    Reg::RegTensor<float> &dst, Reg::RegTensor<uint16_t> &src, Reg::MaskReg &mask) {
    // 规避utest GCC ICE bug
    static constexpr Reg::CastTrait trait = {
        Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    Reg::Cast<float, bfloat16_t, trait>(dst, (Reg::RegTensor<bfloat16_t> &)src, mask);
}

template <bool CeilScale>
__aicore__ inline void ComputeGroupMaxBf16(
    __ubuf__ bfloat16_t *xLocalAddr, __ubuf__ uint16_t *maxExpAddr, uint16_t loopNum2VF) {
    __VEC_SCOPE__ {
        Reg::RegTensor<bfloat16_t> x0;
        Reg::RegTensor<bfloat16_t> x1;
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> xExpExtract0;
        Reg::RegTensor<uint16_t> xExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, CeilScale ? uint16_t(0x7fff) : BF16_MAX_EXP);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignRegForStore ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
            Reg::And(xExpExtract0, (Reg::RegTensor<uint16_t> &)x0, expMaskBF16, Mask);
            Reg::And(xExpExtract1, (Reg::RegTensor<uint16_t> &)x1, expMaskBF16, Mask);
            Reg::Max(xMaxExp, xExpExtract0, xExpExtract1, Mask);
            Reg::ReduceDataBlock<ReduceType::MAX>(xMaxExp, xMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, xMaxExp, ureg, ELEMENT_AFTER_REDUCE_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

template <typename MaxType>
__aicore__ inline void ComputeGroupMaxFp32(
    __ubuf__ float *xLocalAddr, __ubuf__ MaxType *maxExpAddr, uint16_t loopNum2VF) {
    __VEC_SCOPE__ {
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<uint32_t> x0MaxExpB32;
        Reg::RegTensor<uint32_t> x1MaxExpB32;
        Reg::RegTensor<uint32_t> xMaxExpB32;
        Reg::RegTensor<uint16_t> xMaxExpB16;

        Reg::RegTensor<uint32_t> expMask32Bit;
        Reg::Duplicate(expMask32Bit, sizeof(MaxType) == sizeof(uint32_t) ? 0x7fffffffU : FP32_MX_MAX_EXP);

        Reg::MaskReg MaskB32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignRegForStore ureg;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            // x0ZeroFP32: 1,3,5,7,9,...,127        x1ZeroFP32: 2,4,6,8,10,...,128
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0ZeroFP32, x1ZeroFP32, xLocalAddr, VF_LEN_32_DOUBLE);
            // x0OneFP32: 129,131,133,135,...,255  x1OneFP32: 130,132,134,136,...,256
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                x0OneFP32, x1OneFP32, xLocalAddr, VF_LEN_32_DOUBLE);
            // x0ZeroFP32: 1,5,9,13,...,253    x0OneFP32: 3,7,11,15,...,255
            Reg::DeInterleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
            // x1ZeroFP32: 2,6,10,14,...,254    x1OneFP32: 4,8,12,16,...,256
            Reg::DeInterleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
            Reg::And(
                (Reg::RegTensor<uint32_t> &)x0ZeroFP32, (Reg::RegTensor<uint32_t> &)x0ZeroFP32, expMask32Bit, MaskB32);
            Reg::And(
                (Reg::RegTensor<uint32_t> &)x0OneFP32, (Reg::RegTensor<uint32_t> &)x0OneFP32, expMask32Bit, MaskB32);
            Reg::And(
                (Reg::RegTensor<uint32_t> &)x1ZeroFP32, (Reg::RegTensor<uint32_t> &)x1ZeroFP32, expMask32Bit, MaskB32);
            Reg::And(
                (Reg::RegTensor<uint32_t> &)x1OneFP32, (Reg::RegTensor<uint32_t> &)x1OneFP32, expMask32Bit, MaskB32);
            Reg::Max(
                x0MaxExpB32, (Reg::RegTensor<uint32_t> &)x0ZeroFP32, (Reg::RegTensor<uint32_t> &)x0OneFP32, MaskB32);
            Reg::Max(
                x1MaxExpB32, (Reg::RegTensor<uint32_t> &)x1ZeroFP32, (Reg::RegTensor<uint32_t> &)x1OneFP32, MaskB32);
            Reg::Max(xMaxExpB32, x0MaxExpB32, x1MaxExpB32, MaskB32);
            Reg::ReduceDataBlock<ReduceType::MAX>(xMaxExpB32, xMaxExpB32, MaskB32);
            if constexpr (sizeof(MaxType) == sizeof(uint32_t)) {
                Reg::StoreUnAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, xMaxExpB32, ureg, ELEMENT_AFTER_REDUCE_);
            } else {
                Reg::ShiftRights(xMaxExpB32, xMaxExpB32, FP32_PACK_SHR_NUM, MaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(xMaxExpB16, xMaxExpB32);
                Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    maxExpAddr, xMaxExpB16, ureg, ELEMENT_AFTER_REDUCE_);
            }
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
}

__aicore__ inline void ComputeScaleOcp(__ubuf__ uint16_t *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
    __ubuf__ uint16_t *recipScaleLocalAddr, uint16_t loopNum1VF, uint32_t totalScaleInUB) {
    __VEC_SCOPE__ {
        Reg::RegTensor<uint16_t> xMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, FP8_E4M3_MAX_EXP);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanU16;
        Reg::Duplicate(fp8NanU16, FP8_DEFAULT_MAX_EXP);
        Reg::RegTensor<uint16_t> zeroU16;
        Reg::Duplicate(zeroU16, 0);
        Reg::RegTensor<uint16_t> nanU16;
        Reg::Duplicate(nanU16, BF16_NAN_CUSTOM);
        Reg::RegTensor<uint16_t> specialExpU16;
        Reg::Duplicate(specialExpU16, BF16_SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < loopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(xMaxExp, maxExpAddr, VF_LEN_16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, xMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, xMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(xMaxExp, maxExpValue, xMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, xMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, BF16_SHR_NUM, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanU16, cmpResult);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, VF_LEN_32,
                preMaskScale); // 128 个scale，占用 128 * 1 Btyes = VF_LEN_32 * sizeof(uint16_t)

            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, sharedExp, zeroU16, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanU16, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroU16, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpU16, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                recipScaleLocalAddr, halfScale, VF_LEN_16, preMaskScale);
        }
    }
}

// FP32 encoding of 1 / 448, the inverse maximum finite E4M3FN value.
constexpr uint32_t E4M3_INV_MAX_BITS = 0x3b124925;

template <typename MaxType>
__aicore__ inline void ComputeScaleCeil(__ubuf__ MaxType *maxExpAddr, __ubuf__ uint16_t *mxScaleLocalAddr,
    __ubuf__ uint16_t *recipScaleLocalAddr, uint16_t loopNumHalfVF) {
    __VEC_SCOPE__ {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, E4M3_INV_MAX_BITS);
        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, uint32_t(0x007fffff));
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, FP32_MX_MAX_EXP);
        Reg::RegTensor<uint32_t> zeroU32;
        Reg::Duplicate(zeroU32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, uint32_t(BF16_EXP_BIAS));
        Reg::RegTensor<uint32_t> nanU32;
        Reg::Duplicate(nanU32, uint32_t(BF16_NAN_CUSTOM));
        Reg::RegTensor<uint32_t> fp8NanU32;
        Reg::Duplicate(fp8NanU32, uint32_t(FP8_DEFAULT_MAX_EXP));

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);
        Reg::MaskReg maskFloat = Reg::CreateMask<uint32_t>();

        for (uint16_t i = 0; i < loopNumHalfVF; i++) {
            if constexpr (sizeof(MaxType) == sizeof(uint32_t)) {
                Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_NORM>(
                    max32, maxExpAddr, VF_LEN_32);
            } else {
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(
                    max16, maxExpAddr, VF_LEN_32);
                CastBf16ScaleToFloat((Reg::RegTensor<float> &)max32, max16, maskFloat);
            }
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroU32, maskFloat);

            Reg::Mul((Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)max32, (Reg::RegTensor<float> &)invMax,
                maskFloat);
            Reg::ShiftRights(exp32, max32, FP32_SHR_NUM, maskFloat);
            Reg::And(man32, max32, manMaskFP32, maskFloat);

            Reg::Compares<uint32_t, CMPMODE::GT>(p0, exp32, 0U, maskFloat);
            Reg::Compares<uint32_t, CMPMODE::LT>(p1, exp32, 254U, maskFloat);
            Reg::Compares<uint32_t, CMPMODE::GT>(p2, man32, 0U, maskFloat);
            Reg::And(p0, p0, p1, maskFloat);
            Reg::And(p0, p0, p2, maskFloat);

            Reg::Compares<uint32_t, CMPMODE::EQ>(p1, exp32, 0U, maskFloat);
            Reg::Compares<uint32_t, CMPMODE::GT>(p2, man32, 0x00400000U, maskFloat);
            Reg::And(p1, p1, p2, maskFloat);
            Reg::Or(p0, p0, p1, maskFloat);

            Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanU32, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            // 64 个 max 计算得到 64 * 1 Bytes = 32 * sizeof(uint16_t) Bytes
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * 32, expOut, dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, BF16_SHR_NUM, maskFloat);
            Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
            Reg::Select<uint32_t>(halfScale, halfScale, nanU32, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroU32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(recipScaleLocalAddr + i * VF_LEN_32, recExpOut, dataMaskB16Half);
        }
    }
}

// Preserve NaNs while saturating finite values outside the representable E4M3FN range.
__simd_callee__ inline void ClampE4M3(
    Reg::RegTensor<float> &value, Reg::RegTensor<float> &lo, Reg::RegTensor<float> &hi, Reg::MaskReg &mask) {
    Reg::MaskReg outside;
    Reg::Compare<float, CMPMODE::GT>(outside, value, hi, mask);
    Reg::Select<float>(value, hi, value, outside);
    Reg::Compare<float, CMPMODE::LT>(outside, value, lo, mask);
    Reg::Select<float>(value, lo, value, outside);
}

template <typename T, typename U = fp8_e4m3fn_t>
__aicore__ inline void ComputeData(
    __ubuf__ T *xLocalAddr, __ubuf__ uint16_t *recipScaleLocalAddr, __ubuf__ int8_t *yLocalAddr, uint16_t loopNum2VF) {
    __VEC_SCOPE__ {
        Reg::MaskReg dataMask1 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask2 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask3 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask4 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask5 = Reg::CreateMask<U>();
        Reg::MaskReg maskFP32 = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        Reg::RegTensor<float> lo, hi;
        Reg::Duplicate(lo, -448.0f);
        Reg::Duplicate(hi, 448.0f);
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<U> x0ZeroFP8;
        Reg::RegTensor<U> x0OneFP8;
        Reg::RegTensor<U> x1ZeroFP8;
        Reg::RegTensor<U> x1OneFP8;

        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to80 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to81 = {
            Reg::RegLayout::ONE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to82 = {
            Reg::RegLayout::TWO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to83 = {
            Reg::RegLayout::THREE, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            if constexpr (sizeof(T) == sizeof(float)) {
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
                CastBf16ScaleToFloat(floatScaleForMul, halfScaleForMul, maskFP32);
                // x0ZeroFP32: 1,3,5,7,9,...,127        x1ZeroFP32: 2,4,6,8,10,...,128
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    x0ZeroFP32, x1ZeroFP32, xLocalAddr, VF_LEN_32_DOUBLE);
                // x0OneFP32: 129,131,133,135,...,255  x1OneFP32: 130,132,134,136,...,256
                Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                    x0OneFP32, x1OneFP32, xLocalAddr, VF_LEN_32_DOUBLE);
                // x0ZeroFP32: 1,5,9,13,...,253    x0OneFP32: 3,7,11,15,...,255
                Reg::DeInterleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
                // x1ZeroFP32: 2,6,10,14,...,254    x1OneFP32: 4,8,12,16,...,256
                Reg::DeInterleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x0OneFP32, x0OneFP32, floatScaleForMul, maskFP32);
                Reg::Mul(x1OneFP32, x1OneFP32, floatScaleForMul, maskFP32);
            } else {
                Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                    x0, x1, xLocalAddr, VF_LEN_16_DOUBLE);
                Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                    halfScaleForMul, recipScaleLocalAddr, ELEMENT_AFTER_REDUCE_);
                Reg::Mul(x0, x0, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);
                Reg::Mul(x1, x1, (Reg::RegTensor<T> &)halfScaleForMul, dataMask1);

                Reg::Cast<float, T, castTraitZero>(x0ZeroFP32, x0, dataMask1);
                Reg::Cast<float, T, castTraitOne>(x0OneFP32, x0, dataMask1);
                Reg::Cast<float, T, castTraitZero>(x1ZeroFP32, x1, dataMask2);
                Reg::Cast<float, T, castTraitOne>(x1OneFP32, x1, dataMask2);
            }
            ClampE4M3(x0ZeroFP32, lo, hi, maskFP32);
            ClampE4M3(x1ZeroFP32, lo, hi, maskFP32);
            ClampE4M3(x0OneFP32, lo, hi, maskFP32);
            ClampE4M3(x1OneFP32, lo, hi, maskFP32);
            Reg::Cast<U, float, castTrait32to80>(x0ZeroFP8, x0ZeroFP32, dataMask3);
            Reg::Cast<U, float, castTrait32to81>(x1ZeroFP8, x1ZeroFP32, dataMask4);
            Reg::Cast<U, float, castTrait32to82>(x0OneFP8, x0OneFP32, dataMask3);
            Reg::Cast<U, float, castTrait32to83>(x1OneFP8, x1OneFP32, dataMask4);

            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                (Reg::RegTensor<uint8_t> &)x0OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t> &)x1ZeroFP8, (Reg::RegTensor<uint8_t> &)x1ZeroFP8,
                (Reg::RegTensor<uint8_t> &)x1OneFP8, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t> &)x0ZeroFP8, (Reg::RegTensor<uint8_t> &)x0ZeroFP8,
                (Reg::RegTensor<uint8_t> &)x1ZeroFP8, dataMask5);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yLocalAddr, (Reg::RegTensor<int8_t> &)x0ZeroFP8, VF_LEN_16_DOUBLE, dataMask5);
        }
    }
}

} // namespace SigmoidMulMx
