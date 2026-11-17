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

#pragma once
#include "eagle_split_norm_rope_concat_reg_950.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace eagle_snrc {
namespace reg950 {
enum class RotarySpan { LowPartial, LowFull, BothPartial, BothFull };

template <RotarySpan span, bool halfMode, typename T>
__simd_callee__ inline void RotateStore(__ubuf__ T *out, V::RegTensor<float> &lo, V::RegTensor<float> &hi,
    V::RegTensor<uint32_t> &piLo, V::RegTensor<uint32_t> &piHi, V::MaskReg &mixed, V::MaskReg &activeLo,
    V::MaskReg &activeHi, V::MaskReg &all, V::RegTensor<float> &cl, V::RegTensor<float> &ch, V::RegTensor<float> &sl,
    V::RegTensor<float> &sh) {
    // HALF/R128 pairs matching lanes of lo and hi directly. At R96 only the
    // low half can need partners from both registers; adjacent INTERLEAVE
    // pairs always stay in one register.
    V::RegTensor<float> a, b, direct;
    if constexpr (halfMode && span == RotarySpan::BothFull) {
        V::Mul(a, hi, sl, all);
        V::Mul(b, lo, sh, all);
    } else {
        if constexpr (halfMode && span == RotarySpan::BothPartial) {
            Gather128(a, lo, hi, piLo, mixed);
        } else {
            V::Gather(a, lo, piLo);
        }
        V::Mul(a, a, sl, activeLo);
        if constexpr (span == RotarySpan::BothFull || span == RotarySpan::BothPartial) {
            if constexpr (halfMode) {
                V::Gather(b, lo, piHi);
            } else {
                V::Gather(b, hi, piHi);
            }
            V::Mul(b, b, sh, activeHi);
        }
    }
    V::Mul(direct, lo, cl, activeLo);
    V::Add(a, direct, a, activeLo);
    if constexpr (span == RotarySpan::LowPartial) {
        V::Select(a, a, lo, activeLo);
    }
    Store64(out, a, all);
    if constexpr (span == RotarySpan::BothFull || span == RotarySpan::BothPartial) {
        V::Mul(direct, hi, ch, activeHi);
        V::Add(b, direct, b, activeHi);
        if constexpr (span == RotarySpan::BothPartial) {
            V::Select(b, b, hi, activeHi);
        }
        Store64(out + LANES, b, all);
    } else {
        Store64(out + LANES, hi, all);
    }
}

template <typename T, RotarySpan span, bool halfMode, bool affine>
__simd_vf__ inline void ComputeFour(__ubuf__ T *output, __ubuf__ T *input, __ubuf__ T *weight, __ubuf__ T *cosine,
    __ubuf__ T *sine, __ubuf__ uint32_t *partnerIndices, __ubuf__ uint32_t *cacheIndices, __ubuf__ float *signs,
    uint16_t rows, uint32_t rotaryDim, uint32_t ropeWidth, float eps, uint32_t inputHeadStride,
    uint32_t outputHeadStride) {
    V::MaskReg all = V::CreateMask<float, V::MaskPattern::ALL>();
    uint32_t remaining = rotaryDim;
    V::MaskReg activeLo = V::UpdateMask<float>(remaining);
    V::MaskReg activeHi = V::UpdateMask<float>(remaining);
    V::RegTensor<float> cosLo, cosHi, sinLo, sinHi;
    ExpandCache(cosLo, cosHi, cosine, cacheIndices, ropeWidth, all);
    ExpandCache(sinLo, sinHi, sine, cacheIndices, ropeWidth, all);
    {
        V::RegTensor<float> sign;
        V::LoadAlign(sign, signs);
        V::Mul(sinLo, sinLo, sign, all);
        V::LoadAlign(sign, signs + LANES);
        V::Mul(sinHi, sinHi, sign, all);
    }
    V::RegTensor<float> weightLo, weightHi;
    if constexpr (affine) {
        Load64(weightLo, weight, all);
        Load64(weightHi, weight + LANES, all);
    }
    V::RegTensor<uint32_t> partnerLo, partnerHi;
    V::MaskReg mixed;
    if constexpr (!(halfMode && span == RotarySpan::BothFull)) {
        V::LoadAlign(partnerLo, partnerIndices);
    }
    if constexpr (span == RotarySpan::BothPartial || (!halfMode && span == RotarySpan::BothFull)) {
        V::LoadAlign(partnerHi, partnerIndices + LANES);
    }
    if constexpr (halfMode && span == RotarySpan::BothPartial) {
        V::Compares<uint32_t, CMPMODE::GE>(mixed, partnerLo, LANES, all);
    }
    V::RegTensor<float> one;
    V::Duplicate(one, 1.0f);
    // Four heads from one Q or K token. Keep their RMS chains independent;
    // the head stride can differ between the input and the BNSD output.
    const uint16_t groups = rows / 4;
    const uint16_t tail = rows % 4;
    for (uint16_t g = 0; g < groups; ++g) {
        const uint32_t first = 4U * g;
        V::RegTensor<float> tmp;
        V::RegTensor<float> lo0, hi0, scale0;
        V::RegTensor<float> lo1, hi1, scale1;
        V::RegTensor<float> lo2, hi2, scale2;
        V::RegTensor<float> lo3, hi3, scale3;
        Load64(lo0, input + first * inputHeadStride, all);
        Load64(lo1, input + (first + 1) * inputHeadStride, all);
        Load64(lo2, input + (first + 2) * inputHeadStride, all);
        Load64(lo3, input + (first + 3) * inputHeadStride, all);
        Load64(hi0, input + first * inputHeadStride + LANES, all);
        Load64(hi1, input + (first + 1) * inputHeadStride + LANES, all);
        Load64(hi2, input + (first + 2) * inputHeadStride + LANES, all);
        Load64(hi3, input + (first + 3) * inputHeadStride + LANES, all);
        V::Mul(scale0, lo0, lo0, all);
        V::Mul(scale1, lo1, lo1, all);
        V::Mul(scale2, lo2, lo2, all);
        V::Mul(scale3, lo3, lo3, all);
        V::Mul(tmp, hi0, hi0, all);
        V::Add(scale0, scale0, tmp, all);
        V::Mul(tmp, hi1, hi1, all);
        V::Add(scale1, scale1, tmp, all);
        V::Mul(tmp, hi2, hi2, all);
        V::Add(scale2, scale2, tmp, all);
        V::Mul(tmp, hi3, hi3, all);
        V::Add(scale3, scale3, tmp, all);
        V::Reduce<V::ReduceType::SUM>(scale0, scale0, all);
        V::Reduce<V::ReduceType::SUM>(scale1, scale1, all);
        V::Reduce<V::ReduceType::SUM>(scale2, scale2, all);
        V::Reduce<V::ReduceType::SUM>(scale3, scale3, all);
        V::Duplicate(scale0, scale0, all);
        V::Duplicate(scale1, scale1, all);
        V::Duplicate(scale2, scale2, all);
        V::Duplicate(scale3, scale3, all);
        V::Muls(scale0, scale0, 1.0f / DIM, all);
        V::Muls(scale1, scale1, 1.0f / DIM, all);
        V::Muls(scale2, scale2, 1.0f / DIM, all);
        V::Muls(scale3, scale3, 1.0f / DIM, all);
        V::Adds(scale0, scale0, eps, all);
        V::Adds(scale1, scale1, eps, all);
        V::Adds(scale2, scale2, eps, all);
        V::Adds(scale3, scale3, eps, all);
        V::Sqrt(scale0, scale0, all);
        V::Sqrt(scale1, scale1, all);
        V::Sqrt(scale2, scale2, all);
        V::Sqrt(scale3, scale3, all);
        V::Div(scale0, one, scale0, all);
        V::Div(scale1, one, scale1, all);
        V::Div(scale2, one, scale2, all);
        V::Div(scale3, one, scale3, all);
        V::Mul(lo0, lo0, scale0, all);
        V::Mul(lo1, lo1, scale1, all);
        V::Mul(lo2, lo2, scale2, all);
        V::Mul(lo3, lo3, scale3, all);
        V::Mul(hi0, hi0, scale0, all);
        V::Mul(hi1, hi1, scale1, all);
        V::Mul(hi2, hi2, scale2, all);
        V::Mul(hi3, hi3, scale3, all);
        if constexpr (affine) {
            V::Mul(lo0, lo0, weightLo, all);
            V::Mul(lo1, lo1, weightLo, all);
            V::Mul(lo2, lo2, weightLo, all);
            V::Mul(lo3, lo3, weightLo, all);
            V::Mul(hi0, hi0, weightHi, all);
            V::Mul(hi1, hi1, weightHi, all);
            V::Mul(hi2, hi2, weightHi, all);
            V::Mul(hi3, hi3, weightHi, all);
        }
        RotateStore<span, halfMode>(output + first * outputHeadStride, lo0, hi0, partnerLo, partnerHi, mixed, activeLo,
            activeHi, all, cosLo, cosHi, sinLo, sinHi);
        RotateStore<span, halfMode>(output + (first + 1) * outputHeadStride, lo1, hi1, partnerLo, partnerHi, mixed,
            activeLo, activeHi, all, cosLo, cosHi, sinLo, sinHi);
        RotateStore<span, halfMode>(output + (first + 2) * outputHeadStride, lo2, hi2, partnerLo, partnerHi, mixed,
            activeLo, activeHi, all, cosLo, cosHi, sinLo, sinHi);
        RotateStore<span, halfMode>(output + (first + 3) * outputHeadStride, lo3, hi3, partnerLo, partnerHi, mixed,
            activeLo, activeHi, all, cosLo, cosHi, sinLo, sinHi);
    }
    // Do not read or write padding to complete a four-head group.
    for (uint16_t t = 0; t < tail; ++t) {
        const uint32_t first = 4U * groups + t;
        V::RegTensor<float> tmp;
        V::RegTensor<float> lo0, hi0, scale0;
        Load64(lo0, input + first * inputHeadStride, all);
        Load64(hi0, input + first * inputHeadStride + LANES, all);
        V::Mul(scale0, lo0, lo0, all);
        V::Mul(tmp, hi0, hi0, all);
        V::Add(scale0, scale0, tmp, all);
        V::Reduce<V::ReduceType::SUM>(scale0, scale0, all);
        V::Duplicate(scale0, scale0, all);
        V::Muls(scale0, scale0, 1.0f / DIM, all);
        V::Adds(scale0, scale0, eps, all);
        V::Sqrt(scale0, scale0, all);
        V::Div(scale0, one, scale0, all);
        V::Mul(lo0, lo0, scale0, all);
        V::Mul(hi0, hi0, scale0, all);
        if constexpr (affine) {
            V::Mul(lo0, lo0, weightLo, all);
            V::Mul(hi0, hi0, weightHi, all);
        }
        RotateStore<span, halfMode>(output + first * outputHeadStride, lo0, hi0, partnerLo, partnerHi, mixed, activeLo,
            activeHi, all, cosLo, cosHi, sinLo, sinHi);
    }
}
} // namespace reg950
} // namespace eagle_snrc
#endif
