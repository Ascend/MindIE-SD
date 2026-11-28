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
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_mxfp4_attenout_dn.h
 * \brief
 */

#ifndef VF_COMMON_DEF_H_
#define VF_COMMON_DEF_H_
#include "kernel_tensor.h"

namespace Mxfp4Api {
using namespace AscendC;
using namespace MicroAPI;

#define VMULSCVT false
#define DROPOUT false

constexpr static AscendC::MicroAPI::CastTrait h2iZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait h2iOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitZero = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitOne = {
    AscendC::MicroAPI::RegLayout::ONE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitTwo = {
    AscendC::MicroAPI::RegLayout::TWO,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr static AscendC::MicroAPI::CastTrait castTraitThree = {
    AscendC::MicroAPI::RegLayout::THREE,
    AscendC::MicroAPI::SatMode::SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND,
};

constexpr half NUM_127 = static_cast<half>(127.0f);
constexpr half NUM_NEG_125 = static_cast<half>(-125.0f);
constexpr half ZERO_VALUE = static_cast<half>(0.0f);
constexpr int16_t SHIFT_VALUE = 23;

constexpr uint8_t NUM_128 = static_cast<uint8_t>(128);
constexpr int16_t NUM_2 = static_cast<int16_t>(2);
constexpr int8_t indexSubLength = static_cast<int8_t>(32);

constexpr half LN2 = static_cast<half>(0.6931471806f);
constexpr half INV_LN2 = static_cast<half>(1.4426950409f);
constexpr half NEG_TWO_VALE = static_cast<half>(-2.0f);
constexpr half TWO_VALE = static_cast<half>(2.0f);
constexpr half MIN_VALUE = static_cast<half>(-65504.0f);

constexpr float LN2_F32 = 0.6931471805599453f;
constexpr float INV_LN2_F32 = 1.4426950408889634f;

template <const CastTrait &Lane>
__simd_callee__ inline void LogitExponent(RegTensor<float> &exponent, RegTensor<half> &logit) {
    MaskReg all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg all32 = CreateMask<float, MaskPattern::ALL>();
    Cast<float, half, Lane>(exponent, logit, all16);
    Muls(exponent, exponent, INV_LN2_F32, all32);
    Truncate<float, RoundMode::CAST_FLOOR>(exponent, exponent, all32);
}

__simd_callee__ inline void PackHalfLanes(RegTensor<half> &packed, RegTensor<float> &even, RegTensor<float> &odd) {
    MaskReg all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg all32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<half> oddPacked;
    Cast<half, float, castTraitZero>(packed, even, all32);
    Cast<half, float, castTraitOne>(oddPacked, odd, all32);
    Or((RegTensor<uint16_t> &)packed, (RegTensor<uint16_t> &)packed, (RegTensor<uint16_t> &)oddPacked, all16);
}

template <const CastTrait &Lane>
__simd_callee__ inline void LogitRemainder(RegTensor<float> &remainder, RegTensor<half> &maximum) {
    MaskReg all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg all32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> exponent;
    LogitExponent<Lane>(exponent, maximum);
    Cast<float, half, Lane>(remainder, maximum, all16);
    Muls(exponent, exponent, LN2_F32, all32);
    Sub(remainder, remainder, exponent, all32);
    Adds(remainder, remainder, 2.0f * LN2_F32, all32);
}

__simd_callee__ inline void GroupLogitOffset(RegTensor<half> &offset, RegTensor<half> &maximum) {
    RegTensor<float> even, odd;
    LogitRemainder<castTraitZero>(even, maximum);
    LogitRemainder<castTraitOne>(odd, maximum);
    PackHalfLanes(offset, even, odd);
}

// Keep absolute maxima in the existing half buffers. The integer exponent
// difference fits the packed format after clamping, even at large logits.
__simd_callee__ inline void PackBiasedExponentDelta(RegTensor<half> &packed, RegTensor<half> &localMax,
    RegTensor<float> &globalEven, RegTensor<float> &globalOdd, float bias) {
    MaskReg all32 = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> even, odd;
    LogitExponent<castTraitZero>(even, localMax);
    LogitExponent<castTraitOne>(odd, localMax);
    Sub(even, even, globalEven, all32);
    Sub(odd, odd, globalOdd, all32);
    Adds(even, even, bias, all32);
    Adds(odd, odd, bias, all32);
    Maxs(even, even, 0.0f, all32);
    Maxs(odd, odd, 0.0f, all32);
    PackHalfLanes(packed, even, odd);
}

}
#endif // VF_COMMON_DEF_H_
