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
 * \file dynamic_mx_quant_common.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_COMMON_H
#define DYNAMIC_MX_QUANT_COMMON_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/platform.h"
#include "quant_four_over_six_a5_tilingdata.h"

namespace QuantFourOverSixA5 {

template <typename Tp, Tp v>
struct IntegralConstant {
    static constexpr Tp value = v;
};
using trueType = IntegralConstant<bool, true>;
using falseType = IntegralConstant<bool, false>;
template <typename, typename>
struct IsSame : public falseType {};
template <typename Tp>
struct IsSame<Tp, Tp> : public trueType {};

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIM2 = 2;
constexpr int64_t DIGIT_ZERO = 0;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;
constexpr int64_t DIGIT_SIXTY_THREE = 63;
constexpr float DIGIT_ZERO_FLOAT = 0.0;
constexpr float DIGIT_FOUR_FLOAT = 4.0; // dst_type_max=4: per-block 4/6 自适应触发值
constexpr float DIGIT_SIX_FLOAT = 6.0;
constexpr float DIGIT_SEVEN_FLOAT = 7.0;
constexpr int64_t ModeZero = 0;
constexpr int64_t ModeOne = 1;
constexpr int64_t ModeTwo = 2;
constexpr int64_t ModeThree = 3;

constexpr uint32_t vfLen16 = platform::GetVRegSize() / sizeof(uint16_t);
constexpr uint32_t vfLen16Double = vfLen16 * 2;
constexpr uint32_t vfLen32 = platform::GetVRegSize() / sizeof(uint32_t);
constexpr int64_t UBBlockSize_ = platform::GetUbBlockSize();
constexpr uint16_t elementAfterReduce_ = platform::GetVRegSize() / UBBlockSize_;
constexpr uint16_t elementAfterReduceHalf_ = elementAfterReduce_ / 2;   // DeInterleave<uint8_t>→StoreUnAlign<uint16_t> 半量写出

constexpr uint16_t ADD_VALUE_FOR_BF16_MAN1 = 0x003f; // dst_type_max=6 (ceil)
constexpr uint16_t ADD_VALUE_FOR_BF16_MAN2 = 0x001f; // dst_type_max=7
constexpr uint16_t ADD_VALUE_FOR_BF16_D4 = 0x007f;   // dst_type_max=4 (ceil)
// round-to-nearest scale 常量(替代 ceil, 使 D6 有时饱和 → D4 有机会胜出)
constexpr uint16_t ADD_VALUE_FOR_BF16_D6_ROUND = 0x0078; // D=6 round: carry @ mant≥8
constexpr uint16_t ADD_VALUE_FOR_BF16_D4_ROUND = 0x004A; // D=4 round: carry @ mant≥54
constexpr uint16_t SUB_OFFSET_FOR_D6_ROUND     = 0x0180; // round(log2(6))=3
constexpr uint16_t SUB_OFFSET_FOR_D4_ROUND     = 0x0100; // round(log2(4))=2
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr int64_t OUT_ELE_NUM_TWO_BLK = 2 * OUT_ELE_NUM_ONE_BLK;   // 128: 单次 PACK_B16 store 覆盖两半(uint16 DeInterleave 后 Select 一次)
constexpr int64_t OUT_ELE_NUM_TWO_BLK_X2 = 2 * OUT_ELE_NUM_TWO_BLK; // 256: DINTLV_B16 y46 load stride (2×128 uint16)
constexpr int64_t OUT_ELE_NUM_ONE_BLK_FP8 = 32;
constexpr uint16_t elementAfterReduceDouble_ = elementAfterReduce_ * 2; // 16: INTLV_B16 half/suint stride (2×elementAfterReduce_)
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint16_t MAX_EXP_FOR_FP8 = 0x00ff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t THRESHOLD_E2M1 = 0x0100;
constexpr uint16_t THRESHOLD_E1M2 = 0x0080;
constexpr uint16_t NEW_MANTISSA = 0x0008;
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040;
constexpr uint32_t SPECIAL_EXP_THRESHOLD_FP32 = 0x00400000;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr int16_t SHR_NUM_FOR_BF16_EXP = 8;   // BF16 指数位左移(StoreScaleSelect: scale<<8 放 BF16 指数位; int16_t 满足 ShiftLefts scalar 要求)
// 1-bit 决策阈值(ComputeScaleDdr46 派生 D4 from D6): decision = (m<8) OR (m≥54)
//   m = BF16 尾数(bits 0-6). m<8 或 m≥54 → scaleE8M0_4 = scaleE8M0_6+1; 否则相等.
//   数学: scaleE8M0_4 − scaleE8M0_6 = 1 + [m≥54] − [m≥8] ∈ {0,1}(因 m≥54⇒m≥8). 见 docs/dst_type_max_4_scale_1bit_decision.md
//   uint16_t 以与输入 tensor(uint16_t)及 NEW_MANTISSA(0x0008)模式一致: CompareScalar<T> 标量类型应匹配 tensor 类型 T.
constexpr uint16_t MANT_THR_D6_CARRY = 8;    // D6 round carry 阈值(m≥8 进位); 1-bit 决策用 m<MANT_THR_D6_CARRY
constexpr uint16_t MANT_THR_D4_CARRY = 54;   // D4 round carry 阈值(m≥54 进位); 1-bit 决策用 m≥MANT_THR_D4_CARRY
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint32_t FP4_E2M1_FP32_MAX_EXP = 0x01000000;
constexpr uint16_t EXP_BF16_BIAS = 0x7f00;
constexpr uint32_t EXP_FP32_BIAS = 0x7f000000;
constexpr int64_t MODE_ROUND = 0;
constexpr int64_t MODE_FLOOR = 1;
constexpr int64_t MODE_RINT = 4;
constexpr uint16_t FP4_E1M2_MAX_EXP = 0x0000;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7)
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t NEG_ONE = -1;
constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr int32_t SCALE_BUFFER_SIZE = 16 * 1024;
constexpr int32_t MAX_MTE_BLOCK_COUNT = 4095;
constexpr uint16_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr uint16_t ABS_MASK_FOR_16BIT = 0x7fff;
constexpr uint32_t ABS_MASK_FOR_32BIT = 0x7fffffff;
constexpr uint32_t SUB_NUM_FOR_SCALE_32BIT = 0x000000e1;
constexpr uint16_t SUB_NUM_FOR_SCALE_16BIT = 0x00e1;
constexpr uint32_t MAN_MASK_FLOAT = 0x007fffff;
constexpr uint32_t FP32_EXP_BIAS_CUBLAS = 0x00007f00;
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr float FP8_E4M3_INV_MAX = 0.002232142857;
constexpr float FP8_E5M2_INV_MAX = 0.000017438616;
constexpr uint16_t BF16_MAX = 0x7fff;
constexpr uint16_t HALF_INF = 0x7c00;
constexpr uint16_t MAX_EXP_FOR_FP16 = 0x7c00;
constexpr uint16_t INVALID_FLOAT16 = 0x7c00;
constexpr uint16_t HALF_NEG_INF = 0xfc00;
constexpr uint16_t BF16_INF = 0x7f80;
constexpr uint16_t BF16_NEG_INF = 0xff80;
constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
constexpr uint32_t MAN_FOR_FP32 = 0x007fffff;
constexpr float INV_DTYPE_MAX = 0;
constexpr uint32_t MAN_MASK_FP32 = 0x007fffff;

template <typename T>
__aicore__ inline constexpr int16_t GetShrNum()
{
    if constexpr (IsSame<T, bfloat16_t>::value) {
        return SHR_NUM_FOR_BF16;
    } else {
        return SHR_NUM_FOR_FP32;
    }
}

template <typename T>
__aicore__ inline constexpr T GetMaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return MAX_EXP_FOR_BF16;
    } else {
        return MAX_EXP_FOR_FP32;
    }
}

template <typename T>
__aicore__ inline constexpr T GetabsForX()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return ABS_MASK_FOR_16BIT;
    } else {
        return ABS_MASK_FOR_32BIT;
    }
}

template <typename T>
__aicore__ inline constexpr T GetSubNumForScale()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return SUB_NUM_FOR_SCALE_32BIT;
    } else {
        return SUB_NUM_FOR_SCALE_16BIT;
    }
}

template <typename T>
__aicore__ inline constexpr T GetFp4MaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return FP4_E2M1_BF16_MAX_EXP;
    } else {
        return FP4_E2M1_FP32_MAX_EXP;
    }
}

template <typename T>
__aicore__ inline constexpr T GetFp8MaxExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return MAX_EXP_FOR_FP8;
    } else {
        return MAX_EXP_FOR_FP8_IN_FP32;
    }
}

template <typename T>
__aicore__ inline constexpr T GetMaxBias()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return EXP_BF16_BIAS;
    } else {
        return EXP_FP32_BIAS;
    }
}

template <typename T>
__aicore__ inline constexpr T GetNanValue()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return NAN_CUSTOMIZATION;
    } else {
        return NAN_CUSTOMIZATION_FP32;
    }
}

template <typename T>
__aicore__ inline constexpr T GetSpecialExp()
{
    if constexpr (IsSame<T, uint16_t>::value) {
        return SPECIAL_EXP_THRESHOLD;
    } else {
        return SPECIAL_EXP_THRESHOLD_FP32;
    }
}

template <AscendC::RoundMode roundMode, typename outType, typename inType>
__aicore__ inline void CalcElement(
    AscendC::Reg::RegTensor<inType>& in, AscendC::Reg::RegTensor<int32_t>& maxEle, AscendC::Reg::MaskReg mask)
{
    AscendC::Reg::RegTensor<float> y1;
    AscendC::Reg::MaskReg negValueMask;
    AscendC::Reg::MaskReg zeroMask;
    AscendC::Reg::MaskReg negZeroMask;
    AscendC::Reg::MaskReg zeroNegMask;
    AscendC::Reg::RegTensor<int32_t> negZero;
    AscendC::Reg::Duplicate(negZero, NEG_ZERO);
    AscendC::Reg::CompareScalar<int32_t, AscendC::CMPMODE::EQ>(
        zeroNegMask, (AscendC::Reg::RegTensor<int32_t>&)in, NEG_ZERO, mask);
    if constexpr (IsSame<outType, fp4x2_e2m1_t>::value) {
        AscendC::Reg::RegTensor<int32_t> exp1;
        AscendC::Reg::RegTensor<int32_t> exp2;
        AscendC::Reg::And(exp1, (AscendC::Reg::RegTensor<int32_t>&)in, maxEle, mask);
        AscendC::Reg::ShiftRights(exp1, exp1, SHR_NUM_FOR_FP32, mask);
        AscendC::Reg::Adds(exp1, exp1, FP32_BIAS_NEG, mask);
        AscendC::Reg::Maxs(exp1, exp1, 0, mask);
        AscendC::Reg::Adds(exp1, exp1, NEG_ONE, mask);
        AscendC::Reg::Muls(exp2, exp1, NEG_ONE, mask);
        AscendC::Reg::Adds(exp2, exp2, FP32_BIAS, mask);
        AscendC::Reg::ShiftLefts(exp2, exp2, SHR_NUM_FOR_FP32, mask);

        AscendC::Reg::Mul(y1, in, (AscendC::Reg::RegTensor<float>&)exp2, mask);
        AscendC::Reg::Adds(exp1, exp1, FP32_BIAS, mask);
        AscendC::Reg::ShiftLefts(exp1, exp1, SHR_NUM_FOR_FP32, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::Reg::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::Reg::Mul(in, y1, (AscendC::Reg::RegTensor<float>&)exp1, mask);
    } else {
        AscendC::Reg::Muls(y1, in, FOUR, mask);
        AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(negValueMask, y1, 0, mask);
        AscendC::Reg::Truncate<float, roundMode>(y1, y1, mask);
        AscendC::Reg::Muls(in, y1, ONE_FOURTH, mask);
    }
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(zeroMask, in, 0, mask);
    AscendC::Reg::MaskAnd(negZeroMask, zeroMask, negValueMask, mask);
    AscendC::Reg::MaskOr(zeroMask, negZeroMask, zeroNegMask, mask);
    AscendC::Reg::Copy((AscendC::Reg::RegTensor<int32_t>&)in, negZero, zeroMask);
}

template <AscendC::RoundMode roundMode, typename outType, typename inType, typename calcTypeInt>
__aicore__ inline void CalcElement(
    AscendC::Reg::RegTensor<inType>& in, AscendC::Reg::RegTensor<calcTypeInt>& scaleReprocal,
    AscendC::Reg::RegTensor<calcTypeInt>& maxEle, AscendC::Reg::RegTensor<uint8_t>& out, AscendC::Reg::MaskReg mask)
{
    static constexpr AscendC::Reg::CastTrait castTrait = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr AscendC::Reg::CastTrait castTraitFp32ToBf16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING, roundMode};
    AscendC::Reg::RegTensor<bfloat16_t> valueRegTensor;
    AscendC::Reg::RegTensor<outType> y;
    AscendC::Reg::RegTensor<uint16_t> yRegTensor;
    AscendC::Reg::Mul(in, in, (AscendC::Reg::RegTensor<inType>&)scaleReprocal, mask);
    if constexpr (IsSame<inType, float>::value) {
        CalcElement<roundMode, outType, inType>(in, (AscendC::Reg::RegTensor<int32_t>&)maxEle, mask);
        AscendC::Reg::Cast<bfloat16_t, inType, castTraitFp32ToBf16>(valueRegTensor, in, mask);
        AscendC::Reg::Pack(
            (AscendC::Reg::RegTensor<uint16_t>&)valueRegTensor, (AscendC::Reg::RegTensor<uint32_t>&)valueRegTensor);
        AscendC::Reg::Cast<outType, bfloat16_t, castTrait>(y, valueRegTensor, mask);
    } else {
        AscendC::Reg::Cast<outType, inType, castTrait>(y, in, mask);
    }

    AscendC::Reg::Pack(yRegTensor, (AscendC::Reg::RegTensor<uint32_t>&)y);
    AscendC::Reg::Pack(out, yRegTensor);
}

} // namespace QuantFourOverSixA5
#endif // DYNAMIC_MX_QUANT_COMMON_H
