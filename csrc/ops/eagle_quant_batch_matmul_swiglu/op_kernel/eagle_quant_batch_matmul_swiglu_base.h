/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
 * \file eagle_quant_batch_matmul_swiglu_base.h
 * \brief
 */
#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASE_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASE_H

#include "deps/common/inc/op_kernel/macro.h"
#include <cstdint>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "quantization/ascend_dequant_utils.h"
#include "pad/broadcast.h"
#include "quantization/ascend_dequant.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "kernel_type.h"
#include "lib/matmul_intf.h"
#include "eagle_quant_batch_matmul_swiglu_kernel_tiling_data.h"

#define TemplateBasicTypeForClass \
    typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, int x2Format, bool aTrans, \
        bool bTrans, class UPDATE_TYPE, class EpilogueType = EpilogueDequant
#define TemplateBasicType \
    typename x1Type, typename x2Type, typename scaleType, typename yType, int x1Format, int x2Format, bool aTrans, \
        bool bTrans, class UPDATE_TYPE, class EpilogueType
#define TemplateBasicValue \
    x1Type, x2Type, scaleType, yType, x1Format, x2Format, aTrans, bTrans, UPDATE_TYPE, EpilogueType

constexpr uint32_t BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_INT4 = 64;
constexpr uint32_t K0_INT8 = 32;
constexpr uint32_t k0_FLOAT16 = 16;
constexpr uint32_t k0_FLOAT32 = 8;
constexpr int FORMAT_FRACTAL_NZ_INT = 29;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t M_N_TWO_DIMS = 2;

const uint32_t ROW_FIRST = 1;
const uint32_t COL_FIRST = 2;

constexpr uint16_t C2V_PING_FLAG = 0x4;
constexpr uint16_t C2V_PONG_FLAG = 0x5;
constexpr uint16_t V2C_PING_FLAG = 0x6;
// asw const
constexpr uint16_t FLAG_ID_MAX = 16;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 8;
constexpr uint8_t AIC_SYNC_AIV_MODE = 4;
constexpr uint8_t CV_RATIO = 2;
constexpr int32_t MXFP_GROUP_SIZE = 32;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr uint16_t DATA_BLOCK = 32;
const uint32_t FP32_OUTPUT_TIMES = 4;
const uint64_t INT1_X2_OFFSET_FACTOR_8 = 8;

// iterBatch const
constexpr uint64_t L0C_SIZE_256K = 256 * 1024UL;
constexpr uint8_t A_NEED_BROADCAST = 1;
constexpr uint8_t B_NEED_BROADCAST = 2;
constexpr MatmulConfigMode configMode = MatmulConfigMode::CONFIG_NORM;
constexpr MatmulBatchParams batchParams{false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::MULTI_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH = GetMMConfig<configMode>(batchParams);
constexpr MatmulBatchParams batchParamsNoBatchOut{
    false, BatchMode::BATCH_LESS_THAN_L1, false, BatchOutMode::SINGLE_BATCH};
constexpr MatmulConfig MM_CFG_MULTI_BATCH_NO_BATCH_OUT = GetMMConfig<configMode>(batchParamsNoBatchOut);

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, false, false, false, true);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG =
    GetMDLConfig(true, false, 0, false, false, false, false, true, true, false, false, true);
#else
constexpr MatmulConfig MM_CFG_NO_PRELOAD_OPEN_UNIT_FLAG =
    GetMDLConfig(false, false, 0, false, false, false, true, true, false, false, false);
constexpr MatmulConfig MM_DEFAULT_MDL_CFG = GetMDLConfig();
#endif

struct L2CacheParam {
    uint32_t l2MCnt;
    uint32_t l2NCnt;
    uint32_t l2MCntTail;
    uint32_t l2NCntTail;
    uint32_t l2TotalTileCnt;
    uint32_t l2MCntUse;
    uint32_t l2NCntUse;
};

// 量化mm和mc2都用的输入输出地址结构体，不可随意修改
struct QBmmBlockOffset {
    uint64_t offsetA = 0;
    uint64_t offsetB = 0;
    uint64_t offsetScale = 0;
    uint64_t offsetBias = 0;
    uint64_t offsetPertoken = 0;
    uint64_t offsetC = 0;
};

// 量化mm和mc2都用的block输入信息结构体，不可随意修改
struct QBmmBaseBlockArgs {
    uint64_t index;
    uint64_t totalTileCnt;
    uint64_t singleCoreM; // 当前基本块计算大小
    uint64_t singleCoreN;
    uint64_t mTileCntL2;
    uint64_t nTileCntL2;
    uint64_t mTotalCnt;
    uint64_t nTotalCnt;
    uint64_t mCntUse;
    uint64_t nCntUse;
    uint64_t mTileAddrOffset;
    uint64_t nTileAddrOffset;
};

enum class FusedOpType : uint32_t { NONE = 0U, RELU = 1U, SWIGLU = 2U, GELU_TANH = 3UL, GELU_ERF = 4UL };

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
template <typename yType, typename scaleType> struct EpilogueParams {
    AscendC::GlobalTensor<int32_t> &curMmOutGm;
    AscendC::GlobalTensor<yType> &yGm;
    AscendC::GlobalTensor<bfloat16_t> &biasGmBf16;
    AscendC::GlobalTensor<half> &biasGmFp16;
    AscendC::GlobalTensor<float> &biasGmFp32;
    AscendC::GlobalTensor<scaleType> &scaleGm;
    AscendC::GlobalTensor<float> &pertokenScaleGm;

    AscendC::TBuf<AscendC::TPosition::VECCALC> &outFp32Tmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> &vecQueTmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> &biasFp32Tmp;
    AscendC::TBuf<AscendC::TPosition::VECCALC> &broadcastFp32Tmp;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> &vecQueOut;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueBias;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueScale;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQuePertokenScale;

    uint32_t curAicM;
    uint32_t curAicN;
    uint32_t subBlockIdx;
    uint32_t ubCalcM;
    uint32_t ubCalcN;
    uint64_t offsetWorkspaceC;
    uint32_t biasDtype;
    uint32_t biasDtypeSize;
    bool isPerTensor;
    scaleType scaleScalar;
    QBmmBlockOffset offset;
    uint32_t n;

    __aicore__ EpilogueParams(AscendC::GlobalTensor<int32_t> &curMmOutGm, AscendC::GlobalTensor<yType> &yGm,
        AscendC::GlobalTensor<bfloat16_t> &biasGmBf16, AscendC::GlobalTensor<half> &biasGmFp16,
        AscendC::GlobalTensor<float> &biasGmFp32, AscendC::GlobalTensor<scaleType> &scaleGm,
        AscendC::GlobalTensor<float> &pertokenScaleGm,

        AscendC::TBuf<AscendC::TPosition::VECCALC> &outFp32Tmp, AscendC::TBuf<AscendC::TPosition::VECCALC> &vecQueTmp,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &biasFp32Tmp,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &broadcastFp32Tmp,

        AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueSrc,
        AscendC::TQue<AscendC::QuePosition::VECOUT, 1> &vecQueOut,
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueBias,
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQueScale,
        AscendC::TQue<AscendC::QuePosition::VECIN, 1> &vecQuePertokenScale,

        uint32_t curAicM, uint32_t curAicN, uint32_t subBlockIdx, uint32_t ubCalcM, uint32_t ubCalcN,
        uint64_t offsetWorkspaceC, uint32_t biasDtype, uint32_t biasDtypeSize, bool isPerTensor, scaleType scaleScalar,
        QBmmBlockOffset offset, uint32_t n)
        : curMmOutGm(curMmOutGm), yGm(yGm), biasGmBf16(biasGmBf16), biasGmFp16(biasGmFp16), biasGmFp32(biasGmFp32),
          scaleGm(scaleGm), pertokenScaleGm(pertokenScaleGm), outFp32Tmp(outFp32Tmp), vecQueTmp(vecQueTmp),
          biasFp32Tmp(biasFp32Tmp), broadcastFp32Tmp(broadcastFp32Tmp), vecQueSrc(vecQueSrc), vecQueOut(vecQueOut),
          vecQueBias(vecQueBias), vecQueScale(vecQueScale), vecQuePertokenScale(vecQuePertokenScale), curAicM(curAicM),
          curAicN(curAicN), subBlockIdx(subBlockIdx), ubCalcM(ubCalcM), ubCalcN(ubCalcN),
          offsetWorkspaceC(offsetWorkspaceC), biasDtype(biasDtype), biasDtypeSize(biasDtypeSize),
          isPerTensor(isPerTensor), scaleScalar(scaleScalar), offset(offset), n(n) {}
};
#endif

enum class BasicQuantMode : uint32_t {
    DEFAULT = 0x0U,
    PERTENSOR_MODE = 0x1U,
    PERCHANNEL_MODE = 0x1U << 1,
    PERTOKEN_MODE = 0x1U << 2,
    MX_PERGROUP_MODE = 0x1U << 3,
    PERBLOCK_MODE = 0x1U << 4,
    PERGROUP_MODE = 0x1U << 5,
};

namespace DequantBmm {
template <typename T> __aicore__ inline T Max(T a, T b) { return a > b ? a : b; }

template <typename T> __aicore__ inline T Min(T a, T b) { return a > b ? b : a; }

__aicore__ inline uint64_t Align(uint64_t a, uint64_t b = 16) { return (a + b - 1) / b * b; }

__aicore__ inline uint64_t FloorAlign(uint64_t a, uint64_t b = 16) {
    if (b == 0) {
        return a;
    }
    return a / b * b;
}

__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b) {
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t Align16(uint64_t num) { return (num + 15UL) & 0xFFFFFFFFFFFFFFF0UL; }

__aicore__ inline uint64_t Align32(uint64_t num) { return (num + 31UL) & 0xFFFFFFFFFFFFFFE0UL; }

__aicore__ inline uint64_t Align64(uint64_t num) { return (num + 31UL) & 0xFFFFFFFFFFFFFFC0UL; }

__aicore__ inline constexpr CubeFormat GetFormat(int format) {
    if (format == FORMAT_FRACTAL_NZ_INT) {
        return CubeFormat::NZ;
    }
    return CubeFormat::ND;
}

template <typename T, bool isLut = false> __aicore__ inline constexpr uint32_t GetC0Size() {
    // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
    if constexpr (isLut && AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        return K0_INT8;
#if __LUT_SUPPORT__
    } else if constexpr (isLut &&
        (AscendC::IsSameType<T, AscendC::int2b_t>::value || AscendC::IsSameType<T, AscendC::uint1b_t>::value)) {
        return K0_INT4;
#endif
    } else if constexpr (AscendC::IsSameType<T, AscendC::int4b_t>::value) {
        return K0_INT4;
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return K0_INT4;
#endif
    } else if constexpr (sizeof(T) == sizeof(float)) {
        return k0_FLOAT32;
    } else if constexpr (sizeof(T) == sizeof(int8_t)) {
        return K0_INT8;
    } else {
        return k0_FLOAT16;
    }
}

template <typename T, bool isLut = false> __aicore__ inline constexpr uint64_t GetSizeWithDataType(uint64_t shapeSize) {
    bool is4BitInput = false;
    bool is8BitInput = false;
    if constexpr (isLut) {
        // lut查表逻辑: 原始数据DT_INT2和DT_UINT1，查表后转DT_INT4; 原始数据DT_INT4, 查表后转DT_INT8
#if __LUT_SUPPORT__
        is4BitInput =
            (AscendC::IsSameType<T, AscendC::int2b_t>::value || AscendC::IsSameType<T, AscendC::uint1b_t>::value);
#endif
        is8BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value);
    } else {
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
        is4BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value || AscendC::IsSameType<T, fp4x2_e2m1_t>::value);
#else
        is4BitInput = (AscendC::IsSameType<T, AscendC::int4b_t>::value);
#endif
    }
    if (is4BitInput) {
        return shapeSize / 2;
    } else if (is8BitInput) {
        return shapeSize;
    } else {
        return shapeSize * sizeof(T);
    }
}

template <typename T, bool trans> __aicore__ inline uint64_t CalcAL1Size(uint64_t mL1, uint64_t kL1) {
    uint64_t innerAlignedBlock = ONE_BLOCK_SIZE / sizeof(T);
    uint64_t mAligned = 0;
    uint64_t kAligned = 0;
    if constexpr (trans) {
        mAligned = Align(mL1, innerAlignedBlock);
        kAligned = Align(kL1, BMM_BLOCK_NUM);
    } else {
        mAligned = Align(mL1, BMM_BLOCK_NUM);
        kAligned = Align(kL1, innerAlignedBlock);
    }
    return mAligned * kAligned * sizeof(T);
}

/**
 * Get the aiv corenum in different platforms
 */
__aicore__ inline constexpr uint32_t GetTaskRation() {
#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113))
    return 2U; // aiv corenum is 2 in C310 platform
#else
    return 1U;
#endif
}

#if ((defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)) && !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3113)) || \
    __CUBE_S8S4_S4S4__
template <typename T> __aicore__ inline constexpr bool IsMxType() {
    return AscendC::IsSameType<T, AscendC::fp8_e8m0_t>::value;
}

template <typename T> __aicore__ inline constexpr bool IsFp4() { return AscendC::IsSameType<T, fp4x2_e2m1_t>::value; }
#endif

#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
__aicore__ inline void CalcDequantParams(
    uint32_t curAivM, uint32_t curAivN, AscendC::DequantParams &dequantParams, bool needUpdate = true) {
    if (!needUpdate) {
        return;
    }
    uint32_t computedAivN = Align(curAivN, 8U); // 8: 32B aligned for int32_t
    uint32_t ubResAlignedN = Align(curAivN); // 16: sizeof(yType) is 2, 32B / 2
    if (computedAivN == ubResAlignedN) {
        // choose ddequat high performance
        dequantParams.m = 1;
        dequantParams.n = curAivM * computedAivN;
        dequantParams.calCount = computedAivN;
    } else {
        // general
        dequantParams.m = curAivM;
        dequantParams.n = computedAivN;
        dequantParams.calCount = curAivN;
    }
}

__aicore__ inline void SetGm2UbParams(AscendC::DataCopyParams &gm2UbParams, uint32_t curAivM, uint32_t curAivN) {
    gm2UbParams.blockLen = curAivN * sizeof(int32_t);
    gm2UbParams.blockCount = curAivM;
    gm2UbParams.srcStride = 0;
}

template <typename yType>
__aicore__ inline void SetUb2GmParams(
    AscendC::DataCopyExtParams &ub2GmParams, uint32_t curAivM, uint32_t curAivN, uint32_t n) {
    ub2GmParams.blockLen = curAivN * sizeof(yType);
    ub2GmParams.blockCount = curAivM;
    ub2GmParams.dstStride = (n - curAivN) * sizeof(yType);
}

__aicore__ inline void CopyMmOutToLocal(AscendC::LocalTensor<int32_t> &srcLocal,
    AscendC::GlobalTensor<int32_t> &curMmOutGm, AscendC::DataCopyParams &gm2UbParams,
    AscendC::DataCopyPadParams &padParams, uint32_t curAicAivOffset) {
    DataCopyPad(srcLocal, curMmOutGm[curAicAivOffset], gm2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
}

template <typename yType>
__aicore__ inline void CopyUbToGm(uint64_t yGmOffset, AscendC::DataCopyExtParams &ub2GmParams,
    AscendC::LocalTensor<yType> &dstLocal, AscendC::GlobalTensor<yType> &yGm,
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> &vecQueOut) {
    DataCopyPad(yGm[yGmOffset], dstLocal, ub2GmParams);
    vecQueOut.FreeTensor(dstLocal);
}

template <typename scaleType>
__aicore__ inline void Bf16ScaleGm2Ub(AscendC::LocalTensor<scaleType> &scaleLocal,
    AscendC::GlobalTensor<scaleType> &scaleGm, AscendC::DataCopyPadParams &padParams, uint32_t curAivN,
    uint64_t offsetScale) {
    AscendC::DataCopyParams scale2UbParams{1, 0, 0, 0};
    scale2UbParams.blockLen = curAivN * sizeof(scaleType);
    DataCopyPad(scaleLocal, scaleGm[offsetScale], scale2UbParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
}
#endif
} // namespace DequantBmm

#endif // EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASE_H
