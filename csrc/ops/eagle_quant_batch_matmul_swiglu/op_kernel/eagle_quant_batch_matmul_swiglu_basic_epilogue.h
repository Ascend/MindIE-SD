/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file eagle_quant_batch_matmul_swiglu_basic_epilogue.h
 * \brief
 */
#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASIC_EPILOGUE_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASIC_EPILOGUE_H

#include "eagle_quant_batch_matmul_swiglu_base.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace AscendC {

// Gelu 计算参数
constexpr float SCALAR_ONE = 1.0f;
constexpr float BETA = 0.044715f;
constexpr float ALPHA = -1.5957691f;

constexpr float ERF_PARAM1 = -0.3512339572e-8f;
constexpr float ERF_PARAM2 = 0.2645266170e-6f;
constexpr float ERF_PARAM3 = -0.7929488134e-5f;
constexpr float ERF_PARAM4 = 0.1106123840e-3f;
constexpr float ERF_PARAM5 = 0.6518995814e-4f;
constexpr float ERF_PARAM6 = -0.7266616915e-1f;
constexpr float ERF_PARAM7 = -0.1595769883e1f;
constexpr float ERF_MIN = 5.751f;
constexpr float ERF_MAX = -13.15f;

template <typename yType, typename scaleType> class EpilogueDequantCommon {
  private:
    // GlobalTensor
    GlobalTensor<int32_t> curMmOutGm_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<half> biasGmFp16_;
    GlobalTensor<float> biasGmFp32_;
    GlobalTensor<scaleType> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    // TBuf
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;
    // TQue
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    // 常量
    uint32_t ubCalcM_;
    uint64_t offsetWorkspaceC_;
    uint32_t biasDtype_;
    uint32_t biasDtypeSize_;
    QBmmBlockOffset offset_;
    bool isPerTensor_;
    scaleType scaleScalar_;
    uint32_t curAicN_;

  public:
    __aicore__ inline EpilogueDequantCommon() {}

    __aicore__ inline void SetupParams(EpilogueParams<yType, scaleType> &params) {
        // GlobalTensor
        curMmOutGm_ = params.curMmOutGm;
        biasGmBf16_ = params.biasGmBf16;
        biasGmFp16_ = params.biasGmFp16;
        biasGmFp32_ = params.biasGmFp32;
        scaleGm_ = params.scaleGm;
        pertokenScaleGm_ = params.pertokenScaleGm;
        // TBuf
        outFp32Tmp_ = params.outFp32Tmp;
        vecQueTmp_ = params.vecQueTmp;
        biasFp32Tmp_ = params.biasFp32Tmp;
        broadcastFp32Tmp_ = params.broadcastFp32Tmp;
        // TQue
        vecQueSrc_ = params.vecQueSrc;
        vecQueBias_ = params.vecQueBias;
        vecQueScale_ = params.vecQueScale;
        vecQuePertokenScale_ = params.vecQuePertokenScale;
        // 常量
        ubCalcM_ = params.ubCalcM;
        offsetWorkspaceC_ = params.offsetWorkspaceC;
        biasDtype_ = params.biasDtype;
        biasDtypeSize_ = params.biasDtypeSize;
        offset_ = params.offset;
        isPerTensor_ = params.isPerTensor;
        scaleScalar_ = params.scaleScalar;
        curAicN_ = params.curAicN;
    }

    __aicore__ inline void PertokenCalculate(uint32_t basicBlockComputeInfo[], uint32_t mUbLoopIdx,
        DataCopyPadParams &padParams, LocalTensor<float> &dstLocalFp32, LocalTensor<float> &tmpdstLocal,
        bool reuseBroadcast = false) {
        uint32_t curAivN = basicBlockComputeInfo[0];
        uint32_t curAivM = basicBlockComputeInfo[1];
        uint32_t ubResAlignedN = basicBlockComputeInfo[2];
        uint32_t subBlockoffset = basicBlockComputeInfo[3]; // 数组下标3

        DataCopyParams scale2UbParams{1, 0, 0, 0};
        scale2UbParams.blockLen = curAivM * sizeof(float);
        uint64_t offsetPertoken = offset_.offsetPertoken + mUbLoopIdx * ubCalcM_ + subBlockoffset;
        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U); // 8: 32B aligned for float

        const uint32_t broadCastDst[M_N_TWO_DIMS] = {curAivM, computedAivN};
        const uint32_t broadCastSrc[M_N_TWO_DIMS] = {curAivM, 1};

        LocalTensor<float> broadcastFp32 = broadcastFp32Tmp_.Get<float>();
        LocalTensor<float> pertokenScaleLocal;
        if (!reuseBroadcast) {
            pertokenScaleLocal = vecQuePertokenScale_.AllocTensor<float>();
            DataCopyPad(pertokenScaleLocal, pertokenScaleGm_[offsetPertoken], scale2UbParams, padParams);
            vecQuePertokenScale_.EnQue<float>(pertokenScaleLocal);
            pertokenScaleLocal = vecQuePertokenScale_.DeQue<float>();
            BroadCast<float, M_N_TWO_DIMS, 1>(broadcastFp32, pertokenScaleLocal, broadCastDst, broadCastSrc);
        }

        AscendC::PipeBarrier<PIPE_V>();

        if (computedAivN == ubResAlignedN) {
            Mul(tmpdstLocal, broadcastFp32, dstLocalFp32, computedAivN * curAivM);
        } else {
            for (auto i = 0; i < curAivM; i++) {
                Mul(tmpdstLocal[ubResAlignedN * i], broadcastFp32[computedAivN * i], dstLocalFp32[computedAivN * i],
                    computedAivN);
            }
        }
        if (!reuseBroadcast) {
            vecQuePertokenScale_.FreeTensor(pertokenScaleLocal);
        }
    }

    __aicore__ inline void CalBiasAdd(LocalTensor<float> &dstLocalFp32, LocalTensor<float> &biasFp32,
        LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16, LocalTensor<float> &oriBiasFp32,
        uint32_t curAivN, uint32_t curAivM) {
        uint32_t computedAivN = DequantBmm::Align(curAivN, 8U); // 8: 32B aligened for int32_t
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN); // 16: sizeof(ytype) is 2 , 32B / 2
        AscendC::PipeBarrier<PIPE_V>();
        if (biasDtype_ == DT_BF16) {
            Cast(biasFp32, oriBiasBf16, RoundMode::CAST_NONE, ubResAlignedN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasBf16);
        } else if (biasDtype_ == DT_FLOAT16) {
            Cast(biasFp32, oriBiasFp16, RoundMode::CAST_NONE, ubResAlignedN);
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp16);
        } else if (biasDtype_ == DT_FLOAT) {
            biasFp32 = oriBiasFp32;
            AscendC::PipeBarrier<PIPE_V>();
            vecQueBias_.FreeTensor(oriBiasFp32);
        }
        for (int32_t mIdx = 0; mIdx < curAivM; ++mIdx) {
            Add(dstLocalFp32[mIdx * ubResAlignedN], dstLocalFp32[mIdx * ubResAlignedN], biasFp32, ubResAlignedN);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BiasTensorInit(LocalTensor<float> & /* dstLocalFp32 */, LocalTensor<float> &biasFp32,
        LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16, LocalTensor<float> &oriBiasFp32) {
        biasFp32 = biasFp32Tmp_.Get<float>();
        if (biasDtype_ == DT_BF16) {
            oriBiasBf16 = vecQueBias_.AllocTensor<bfloat16_t>(); // free in CalBiasAdd
        } else if (biasDtype_ == DT_FLOAT16) {
            oriBiasFp16 = vecQueBias_.AllocTensor<half>(); // free in CalBiasAdd
        } else if (biasDtype_ == DT_FLOAT) {
            oriBiasFp32 = vecQueBias_.AllocTensor<float>(); // free in CalBiasAdd
        }
    }

    __aicore__ inline void BiasGm2Ub(LocalTensor<bfloat16_t> &oriBiasBf16, LocalTensor<half> &oriBiasFp16,
        LocalTensor<float> &oriBiasFp32, DataCopyPadParams padParams, uint32_t curAivN) {
        DataCopyParams bias2UbParams{1, 0, 0, 0};
        bias2UbParams.blockLen = curAivN * biasDtypeSize_;

        if (biasDtype_ == DT_BF16) {
            DataCopyPad(oriBiasBf16, biasGmBf16_[offset_.offsetBias], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT16) {
            DataCopyPad(oriBiasFp16, biasGmFp16_[offset_.offsetBias], bias2UbParams, padParams);
        } else if (biasDtype_ == DT_FLOAT) {
            DataCopyPad(oriBiasFp32, biasGmFp32_[offset_.offsetBias], bias2UbParams, padParams);
        }
    }

    __aicore__ inline void DequantCompute(EpilogueParams<yType, scaleType> &params, uint32_t dequantComputeInfo[],
        DataCopyPadParams &padParams, DequantParams &dequantParams, LocalTensor<float> &tmpdstLocal,
        const LocalTensor<scaleType> *residentScale = nullptr, bool reusePertokenBroadcast = false) {
        SetupParams(params);
        uint32_t curAivM = dequantComputeInfo[0];
        uint32_t curAivN = dequantComputeInfo[1];
        uint32_t mUbLoopIdx = dequantComputeInfo[2];
        uint32_t subBlockoffset = dequantComputeInfo[3];
        LocalTensor<float> dstLocalFp32 = outFp32Tmp_.Get<float>();
        LocalTensor<float> biasFp32;
        LocalTensor<bfloat16_t> oriBiasBf16;
        LocalTensor<half> oriBiasFp16;
        LocalTensor<float> oriBiasFp32;

        LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
        LocalTensor<uint8_t> tmpLocal = vecQueTmp_.Get<uint8_t>();
        // datacopypad 32B aligned
        DataCopyParams gm2UbParams{1, 0, 0, 0};
        DequantBmm::SetGm2UbParams(gm2UbParams, curAivM, curAivN);
        DequantBmm::CopyMmOutToLocal(srcLocal, curMmOutGm_, gm2UbParams, padParams,
            offsetWorkspaceC_ + mUbLoopIdx * ubCalcM_ * curAicN_ + subBlockoffset * curAicN_);
        if (biasDtype_ != DT_INT32) {
            BiasTensorInit(dstLocalFp32, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32);
            BiasGm2Ub(oriBiasBf16, oriBiasFp16, oriBiasFp32, padParams,
                curAicN_); // 当bias不是int32时，在vector做+bias，将bias从gm搬到ub
        }
        if (isPerTensor_) {
            AscendDequant(dstLocalFp32, srcLocal, scaleScalar_, tmpLocal, dequantParams);
        } else if (residentScale != nullptr) {
            AscendDequant(dstLocalFp32, srcLocal, *residentScale, tmpLocal, dequantParams);
        } else {
            LocalTensor<scaleType> scaleLocal = vecQueScale_.AllocTensor<scaleType>();
            DequantBmm::Bf16ScaleGm2Ub<scaleType>(scaleLocal, scaleGm_, padParams, curAicN_, offset_.offsetScale);
            AscendDequant(dstLocalFp32, srcLocal, scaleLocal, tmpLocal, dequantParams);
            vecQueScale_.FreeTensor(scaleLocal);
        }
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN); // 16: sizeof(yType) is 2, 32B / 2
        uint32_t basicBlockComputeInfo[4] = {curAivN, curAivM, ubResAlignedN, subBlockoffset};
        PertokenCalculate(
            basicBlockComputeInfo, mUbLoopIdx, padParams, dstLocalFp32, tmpdstLocal, reusePertokenBroadcast);

        if (biasDtype_ != DT_INT32) {
            CalBiasAdd(tmpdstLocal, biasFp32, oriBiasBf16, oriBiasFp16, oriBiasFp32, curAivN, curAivM);
        }
        vecQueSrc_.FreeTensor(srcLocal);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DequantComputeInPlace(EpilogueParams<yType, scaleType> &params, uint32_t computeInfo[],
        DataCopyPadParams &padParams, LocalTensor<float> &work, const LocalTensor<float> &scale,
        bool reusePertokenBroadcast) {
        SetupParams(params);
        uint32_t rows = computeInfo[0];
        uint32_t rowTile = computeInfo[2];
        uint32_t subBlockoffset = computeInfo[3];
        uint32_t cols = computeInfo[1];
        uint32_t pitch = DequantBmm::Align(cols);
        constexpr uint32_t FLOATS_PER_REPEAT = 64;
        LocalTensor<int32_t> srcLocal = vecQueSrc_.AllocTensor<int32_t>();
        DataCopyParams copyParams{1, 0, 0, 0};
        DequantBmm::SetGm2UbParams(copyParams, rows, cols);
        // FP32 and final FP16/BF16 rows share a 16-element pitch, including N tails.
        copyParams.dstStride = (pitch - DequantBmm::Align(cols, 8U)) / 8;
        DequantBmm::CopyMmOutToLocal(srcLocal, curMmOutGm_, copyParams, padParams,
            offsetWorkspaceC_ + (rowTile * ubCalcM_ + subBlockoffset) * curAicN_);
        Cast(work, srcLocal, RoundMode::CAST_NONE, rows * pitch);
        PipeBarrier<PIPE_V>();
        // Each repeat advances one matrix row and reuses the resident channel scale.
        BinaryRepeatParams mulParams(1, 1, 1, pitch / 8, pitch / 8, 0);
        for (uint32_t col = 0; col < cols; col += FLOATS_PER_REPEAT) {
            uint32_t mask = cols - col < FLOATS_PER_REPEAT ? cols - col : FLOATS_PER_REPEAT;
            Mul(work[col], work[col], scale[col], mask, static_cast<uint8_t>(rows), mulParams);
        }
        PipeBarrier<PIPE_V>();
        uint32_t tokenInfo[4] = {pitch, rows, pitch, subBlockoffset};
        PertokenCalculate(tokenInfo, rowTile, padParams, work, work, reusePertokenBroadcast);
        PipeBarrier<PIPE_V>();
        vecQueSrc_.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOutToGM(EpilogueParams<yType, scaleType> &params, LocalTensor<float> &tmpdstLocal,
        uint32_t curAivM, uint32_t curAivN, uint32_t mUbLoopIdx, uint32_t subBlockoffset, bool swiglu = false) {
        LocalTensor<yType> dstLocal = params.vecQueOut.template AllocTensor<yType>();
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN); // 16: sizeof(yType) is 2, 32B / 2
        DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
        Cast(dstLocal, tmpdstLocal, RoundMode::CAST_RINT, curAivM * ubResAlignedN);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
        // dst from ub -> gm
        uint64_t outputN = swiglu ? params.n / 2 : params.n;
        uint64_t outputOffset = swiglu ? (params.offset.offsetC / params.n) * outputN + params.offset.offsetC % params.n
                                       : params.offset.offsetC;
        DequantBmm::SetUb2GmParams<yType>(ub2GmParams, curAivM, curAivN, outputN);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
        DequantBmm::CopyUbToGm<yType>(outputOffset + (mUbLoopIdx * params.ubCalcM + subBlockoffset) * outputN,
            ub2GmParams, dstLocal, params.yGm, params.vecQueOut);
    }
};

class EpilogueDequant {
  public:
    __aicore__ inline EpilogueDequant() {}

    template <typename yType, typename scaleType>
    __aicore__ inline void ProcessPair(EpilogueParams<yType, scaleType> &left, EpilogueParams<yType, scaleType> &right,
        bool inPlaceDequant, TBuf<TPosition::VECCALC> &workB, TBuf<TPosition::VECCALC> &activationTmp) {
        // Paired N branches have identical M/N tails and share the same per-token rows.
        int64_t vecNum = GetTaskRation();
        vecNum = (vecNum == 0) ? 1 : vecNum;
        uint32_t subBlockoffset = left.subBlockIdx * left.curAicM / vecNum;
        uint32_t aivM = left.curAicM / vecNum + left.subBlockIdx * (left.curAicM % vecNum);
        if (inPlaceDequant && vecNum == 2) {
            // Assign whole UB tiles to each AIV, including short M tails.
            uint32_t totalTiles = DequantBmm::CeilDiv(left.curAicM, left.ubCalcM);
            uint32_t firstRows = DequantBmm::CeilDiv(totalTiles, 2U) * left.ubCalcM;
            firstRows = firstRows < left.curAicM ? firstRows : left.curAicM;
            subBlockoffset = left.subBlockIdx == 0 ? 0 : firstRows;
            aivM = left.subBlockIdx == 0 ? firstRows : left.curAicM - firstRows;
        }
        if (aivM == 0) {
            return;
        }
        EpilogueParams<yType, scaleType> *branches[2] = {&left, &right};
        LocalTensor<scaleType> scales[2];
        DataCopyPadParams padParams{};
        if (!left.isPerTensor) {
            // Reuse the two existing scale buffers as left/right residents for the entire pair.
            for (uint32_t branch = 0; branch < 2; ++branch) {
                scales[branch] = left.vecQueScale.template AllocTensor<scaleType>();
                DequantBmm::Bf16ScaleGm2Ub<scaleType>(scales[branch], branches[branch]->scaleGm, padParams,
                    left.curAicN, branches[branch]->offset.offsetScale);
            }
        }
        LocalTensor<float> tmpdstLocal = left.vecQueTmp.template Get<float>();
        EpilogueDequantCommon<yType, scaleType> commonDequant;
        uint32_t mUbLoops = DequantBmm::CeilDiv(aivM, left.ubCalcM);
        for (uint32_t row = 0; row < mUbLoops; ++row) {
            uint32_t rows = row + 1 == mUbLoops ? aivM - row * left.ubCalcM : left.ubCalcM;
            // This descriptor is new on every row iteration, including full and single-row tiles.
            DequantParams dequantParams;
            DequantBmm::CalcDequantParams(rows, left.curAicN, dequantParams);
            uint32_t computeInfo[4] = {rows, left.curAicN, row, subBlockoffset};
            for (uint32_t branch = 0; branch < 2; ++branch) {
                if constexpr (AscendC::IsSameType<scaleType, float>::value) {
                    if (inPlaceDequant) {
                        LocalTensor<float> branchWork = branch == 0 ? tmpdstLocal : workB.Get<float>();
                        commonDequant.DequantComputeInPlace(
                            *branches[branch], computeInfo, padParams, branchWork, scales[branch], branch != 0);
                    } else {
                        commonDequant.DequantCompute(*branches[branch], computeInfo, padParams, dequantParams,
                            tmpdstLocal, left.isPerTensor ? nullptr : &scales[branch], branch != 0);
                    }
                } else {
                    commonDequant.DequantCompute(*branches[branch], computeInfo, padParams, dequantParams, tmpdstLocal,
                        left.isPerTensor ? nullptr : &scales[branch], branch != 0);
                }
                if (!inPlaceDequant) {
                    commonDequant.CopyOutToGM(*branches[branch], tmpdstLocal, rows, left.curAicN, row, subBlockoffset);
                }
            }
            if (inPlaceDequant) {
                LocalTensor<float> b = workB.Get<float>();
                LocalTensor<float> scratch = activationTmp.Get<float>();
                uint32_t count = rows * DequantBmm::Align(left.curAicN);
                Muls(scratch, tmpdstLocal, -1.0f, count);
                PipeBarrier<PIPE_V>();
                Exp(scratch, scratch, count);
                PipeBarrier<PIPE_V>();
                Adds(scratch, scratch, 1.0f, count);
                PipeBarrier<PIPE_V>();
                Div(tmpdstLocal, tmpdstLocal, scratch, count);
                PipeBarrier<PIPE_V>();
                Mul(tmpdstLocal, tmpdstLocal, b, count);
                PipeBarrier<PIPE_V>();
                commonDequant.CopyOutToGM(left, tmpdstLocal, rows, left.curAicN, row, subBlockoffset, true);
            }
        }
        if (!left.isPerTensor) {
            for (uint32_t branch = 0; branch < 2; ++branch) {
                left.vecQueScale.FreeTensor(scales[branch]);
            }
        }
    }

    template <typename yType, typename scaleType>
    __aicore__ inline void Process(EpilogueParams<yType, scaleType> &params) {
        LocalTensor<float> tmpdstLocal = params.vecQueTmp.template Get<float>();
        uint32_t subBlockoffset = 0;

        int64_t vecNum = GetTaskRation();
        vecNum = (vecNum == 0) ? 1 : vecNum;
        subBlockoffset = params.subBlockIdx * params.curAicM / vecNum;
        params.curAicM = params.curAicM / vecNum + params.subBlockIdx * (params.curAicM % vecNum);
        uint32_t curAivM = params.ubCalcM;
        // calcN in ub is equal to aicN
        uint32_t curAivN = params.curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(params.curAicM, params.ubCalcM);
        DequantParams dequantParams;
        DataCopyPadParams padParams;
        DequantBmm::CalcDequantParams(mUbLoops == 1 ? params.curAicM : params.ubCalcM, params.curAicN, dequantParams);
        EpilogueDequantCommon<yType, scaleType> commonDequant;
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = params.curAicM - params.ubCalcM * (mUbLoops - 1);
                DequantBmm::CalcDequantParams(
                    curAivM, params.curAicN, dequantParams, mUbLoops != 1 && curAivM != params.ubCalcM);
            }
            uint32_t dequantComputeInfo[4] = {curAivM, curAivN, mUbLoopIdx, subBlockoffset};
            commonDequant.DequantCompute(params, dequantComputeInfo, padParams, dequantParams, tmpdstLocal);

            commonDequant.CopyOutToGM(params, tmpdstLocal, curAivM, curAivN, mUbLoopIdx, subBlockoffset);
        }
    }
};

class EpilogueDequantGeluTanh {
  public:
    __aicore__ inline EpilogueDequantGeluTanh() {}

    template <typename yType, typename scaleType>
    __aicore__ inline void Process(EpilogueParams<yType, scaleType> &params) {
        LocalTensor<float> tmpdstLocal = params.vecQueTmp.template Get<float>();
        uint32_t subBlockoffset = 0;

        int64_t vecNum = GetTaskRation();
        vecNum = (vecNum == 0) ? 1 : vecNum;
        subBlockoffset = params.subBlockIdx * params.curAicM / vecNum;
        params.curAicM = params.curAicM / vecNum + params.subBlockIdx * (params.curAicM % vecNum);
        uint32_t curAivM = params.ubCalcM;
        // calcN in ub is equal to aicN
        uint32_t curAivN = params.curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(params.curAicM, params.ubCalcM);
        DequantParams dequantParams;
        DataCopyPadParams padParams;
        DequantBmm::CalcDequantParams(mUbLoops == 1 ? params.curAicM : params.ubCalcM, params.curAicN, dequantParams);
        EpilogueDequantCommon<yType, scaleType> commonDequant;
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = params.curAicM - params.ubCalcM * (mUbLoops - 1);
                DequantBmm::CalcDequantParams(
                    curAivM, params.curAicN, dequantParams, mUbLoops != 1 && curAivM != params.ubCalcM);
            }
            uint32_t dequantComputeInfo[4] = {curAivM, curAivN, mUbLoopIdx, subBlockoffset};
            commonDequant.DequantCompute(params, dequantComputeInfo, padParams, dequantParams, tmpdstLocal);

            CalGeluTanh(params, tmpdstLocal, curAivM, curAivN);

            commonDequant.CopyOutToGM(params, tmpdstLocal, curAivM, curAivN, mUbLoopIdx, subBlockoffset);
        }
    }

    template <typename yType, typename scaleType>
    __aicore__ inline void CalGeluTanh(
        EpilogueParams<yType, scaleType> &params, LocalTensor<float> &tmpdstLocal, uint32_t curAivM, uint32_t curAivN) {
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN); // 16: sizeof(ytype) is 2 , 32B / 2
        LocalTensor<float> tmpResLocal = params.outFp32Tmp.template Get<float>(); // 复用了outFp32Tmp的空间
        ComputeGeluTanh(tmpdstLocal, tmpResLocal, curAivM * ubResAlignedN);
    }

    __aicore__ inline void ComputeGeluTanh(
        const LocalTensor<float> &castFp32, const LocalTensor<float> &tempRes, int64_t calCount) {
        Mul(tempRes, castFp32, castFp32, calCount); // x^2
        PipeBarrier<PIPE_V>();
        Mul(tempRes, castFp32, tempRes, calCount); // x^3
        PipeBarrier<PIPE_V>();
        Muls(tempRes, tempRes, BETA, calCount); // 0.044715 * x^3
        PipeBarrier<PIPE_V>();
        Add(tempRes, castFp32, tempRes, calCount); // x + 0.044715 * x^3
        PipeBarrier<PIPE_V>();
        Muls(tempRes, tempRes, ALPHA, calCount); // -sqrt(8/pi)(x + 0.044715 * x^3)
        PipeBarrier<PIPE_V>();
        Exp(tempRes, tempRes, calCount); // exp(-sqrt(8/pi)(x + 0.044715 * x^3))
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, SCALAR_ONE, calCount); // 1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3))
        PipeBarrier<PIPE_V>();
        Div(castFp32, castFp32, tempRes, calCount); // x / (1 + exp(-sqrt(8/pi)(x + 0.044715 * x^3)))
        PipeBarrier<PIPE_V>();
    }
};

class EpilogueDequantGeluErf {
  public:
    __aicore__ inline EpilogueDequantGeluErf() {}

    template <typename yType, typename scaleType>
    __aicore__ inline void Process(EpilogueParams<yType, scaleType> &params) {
        LocalTensor<float> tmpdstLocal = params.vecQueTmp.template Get<float>();
        uint32_t subBlockoffset = 0;

        int64_t vecNum = GetTaskRation();
        vecNum = (vecNum == 0) ? 1 : vecNum;
        subBlockoffset = params.subBlockIdx * params.curAicM / vecNum;
        params.curAicM = params.curAicM / vecNum + params.subBlockIdx * (params.curAicM % vecNum);
        uint32_t curAivM = params.ubCalcM;
        // calcN in ub is equal to aicN
        uint32_t curAivN = params.curAicN;
        uint32_t mUbLoops = DequantBmm::CeilDiv(params.curAicM, params.ubCalcM);
        DequantParams dequantParams;
        DataCopyPadParams padParams;
        DequantBmm::CalcDequantParams(mUbLoops == 1 ? params.curAicM : params.ubCalcM, params.curAicN, dequantParams);
        EpilogueDequantCommon<yType, scaleType> commonDequant;
        for (uint32_t mUbLoopIdx = 0; mUbLoopIdx < mUbLoops; ++mUbLoopIdx) {
            if (mUbLoopIdx == mUbLoops - 1) {
                curAivM = params.curAicM - params.ubCalcM * (mUbLoops - 1);
                DequantBmm::CalcDequantParams(
                    curAivM, params.curAicN, dequantParams, mUbLoops != 1 && curAivM != params.ubCalcM);
            }
            uint32_t dequantComputeInfo[4] = {curAivM, curAivN, mUbLoopIdx, subBlockoffset};
            commonDequant.DequantCompute(params, dequantComputeInfo, padParams, dequantParams, tmpdstLocal);

            CalGeluErf(params, tmpdstLocal, curAivM, curAivN);

            commonDequant.CopyOutToGM(params, tmpdstLocal, curAivM, curAivN, mUbLoopIdx, subBlockoffset);
        }
    }

    template <typename yType, typename scaleType>
    __aicore__ inline void CalGeluErf(
        EpilogueParams<yType, scaleType> &params, LocalTensor<float> &tmpdstLocal, uint32_t curAivM, uint32_t curAivN) {
        uint32_t ubResAlignedN = DequantBmm::Align(curAivN); // 16: sizeof(ytype) is 2 , 32B / 2
        LocalTensor<float> tmpResLocal = params.outFp32Tmp.template Get<float>(); // 复用了outFp32Tmp的空间
        LocalTensor<float> xSquaredTensor =
            params.broadcastFp32Tmp.template Get<float>(); // 复用了broadcastFp32Tmp的空间
        ComputeGeluErf(tmpdstLocal, tmpResLocal, xSquaredTensor, curAivM * ubResAlignedN);
    }

    __aicore__ inline void ComputeGeluErf(const LocalTensor<float> &castFp32, const LocalTensor<float> &tempRes,
        const LocalTensor<float> &xSquared, int64_t calCount) {
        Maxs(castFp32, castFp32, ERF_MAX, calCount);
        PipeBarrier<PIPE_V>();
        Mins(tempRes, castFp32, ERF_MIN, calCount);
        PipeBarrier<PIPE_V>();

        Mul(xSquared, tempRes, tempRes, calCount); // x^2
        PipeBarrier<PIPE_V>();

        Muls(tempRes, xSquared, ERF_PARAM1, calCount); // a1*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM2, calCount); // a1*x^2+a2
        PipeBarrier<PIPE_V>();

        Mul(tempRes, tempRes, xSquared, calCount); // (a1*x^2+a2)*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM3, calCount); //(a1*x^2+a2)*x^2 + a3
        PipeBarrier<PIPE_V>();

        Mul(tempRes, tempRes, xSquared, calCount); // ((a1*x^2+a2)*x^2 + a3)*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM4, calCount); // ((a1*x^2+a2)*x^2 + a3)*x^2 + a4
        PipeBarrier<PIPE_V>();

        Mul(tempRes, tempRes, xSquared, calCount); // (((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM5, calCount); // (((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5
        PipeBarrier<PIPE_V>();

        Mul(tempRes, tempRes, xSquared, calCount); // ((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM6, calCount); // ((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6
        PipeBarrier<PIPE_V>();

        Mul(tempRes, tempRes, xSquared, calCount); // (((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2
        PipeBarrier<PIPE_V>();
        Adds(tempRes, tempRes, ERF_PARAM7,
            calCount); // (((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2 + a7
        PipeBarrier<PIPE_V>();

        Mins(xSquared, castFp32, ERF_MIN, calCount);
        PipeBarrier<PIPE_V>();
        Mul(tempRes, tempRes, xSquared,
            calCount); // ((((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2 + a7)*x
        PipeBarrier<PIPE_V>();

        Exp(tempRes, tempRes, calCount); // exp(((((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2 + a7)*x)
        PipeBarrier<PIPE_V>();

        Adds(tempRes, tempRes, 1.0f,
            calCount); // 1+exp(((((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2 + a7)*x)
        PipeBarrier<PIPE_V>();

        Div(castFp32, castFp32, tempRes,
            calCount); // x/(1+exp(((((((a1*x^2+a2)*x^2 + a3)*x^2 + a4)*x^2 + a5)*x^2 + a6)*x^2 + a7)*x))
        PipeBarrier<PIPE_V>();
    }
};

} // namespace AscendC

#endif // EAGLE_QUANT_BATCH_MATMUL_SWIGLU_BASIC_EPILOGUE_H
