/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
 * \file eagle_quant_batch_matmul_swiglu_pertoken_basic.h
 * \brief
 */
#ifndef EAGLE_QUANT_BATCH_MATMUL_SWIGLU_PERTOKEN_BASIC_H
#define EAGLE_QUANT_BATCH_MATMUL_SWIGLU_PERTOKEN_BASIC_H

#include "eagle_quant_batch_matmul_swiglu_block.h"
#include "eagle_quant_batch_matmul_swiglu_update.h"
#include "eagle_quant_batch_matmul_swiglu_basic_epilogue.h"

namespace AscendC {

template <FusedOpType OpType> struct EpilogueTypeTraits;

template <> struct EpilogueTypeTraits<FusedOpType::GELU_TANH> {
    using Type = EpilogueDequantGeluTanh;
};

template <> struct EpilogueTypeTraits<FusedOpType::GELU_ERF> {
    using Type = EpilogueDequantGeluErf;
};

template <TemplateBasicTypeForClass> class BmmDequantPertokenBasic {
  public:
    __aicore__ inline BmmDequantPertokenBasic() {}
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y,
        GM_ADDR workSpace, const EagleQuantBatchMatmulSwigluTilingData *__restrict tilingData, TPipe *tPipe) {
        blockIdx_ = GetBlockIdx();
        blockIdx_ /= GetTaskRation();
        subBlockIdx_ = GetSubBlockIdx();
        usedCoreNum_ = tilingData->matmulTiling.usedCoreNum;
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        pipe_ = tPipe;
        mm_.SetSubBlockIdx(0);
        mm_.Init(&(tilingData->matmulTiling), pipe_);
        InitTilingData(tilingData);
        SetMatmulOrgShape();

        InitGlobalBuffers(x1, x2, scale, bias, pertokenScale, y, workSpace);
        InitLocalBuffers();
        offsetWorkspaceC_ = BUFFER_NUM * (nPairSize_ == 0 ? 1UL : 2UL) * blockIdx_ * baseM_ * baseN_;

        block_.Init(tilingData);
        update_.template Init<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
            &tilingData->matmulTiling, block_.params_, nPairSize_);
        loop_ = 0; // all_gather_quant_batch_mat_mul.h循环调Init和Process，管理CV同步的计数器每次都要清零
    }

    __aicore__ inline void UpdateBatchOffset(uint64_t batchIndex, QBmmBlockOffset &offset) {
        uint64_t batchAOffset = batchIndex % batchA_;
        uint64_t batchBOffset = batchIndex % batchB_;
        uint64_t batchCOffset = batchIndex;
        offset.offsetA = offset.offsetA + batchAOffset * ka_ * m_;
        uint64_t weightBatchSize = static_cast<uint64_t>(ka_) * n_;
        if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
            weightBatchSize = bTrans
                ? static_cast<uint64_t>(DequantBmm::Align(n_, BMM_BLOCK_NUM)) * DequantBmm::Align(ka_, K0_INT8)
                : static_cast<uint64_t>(DequantBmm::Align(n_, K0_INT8)) * DequantBmm::Align(ka_, BMM_BLOCK_NUM);
        }
        offset.offsetB = offset.offsetB + batchBOffset * weightBatchSize;
        offset.offsetC = offset.offsetC + batchCOffset * m_ * n_;
        if (biasThreeDim_) {
            offset.offsetBias = offset.offsetBias + batchCOffset * n_;
        }
    }

    __aicore__ inline void Process() {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        // 带batch场景只需要在最开始初始化offsetWorkspaceC_和控制同步的loop
        bool pongSwitch = false;
        offsetWorkspaceC_ = BUFFER_NUM * (nPairSize_ == 0 ? 1UL : 2UL) * blockIdx_ * baseM_ * baseN_;
        loop_ = 0;
        uint64_t pingOffsetC = offsetWorkspaceC_;
        for (uint64_t batchIndex = 0; batchIndex < batch_; batchIndex++) {
            bool reverse = true;
            if (nPairSize_ != 0 && batchIndex != 0) {
                // 恢复首区块数，延续上一batch的余量分核游标。
                block_.UpdateBlockCnt(0, 0);
                block_.InitBlockIndex();
            } else {
                block_.InitFirstTileBlockIndex();
            }
            // 首块计算，兼容无L2cache切分场景，减少scalar计算
            OneTileCompute(0, 0, pingOffsetC, batchIndex, pongSwitch);

            for (uint64_t mTileIndex = 0; mTileIndex < block_.params_.mTileCntL2; mTileIndex++) {
                reverse = !reverse;
                for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < block_.params_.nTileCntL2; nTileIndexTemp++) {
                    uint64_t nTileIndex = reverse ? (block_.params_.nTileCntL2 - nTileIndexTemp - 1) : nTileIndexTemp;
                    if (mTileIndex > 0 || nTileIndex > 0) { // 跳过首块
                        block_.UpdateBlockCnt(mTileIndex, nTileIndex);
                        block_.InitBlockIndex();
                        OneTileCompute(mTileIndex, nTileIndex, pingOffsetC, batchIndex, pongSwitch);
                    }
                }
            }
        }

        End();
    }

    __aicore__ inline UPDATE_TYPE &GetUpdateObj() { return update_; }

  private:
    __aicore__ inline void InitTilingData(const EagleQuantBatchMatmulSwigluTilingData *tilingData) {
        isPerTensor_ = tilingData->params.isPerTensor;
        m_ = tilingData->matmulTiling.M;
        n_ = tilingData->matmulTiling.N;
        nPairSize_ = tilingData->tileL2cacheTiling.nPairSize;
        ka_ = tilingData->matmulTiling.Ka;

        baseM_ = tilingData->matmulTiling.baseM;
        baseN_ = tilingData->matmulTiling.baseN;
        batchA_ = tilingData->params.batchA;
        batchB_ = tilingData->params.batchB;
        batch_ = tilingData->params.batchC;
        biasThreeDim_ = tilingData->params.biasThreeDim;

        hasBias_ = tilingData->matmulTiling.isBias;
        biasDtype_ = tilingData->params.biasDtype;
        if (biasDtype_ == DT_INT32 || biasDtype_ == DT_FLOAT) {
            biasDtypeSize_ = sizeof(int32_t);
        } else {
            biasDtypeSize_ = sizeof(half);
        }

        ubCalcM_ = tilingData->params.ubCalcM;
        ubCalcN_ = tilingData->params.ubCalcN;
        ubTmpBuffer_ = tilingData->params.needUbBuffer;
    }

    __aicore__ inline void SetMatmulOrgShape() {
        if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ &&
            DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
            mm_.SetOrgShape(DequantBmm::Align(m_, BMM_BLOCK_NUM), DequantBmm::Align(n_, BMM_BLOCK_NUM),
                DequantBmm::Align(ka_, K0_INT8), DequantBmm::Align(ka_, K0_INT8), n_);
        } else if constexpr (DequantBmm::GetFormat(x1Format) == CubeFormat::NZ) {
            mm_.SetOrgShape(DequantBmm::Align(m_, K0_INT8), n_, DequantBmm::Align(ka_, BMM_BLOCK_NUM), ka_, n_);
        } else if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ) {
            if constexpr (bTrans) {
                // k1,n1,n0,k0 layout: orgN aligned, last param is full N (see mat_mul_v3 / weight_quant NZ)
                mm_.SetOrgShape(m_, DequantBmm::Align(n_, BMM_BLOCK_NUM), ka_, DequantBmm::Align(ka_, K0_INT8), n_);
            } else {
                // n1,k1,k0,n0 layout (k0=16): orgK aligned to k fractal
                mm_.SetOrgShape(m_, DequantBmm::Align(n_, K0_INT8), ka_, DequantBmm::Align(ka_, BMM_BLOCK_NUM), n_);
            }
        } else {
            mm_.SetOrgShape(m_, n_, ka_);
        }
    }

    __aicore__ inline void InitGlobalBuffers(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y, GM_ADDR workSpace) {
        if (isPerTensor_) {
            scaleScalar_ = *((__gm__ scaleType *)scale);
        }
        xGm_.SetGlobalBuffer((__gm__ x1Type *)x1);
        weightGm_.SetGlobalBuffer((__gm__ x2Type *)x2);
        if (m_ <= baseM_) {
            weightGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if (hasBias_ != 0U) {
            if (biasDtype_ == DT_BF16) {
                biasGmBf16_.SetGlobalBuffer((__gm__ bfloat16_t *)bias);
            } else if (biasDtype_ == DT_FLOAT16) {
                biasGmFp16_.SetGlobalBuffer((__gm__ half *)bias);
            } else if (biasDtype_ == DT_FLOAT) {
                biasGmFp32_.SetGlobalBuffer((__gm__ float *)bias);
            } else {
                biasGmInt32_.SetGlobalBuffer((__gm__ int32_t *)bias);
            }
        }
        yGm_.SetGlobalBuffer((__gm__ yType *)y);
        scaleGm_.SetGlobalBuffer((__gm__ scaleType *)scale);
        pertokenScaleGm_.SetGlobalBuffer((__gm__ float *)pertokenScale);
        mmOutGm_.SetGlobalBuffer(
            (__gm__ int32_t *)workSpace, BUFFER_NUM * (nPairSize_ == 0 ? 1UL : 2UL) * usedCoreNum_ * baseM_ * baseN_);
    }

    __aicore__ inline void InitLocalBuffers() {
        pipe_->InitBuffer(vecQueSrc_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(int32_t));
        pipe_->InitBuffer(vecQueTmp_, ubTmpBuffer_);
        if (UseInPlaceDequant()) {
            pipe_->InitBuffer(vecQueTmp2_, ubTmpBuffer_);
            pipe_->InitBuffer(swigluTmp_, ubTmpBuffer_);
        }
        pipe_->InitBuffer(vecQueOut_, BUFFER_NUM, ubCalcM_ * ubCalcN_ * sizeof(yType));
        if (biasDtype_ != DT_INT32) {
            pipe_->InitBuffer(biasFp32Tmp_, ubCalcN_ * sizeof(float));
            pipe_->InitBuffer(vecQueBias_, BUFFER_NUM, ubCalcN_ * biasDtypeSize_);
        }
        if (!isPerTensor_) {
            pipe_->InitBuffer(vecQueScale_, BUFFER_NUM, ubCalcN_ * sizeof(scaleType));
        }
        if (!UseInPlaceDequant()) {
            pipe_->InitBuffer(outFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
        }
        pipe_->InitBuffer(vecQuePertokenScale_, BUFFER_NUM, DequantBmm::Align(ubCalcM_, 8U) * sizeof(float));
        pipe_->InitBuffer(broadcastFp32Tmp_, ubCalcM_ * ubCalcN_ * sizeof(float));
    }

    __aicore__ inline bool UseInPlaceDequant() const {
        return AscendC::IsSameType<scaleType, float>::value &&
            AscendC::IsSameType<EpilogueType, EpilogueDequant>::value && !isPerTensor_ && nPairSize_ != 0;
    }

    __aicore__ inline void OneTileCompute(
        uint64_t mTileIndex, uint64_t nTileIndex, uint64_t pingOffsetC, uint64_t batchIndex, bool &pongSwitch) {
        for (uint64_t j = 0; j < block_.realRound_; j++) {
            if (nPairSize_ != 0) {
                PairedMMCVFusionCompute(mTileIndex, nTileIndex, pingOffsetC, batchIndex, pongSwitch);
            } else {
                update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
                    block_.params_, offset_, mTileIndex, nTileIndex);

                UpdateBatchOffset(batchIndex, offset_);
                offsetWorkspaceC_ = pingOffsetC + pongSwitch * baseM_ * baseN_;
                BasicMMCVFusionCompute(block_.params_.singleCoreM, block_.params_.singleCoreN,
                    C2V_PING_FLAG | pongSwitch, V2C_PING_FLAG | pongSwitch);
            }
            pongSwitch = !pongSwitch;
            block_.UpdateBlockIndex();
        }
    }

    __aicore__ inline void PairedMMCVFusionCompute(
        uint64_t mTileIndex, uint64_t nTileIndex, uint64_t pingOffsetC, uint64_t batchIndex, bool pongSwitch) {
        // Keep the existing event IDs: 4/5 release workspace, 6/7 publish Fixpipe results.
        uint16_t freeFlag = C2V_PING_FLAG | pongSwitch;
        uint16_t readyFlag = V2C_PING_FLAG | pongSwitch;
        uint64_t pairOffset = pingOffsetC + static_cast<uint64_t>(pongSwitch) * 2 * baseM_ * baseN_;
        update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
            block_.params_, offset_, mTileIndex, nTileIndex, 0);
        UpdateBatchOffset(batchIndex, offset_);
        QBmmBlockOffset leftOffset = offset_;
        uint32_t curM = block_.params_.singleCoreM;
        uint32_t curN = block_.params_.singleCoreN;
        if ASCEND_IS_AIC {
            if (++loop_ > BUFFER_NUM) { // Count pairs; the first ping and pong are initially free.
                WaitEvent(freeFlag);
            }
            // Keep one Matmul call site for both branches to limit duplicated inline code.
#pragma clang loop unroll(disable)
            for (uint32_t branch = 0; branch < 2; ++branch) {
                if (branch != 0) {
                    update_
                        .template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
                            block_.params_, offset_, mTileIndex, nTileIndex, nPairSize_);
                    UpdateBatchOffset(batchIndex, offset_);
                }
                offsetWorkspaceC_ = pairOffset + static_cast<uint64_t>(branch) * baseM_ * baseN_;
                BasicMMCompute(curM, curN);
            }
            NotifyEvent<PIPE_FIX>(readyFlag); // Both branches have reached GM before AIV consumes the pair.
        }
        if ASCEND_IS_AIV {
            update_.template UpdateBlockParamsAndCalcGmOffset<x1Format, x2Format, aTrans, bTrans, x1Type, x2Type>(
                block_.params_, offset_, mTileIndex, nTileIndex, nPairSize_);
            UpdateBatchOffset(batchIndex, offset_);
            offsetWorkspaceC_ = pairOffset + static_cast<uint64_t>(baseM_) * baseN_;
            WaitEvent(readyFlag);
            auto rightParams = InitEpilogueParams(curM, curN);
            offset_ = leftOffset;
            offsetWorkspaceC_ = pairOffset;
            auto leftParams = InitEpilogueParams(curM, curN);
            if constexpr (AscendC::IsSameType<EpilogueType, EpilogueDequant>::value) {
                epilogue_.ProcessPair(leftParams, rightParams, UseInPlaceDequant(), vecQueTmp2_, swigluTmp_);
            } else {
                // Preserve the existing Process-only contract of other fused epilogues.
                epilogue_.Process(leftParams);
                epilogue_.Process(rightParams);
            }
            // Both AIVs must notify, including an AIV with no rows in a short M tail.
            // MODE 2 releases the workspace only after both AIVs finish reading both branches.
            NotifyEvent<PIPE_MTE2>(freeFlag);
        }
    }

    __aicore__ inline EpilogueParams<yType, scaleType> InitEpilogueParams(uint32_t CurAicM, uint32_t CurAicN) {
        return EpilogueParams<yType, scaleType>(mmOutGm_, yGm_, biasGmBf16_, biasGmFp16_, biasGmFp32_, scaleGm_,
            pertokenScaleGm_, // GlobalTensor

            outFp32Tmp_, vecQueTmp_, biasFp32Tmp_,
            broadcastFp32Tmp_, // TBuf

            vecQueSrc_, vecQueOut_, vecQueBias_, vecQueScale_,
            vecQuePertokenScale_, // TQue

            CurAicM, CurAicN, subBlockIdx_, ubCalcM_, ubCalcN_, offsetWorkspaceC_, biasDtype_, biasDtypeSize_,
            isPerTensor_, scaleScalar_, offset_,
            n_ // 常量
        );
    }

    __aicore__ inline void BasicMMCVFusionCompute(
        uint32_t CurAicM, uint32_t CurAicN, uint16_t v2cSyncFlag, uint16_t c2vSyncFlag) {
        if ASCEND_IS_AIC {
            if (++loop_ > 2) { // 2表示跳过第一次ping和第一次pong
                WaitEvent(v2cSyncFlag);
            }
            BasicMMCompute(CurAicM, CurAicN);
            NotifyEvent<PIPE_FIX>(c2vSyncFlag);
        }

        if ASCEND_IS_AIV {
            WaitEvent(c2vSyncFlag);
            auto epilogueParams = InitEpilogueParams(CurAicM, CurAicN);
            epilogue_.Process(epilogueParams);
            NotifyEvent<PIPE_MTE2>(v2cSyncFlag);
        }
    }

    __aicore__ inline void BasicMMCompute(uint32_t baseM, uint32_t baseN) {
        if constexpr (DequantBmm::GetFormat(x2Format) == CubeFormat::NZ && !bTrans) {
            mm_.SetOrgShape(m_, baseN, ka_, DequantBmm::Align(ka_, BMM_BLOCK_NUM), n_);
        }
        mm_.SetSingleShape(baseM, baseN, ka_);
        mm_.SetTail(baseM, baseN, ka_);
        mm_.SetTensorA(xGm_[offset_.offsetA], aTrans);
        mm_.SetTensorB(weightGm_[offset_.offsetB], bTrans);

        if (hasBias_ != 0 && biasDtype_ == DT_INT32) {
            mm_.SetBias(biasGmInt32_[offset_.offsetBias]);
        }
        mm_.Iterate();
        mm_.GetTensorC(mmOutGm_[offsetWorkspaceC_], 0, true);
    }

    __aicore__ inline void End() {
        if ASCEND_IS_AIC {
            // AIC跳过前两次Wait，也就是一次ping一次pong，这里补上
            if (loop_ > 0) { // 大于0表示需要补上开头跳过的ping
                WaitEvent(C2V_PING_FLAG);
            }

            if (loop_ > 1) { // 大于1表示需要补上开头跳过的pong
                WaitEvent(C2V_PONG_FLAG);
            }
            mm_.End();
        }
    }

  private:
    GlobalTensor<x1Type> xGm_;
    GlobalTensor<x2Type> weightGm_;
    GlobalTensor<int32_t> biasGmInt32_;
    GlobalTensor<bfloat16_t> biasGmBf16_;
    GlobalTensor<half> biasGmFp16_;
    GlobalTensor<float> biasGmFp32_;
    GlobalTensor<yType> yGm_;
    GlobalTensor<scaleType> scaleGm_;
    GlobalTensor<float> pertokenScaleGm_;
    GlobalTensor<int32_t> mmOutGm_;

    TPipe *pipe_;
    // define the que
    TQue<QuePosition::VECIN, 1> vecQueSrc_;
    TQue<QuePosition::VECIN, 1> vecQueScale_;
    TQue<QuePosition::VECIN, 1> vecQueBias_;
    TBuf<TPosition::VECCALC> vecQueTmp_;
    TBuf<TPosition::VECCALC> vecQueTmp2_;
    TBuf<TPosition::VECCALC> swigluTmp_;
    TQue<QuePosition::VECOUT, 1> vecQueOut_;
    // used when bias type is bf16/fp16/fp32, dequant result should be fp32
    TBuf<TPosition::VECCALC> biasFp32Tmp_;
    TBuf<TPosition::VECCALC> outFp32Tmp_;
    TQue<QuePosition::VECIN, 1> vecQuePertokenScale_;
    TBuf<TPosition::VECCALC> broadcastFp32Tmp_;

    scaleType scaleScalar_;

    // tilling data
    bool isPerTensor_;
    uint32_t usedCoreNum_;
    uint32_t m_;
    uint32_t n_;
    uint32_t nPairSize_ = 0;
    uint32_t ka_;
    uint32_t baseM_;
    uint32_t baseN_;
    uint32_t hasBias_;
    uint32_t biasDtype_ = 0;
    uint32_t biasDtypeSize_ = 0;
    uint32_t batch_ = 1;
    uint32_t batchA_ = 1;
    uint32_t batchB_ = 1;
    uint32_t biasThreeDim_;
    // vector
    uint32_t ubCalcM_;
    uint32_t ubCalcN_;
    uint32_t ubTmpBuffer_;

    uint32_t blockIdx_;
    uint32_t subBlockIdx_ = 0;
    uint64_t offsetWorkspaceC_ = 0;
    uint64_t loop_ = 0;

    EagleQuantBatchMatmulSwigluBaseBlock block_;
    UPDATE_TYPE update_; // 量化mm或mc2的更新计算大小和地址的接口
    QBmmBlockOffset offset_;
    EpilogueType epilogue_;

    using AMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, x1Type, aTrans>;
    using BMatmulType = matmul::MatmulType<TPosition::GM, DequantBmm::GetFormat(x2Format), x2Type, bTrans>;
    using BiasMatmulType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
    // notice: the TPos of ctype must be ub given by mm api when iterate<false>,
    // but actually we can move data to gm then to ub.
    using CMatmulType = matmul::MatmulType<TPosition::VECIN, CubeFormat::ND, int32_t>;
    matmul::MatmulImpl<AMatmulType, BMatmulType, CMatmulType, BiasMatmulType, MM_DEFAULT_MDL_CFG> mm_;
};

} // namespace AscendC

#endif // EAGLE_QUANT_BATCH_MATMUL_SWIGLU_PERTOKEN_BASIC_H
