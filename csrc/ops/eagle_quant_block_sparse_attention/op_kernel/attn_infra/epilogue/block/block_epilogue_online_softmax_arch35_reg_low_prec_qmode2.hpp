/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_QMODE2_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_QMODE2_HPP

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/tile_copy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "../../../tla/tensor.hpp"
#include "../../../tla/layout.hpp"
#include "vf/vf_softmax_mxfp4_postupdate.h"
#include "vf/vf_softmax_mxfp4_postupdate_approx.h"

namespace NpuArch::Epilogue::Block {

template <QMode2SoftmaxAlgo ALGO> struct Mxfp4Dispatcher {
    template <typename T>
    static __aicore__ inline void SoftmaxPadCallVF(const AscendC::LocalTensor<T> &src, uint16_t n) {
        if constexpr (ALGO == QMode2SoftmaxAlgo::APPROX) {
            Mxfp4ApiApprox::SoftmaxPadCallVF(src, n);
        } else {
            Mxfp4Api::SoftmaxPadCallVF(src, n);
        }
    }

    template <typename T, typename T2>
    static __aicore__ inline void SoftmaxCallVF256TileRaw(__ubuf__ T2 *pDest, __ubuf__ T *src,
        __ubuf__ T *localGroupMax, __ubuf__ T *globalMaxCur, __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr,
        __ubuf__ T *tmp, __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff, bool isFirst, uint16_t forceUpdate, uint16_t n,
        half clip, uint32_t qtail) {
        if constexpr (ALGO == QMode2SoftmaxAlgo::APPROX) {
            Mxfp4ApiApprox::SoftmaxCallVF256TileRaw(pDest, src, localGroupMax, globalMaxCur, globalMaxHis, idxGroupPtr,
                tmp, mxDest, expDiff, isFirst, forceUpdate, n, clip, qtail);
        } else {
            Mxfp4Api::SoftmaxCallVF256TileRaw(pDest, src, localGroupMax, globalMaxCur, globalMaxHis, idxGroupPtr, tmp,
                mxDest, expDiff, isFirst, forceUpdate, n, clip, qtail);
        }
    }

    template <typename T, typename T2>
    static __aicore__ inline void SoftmaxCallVFDeferredRaw(__ubuf__ T2 *pDest, __ubuf__ T *src,
        __ubuf__ T *localGroupMax, __ubuf__ T *globalMaxCur, __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr,
        __ubuf__ T *tmp, __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff, bool isFirst, uint16_t forceUpdate, uint16_t n,
        half clip, uint32_t qtail) {
        if constexpr (ALGO == QMode2SoftmaxAlgo::APPROX) {
            Mxfp4ApiApprox::SoftmaxCallVFDeferredRaw(pDest, src, localGroupMax, globalMaxCur, globalMaxHis, idxGroupPtr,
                tmp, mxDest, expDiff, isFirst, forceUpdate, n, clip, qtail);
        } else {
            Mxfp4Api::SoftmaxCallVFDeferredRaw(pDest, src, localGroupMax, globalMaxCur, globalMaxHis, idxGroupPtr, tmp,
                mxDest, expDiff, isFirst, forceUpdate, n, clip, qtail);
        }
    }
};

template <class OutputType_, class LayoutS_, QMode2SoftmaxAlgo ALGO>
class BlockEpilogue<EpilogueOnlineSoftmaxBsaQMode2<ALGO>, OutputType_, Gemm::GemmType<half, LayoutS_>> {
  public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsaQMode2<ALGO>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element;
    using ElementOutputTmp = half;
    using ElementInput = half;
    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = LayoutS_;

    static constexpr uint32_t MX_BLOCK_SIZE = 32;
    static constexpr uint32_t EVEN = 2;
    static constexpr uint32_t DB = 2;
    static constexpr uint32_t FP4_BLOCK_SIZE = 64;
    static constexpr uint32_t VL_INT8 = 256;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64 * 2;
    static constexpr ElementInput MIN_VALUE = -60000.0f;
    static constexpr uint32_t SM_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t SM_COL_MAX_ELEM_NUM = 256;
    static constexpr uint32_t MAX_UB_S_ELEM_NUM = SM_COL_MAX_ELEM_NUM * SM_ROW_MAX_ELEM_NUM;
    static constexpr uint32_t MAX_P_L1_BUF_STAGES = 4;
    static constexpr uint32_t S_UB_SLOT_STAGES = 3;
    static constexpr uint32_t O_UB_TMP_STAGES = 2;
    static constexpr uint32_t S_UB_SLOT_BYTES = MAX_UB_S_ELEM_NUM * sizeof(ElementInput);
    static constexpr uint32_t P_UB_DATA_BYTES = MAX_UB_S_ELEM_NUM / 2;
    static constexpr uint32_t P_UB_SCALE_BYTES = MAX_UB_S_ELEM_NUM / MX_BLOCK_SIZE;
    static constexpr uint32_t P_UB_PAYLOAD_BYTES = P_UB_DATA_BYTES + P_UB_SCALE_BYTES;
    static constexpr uint32_t P_UB_GROUP_MAX_BYTES = MAX_UB_S_ELEM_NUM / MX_BLOCK_SIZE * sizeof(ElementInput);
    static constexpr uint32_t TMP_UB_SLOT_BYTES = 32;
    static constexpr ElementInput MX_MAX = 100;
    // Mode-4 AIV flag IDs are limited to [0, 10]. IDs 0..8 are occupied by
    // QK/Softmax/PV/Rescale handshakes.
    static constexpr uint16_t AIV_UPDATE_SYNC_FLAG_ID = 9;
    static constexpr uint16_t SSBUF_READY_FLAG_ID = 10;
    static constexpr uint16_t SSBUF_SYNC_EVENT_ID = EVENT_ID7;
    // SSBUF loads and stores must use 32-byte aligned addresses.
    static constexpr uint32_t SSBUF_SLOT_BYTES = 32;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {
        constexpr uint32_t LS_UB_TENSOR_OFFSET = 0;
        constexpr uint32_t HEAD_DIM = 128, SUM_DIM = 8;
        constexpr uint32_t O_CURR_UB_TENSOR_OFFSET = S_UB_SLOT_BYTES * S_UB_SLOT_STAGES; // 32 * 3
        constexpr uint32_t O_HIS_UB_TENSOR_OFFSET = O_CURR_UB_TENSOR_OFFSET +
            SM_ROW_MAX_ELEM_NUM * (HEAD_DIM + SUM_DIM) * O_UB_TMP_STAGES * sizeof(float); // 34 * 2
        constexpr uint32_t P_PAYLOAD_UB_TENSOR_OFFSET =
            O_HIS_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * HEAD_DIM * sizeof(float); // 32
        constexpr uint32_t GROUP_MAX_UB_TENSOR_OFFSET =
            P_PAYLOAD_UB_TENSOR_OFFSET + P_UB_PAYLOAD_BYTES * S_UB_SLOT_STAGES; // 8.5 * 3
        constexpr uint32_t GATHER_INDEX_OFFSET = GROUP_MAX_UB_TENSOR_OFFSET + P_UB_GROUP_MAX_BYTES; // 2
        constexpr uint32_t LM_UB_TENSOR_OFFSET = GATHER_INDEX_OFFSET + VL_INT8; // gather index 256B
        constexpr uint32_t GM_UB_TENSOR_OFFSET =
            LM_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * EVEN * sizeof(ElementInput); // local_max 256B
        constexpr uint32_t VSCALE_UB_TENSOR_OFFSET =
            GM_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * EVEN * sizeof(ElementInput); // global_max 256B
        constexpr uint32_t GL_UB_TENSOR_OFFSET = VSCALE_UB_TENSOR_OFFSET + HEAD_DIM * sizeof(float) * DB; // VSCALE 1k
        constexpr uint32_t DM_UB_TENSOR_OFFSET = GL_UB_TENSOR_OFFSET + SM_ROW_MAX_ELEM_NUM * sizeof(float); // sum 256B
        constexpr uint32_t TMP_UB_TENSOR_OFFSET =
            DM_UB_TENSOR_OFFSET + MAX_P_L1_BUF_STAGES * DM_UB_GLOBAL_ELEM_NUM * sizeof(ElementInput);
        constexpr uint32_t UB_LAYOUT_END = TMP_UB_TENSOR_OFFSET + TMP_UB_SLOT_BYTES * S_UB_SLOT_STAGES;

        subBlockIdx_ = AscendC::GetSubBlockIdx();
        lsUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LS_UB_TENSOR_OFFSET);
        pPayloadUbTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(P_PAYLOAD_UB_TENSOR_OFFSET);
        gatherIndexTensor = resource.ubBuf.template GetBufferByByte<uint8_t>(GATHER_INDEX_OFFSET);
        groupMaxUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(GROUP_MAX_UB_TENSOR_OFFSET);
        curMaxUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(LM_UB_TENSOR_OFFSET);
        hisMaxUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(GM_UB_TENSOR_OFFSET);
        sumUbTensor = resource.ubBuf.template GetBufferByByte<float>(GL_UB_TENSOR_OFFSET);
        expDiffUbTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(DM_UB_TENSOR_OFFSET);
        tmpTensor = resource.ubBuf.template GetBufferByByte<ElementInput>(TMP_UB_TENSOR_OFFSET);
        // for (uint32_t slot = 0; slot < S_UB_SLOT_STAGES; ++slot) {
        //     softmaxPending_[slot] = false;
        // }
        Mxfp4Api::GetIdxGroup(gatherIndexTensor);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class Mx>
    __aicore__ inline void CopyPUbToPL1(const AscendC::LocalTensor<uint8_t> &l1p, const AscendC::LocalTensor<Mx> &l1MxP,
        const AscendC::LocalTensor<uint8_t> &ubP, const AscendC::LocalTensor<uint8_t> &ubMx, const uint32_t mRound,
        const uint32_t nRound) {
        //! /2 for fp4x2
        auto offset = SM_ROW_MAX_ELEM_NUM * nRound * subBlockIdx_;
        auto l1MxPu8 = l1MxP.template ReinterpretCast<uint8_t>();
        DataCopy(l1p[offset / 2], ubP, mRound * nRound / 2);
        DataCopy(l1MxPu8[offset / MX_BLOCK_SIZE], ubMx, mRound * nRound / MX_BLOCK_SIZE);
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void SetCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreSetFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    template <uint32_t MODE, pipe_t PIPE> __aicore__ inline void WaitCrossCoreSync(Arch::CrossCoreFlag &crossCoreFlag) {
        if constexpr (MODE == 4U) {
            Arch::CrossCoreWaitFlag<MODE, PIPE>(crossCoreFlag);
        }
    }

    __aicore__ inline void IssueSoftmax(GemmCoord actualBlockShape, bool isFirstKvSTile, uint32_t ubSBufId,
        uint32_t l1PBufId, Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag) {
        const uint32_t mTot = actualBlockShape.m();
        const uint32_t m = (mTot <= FP4_BLOCK_SIZE) ? ((subBlockIdx_ == 0) ? mTot : 0U)
                                                    : ((subBlockIdx_ == 0) ? FP4_BLOCK_SIZE : mTot - FP4_BLOCK_SIZE);

        // An inactive AIV has no Vector work to defer.  Preserve the existing
        // zero message and return both cross-core resources synchronously.
        if (m == 0U) {
            WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            SetCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
            WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            auto ssbufBase = reinterpret_cast<__ssbuf__ uint8_t *>(AscendC::GetSsbufBaseAddr());
            const uint32_t msgSlot = subBlockIdx_ + 2U * l1PBufId;
            auto updateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(ssbufBase + msgSlot * SSBUF_SLOT_BYTES);
            *updateMsg = 0U;
            isUpdateArr[l1PBufId] = 0U;
            AscendC::CrossCoreSetFlag<4, PIPE_S>(SSBUF_READY_FLAG_ID);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SSBUF_SYNC_EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SSBUF_SYNC_EVENT_ID);
            AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(AIV_UPDATE_SYNC_FLAG_ID);
            SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
            // softmaxPending_[ubSBufId] = false;
            return;
        }

        const uint32_t n = actualBlockShape.n();
        const uint32_t nRound = RoundUp(n, 128);
        WaitCrossCoreSync<4, PIPE_V>(mm1ToSmFlag);
        auto sSlot = lsUbTensor[ubSBufId * MAX_UB_S_ELEM_NUM];
        if (unlikely(n < nRound)) {
            Mxfp4Dispatcher<ALGO>::SoftmaxPadCallVF(sSlot, n);
        }

        auto payload = pPayloadUbTensor[ubSBufId * P_UB_PAYLOAD_BYTES];
        auto tmpSlot = tmpTensor[ubSBufId * (TMP_UB_SLOT_BYTES / sizeof(ElementInput))];
        if (unlikely(isFirstKvSTile)) {
            AscendC::Duplicate(hisMaxUbTensor, MIN_VALUE, SM_ROW_MAX_ELEM_NUM * EVEN);
            AscendC::Duplicate(sumUbTensor, 0.0f, SM_ROW_MAX_ELEM_NUM);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if (nRound == SM_COL_MAX_ELEM_NUM) {
            Mxfp4Dispatcher<ALGO>::SoftmaxCallVF256TileRaw(
                reinterpret_cast<__ubuf__ float4_e2m1x2_t *>(payload.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(sSlot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(groupMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(curMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(hisMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint8_t *>(gatherIndexTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(tmpSlot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint8_t *>(payload.GetPhyAddr()) + P_UB_DATA_BYTES,
                reinterpret_cast<__ubuf__ ElementInput *>(
                    expDiffUbTensor[DM_UB_GLOBAL_ELEM_NUM * l1PBufId].GetPhyAddr()),
                isFirstKvSTile, 0, (uint16_t)nRound, MX_MAX, m);
        } else {
            Mxfp4Dispatcher<ALGO>::SoftmaxCallVFDeferredRaw(
                reinterpret_cast<__ubuf__ float4_e2m1x2_t *>(payload.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(sSlot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(groupMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(curMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(hisMaxUbTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint8_t *>(gatherIndexTensor.GetPhyAddr()),
                reinterpret_cast<__ubuf__ ElementInput *>(tmpSlot.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint8_t *>(payload.GetPhyAddr()) + P_UB_DATA_BYTES,
                reinterpret_cast<__ubuf__ ElementInput *>(
                    expDiffUbTensor[DM_UB_GLOBAL_ELEM_NUM * l1PBufId].GetPhyAddr()),
                isFirstKvSTile, 0, (uint16_t)nRound, MX_MAX, m);
        }
        if (unlikely(n < nRound)) {
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(payload[n * 32], (uint8_t)0, (nRound - n) * 32);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(ubSBufId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        // softmaxPending_[ubSBufId] = true;
    }

    template <class TensorP, class TensorMx>
    __aicore__ inline void ResolveSoftmax(TensorP const &l1PTensor, TensorMx const &l1MxPTensor,
        uint32_t kvSTileSizeAct, uint32_t mTot, bool isFirstKvSTile, uint32_t ubSBufId, uint32_t l1PBufId,
        Arch::CrossCoreFlag mm1ToSmFlag, Arch::CrossCoreFlag smToMm2Flag) {
        const uint32_t m = (mTot <= FP4_BLOCK_SIZE) ? ((subBlockIdx_ == 0) ? mTot : 0U)
                                                    : ((subBlockIdx_ == 0) ? FP4_BLOCK_SIZE : mTot - FP4_BLOCK_SIZE);
        if (unlikely(m == 0U)) { // unlikely(!softmaxPending_[ubSBufId])
            return;
        }
        const uint16_t mRound = RoundUp(m, FP4_BLOCK_SIZE);
        const uint16_t nRound = RoundUp(kvSTileSizeAct, 128); // n必须按照128对齐

        AscendC::WaitFlag<AscendC::HardEvent::V_S>(ubSBufId);
        auto tmpSlot = tmpTensor[ubSBufId * (TMP_UB_SLOT_BYTES / sizeof(ElementInput))];
        const auto tmpPtr = reinterpret_cast<__ubuf__ ElementInput *>(tmpSlot.GetPhyAddr());
        const uint32_t isUpdateO = isFirstKvSTile ? 0U : (tmpPtr[0] >= static_cast<ElementInput>(MX_MAX) ? 1U : 0U);
        isUpdateArr[l1PBufId] = isUpdateO;

        auto ssbufBase = reinterpret_cast<__ssbuf__ uint8_t *>(AscendC::GetSsbufBaseAddr());
        const uint32_t msgSlot = subBlockIdx_ + 2U * l1PBufId;
        auto updateMsg = reinterpret_cast<__ssbuf__ uint32_t *>(ssbufBase + msgSlot * SSBUF_SLOT_BYTES);
        *updateMsg = isUpdateO;
        AscendC::CrossCoreSetFlag<4, PIPE_S>(SSBUF_READY_FLAG_ID);
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(SSBUF_SYNC_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(SSBUF_SYNC_EVENT_ID);

        WaitCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(ubSBufId);
        auto payload = pPayloadUbTensor[ubSBufId * P_UB_PAYLOAD_BYTES];
        auto slotP = payload.template ReinterpretCast<uint8_t>();
        auto slotMx = payload[P_UB_DATA_BYTES];
        CopyPUbToPL1(l1PTensor.template ReinterpretCast<uint8_t>(), l1MxPTensor, slotP, slotMx, mRound, nRound);
        AscendC::CrossCoreSetFlag<1, PIPE_MTE3>(AIV_UPDATE_SYNC_FLAG_ID);
        SetCrossCoreSync<4, PIPE_MTE3>(mm1ToSmFlag);
        SetCrossCoreSync<4, PIPE_MTE3>(smToMm2Flag);
        // softmaxPending_[ubSBufId] = false;
    }

  private:
    uint32_t subBlockIdx_;
    uint32_t isUpdateArr[MAX_P_L1_BUF_STAGES];
    AscendC::LocalTensor<ElementInput> lsUbTensor;
    AscendC::LocalTensor<uint8_t> pPayloadUbTensor;
    AscendC::LocalTensor<uint8_t> gatherIndexTensor;
    AscendC::LocalTensor<ElementInput> groupMaxUbTensor;
    AscendC::LocalTensor<ElementInput> curMaxUbTensor;
    AscendC::LocalTensor<ElementInput> hisMaxUbTensor;
    AscendC::LocalTensor<float> sumUbTensor;
    AscendC::LocalTensor<ElementInput> expDiffUbTensor;
    AscendC::LocalTensor<ElementInput> tmpTensor;
    // bool softmaxPending_[S_UB_SLOT_STAGES];
};
}

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_BSA_LOW_PREC_HPP
