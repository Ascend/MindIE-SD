/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */

#ifndef ADA_RELU2_BLOCK_ESTIMATE_REGULAR_H
#define ADA_RELU2_BLOCK_ESTIMATE_REGULAR_H

#include "ada_relu2_block_estimate.h"

struct AdaRelu2Params {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR actualSeqLengths;
    GM_ADDR actualSeqLengthsKV;
    GM_ADDR qScale;
    GM_ADDR kScale;
    GM_ADDR score;
    GM_ADDR rowSum;
    GM_ADDR workspace;
    GM_ADDR tiling;

    __aicore__ inline AdaRelu2Params(GM_ADDR queryAddr, GM_ADDR keyAddr, GM_ADDR actualSeqLengthsAddr,
        GM_ADDR actualSeqLengthsKVAddr, GM_ADDR qScaleAddr, GM_ADDR kScaleAddr, GM_ADDR scoreAddr, GM_ADDR rowSumAddr,
        GM_ADDR workspaceAddr, GM_ADDR tilingAddr)
        : query(queryAddr), key(keyAddr), actualSeqLengths(actualSeqLengthsAddr),
          actualSeqLengthsKV(actualSeqLengthsKVAddr), qScale(qScaleAddr), kScale(kScaleAddr), score(scoreAddr),
          rowSum(rowSumAddr), workspace(workspaceAddr), tiling(tilingAddr) {}
};

template <typename ElementQ_, typename ElementK_, typename ElementS_> class AdaRelu2RegularKernel {
  public:
    using ElementQ = ElementQ_;
    using ElementK = ElementK_;
    using ElementS = ElementS_;

    static constexpr uint32_t CUBE_BLOCK_SIZE = 16;
    static constexpr uint32_t MAX_Q_BLOCK_NUM_PER_TILE = 8;
    static constexpr uint32_t Q_BLOCK_NUM_PER_VEC = MAX_Q_BLOCK_NUM_PER_TILE / 2U;
    static constexpr uint32_t M_L0_SLICE_NUM = 2;
    static constexpr uint32_t L0_S_BUF_NUM = 4;
    static constexpr uint32_t L0_S_BUF_SIZE = 128U * 128U * sizeof(ElementS);
    static constexpr uint32_t S_UB_BUF_NUM = L0_S_BUF_NUM;
    static constexpr uint32_t K_L1_BUF_NUM = 8;
    static constexpr uint32_t MAX_KV_BLOCK_NUM_PER_L1_TILE = 16;
    static constexpr uint32_t S_UB_BUF_SIZE = 128U * 64U;
    static constexpr uint32_t S_UB_BUF_BYTES = S_UB_BUF_SIZE * sizeof(ElementS);
    static constexpr uint32_t S_UB_TOTAL_BYTES = S_UB_BUF_NUM * S_UB_BUF_BYTES;
    static constexpr uint32_t UB_BYTES = 248 * 1024;
    static constexpr uint32_t SCORE_CHUNK_SIZE = Q_BLOCK_NUM_PER_VEC * MAX_KV_BLOCK_NUM_PER_L1_TILE * CUBE_BLOCK_SIZE;
    static constexpr uint32_t SCORE_CHUNK_BYTES = SCORE_CHUNK_SIZE * sizeof(ElementS);
    static constexpr uint32_t SCORE_BUF_NUM = 2;
    static constexpr uint32_t ROW_SUM_SIZE = Q_BLOCK_NUM_PER_VEC * CUBE_BLOCK_SIZE;
    static constexpr uint32_t ROW_SUM_BYTES = ROW_SUM_SIZE * sizeof(ElementS);
    // B/C lifetime: two compact [4, 16, 16] score tiles, [4, 16] row sum, then four 32 KB S buffers.
    static constexpr uint32_t SCORE_OFFSET = 0;
    static constexpr uint32_t ROW_SUM_OFFSET = SCORE_OFFSET + SCORE_BUF_NUM * SCORE_CHUNK_BYTES;
    static constexpr uint32_t S_UB_OFFSET = ROW_SUM_OFFSET + ROW_SUM_BYTES;
    static constexpr uint64_t qFlip2mmFlag = 0;
    static constexpr uint64_t mm2ReluFlag = 2;
    static constexpr uint64_t relu2mmFlag = 3;
    static constexpr uint64_t AIV1_FLAG_OFFSET = 16;
    static constexpr AscendC::FixpipeConfig FIX_TO_NZ_UB = {AscendC::CO2Layout::NZ, true};

    static_assert(S_UB_OFFSET + S_UB_TOTAL_BYTES <= UB_BYTES, "nL1 S UB buffers exceed 248 KB");

    __aicore__ inline AdaRelu2RegularKernel() {
#ifdef __DAV_CUBE__
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event);
        }
        for (uint32_t event = 0; event < L0_S_BUF_NUM; ++event) {
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(event);
        }
        for (uint32_t event = 0; event < K_L1_BUF_NUM; ++event) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        }
#endif
    }

    __aicore__ inline ~AdaRelu2RegularKernel() {
#ifdef __DAV_CUBE__
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event);
        }
        for (uint32_t event = 0; event < L0_S_BUF_NUM; ++event) {
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(event);
        }
        for (uint32_t event = 0; event < K_L1_BUF_NUM; ++event) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        }
#endif
    }

    __aicore__ inline void Init(const AdaRelu2Params &params) {
        queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementQ *>(params.query));
        keyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementK *>(params.key));
        // Bind the optional actual Q-length tensor only when the caller provides it.
        if (params.actualSeqLengths != nullptr) {
            actualSeqLengthsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.actualSeqLengths));
        }
        // Bind the optional actual KV-length tensor only when the caller provides it.
        if (params.actualSeqLengthsKV != nullptr) {
            actualSeqLengthsKVGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(params.actualSeqLengthsKV));
        }
        scoreGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS *>(params.score));
        rowSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementS *>(params.rowSum));
        workspace_ = params.workspace;

        GET_TILING_DATA(tilingData, params.tiling);
        batchSize_ = tilingData.batchSize;
        seqLenQ_ = tilingData.seqLenQ;
        seqLenKV_ = tilingData.seqLenKV;
        headNumQ_ = tilingData.headNumQ;
        headNumKV_ = tilingData.headNumKV;
        headDim_ = tilingData.headDim;
        actualSeqLengthsSize_ = tilingData.actualSeqLengthsSize;
        actualSeqLengthsKVSize_ = tilingData.actualSeqLengthsKVSize;
        stride_ = tilingData.stride;
        sparseSize_ = tilingData.sparseSize;
        scaleValue_ = tilingData.scaleValue;
        threshold_ = tilingData.threshold;
        qTileSize_ = tilingData.qTileSize;
        qLoadChunk_ = tilingData.qLoadChunk;
        kvTileSize_ = tilingData.kvTileSize;
        mL0TileSize_ = tilingData.mL0TileSize;
        nL0TileSize_ = tilingData.nL0TileSize;
        kL0TileSize_ = tilingData.kL0TileSize;
        kL0LoopNum_ = tilingData.kL0LoopNum;
        mL1SliceNum_ = tilingData.mL1SliceNum;
        qBlockNum_ = tilingData.qBlockNum;
        scoredKvBlockNum_ = tilingData.scoredKvBlockNum;
        usedCoreNum_ = tilingData.usedCoreNum;
        totalTaskNum_ = tilingData.totalTaskNum;

        kL1TileSize_ = stride_ * headDim_;
        qSTileSizeAlign_ =
            ((qTileSize_ + stride_ - 1U) / stride_ + CUBE_BLOCK_SIZE - 1U) / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
        qBlockNumPerTile_ = qTileSize_ / sparseSize_;
        kvBlockRowNum_ = sparseSize_ / stride_;
        kvBlockNumPerTile_ = kvTileSize_ / sparseSize_;
        kvSTileSizeAlign_ = kvTileSize_ / stride_;
        qRawBufSize_ = qLoadChunk_ * headDim_ * sizeof(ElementQ);
        l1QBufSize_ = qSTileSizeAlign_ * kL1TileSize_ * sizeof(ElementQ);
        l1KBufSize_ = mL0TileSize_ * kL0TileSize_ * sizeof(ElementK);
        l0QKBufSize_ = mL0TileSize_ * kL0TileSize_ * sizeof(ElementQ);

        // Q occupies 256 KB of L1; eight 32 KB K buffers occupy the following 256 KB.
        l1QTensor_ = AscendC::LocalTensor<ElementQ>(AscendC::TPosition::B1, 0, qSTileSizeAlign_ * kL1TileSize_);
        for (uint32_t i = 0; i < S_UB_BUF_NUM; ++i) {
            sTensorSave_[i] = AscendC::LocalTensor<ElementS>(
                AscendC::TPosition::VECOUT, S_UB_OFFSET + i * S_UB_BUF_BYTES, S_UB_BUF_SIZE);
        }
#ifdef __DAV_CUBE__
        for (uint32_t i = 0; i < K_L1_BUF_NUM; ++i) {
            l1KTensor_[i] = AscendC::LocalTensor<ElementK>(
                AscendC::TPosition::A1, l1QBufSize_ + i * l1KBufSize_, mL0TileSize_ * kL0TileSize_);
        }
        for (uint32_t i = 0; i < 2; ++i) {
            l0QTensor_[i] =
                AscendC::LocalTensor<ElementQ>(AscendC::TPosition::B2, i * l0QKBufSize_, mL0TileSize_ * kL0TileSize_);
        }
        l0KTensor_[0] = AscendC::LocalTensor<ElementK>(AscendC::TPosition::A2, 0, mL0TileSize_ * kL0TileSize_);
        l0KTensor_[1] =
            AscendC::LocalTensor<ElementK>(AscendC::TPosition::A2, l0QKBufSize_, mL0TileSize_ * kL0TileSize_);
        for (uint32_t i = 0; i < L0_S_BUF_NUM; ++i) {
            l0STensor_[i] =
                AscendC::LocalTensor<ElementS>(AscendC::TPosition::CO1, i * L0_S_BUF_SIZE, mL0TileSize_ * nL0TileSize_);
        }
#endif
#ifdef __DAV_VEC__
        qRawTensor_[0] = AscendC::LocalTensor<ElementQ>(AscendC::TPosition::VECIN, 0, qRawBufSize_ / sizeof(ElementQ));
        qRawTensor_[1] =
            AscendC::LocalTensor<ElementQ>(AscendC::TPosition::VECIN, qRawBufSize_, qRawBufSize_ / sizeof(ElementQ));
        qTmpTensor_ = AscendC::LocalTensor<ElementQ>(
            AscendC::TPosition::VECOUT, 2U * qRawBufSize_, qRawBufSize_ / sizeof(ElementQ));
        for (uint32_t scoreBufId = 0; scoreBufId < SCORE_BUF_NUM; ++scoreBufId) {
            sScoreTensor_[scoreBufId] = AscendC::LocalTensor<ElementS>(
                AscendC::TPosition::VECOUT, SCORE_OFFSET + scoreBufId * SCORE_CHUNK_BYTES, SCORE_CHUNK_SIZE);
        }
        sSumTensor_ = AscendC::LocalTensor<ElementS>(AscendC::TPosition::VECIN, ROW_SUM_OFFSET, ROW_SUM_SIZE);
#endif
    }

    __aicore__ inline void Process() {
        curBatch_ = 0;
        qSeqLen_ =
            actualSeqLengthsSize_ != 0 ? static_cast<uint32_t>(actualSeqLengthsGm_.GetValue(curBatch_)) : seqLenQ_;
        kvSeqLen_ =
            actualSeqLengthsKVSize_ != 0 ? static_cast<uint32_t>(actualSeqLengthsKVGm_.GetValue(curBatch_)) : seqLenKV_;
        qSeqLen_ = qSeqLen_ > seqLenQ_ ? seqLenQ_ : qSeqLen_;
        kvSeqLen_ = kvSeqLen_ > seqLenKV_ ? seqLenKV_ : kvSeqLen_;
        uint32_t curQSTileNum = (qSeqLen_ + qTileSize_ - 1U) / qTileSize_;
        uint64_t curTotalTaskNum = static_cast<uint64_t>(curQSTileNum) * headNumQ_;
        uint64_t preTotalTaskNum = 0;

        uint32_t logicCoreIdx = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        // Map the independently numbered AIVs back to their shared logical AI Core index.
        logicCoreIdx /= static_cast<uint32_t>(AscendC::GetSubBlockNum());
#endif
        for (uint64_t taskIdx = logicCoreIdx; taskIdx < totalTaskNum_; taskIdx += usedCoreNum_) {
            while (taskIdx >= preTotalTaskNum + curTotalTaskNum) {
                preTotalTaskNum += curTotalTaskNum;
                ++curBatch_;
                qSeqLen_ = actualSeqLengthsSize_ != 0 ? static_cast<uint32_t>(actualSeqLengthsGm_.GetValue(curBatch_))
                                                      : seqLenQ_;
                kvSeqLen_ = actualSeqLengthsKVSize_ != 0
                    ? static_cast<uint32_t>(actualSeqLengthsKVGm_.GetValue(curBatch_))
                    : seqLenKV_;
                qSeqLen_ = qSeqLen_ > seqLenQ_ ? seqLenQ_ : qSeqLen_;
                kvSeqLen_ = kvSeqLen_ > seqLenKV_ ? seqLenKV_ : kvSeqLen_;
                curQSTileNum = (qSeqLen_ + qTileSize_ - 1U) / qTileSize_;
                curTotalTaskNum = static_cast<uint64_t>(curQSTileNum) * headNumQ_;
            }

            const uint64_t curTaskIdx = taskIdx - preTotalTaskNum;
            qHeadIdx_ = static_cast<uint32_t>(curTaskIdx / curQSTileNum);
            qSTileIdx_ = static_cast<uint32_t>(curTaskIdx % curQSTileNum);
            kvHeadIdx_ = qHeadIdx_ / (headNumQ_ / headNumKV_);
            curKVBlockNum_ = (kvSeqLen_ + sparseSize_ - 1U) / sparseSize_;
            curScoredKVBlockNum_ = curKVBlockNum_ > 0 ? curKVBlockNum_ - 1U : 0U;
            const uint32_t qTokenStart = qSTileIdx_ * qTileSize_;
            curQTokenNum_ = qSeqLen_ - qTokenStart > qTileSize_ ? qTileSize_ : qSeqLen_ - qTokenStart;
            curQBlockWriteNum_ = (curQTokenNum_ + sparseSize_ - 1U) / sparseSize_;
            curQSTileSizeAlign_ = qSTileSizeAlign_;

            ProcessQTileFlip();
            ProcessQKMatmul();
            ProcessShortRelu2AndScore();
        }
    }

    __aicore__ inline void ProcessQTileFlip() {
#ifdef __DAV_VEC__
        // No Q transform is needed when this task has no valid Q rows or no KV block requiring a score.
        if (curScoredKVBlockNum_ == 0 || curQSTileSizeAlign_ == 0) {
            return;
        }
        const uint64_t qHeadBase = (static_cast<uint64_t>(curBatch_) * headNumQ_ + qHeadIdx_) * seqLenQ_ * headDim_;
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t qVecTokenNum = qTileSize_ / 2U;
        const uint32_t qVecTokenStart = subBlockIdx * qVecTokenNum;
        const uint32_t qTileStart = qSTileIdx_ * qTileSize_ + qVecTokenStart;
        const uint32_t qChunkLoopNum = qVecTokenNum / qLoadChunk_;
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

        for (uint32_t qChunkIdx = 0; qChunkIdx < qChunkLoopNum; ++qChunkIdx) {
            const uint32_t qRawBufId = qChunkIdx & 1U;
            const uint32_t qChunkTokenStart = qVecTokenStart + qChunkIdx * qLoadChunk_;
            uint32_t curLoadTokenNum = 0;
            // Each AIV owns one half of the Q tile; tokens beyond the actual tail are zero padded.
            if (qChunkTokenStart < curQTokenNum_) {
                curLoadTokenNum = curQTokenNum_ - qChunkTokenStart;
                if (curLoadTokenNum > qLoadChunk_) {
                    curLoadTokenNum = qLoadChunk_;
                }
            }
            const uint32_t qChunkMAlign = qLoadChunk_ / stride_;
            const uint32_t qChunkTokenNumAlign = qLoadChunk_;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(qRawBufId);
            // Avoid issuing a zero-length MTE2 request for the inactive AIV on a short Q tail.
            if (curLoadTokenNum != 0) {
                AscendC::DataCopyParams qCopyInParams;
                qCopyInParams.blockCount = 1;
                qCopyInParams.blockLen = curLoadTokenNum * headDim_ * sizeof(ElementQ) / 32U;
                qCopyInParams.srcStride = 0;
                qCopyInParams.dstStride = 0;
                const uint64_t qCopyInOffset = static_cast<uint64_t>(qTileStart + qChunkIdx * qLoadChunk_) * headDim_;
                AscendC::DataCopy(qRawTensor_[qRawBufId], queryGm_[qHeadBase + qCopyInOffset], qCopyInParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(qRawBufId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(qRawBufId);

            // Zero padding makes the final partial eight-token group and M-aligned rows score as zero.
            if (curLoadTokenNum < qChunkTokenNumAlign) {
                AscendC::Duplicate(qRawTensor_[qRawBufId][curLoadTokenNum * headDim_], static_cast<ElementQ>(0),
                    (qChunkTokenNumAlign - curLoadTokenNum) * headDim_);
                AscendC::PipeBarrier<PIPE_V>();
            }

            __ubuf__ ElementQ *qRawUb = reinterpret_cast<__ubuf__ ElementQ *>(qRawTensor_[qRawBufId].GetPhyAddr());
            __ubuf__ ElementQ *qTmpUb = reinterpret_cast<__ubuf__ ElementQ *>(qTmpTensor_.GetPhyAddr());
            const uint16_t qChunkMReg = static_cast<uint16_t>(qChunkMAlign);
            const uint16_t flipPairNumReg = static_cast<uint16_t>(stride_ / 2U);
            const uint32_t qChunkMStrideReg = qChunkMAlign;
            const uint32_t headDimReg = headDim_;
            const uint32_t strideReg = stride_;
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<ElementQ> qReg0;
                AscendC::Reg::RegTensor<ElementQ> qReg1;
                AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<ElementQ, AscendC::Reg::MaskPattern::ALL>();
                for (uint16_t m = 0; m < qChunkMReg; ++m) {
                    for (uint16_t pair = 0; pair < flipPairNumReg; ++pair) {
                        const uint32_t highTokenIdx = strideReg - 1U - static_cast<uint32_t>(pair) * 2U;
                        const uint32_t lowTokenIdx = highTokenIdx - 1U;
                        const uint32_t rawGroupOffset = static_cast<uint32_t>(m) * strideReg * headDimReg;
                        AscendC::Reg::LoadAlign<ElementQ, AscendC::Reg::LoadDist::DIST_NORM>(
                            qReg0, qRawUb + rawGroupOffset + highTokenIdx * headDimReg);
                        AscendC::Reg::LoadAlign<ElementQ, AscendC::Reg::LoadDist::DIST_NORM>(
                            qReg1, qRawUb + rawGroupOffset + lowTokenIdx * headDimReg);
                        const uint32_t highNzOffset = static_cast<uint32_t>(pair) * 2U * headDimReg * qChunkMStrideReg;
                        const uint32_t lowNzOffset = highNzOffset + headDimReg * qChunkMStrideReg;
                        AscendC::Reg::StoreAlign<ElementQ, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                            qTmpUb + highNzOffset + static_cast<uint32_t>(m) * CUBE_BLOCK_SIZE, qReg0, qChunkMStrideReg,
                            fullMask);
                        AscendC::Reg::StoreAlign<ElementQ, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                            qTmpUb + lowNzOffset + static_cast<uint32_t>(m) * CUBE_BLOCK_SIZE, qReg1, qChunkMStrideReg,
                            fullMask);
                    }
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(qRawBufId);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopyParams qUbToL1Params;
            qUbToL1Params.blockCount = kL1TileSize_ / CUBE_BLOCK_SIZE;
            qUbToL1Params.blockLen = qChunkMAlign;
            qUbToL1Params.srcStride = 0;
            qUbToL1Params.dstStride = curQSTileSizeAlign_ - qChunkMAlign;
            const uint32_t qChunkMStart = subBlockIdx * (qSTileSizeAlign_ / 2U) + qChunkIdx * (qLoadChunk_ / stride_);
            const uint32_t qL1Offset = qChunkMStart * CUBE_BLOCK_SIZE;
            AscendC::DataCopy(l1QTensor_[qL1Offset], qTmpTensor_, qUbToL1Params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(qFlip2mmFlag);
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

    __aicore__ inline void ProcessQKMatmul() {
#ifdef __DAV_CUBE__
        // Skip Cube work when there is no KV block requiring a score.
        if (curScoredKVBlockNum_ == 0) {
            return;
        }
        // Mode 4 releases Cube only after both AIVs have produced their half of Q.
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(qFlip2mmFlag);
        AscendC::CrossCoreWaitFlag<4, PIPE_MTE1>(qFlip2mmFlag + AIV1_FLAG_OFFSET);

        const uint32_t mL1LoopNum = (curScoredKVBlockNum_ + kvBlockNumPerTile_ - 1U) / kvBlockNumPerTile_;

        const uint64_t keyHeadBase =
            (static_cast<uint64_t>(curBatch_) * headNumKV_ + kvHeadIdx_) * seqLenKV_ * headDim_;

        uint32_t l0ABufId = 0;
        uint32_t l0BBufId = 0;
        uint32_t l1KBufId = 0;
        uint32_t prefetchML1Idx = 0;
        uint32_t prefetchKL0Idx = 0;
        uint32_t prefetchML0Idx = 0;

        // Fill the eight-slot K L1 ring before Cube starts consuming it. Logical K tiles are ordered
        // exactly like the consumer: kL0Idx first, then mL0Idx, so Q remains reusable across M tiles.
        for (uint32_t primeL1KBufId = 0; primeL1KBufId < K_L1_BUF_NUM && prefetchML1Idx < mL1LoopNum; ++primeL1KBufId) {
            const uint32_t prefetchML1BlockStart = prefetchML1Idx * kvBlockNumPerTile_;
            const uint32_t prefetchML1BlockNumAct = curScoredKVBlockNum_ - prefetchML1BlockStart > kvBlockNumPerTile_
                ? kvBlockNumPerTile_
                : curScoredKVBlockNum_ - prefetchML1BlockStart;
            const uint32_t prefetchMSize = prefetchML1BlockNumAct * kvBlockRowNum_;
            const uint32_t prefetchML0LoopNum = (prefetchMSize + mL0TileSize_ - 1U) / mL0TileSize_;
            const uint32_t prefetchActMSize = prefetchMSize - prefetchML0Idx * mL0TileSize_ > mL0TileSize_
                ? mL0TileSize_
                : prefetchMSize - prefetchML0Idx * mL0TileSize_;

            AscendC::Nd2NzParams kNd2NzParams;
            kNd2NzParams.ndNum = 1;
            kNd2NzParams.nValue = prefetchActMSize;
            kNd2NzParams.dValue = kL0TileSize_;
            kNd2NzParams.srcNdMatrixStride = 0;
            kNd2NzParams.srcDValue = kL1TileSize_;
            kNd2NzParams.dstNzC0Stride = (prefetchActMSize + CUBE_BLOCK_SIZE - 1U) / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
            kNd2NzParams.dstNzNStride = 1;
            kNd2NzParams.dstNzMatrixStride = 0;

            const uint64_t kGmOffset = static_cast<uint64_t>(prefetchML1BlockStart) * sparseSize_ * headDim_ +
                static_cast<uint64_t>(prefetchML0Idx) * mL0TileSize_ * kL1TileSize_ + prefetchKL0Idx * kL0TileSize_;
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(primeL1KBufId);
            AscendC::DataCopy(l1KTensor_[primeL1KBufId], keyGm_[keyHeadBase + kGmOffset], kNd2NzParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(primeL1KBufId);

            ++prefetchML0Idx;
            if (prefetchML0Idx == prefetchML0LoopNum) {
                prefetchML0Idx = 0;
                ++prefetchKL0Idx;
                if (prefetchKL0Idx == kL0LoopNum_) {
                    prefetchKL0Idx = 0;
                    ++prefetchML1Idx;
                }
            }
        }

        for (uint32_t mL1Idx = 0; mL1Idx < mL1LoopNum; ++mL1Idx) {
            const uint32_t mL1BlockStart = mL1Idx * kvBlockNumPerTile_;
            const uint32_t mL1BlockNumAct = curScoredKVBlockNum_ - mL1BlockStart > kvBlockNumPerTile_
                ? kvBlockNumPerTile_
                : curScoredKVBlockNum_ - mL1BlockStart;
            const uint32_t curMSize = mL1BlockNumAct * kvBlockRowNum_;
            const uint32_t mL0LoopNum = (curMSize + mL0TileSize_ - 1U) / mL0TileSize_;

            for (uint32_t kL0Idx = 0; kL0Idx < kL0LoopNum_; ++kL0Idx) {
                AscendC::LoadData2DParamsV2 qData;
                qData.mStartPosition = 0;
                qData.kStartPosition = kL0Idx * (kL0TileSize_ / CUBE_BLOCK_SIZE);
                qData.mStep = nL0TileSize_ / CUBE_BLOCK_SIZE;
                qData.kStep = kL0TileSize_ / CUBE_BLOCK_SIZE;
                qData.srcStride = curQSTileSizeAlign_ / CUBE_BLOCK_SIZE;
                qData.dstStride = qData.mStep;
                qData.ifTranspose = false;
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2U + l0BBufId);
                AscendC::LoadData(l0QTensor_[l0BBufId], l1QTensor_, qData);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(2U + l0BBufId);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(2U + l0BBufId);

                for (uint32_t mL0Idx = 0; mL0Idx < mL0LoopNum; ++mL0Idx) {
                    const uint32_t actML0TileSize = curMSize - mL0Idx * mL0TileSize_ > mL0TileSize_
                        ? mL0TileSize_
                        : curMSize - mL0Idx * mL0TileSize_;
                    const uint32_t ML0TileSizeAligned =
                        (actML0TileSize + CUBE_BLOCK_SIZE - 1U) / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
                    const uint32_t curL1KBufId = l1KBufId;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(curL1KBufId);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABufId);
                    AscendC::LoadData2DParamsV2 kData;
                    kData.mStartPosition = 0;
                    kData.kStartPosition = 0;
                    kData.mStep = ML0TileSizeAligned / CUBE_BLOCK_SIZE;
                    kData.kStep = kL0TileSize_ / CUBE_BLOCK_SIZE;
                    kData.srcStride = ML0TileSizeAligned / CUBE_BLOCK_SIZE;
                    kData.dstStride = kData.mStep;
                    kData.ifTranspose = false;
                    AscendC::LoadData(l0KTensor_[l0ABufId], l1KTensor_[curL1KBufId], kData);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABufId);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(curL1KBufId);

                    // Refill the just-consumed ring slot eight logical K tiles ahead. This also
                    // crosses an mL1 boundary, so MTE2 overlaps the final MMADs of the current group.
                    if (prefetchML1Idx < mL1LoopNum) {
                        const uint32_t prefetchML1BlockStart = prefetchML1Idx * kvBlockNumPerTile_;
                        const uint32_t prefetchML1BlockNumAct =
                            curScoredKVBlockNum_ - prefetchML1BlockStart > kvBlockNumPerTile_
                            ? kvBlockNumPerTile_
                            : curScoredKVBlockNum_ - prefetchML1BlockStart;
                        const uint32_t prefetchMSize = prefetchML1BlockNumAct * kvBlockRowNum_;
                        const uint32_t prefetchML0LoopNum = (prefetchMSize + mL0TileSize_ - 1U) / mL0TileSize_;
                        const uint32_t prefetchActMSize = prefetchMSize - prefetchML0Idx * mL0TileSize_ > mL0TileSize_
                            ? mL0TileSize_
                            : prefetchMSize - prefetchML0Idx * mL0TileSize_;

                        AscendC::Nd2NzParams nextKNd2NzParams;
                        nextKNd2NzParams.ndNum = 1;
                        nextKNd2NzParams.nValue = prefetchActMSize;
                        nextKNd2NzParams.dValue = kL0TileSize_;
                        nextKNd2NzParams.srcNdMatrixStride = 0;
                        nextKNd2NzParams.srcDValue = kL1TileSize_;
                        nextKNd2NzParams.dstNzC0Stride =
                            (prefetchActMSize + CUBE_BLOCK_SIZE - 1U) / CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
                        nextKNd2NzParams.dstNzNStride = 1;
                        nextKNd2NzParams.dstNzMatrixStride = 0;

                        const uint64_t nextKGmOffset =
                            static_cast<uint64_t>(prefetchML1BlockStart) * sparseSize_ * headDim_ +
                            static_cast<uint64_t>(prefetchML0Idx) * mL0TileSize_ * kL1TileSize_ +
                            prefetchKL0Idx * kL0TileSize_;
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(curL1KBufId);
                        AscendC::DataCopy(
                            l1KTensor_[curL1KBufId], keyGm_[keyHeadBase + nextKGmOffset], nextKNd2NzParams);
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(curL1KBufId);

                        ++prefetchML0Idx;
                        if (prefetchML0Idx == prefetchML0LoopNum) {
                            prefetchML0Idx = 0;
                            ++prefetchKL0Idx;
                            if (prefetchKL0Idx == kL0LoopNum_) {
                                prefetchKL0Idx = 0;
                                ++prefetchML1Idx;
                            }
                        }
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABufId);
                    const uint32_t sTileIdx = mL1Idx * M_L0_SLICE_NUM + mL0Idx;
                    const uint32_t l0CBufId = sTileIdx & (L0_S_BUF_NUM - 1U);
                    // The first K slice reuses this L0C slot only after its previous Fixpipe completes.
                    if (kL0Idx == 0) {
                        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CBufId);
                    }
                    AscendC::MmadParams params;
                    params.m = actML0TileSize;
                    params.n = nL0TileSize_;
                    params.k = kL0TileSize_;
                    params.cmatrixInitVal = kL0Idx == 0;
                    params.unitFlag = 0;
                    AscendC::Mmad(l0STensor_[l0CBufId], l0KTensor_[l0ABufId], l0QTensor_[l0BBufId], params);
                    // The last K slice completes this L0S tile, so Fixpipe can consume it immediately.
                    if (kL0Idx + 1U == kL0LoopNum_) {
                        // The fifth S tile reuses this S-UB slot only after both AIV consumers release it.
                        if (sTileIdx >= S_UB_BUF_NUM) {
                            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag);
                            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag + AIV1_FLAG_OFFSET);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CBufId);
                        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CBufId);
                        AscendC::FixpipeParamsC310<AscendC::CO2Layout::NZ> fix;
                        fix.mSize = actML0TileSize;
                        fix.nSize = nL0TileSize_;
                        fix.srcStride = ML0TileSizeAligned;
                        fix.dstStride = actML0TileSize * CUBE_BLOCK_SIZE;
                        fix.unitFlag = 0;
                        fix.reluEn = false;
                        fix.quantPre = QuantMode_t::NoQuant;
                        fix.dualDstCtl = 0b10;
                        AscendC::Fixpipe<ElementS, ElementS, FIX_TO_NZ_UB>(
                            sTensorSave_[l0CBufId], l0STensor_[l0CBufId], fix);
                        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CBufId);
                        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(mm2ReluFlag);
                        AscendC::CrossCoreSetFlag<4, PIPE_FIX>(mm2ReluFlag + AIV1_FLAG_OFFSET);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABufId);
                    l0ABufId ^= 1U;
                    l1KBufId = (l1KBufId + 1U) & (K_L1_BUF_NUM - 1U);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2U + l0BBufId);
                l0BBufId ^= 1U;
            }
        }
#endif
    }

    __aicore__ inline void ProcessShortRelu2AndScore() {
        if (stride_ == 8U) {
            ProcessShortRelu2AndScoreStride8();
        } else if (stride_ == 16U) {
            ProcessShortRelu2AndScoreStride16();
        }
    }

    __aicore__ inline void ProcessShortRelu2AndScoreStride8() {
#ifdef __DAV_VEC__
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t localQBlockStart = subBlockIdx * Q_BLOCK_NUM_PER_VEC;
        uint32_t curLocalQBlockNum = 0;
        // Each AIV writes only its own four-QBlock range; the second AIV can be padding-only.
        if (localQBlockStart < curQBlockWriteNum_) {
            curLocalQBlockNum = curQBlockWriteNum_ - localQBlockStart;
            if (curLocalQBlockNum > Q_BLOCK_NUM_PER_VEC) {
                curLocalQBlockNum = Q_BLOCK_NUM_PER_VEC;
            }
        }
        const uint32_t compressedQRowNum = (curQTokenNum_ + stride_ - 1U) / stride_;
        const uint32_t localQRowStart = localQBlockStart * kvBlockRowNum_;
        const uint32_t localValidQRowNum = compressedQRowNum > localQRowStart ? compressedQRowNum - localQRowStart : 0U;
        const uint32_t kvBlockNumPerSUbTile = mL0TileSize_ / kvBlockRowNum_;
        const uint32_t sTileLoopNum = (curScoredKVBlockNum_ + kvBlockNumPerSUbTile - 1U) / kvBlockNumPerSUbTile;
        for (uint32_t sTileIdx = 0; sTileIdx < sTileLoopNum; ++sTileIdx) {
            const uint32_t sUbBufId = sTileIdx & (S_UB_BUF_NUM - 1U);
            const uint32_t scoreBufId = sTileIdx & (SCORE_BUF_NUM - 1U);
            const uint32_t sTileBlockStart = sTileIdx * kvBlockNumPerSUbTile;
            const uint32_t sTileBlockNumAct = curScoredKVBlockNum_ - sTileBlockStart > kvBlockNumPerSUbTile
                ? kvBlockNumPerSUbTile
                : curScoredKVBlockNum_ - sTileBlockStart;
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(mm2ReluFlag);
            // A score slot can be overwritten only after its copy from two iterations ago finishes.
            if (sTileIdx >= SCORE_BUF_NUM && curLocalQBlockNum != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }
            __ubuf__ ElementS *sTensorSaveUb =
                reinterpret_cast<__ubuf__ ElementS *>(sTensorSave_[sUbBufId].GetPhyAddr());
            __ubuf__ ElementS *sScoreUb = reinterpret_cast<__ubuf__ ElementS *>(sScoreTensor_[scoreBufId].GetPhyAddr());
            __ubuf__ ElementS *sSumUb = reinterpret_cast<__ubuf__ ElementS *>(sSumTensor_.GetPhyAddr());
            const uint16_t mL0LoopNumReg = static_cast<uint16_t>(curLocalQBlockNum);
            const uint16_t nL0LoopNumReg = static_cast<uint16_t>(sTileBlockNumAct);
            const ElementS scaleValueReg = static_cast<ElementS>(scaleValue_);
            const bool isFirstSTile = sTileIdx == 0;
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<ElementS> sReg0;
                AscendC::Reg::RegTensor<ElementS> sReg1;
                AscendC::Reg::RegTensor<ElementS> sReg2;
                AscendC::Reg::RegTensor<ElementS> sReg3;
                AscendC::Reg::RegTensor<ElementS> sTmpReg0;
                AscendC::Reg::RegTensor<ElementS> sTmpReg1;
                AscendC::Reg::RegTensor<ElementS> sReduceReg;
                AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<ElementS, AscendC::Reg::MaskPattern::ALL>();
                uint32_t validQRowNum = CUBE_BLOCK_SIZE;
                AscendC::Reg::MaskReg qRowMask = AscendC::Reg::UpdateMask<ElementS>(validQRowNum);
                for (uint16_t mL0Idx = 0; mL0Idx < mL0LoopNumReg; ++mL0Idx) {
                    uint32_t validScoreRowNum = localValidQRowNum - static_cast<uint32_t>(mL0Idx) * CUBE_BLOCK_SIZE;
                    AscendC::Reg::MaskReg validScoreMask = AscendC::Reg::UpdateMask<ElementS>(validScoreRowNum);
                    for (uint16_t nL0Idx = 0; nL0Idx < nL0LoopNumReg; ++nL0Idx) {
                        const uint32_t matrixOffset = (static_cast<uint32_t>(mL0Idx) * sTileBlockNumAct + nL0Idx) *
                            CUBE_BLOCK_SIZE * CUBE_BLOCK_SIZE;
                        // Four independent 64-element chains expose enough instructions for multi-issue.
                        AscendC::Reg::LoadAlign(sReg0, sTensorSaveUb + matrixOffset + 0U * 64U);
                        AscendC::Reg::LoadAlign(sReg1, sTensorSaveUb + matrixOffset + 1U * 64U);
                        AscendC::Reg::LoadAlign(sReg2, sTensorSaveUb + matrixOffset + 2U * 64U);
                        AscendC::Reg::LoadAlign(sReg3, sTensorSaveUb + matrixOffset + 3U * 64U);

                        AscendC::Reg::Muls(sReg0, sReg0, scaleValueReg, fullMask);
                        AscendC::Reg::Muls(sReg1, sReg1, scaleValueReg, fullMask);
                        AscendC::Reg::Muls(sReg2, sReg2, scaleValueReg, fullMask);
                        AscendC::Reg::Muls(sReg3, sReg3, scaleValueReg, fullMask);

                        AscendC::Reg::Adds(sReg0, sReg0, static_cast<ElementS>(1), fullMask);
                        AscendC::Reg::Adds(sReg1, sReg1, static_cast<ElementS>(1), fullMask);
                        AscendC::Reg::Adds(sReg2, sReg2, static_cast<ElementS>(1), fullMask);
                        AscendC::Reg::Adds(sReg3, sReg3, static_cast<ElementS>(1), fullMask);

                        AscendC::Reg::Maxs(sReg0, sReg0, static_cast<ElementS>(0), fullMask);
                        AscendC::Reg::Maxs(sReg1, sReg1, static_cast<ElementS>(0), fullMask);
                        AscendC::Reg::Maxs(sReg2, sReg2, static_cast<ElementS>(0), fullMask);
                        AscendC::Reg::Maxs(sReg3, sReg3, static_cast<ElementS>(0), fullMask);

                        AscendC::Reg::Mul(sReg0, sReg0, sReg0, fullMask);
                        AscendC::Reg::Mul(sReg2, sReg2, sReg2, fullMask);
                        AscendC::Reg::MulAddDst(sReg0, sReg1, sReg1, fullMask);
                        AscendC::Reg::MulAddDst(sReg2, sReg3, sReg3, fullMask);
                        AscendC::Reg::Add(sReg0, sReg0, sReg2, fullMask);

                        AscendC::Reg::Interleave(sTmpReg0, sTmpReg1, sReg0, sReg0);
                        AscendC::Reg::Add(sReg1, sTmpReg0, sTmpReg1, fullMask);
                        AscendC::Reg::Interleave(sTmpReg0, sTmpReg1, sReg1, sReg1);
                        AscendC::Reg::Add(sReduceReg, sTmpReg0, sTmpReg1, fullMask);

                        // Undo the four-way replicated layout so one compact register starts with q0..q15.
                        AscendC::Reg::DeInterleave(sTmpReg0, sTmpReg1, sReduceReg, sReduceReg);
                        AscendC::Reg::DeInterleave(sReg0, sTmpReg1, sTmpReg0, sTmpReg0);
                        // ZEROING clears padded Q rows while keeping the complete score slot initialized.
                        AscendC::Reg::Adds(sReg0, sReg0, static_cast<ElementS>(0), validScoreMask);
                        __ubuf__ ElementS *scoreDst = sScoreUb +
                            (static_cast<uint32_t>(mL0Idx) * MAX_KV_BLOCK_NUM_PER_L1_TILE + nL0Idx) * CUBE_BLOCK_SIZE;
                        AscendC::Reg::StoreAlign(scoreDst, sReg0, qRowMask);
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            // Reload the compact score tile in a separate VF scope. This removes the
            // StoreAlign/register-reuse dependency while keeping row_sum entirely in UB.
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<ElementS> scoreReg;
                AscendC::Reg::RegTensor<ElementS> sumReg;
                uint32_t validQRowNum = CUBE_BLOCK_SIZE;
                AscendC::Reg::MaskReg qRowMask = AscendC::Reg::UpdateMask<ElementS>(validQRowNum);
                for (uint16_t mL0Idx = 0; mL0Idx < mL0LoopNumReg; ++mL0Idx) {
                    __ubuf__ ElementS *sumDst = sSumUb + static_cast<uint32_t>(mL0Idx) * CUBE_BLOCK_SIZE;
                    // The first S tile initializes row_sum; later tiles accumulate it.
                    if (isFirstSTile) {
                        AscendC::Reg::Duplicate(sumReg, static_cast<ElementS>(0), qRowMask);
                    } else {
                        AscendC::Reg::LoadAlign(sumReg, sumDst);
                    }
                    for (uint16_t nL0Idx = 0; nL0Idx < nL0LoopNumReg; ++nL0Idx) {
                        __ubuf__ ElementS *scoreSrc = sScoreUb +
                            (static_cast<uint32_t>(mL0Idx) * MAX_KV_BLOCK_NUM_PER_L1_TILE + nL0Idx) * CUBE_BLOCK_SIZE;
                        AscendC::Reg::LoadAlign(scoreReg, scoreSrc);
                        AscendC::Reg::Add(sumReg, sumReg, scoreReg, qRowMask);
                    }
                    AscendC::Reg::StoreAlign(sumDst, sumReg, qRowMask);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CrossCoreSetFlag<4, PIPE_V>(relu2mmFlag);

            // A padding-only AIV participates in CV synchronization but does not issue a zero-sized GM copy.
            if (curLocalQBlockNum != 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(scoreBufId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(scoreBufId);
                const uint64_t scoreOutputOffset =
                    (((static_cast<uint64_t>(curBatch_) * headNumQ_ + qHeadIdx_) * qBlockNum_ +
                         qSTileIdx_ * qBlockNumPerTile_ + localQBlockStart) *
                            scoredKvBlockNum_ +
                        sTileBlockStart) *
                    CUBE_BLOCK_SIZE;
                AscendC::DataCopyExtParams scoreCopyOutParams;
                scoreCopyOutParams.blockCount = curLocalQBlockNum;
                scoreCopyOutParams.blockLen = sTileBlockNumAct * CUBE_BLOCK_SIZE * sizeof(ElementS);
                scoreCopyOutParams.srcStride =
                    (MAX_KV_BLOCK_NUM_PER_L1_TILE - sTileBlockNumAct) * CUBE_BLOCK_SIZE * sizeof(ElementS) / 32U;
                scoreCopyOutParams.dstStride =
                    (scoredKvBlockNum_ - sTileBlockNumAct) * CUBE_BLOCK_SIZE * sizeof(ElementS);
                AscendC::DataCopyPad(scoreGm_[scoreOutputOffset], sScoreTensor_[scoreBufId], scoreCopyOutParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }
        }

        const uint32_t activeScoreBufNum = sTileLoopNum < SCORE_BUF_NUM ? sTileLoopNum : SCORE_BUF_NUM;
        // Only an AIV with valid Q blocks has outstanding score copies and a row_sum range in GM.
        if (curLocalQBlockNum != 0) {
            // Drain the outstanding score copies before row_sum reuses MTE3 event 0 and shared UB.
            for (uint32_t scoreBufId = 0; scoreBufId < activeScoreBufNum; ++scoreBufId) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }

            // Skipping the final KV block can leave no score tiles to initialize row_sum.
            if (sTileLoopNum == 0) {
                AscendC::Duplicate(sSumTensor_, static_cast<ElementS>(0), curLocalQBlockNum * CUBE_BLOCK_SIZE);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t rowSumOutputOffset =
                ((static_cast<uint64_t>(curBatch_) * headNumQ_ + qHeadIdx_) * qBlockNum_ +
                    qSTileIdx_ * qBlockNumPerTile_ + localQBlockStart) *
                CUBE_BLOCK_SIZE;
            AscendC::DataCopyExtParams rowSumCopyOutParams;
            rowSumCopyOutParams.blockCount = 1;
            rowSumCopyOutParams.blockLen = curLocalQBlockNum * CUBE_BLOCK_SIZE * sizeof(ElementS);
            rowSumCopyOutParams.srcStride = 0;
            rowSumCopyOutParams.dstStride = 0;
            AscendC::DataCopyPad(rowSumGm_[rowSumOutputOffset], sSumTensor_, rowSumCopyOutParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
#endif
#ifdef __DAV_CUBE__
        const uint32_t kvBlockNumPerSUbTile = mL0TileSize_ / kvBlockRowNum_;
        const uint32_t sTileLoopNum = (curScoredKVBlockNum_ + kvBlockNumPerSUbTile - 1U) / kvBlockNumPerSUbTile;
        const uint32_t activeSUbBufNum = sTileLoopNum < S_UB_BUF_NUM ? sTileLoopNum : S_UB_BUF_NUM;
        // Drain the remaining per-AIV consumption tokens before the next QTile reuses these flag IDs.
        for (uint32_t sUbBufId = 0; sUbBufId < activeSUbBufNum; ++sUbBufId) {
            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag);
            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag + AIV1_FLAG_OFFSET);
        }
#endif
    }

    __aicore__ inline void ProcessShortRelu2AndScoreStride16() {
#ifdef __DAV_VEC__
        const uint32_t subBlockIdx = AscendC::GetSubBlockIdx();
        const uint32_t localQBlockStart = subBlockIdx * Q_BLOCK_NUM_PER_VEC;
        uint32_t curLocalQBlockNum = 0;
        // Each AIV writes only its own four-QBlock range; the second AIV can be padding-only.
        if (localQBlockStart < curQBlockWriteNum_) {
            curLocalQBlockNum = curQBlockWriteNum_ - localQBlockStart;
            if (curLocalQBlockNum > Q_BLOCK_NUM_PER_VEC) {
                curLocalQBlockNum = Q_BLOCK_NUM_PER_VEC;
            }
        }
        const uint32_t compressedQRowNum = (curQTokenNum_ + stride_ - 1U) / stride_;
        const uint32_t localQRowStart = localQBlockStart * kvBlockRowNum_;
        const uint32_t localValidQRowNum = compressedQRowNum > localQRowStart ? compressedQRowNum - localQRowStart : 0U;
        const uint32_t kvBlockNumPerSUbTile = mL0TileSize_ / kvBlockRowNum_;
        const uint32_t sTileLoopNum = (curScoredKVBlockNum_ + kvBlockNumPerSUbTile - 1U) / kvBlockNumPerSUbTile;
        for (uint32_t sTileIdx = 0; sTileIdx < sTileLoopNum; ++sTileIdx) {
            const uint32_t sUbBufId = sTileIdx & (S_UB_BUF_NUM - 1U);
            const uint32_t scoreBufId = sTileIdx & (SCORE_BUF_NUM - 1U);
            const uint32_t sTileBlockStart = sTileIdx * kvBlockNumPerSUbTile;
            const uint32_t sTileBlockNumAct = curScoredKVBlockNum_ - sTileBlockStart > kvBlockNumPerSUbTile
                ? kvBlockNumPerSUbTile
                : curScoredKVBlockNum_ - sTileBlockStart;
            AscendC::CrossCoreWaitFlag<4, PIPE_V>(mm2ReluFlag);
            // A score slot can be overwritten only after its copy from two iterations ago finishes.
            if (sTileIdx >= SCORE_BUF_NUM && curLocalQBlockNum != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }
            __ubuf__ ElementS *sTensorSaveUb =
                reinterpret_cast<__ubuf__ ElementS *>(sTensorSave_[sUbBufId].GetPhyAddr());
            __ubuf__ ElementS *sScoreUb = reinterpret_cast<__ubuf__ ElementS *>(sScoreTensor_[scoreBufId].GetPhyAddr());
            __ubuf__ ElementS *sSumUb = reinterpret_cast<__ubuf__ ElementS *>(sSumTensor_.GetPhyAddr());
            const uint16_t mL0LoopNumReg = static_cast<uint16_t>(curLocalQBlockNum);
            const uint16_t nL0LoopNumReg = static_cast<uint16_t>(sTileBlockNumAct);
            const ElementS scaleValueReg = static_cast<ElementS>(scaleValue_);
            const bool isFirstSTile = sTileIdx == 0;
            const uint16_t qPairLoopNumReg = static_cast<uint16_t>((curLocalQBlockNum + 1U) / 2U);
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<ElementS> sReg0;
                AscendC::Reg::RegTensor<ElementS> sReg1;
                AscendC::Reg::RegTensor<ElementS> sTmpReg0;
                AscendC::Reg::RegTensor<ElementS> sTmpReg1;
                AscendC::Reg::RegTensor<ElementS> sReduceReg;
                AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<ElementS, AscendC::Reg::MaskPattern::ALL>();
                for (uint16_t mL0Idx = 0; mL0Idx < qPairLoopNumReg; ++mL0Idx) {
                    const uint32_t localQBlockIdx = static_cast<uint32_t>(mL0Idx) * 2U;
                    uint32_t validQRowNum = localQBlockIdx + 1U < curLocalQBlockNum ? 16U : 8U;
                    AscendC::Reg::MaskReg qRowMask = AscendC::Reg::UpdateMask<ElementS>(validQRowNum);
                    uint32_t validScoreRowNum = localValidQRowNum - static_cast<uint32_t>(mL0Idx) * CUBE_BLOCK_SIZE;
                    AscendC::Reg::MaskReg validScoreMask = AscendC::Reg::UpdateMask<ElementS>(validScoreRowNum);
                    for (uint16_t nL0Idx = 0; nL0Idx < nL0LoopNumReg; ++nL0Idx) {
                        // Fixpipe packs valid M rows; one Q pair occupies a 16-column strip.
                        const uint32_t matrixOffset =
                            (static_cast<uint32_t>(mL0Idx) * sTileBlockNumAct + nL0Idx) * 8U * CUBE_BLOCK_SIZE;
                        AscendC::Reg::LoadAlign(sReg0, sTensorSaveUb + matrixOffset);
                        AscendC::Reg::LoadAlign(sReg1, sTensorSaveUb + matrixOffset + 64U);
                        AscendC::Reg::Muls(sReg0, sReg0, scaleValueReg, fullMask);
                        AscendC::Reg::Muls(sReg1, sReg1, scaleValueReg, fullMask);
                        AscendC::Reg::Adds(sReg0, sReg0, static_cast<ElementS>(1), fullMask);
                        AscendC::Reg::Adds(sReg1, sReg1, static_cast<ElementS>(1), fullMask);
                        AscendC::Reg::Maxs(sReg0, sReg0, static_cast<ElementS>(0), fullMask);
                        AscendC::Reg::Maxs(sReg1, sReg1, static_cast<ElementS>(0), fullMask);
                        AscendC::Reg::Mul(sReg0, sReg0, sReg0, fullMask);
                        AscendC::Reg::MulAddDst(sReg0, sReg1, sReg1, fullMask);
                        AscendC::Reg::Interleave(sTmpReg0, sTmpReg1, sReg0, sReg0);
                        AscendC::Reg::Add(sReg1, sTmpReg0, sTmpReg1, fullMask);
                        AscendC::Reg::Interleave(sTmpReg0, sTmpReg1, sReg1, sReg1);
                        AscendC::Reg::Add(sReduceReg, sTmpReg0, sTmpReg1, fullMask);
                        AscendC::Reg::DeInterleave(sTmpReg0, sTmpReg1, sReduceReg, sReduceReg);
                        AscendC::Reg::DeInterleave(sReg0, sTmpReg1, sTmpReg0, sTmpReg0);
                        // ZEROING clears padded Q rows while keeping the complete score slot initialized.
                        AscendC::Reg::Adds(sReg0, sReg0, static_cast<ElementS>(0), validScoreMask);

                        // Split the two 32-byte results directly into [Q block, KV block, 8].
                        __ubuf__ ElementS *scoreDst =
                            sScoreUb + (localQBlockIdx * MAX_KV_BLOCK_NUM_PER_L1_TILE + nL0Idx) * 8U;
                        AscendC::Reg::StoreAlign<ElementS, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                            scoreDst, sReg0, MAX_KV_BLOCK_NUM_PER_L1_TILE, qRowMask);
                    }
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            // Reload the compact score tile in a separate VF scope. This removes the
            // StoreAlign/register-reuse dependency while keeping row_sum entirely in UB.
            __VEC_SCOPE__ {
                AscendC::Reg::RegTensor<ElementS> scoreReg;
                AscendC::Reg::RegTensor<ElementS> sumReg;
                uint32_t validQRowNum = kvBlockRowNum_;
                AscendC::Reg::MaskReg qRowMask = AscendC::Reg::UpdateMask<ElementS>(validQRowNum);
                for (uint16_t mL0Idx = 0; mL0Idx < mL0LoopNumReg; ++mL0Idx) {
                    __ubuf__ ElementS *sumDst = sSumUb + static_cast<uint32_t>(mL0Idx) * kvBlockRowNum_;
                    // The first S tile initializes row_sum; later tiles accumulate it.
                    if (isFirstSTile) {
                        AscendC::Reg::Duplicate(sumReg, static_cast<ElementS>(0), qRowMask);
                    } else {
                        AscendC::Reg::LoadAlign(sumReg, sumDst);
                    }
                    for (uint16_t nL0Idx = 0; nL0Idx < nL0LoopNumReg; ++nL0Idx) {
                        __ubuf__ ElementS *scoreSrc = sScoreUb +
                            (static_cast<uint32_t>(mL0Idx) * MAX_KV_BLOCK_NUM_PER_L1_TILE + nL0Idx) * kvBlockRowNum_;
                        AscendC::Reg::LoadAlign(scoreReg, scoreSrc);
                        AscendC::Reg::Add(sumReg, sumReg, scoreReg, qRowMask);
                    }
                    AscendC::Reg::StoreAlign(sumDst, sumReg, qRowMask);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CrossCoreSetFlag<4, PIPE_V>(relu2mmFlag);

            // A padding-only AIV participates in CV synchronization but does not issue a zero-sized GM copy.
            if (curLocalQBlockNum != 0) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(scoreBufId);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(scoreBufId);
                const uint64_t scoreOutputOffset =
                    (((static_cast<uint64_t>(curBatch_) * headNumQ_ + qHeadIdx_) * qBlockNum_ +
                         qSTileIdx_ * qBlockNumPerTile_ + localQBlockStart) *
                            scoredKvBlockNum_ +
                        sTileBlockStart) *
                    kvBlockRowNum_;
                AscendC::DataCopyExtParams scoreCopyOutParams;
                scoreCopyOutParams.blockCount = curLocalQBlockNum;
                scoreCopyOutParams.blockLen = sTileBlockNumAct * kvBlockRowNum_ * sizeof(ElementS);
                scoreCopyOutParams.srcStride =
                    (MAX_KV_BLOCK_NUM_PER_L1_TILE - sTileBlockNumAct) * kvBlockRowNum_ * sizeof(ElementS) / 32U;
                scoreCopyOutParams.dstStride =
                    (scoredKvBlockNum_ - sTileBlockNumAct) * kvBlockRowNum_ * sizeof(ElementS);
                AscendC::DataCopyParams scoreCopyParams;
                scoreCopyParams.blockCount = scoreCopyOutParams.blockCount;
                scoreCopyParams.blockLen = scoreCopyOutParams.blockLen / 32U;
                scoreCopyParams.srcStride = scoreCopyOutParams.srcStride;
                scoreCopyParams.dstStride = scoreCopyOutParams.dstStride / 32U;
                AscendC::DataCopy(scoreGm_[scoreOutputOffset], sScoreTensor_[scoreBufId], scoreCopyParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }
        }

        const uint32_t activeScoreBufNum = sTileLoopNum < SCORE_BUF_NUM ? sTileLoopNum : SCORE_BUF_NUM;
        // Only an AIV with valid Q blocks has outstanding score copies and a row_sum range in GM.
        if (curLocalQBlockNum != 0) {
            // Drain the outstanding score copies before row_sum reuses MTE3 event 0 and shared UB.
            for (uint32_t scoreBufId = 0; scoreBufId < activeScoreBufNum; ++scoreBufId) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(scoreBufId);
            }

            // Skipping the final KV block can leave no score tiles to initialize row_sum.
            if (sTileLoopNum == 0) {
                AscendC::Duplicate(sSumTensor_, static_cast<ElementS>(0), curLocalQBlockNum * kvBlockRowNum_);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            const uint64_t rowSumOutputOffset =
                ((static_cast<uint64_t>(curBatch_) * headNumQ_ + qHeadIdx_) * qBlockNum_ +
                    qSTileIdx_ * qBlockNumPerTile_ + localQBlockStart) *
                kvBlockRowNum_;
            AscendC::DataCopyExtParams rowSumCopyOutParams;
            rowSumCopyOutParams.blockCount = 1;
            rowSumCopyOutParams.blockLen = curLocalQBlockNum * kvBlockRowNum_ * sizeof(ElementS);
            rowSumCopyOutParams.srcStride = 0;
            rowSumCopyOutParams.dstStride = 0;
            AscendC::DataCopyPad(rowSumGm_[rowSumOutputOffset], sSumTensor_, rowSumCopyOutParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
#endif
#ifdef __DAV_CUBE__
        const uint32_t kvBlockNumPerSUbTile = mL0TileSize_ / kvBlockRowNum_;
        const uint32_t sTileLoopNum = (curScoredKVBlockNum_ + kvBlockNumPerSUbTile - 1U) / kvBlockNumPerSUbTile;
        const uint32_t activeSUbBufNum = sTileLoopNum < S_UB_BUF_NUM ? sTileLoopNum : S_UB_BUF_NUM;
        // Drain the remaining per-AIV consumption tokens before the next QTile reuses these flag IDs.
        for (uint32_t sUbBufId = 0; sUbBufId < activeSUbBufNum; ++sUbBufId) {
            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag);
            AscendC::CrossCoreWaitFlag<4, PIPE_FIX>(relu2mmFlag + AIV1_FLAG_OFFSET);
        }
#endif
    }

  private:
    AscendC::GlobalTensor<ElementQ> queryGm_;
    AscendC::GlobalTensor<ElementK> keyGm_;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsGm_;
    AscendC::GlobalTensor<int64_t> actualSeqLengthsKVGm_;
    AscendC::GlobalTensor<ElementS> scoreGm_;
    AscendC::GlobalTensor<ElementS> rowSumGm_;
    GM_ADDR workspace_;

    AscendC::LocalTensor<ElementQ> qRawTensor_[2];
    AscendC::LocalTensor<ElementQ> qTmpTensor_;
    AscendC::LocalTensor<ElementQ> l1QTensor_;
    AscendC::LocalTensor<ElementK> l1KTensor_[K_L1_BUF_NUM];
    AscendC::LocalTensor<ElementQ> l0QTensor_[2];
    AscendC::LocalTensor<ElementK> l0KTensor_[2];
    AscendC::LocalTensor<ElementS> l0STensor_[L0_S_BUF_NUM];
    AscendC::LocalTensor<ElementS> sTensorSave_[S_UB_BUF_NUM];
    AscendC::LocalTensor<ElementS> sScoreTensor_[SCORE_BUF_NUM];
    AscendC::LocalTensor<ElementS> sSumTensor_;

    uint32_t batchSize_;
    uint32_t seqLenQ_;
    uint32_t seqLenKV_;
    uint32_t headNumQ_;
    uint32_t headNumKV_;
    uint32_t headDim_;
    uint32_t actualSeqLengthsSize_;
    uint32_t actualSeqLengthsKVSize_;
    uint32_t stride_;
    uint32_t sparseSize_;
    float scaleValue_;
    float threshold_;
    uint32_t qTileSize_;
    uint32_t qLoadChunk_;
    uint32_t kvTileSize_;
    uint32_t mL0TileSize_;
    uint32_t nL0TileSize_;
    uint32_t kL0TileSize_;
    uint32_t kL0LoopNum_;
    uint32_t mL1SliceNum_;
    uint32_t qBlockNum_;
    uint32_t scoredKvBlockNum_;
    uint32_t usedCoreNum_;
    uint64_t totalTaskNum_;

    uint32_t kL1TileSize_;
    uint32_t qSTileSizeAlign_;
    uint32_t qBlockNumPerTile_;
    uint32_t kvBlockRowNum_;
    uint32_t kvBlockNumPerTile_;
    uint32_t kvSTileSizeAlign_;
    uint32_t qRawBufSize_;
    uint32_t l1QBufSize_;
    uint32_t l1KBufSize_;
    uint32_t l0QKBufSize_;

    uint32_t curBatch_;
    uint32_t qHeadIdx_;
    uint32_t kvHeadIdx_;
    uint32_t qSTileIdx_;
    uint32_t qSeqLen_;
    uint32_t kvSeqLen_;
    uint32_t curKVBlockNum_;
    uint32_t curScoredKVBlockNum_;
    uint32_t curQTokenNum_;
    uint32_t curQSTileSizeAlign_;
    uint32_t curQBlockWriteNum_;
};

#endif // ADA_RELU2_BLOCK_ESTIMATE_REGULAR_H
