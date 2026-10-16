/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelMulAdd {
public:
    __aicore__ inline KernelMulAdd() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR out,
                                const MulAddTilingData *tiling)
    {
        this->batchSize = static_cast<int32_t>(tiling->batchSize);
        this->seqLen = static_cast<int32_t>(tiling->seqLen);
        this->hiddenSize = static_cast<int32_t>(tiling->hiddenSize);
        this->hiddenSizeAlign = static_cast<int32_t>(tiling->hiddenSizeAlign);
        this->rowsPerTile = static_cast<int32_t>(tiling->rowsPerTile);
        this->dtypeSize = static_cast<int32_t>(sizeof(T));

        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t formerNum = tiling->formerNum;
        int64_t formerLength = tiling->formerLength;
        int64_t tailLength = tiling->tailLength;

        int64_t offset = 0;
        if (formerNum > 0 && blockIdx < formerNum) {
            this->blockLength = static_cast<int32_t>(formerLength);
            offset = formerLength * blockIdx;
        } else {
            this->blockLength = static_cast<int32_t>(tailLength);
            if (formerNum > 0) {
                offset = formerNum * formerLength + (blockIdx - formerNum) * tailLength;
            }
        }

        this->rowsPerBatch = this->blockLength / this->hiddenSize;
        this->coreOffsetInBatch = offset;

        this->batchStride = static_cast<int64_t>(this->seqLen) * this->hiddenSize;

        this->aBase = a;
        this->bBase = b;
        this->cBase = c;
        this->outBase = out;

        int32_t tileBytes = this->rowsPerTile * this->hiddenSizeAlign * this->dtypeSize;

        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(inQueueC, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileBytes);
        pipe.InitBuffer(tmpBuf0, this->hiddenSizeAlign * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->hiddenSizeAlign * sizeof(float));
        pipe.InitBuffer(bBuf, this->hiddenSizeAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (int32_t batch = 0; batch < this->batchSize; batch++) {
            ProcessBatch(batch);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int32_t batchIdx)
    {
        int64_t batchOff = static_cast<int64_t>(batchIdx) * this->batchStride;
        int64_t coreOff = this->coreOffsetInBatch;

        aGm.SetGlobalBuffer((__gm__ T *)(this->aBase) + batchOff + coreOff, this->blockLength);
        cGm.SetGlobalBuffer((__gm__ T *)(this->cBase) + batchOff + coreOff, this->blockLength);
        outGm.SetGlobalBuffer((__gm__ T *)(this->outBase) + batchOff + coreOff, this->blockLength);

        bGm.SetGlobalBuffer((__gm__ T *)(this->bBase) + static_cast<int64_t>(batchIdx) * this->hiddenSize,
                            this->hiddenSize);

        AscendC::LocalTensor<T> bLoadLocal = tmpBuf0.Get<float>().ReinterpretCast<T>();
        AscendC::DataCopyExtParams copyParamsB{1,
            static_cast<uint32_t>(this->hiddenSize * this->dtypeSize), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParamsB{true, 0,
            static_cast<uint8_t>(this->hiddenSizeAlign - this->hiddenSize), 0};
        AscendC::DataCopyPad(bLoadLocal, bGm, copyParamsB, padParamsB);
        pipe_barrier(PIPE_ALL);

        AscendC::LocalTensor<float> bFp32 = bBuf.Get<float>();
        AscendC::Cast(bFp32, bLoadLocal, AscendC::RoundMode::CAST_NONE, this->hiddenSizeAlign);
        pipe_barrier(PIPE_ALL);

        int32_t totalTiles = (this->rowsPerBatch + this->rowsPerTile - 1) / this->rowsPerTile;
        for (int32_t tile = 0; tile < totalTiles; ++tile) {
            int32_t rowsThisTile = (tile < totalTiles - 1) ?
                this->rowsPerTile : (this->rowsPerBatch - tile * this->rowsPerTile);
            CopyIn(tile, rowsThisTile);
            Compute(rowsThisTile);
            CopyOut(tile, rowsThisTile);
        }
    }

    __aicore__ inline void CopyIn(int32_t tileIdx, int32_t rowsThisTile)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        AscendC::LocalTensor<T> cLocal = inQueueC.AllocTensor<T>();

        int64_t baseOffset = static_cast<int64_t>(tileIdx) * this->rowsPerTile * this->hiddenSize;

        if (this->hiddenSize == this->hiddenSizeAlign) {
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowsThisTile),
                static_cast<uint32_t>(this->hiddenSize * this->dtypeSize),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            AscendC::DataCopyPad(aLocal, aGm[baseOffset], copyParams, padParams);
            AscendC::DataCopyPad(cLocal, cGm[baseOffset], copyParams, padParams);
        } else {
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowsThisTile),
                static_cast<uint32_t>(this->hiddenSize * this->dtypeSize),
                0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{true, 0,
                static_cast<uint8_t>(this->hiddenSizeAlign - this->hiddenSize), 0};
            AscendC::DataCopyPad(aLocal, aGm[baseOffset], copyParams, padParams);
            AscendC::DataCopyPad(cLocal, cGm[baseOffset], copyParams, padParams);
        }

        inQueueA.EnQue(aLocal);
        inQueueC.EnQue(cLocal);
    }

    __aicore__ inline void Compute(int32_t rowsThisTile)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.DeQue<T>();
        AscendC::LocalTensor<T> cLocal = inQueueC.DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueue.AllocTensor<T>();

        AscendC::LocalTensor<float> temp0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> temp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> bFp32 = bBuf.Get<float>();

        for (int32_t r = 0; r < rowsThisTile; ++r) {
            int32_t offset = r * this->hiddenSizeAlign;
            AscendC::Cast(temp0, aLocal[offset], AscendC::RoundMode::CAST_NONE, this->hiddenSizeAlign);
            AscendC::Mul(temp0, temp0, bFp32, this->hiddenSizeAlign);
            AscendC::Cast(temp1, cLocal[offset], AscendC::RoundMode::CAST_NONE, this->hiddenSizeAlign);
            AscendC::Add(temp0, temp0, temp1, this->hiddenSizeAlign);
            AscendC::Cast(outLocal[offset], temp0, AscendC::RoundMode::CAST_RINT, this->hiddenSizeAlign);
        }

        outQueue.EnQue<T>(outLocal);
        inQueueA.FreeTensor(aLocal);
        inQueueC.FreeTensor(cLocal);
    }

    __aicore__ inline void CopyOut(int32_t tileIdx, int32_t rowsThisTile)
    {
        AscendC::LocalTensor<T> outLocal = outQueue.DeQue<T>();
        int64_t baseOffset = static_cast<int64_t>(tileIdx) * this->rowsPerTile * this->hiddenSize;

        if (this->hiddenSize == this->hiddenSizeAlign) {
            AscendC::DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowsThisTile),
                static_cast<uint32_t>(this->hiddenSize * this->dtypeSize),
                0, 0, 0};
            AscendC::DataCopyPad(outGm[baseOffset], outLocal, copyParams);
        } else {
            for (int32_t r = 0; r < rowsThisTile; ++r) {
                int32_t srcOffset = r * this->hiddenSizeAlign;
                int64_t dstOffset = baseOffset + static_cast<int64_t>(r) * this->hiddenSize;
                AscendC::DataCopyExtParams copyParams{1,
                    static_cast<uint32_t>(this->hiddenSize * this->dtypeSize), 0, 0, 0};
                AscendC::DataCopyPad(outGm[dstOffset], outLocal[srcOffset], copyParams);
            }
        }

        outQueue.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueC;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bBuf;
    AscendC::GlobalTensor<T> aGm;
    AscendC::GlobalTensor<T> bGm;
    AscendC::GlobalTensor<T> cGm;
    AscendC::GlobalTensor<T> outGm;
    int32_t blockLength;
    int32_t hiddenSize;
    int32_t hiddenSizeAlign;
    int32_t rowsPerTile;
    int32_t rowsPerBatch;
    int32_t batchSize;
    int32_t seqLen;
    int32_t dtypeSize;
    int64_t coreOffsetInBatch;
    int64_t batchStride;
    GM_ADDR aBase;
    GM_ADDR bBase;
    GM_ADDR cBase;
    GM_ADDR outBase;
};

extern "C" __global__ __aicore__ void mul_add(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR out,
                                               GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // Host API: out = a + b * c
    // a: addend [B, S, H], b: multiplier [B, S, H], c: resident vector [B, 1, H]
    // Kernel internal formula: out = kernel_a * kernel_b + kernel_c
    // Map kernel_a <- b, kernel_b <- c, kernel_c <- a to get b * c + a == a + b * c
    if (tilingData.dtypeFlag == 0) {
        KernelMulAdd<bfloat16_t> op;
        op.Init(b, c, a, out, &tilingData);
        op.Process();
    } else {
        KernelMulAdd<half> op;
        op.Init(b, c, a, out, &tilingData);
        op.Process();
    }
}
