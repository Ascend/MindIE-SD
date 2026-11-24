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

/*
 * Fused Indexed AdaLayerNorm for FLUX.3:
 *   out = LayerNorm(x) * (1 + scale_table[b, mod_index[t], :])
 *                       + shift_table[b, mod_index[t], :]
 *
 * LayerNorm reduction follows the CANN ops-nn ada_layer_norm SingleLayerNorm
 * sequence (CANN Open Software License Agreement V2.0). The K-row tables are
 * loaded once per batch and selected per token by mod_index; the int32
 * low-word mod_index access is specific to this operator.
 */

#include <type_traits>

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t kIdxTile = 256;

template <typename T> class KernelIndexedAdaLayerNorm {
  public:
    __aicore__ inline KernelIndexedAdaLayerNorm() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scaleTable, GM_ADDR shiftTable, GM_ADDR modIndex, GM_ADDR out,
        const IndexedAdaLayerNormTilingData *tiling) {
        seqLen_ = tiling->seqLen;
        hiddenSize_ = tiling->hiddenSize;
        tableK_ = tiling->tableK;
        epsilon_ = tiling->epsilon;
        aveFactor_ = tiling->aveFactor;

        const int64_t blockIdx = static_cast<int64_t>(AscendC::GetBlockIdx());
        const int64_t rowsPerCore = tiling->rowsPerCore;
        const int64_t tailCoreNum = tiling->tailCoreNum;
        rowStart_ = rowsPerCore * blockIdx + (blockIdx < tailCoreNum ? blockIdx : tailCoreNum);
        rowEnd_ = rowStart_ + rowsPerCore + (blockIdx < tailCoreNum ? 1 : 0);

        hiddenBytes_ = hiddenSize_ * static_cast<int64_t>(sizeof(T));
        fp32Bytes_ = hiddenSize_ * static_cast<int64_t>(sizeof(float));
        tableBf16Bytes_ = tableK_ * hiddenBytes_;
        tableFp32Bytes_ = tableK_ * fp32Bytes_;

        xBase_ = reinterpret_cast<__gm__ T *>(x);
        scaleBase_ = reinterpret_cast<__gm__ T *>(scaleTable);
        shiftBase_ = reinterpret_cast<__gm__ T *>(shiftTable);
        indexBase_ = reinterpret_cast<__gm__ int32_t *>(modIndex);
        outBase_ = reinterpret_cast<__gm__ T *>(out);

        pipe_.InitBuffer(xBf16Buf_, static_cast<uint32_t>(hiddenBytes_));
        pipe_.InitBuffer(outBf16Buf_, static_cast<uint32_t>(hiddenBytes_));
        pipe_.InitBuffer(tableBf16Buf_, static_cast<uint32_t>(tableBf16Bytes_));
        pipe_.InitBuffer(xFp32Buf_, static_cast<uint32_t>(fp32Bytes_));
        pipe_.InitBuffer(yFp32Buf_, static_cast<uint32_t>(fp32Bytes_));
        pipe_.InitBuffer(tmpFp32Buf_, static_cast<uint32_t>(fp32Bytes_));
        pipe_.InitBuffer(scaleCacheBuf_, static_cast<uint32_t>(tableFp32Bytes_));
        pipe_.InitBuffer(shiftCacheBuf_, static_cast<uint32_t>(tableFp32Bytes_));
        pipe_.InitBuffer(reduceFp32Buf_, static_cast<uint32_t>(64 * sizeof(float)));
        pipe_.InitBuffer(indexBuf_, static_cast<uint32_t>(2 * kIdxTile * sizeof(int32_t)));

        xBf16_ = xBf16Buf_.Get<T>();
        outBf16_ = outBf16Buf_.Get<T>();
        tableBf16_ = tableBf16Buf_.Get<T>();
        xFp32_ = xFp32Buf_.Get<float>();
        yFp32_ = yFp32Buf_.Get<float>();
        tmpFp32_ = tmpFp32Buf_.Get<float>();
        scaleCache_ = scaleCacheBuf_.Get<float>();
        shiftCache_ = shiftCacheBuf_.Get<float>();
        reduceFp32_ = reduceFp32Buf_.Get<float>();
        index32_ = indexBuf_.Get<int32_t>();

        eventVToS_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_S>());
    }

    __aicore__ inline void Process() {
        if (rowStart_ >= rowEnd_) {
            pipe_.ReleaseEventID<HardEvent::V_S>(eventVToS_);
            return;
        }

        int64_t loadedBatch = -1;
        int64_t loadedTile = -1;
        for (int64_t row = rowStart_; row < rowEnd_; ++row) {
            const int64_t batch = row / seqLen_;
            const int64_t token = row - batch * seqLen_;
            if (batch != loadedBatch) {
                LoadTable(batch);
                loadedBatch = batch;
            }
            const int64_t tileStart = (token / kIdxTile) * kIdxTile;
            if (tileStart != loadedTile) {
                LoadIndexTile(tileStart);
                loadedTile = tileStart;
            }
            int32_t idx = index32_.GetValue(2 * (token - tileStart));
            // Defensive clamp: the Python API validates by default, but a
            // direct op caller must not read outside the K-row UB cache.
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= static_cast<int32_t>(tableK_)) {
                idx = static_cast<int32_t>(tableK_) - 1;
            }
            ProcessRow(row, idx);
        }

        PipeBarrier<PIPE_ALL>();
        pipe_.ReleaseEventID<HardEvent::V_S>(eventVToS_);
    }

  private:
    __aicore__ inline void LoadIndexTile(int64_t tileStart) {
        const int64_t count = (seqLen_ - tileStart) > kIdxTile ? kIdxTile : (seqLen_ - tileStart);
        GlobalTensor<int32_t> idxGm;
        idxGm.SetGlobalBuffer(indexBase_ + 2 * tileStart, 2 * count);
        DataCopyExtParams params{1, static_cast<uint32_t>(2 * count * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(index32_, idxGm, params, pad);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTable(int64_t batch) {
        DataCopyExtParams params{static_cast<uint16_t>(tableK_), static_cast<uint32_t>(hiddenBytes_), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        const int64_t tableOffset = batch * tableK_ * hiddenSize_;

        PipeBarrier<PIPE_ALL>();
        GlobalTensor<T> scaleRows;
        scaleRows.SetGlobalBuffer(scaleBase_ + tableOffset, tableK_ * hiddenSize_);
        DataCopyPad(tableBf16_, scaleRows, params, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(scaleCache_, tableBf16_, 0.0f, tableK_ * hiddenSize_);
        } else {
            Cast(scaleCache_, tableBf16_, RoundMode::CAST_NONE, tableK_ * hiddenSize_);
        }
        Adds(scaleCache_, scaleCache_, 1.0f, tableK_ * hiddenSize_);
        PipeBarrier<PIPE_ALL>();

        GlobalTensor<T> shiftRows;
        shiftRows.SetGlobalBuffer(shiftBase_ + tableOffset, tableK_ * hiddenSize_);
        DataCopyPad(tableBf16_, shiftRows, params, pad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(shiftCache_, tableBf16_, 0.0f, tableK_ * hiddenSize_);
        } else {
            Cast(shiftCache_, tableBf16_, RoundMode::CAST_NONE, tableK_ * hiddenSize_);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRow(int64_t row, int32_t idx) {
        const int64_t rowOffset = row * hiddenSize_;
        DataCopyExtParams xParams{1, static_cast<uint32_t>(hiddenBytes_), 0, 0, 0};
        DataCopyPadExtParams<T> xPad{false, 0, 0, 0};
        GlobalTensor<T> xRow;
        xRow.SetGlobalBuffer(xBase_ + rowOffset, hiddenSize_);
        DataCopyPad(xBf16_, xRow, xParams, xPad);
        PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(xFp32_, xBf16_, 0.0f, hiddenSize_);
        } else {
            Cast(xFp32_, xBf16_, RoundMode::CAST_NONE, hiddenSize_);
        }
        PipeBarrier<PIPE_V>();

        // mean = sum(x) / H (CANN ada_layer_norm SingleLayerNorm sequence).
        Muls(tmpFp32_, xFp32_, aveFactor_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceFp32_, tmpFp32_, tmpFp32_, hiddenSize_);
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        const float mean = reduceFp32_.GetValue(0);

        // var = sum((x - mean)^2) / H.
        Adds(tmpFp32_, xFp32_, -mean, hiddenSize_);
        PipeBarrier<PIPE_V>();
        Mul(yFp32_, tmpFp32_, tmpFp32_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        Muls(xFp32_, yFp32_, aveFactor_, hiddenSize_);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceFp32_, xFp32_, xFp32_, hiddenSize_);
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        const float var = reduceFp32_.GetValue(0);
        const float rstd = 1.0f / sqrt(var + epsilon_);

        LocalTensor<float> scaleRow = scaleCache_[idx * static_cast<int32_t>(hiddenSize_)];
        LocalTensor<float> shiftRow = shiftCache_[idx * static_cast<int32_t>(hiddenSize_)];
        Muls(yFp32_, tmpFp32_, rstd, hiddenSize_);
        PipeBarrier<PIPE_V>();
        Mul(tmpFp32_, yFp32_, scaleRow, hiddenSize_);
        PipeBarrier<PIPE_V>();
        Add(yFp32_, tmpFp32_, shiftRow, hiddenSize_);
        PipeBarrier<PIPE_V>();
        // Wait for the previous MTE3 read of outBf16_ before overwriting it;
        // the next MTE2 work can overlap with this copy.
        PipeBarrier<PIPE_ALL>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(outBf16_, yFp32_, 0.0f, hiddenSize_);
        } else {
            Cast(outBf16_, yFp32_, RoundMode::CAST_RINT, hiddenSize_);
        }
        PipeBarrier<PIPE_ALL>();
        GlobalTensor<T> outRow;
        outRow.SetGlobalBuffer(outBase_ + rowOffset, hiddenSize_);
        DataCopyExtParams outParams{1, static_cast<uint32_t>(hiddenBytes_), 0, 0, 0};
        DataCopyPad(outRow, outBf16_, outParams);
    }

  private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> xBf16Buf_;
    TBuf<TPosition::VECCALC> outBf16Buf_;
    TBuf<TPosition::VECCALC> tableBf16Buf_;
    TBuf<TPosition::VECCALC> xFp32Buf_;
    TBuf<TPosition::VECCALC> yFp32Buf_;
    TBuf<TPosition::VECCALC> tmpFp32Buf_;
    TBuf<TPosition::VECCALC> scaleCacheBuf_;
    TBuf<TPosition::VECCALC> shiftCacheBuf_;
    TBuf<TPosition::VECCALC> reduceFp32Buf_;
    TBuf<TPosition::VECCALC> indexBuf_;

    __gm__ T *xBase_{nullptr};
    __gm__ T *scaleBase_{nullptr};
    __gm__ T *shiftBase_{nullptr};
    __gm__ T *outBase_{nullptr};
    __gm__ int32_t *indexBase_{nullptr};

    LocalTensor<T> xBf16_;
    LocalTensor<T> outBf16_;
    LocalTensor<T> tableBf16_;
    LocalTensor<float> xFp32_;
    LocalTensor<float> yFp32_;
    LocalTensor<float> tmpFp32_;
    LocalTensor<float> scaleCache_;
    LocalTensor<float> shiftCache_;
    LocalTensor<float> reduceFp32_;
    LocalTensor<int32_t> index32_;

    event_t eventVToS_;

    int64_t seqLen_{0};
    int64_t hiddenSize_{0};
    int64_t tableK_{0};
    int64_t rowStart_{0};
    int64_t rowEnd_{0};
    int64_t hiddenBytes_{0};
    int64_t fp32Bytes_{0};
    int64_t tableBf16Bytes_{0};
    int64_t tableFp32Bytes_{0};
    float epsilon_{1e-6f};
    float aveFactor_{0.0f};
};

extern "C" __global__ __aicore__ void indexed_ada_layer_norm(GM_ADDR x, GM_ADDR scale_table, GM_ADDR shift_table,
    GM_ADDR mod_index, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    KernelIndexedAdaLayerNorm<DTYPE_X> op;
    op.Init(x, scale_table, shift_table, mod_index, out, &tilingData);
    op.Process();
}
