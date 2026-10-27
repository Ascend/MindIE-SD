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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_HEAD128_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_HEAD128_H

#include "kernel_operator.h"

namespace eagle_snrc {
using namespace AscendC;

// Batch independent heads of one token. All vector repeat strides are in 32-byte blocks.
template <typename T> class EagleSplitNormRopeConcatHead128 {
  public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qw, GM_ADDR kw, GM_ADDR cos, GM_ADDR sin, GM_ADDR query,
        GM_ADDR key, GM_ADDR value, const EagleSplitNormRopeConcatTilingData &tiling) {
        data_ = tiling;
        tileHeads_ = data_.headTile;
        tileElements_ = tileHeads_ * DIM;
        totalHeads_ = data_.qHeads + data_.kHeads + data_.vHeads;
        packed_.SetGlobalBuffer((__gm__ T *)qkv);
        qw_.SetGlobalBuffer((__gm__ T *)qw);
        kw_.SetGlobalBuffer((__gm__ T *)kw);
        cos_.SetGlobalBuffer((__gm__ T *)cos);
        sin_.SetGlobalBuffer((__gm__ T *)sin);
        query_.SetGlobalBuffer((__gm__ T *)query);
        key_.SetGlobalBuffer((__gm__ T *)key);
        value_.SetGlobalBuffer((__gm__ T *)value);
        pipe_.InitBuffer(input_, 2, (tileElements_ + 2 * DIM) * sizeof(T));
        pipe_.InitBuffer(output_, 2, tileElements_ * sizeof(T));
        pipe_.InitBuffer(valueQueue_, 2, tileElements_ * sizeof(T));
        pipe_.InitBuffer(weightQueue_, 1, 2 * DIM * sizeof(T));
        // Shift tmp, weights and cosine by eight bank groups (256 bytes) relative to x.
        pipe_.InitBuffer(work_, (2 * tileElements_ + BANK_PADDING_FLOATS) * sizeof(float));
        // Keep sine in x's bank phase: HALF RoPE reads it with the opposite half of x.
        pipe_.InitBuffer(params_, (4 * DIM + BANK_PADDING_FLOATS) * sizeof(float));
        // Compact sums, constant ones, and one broadcast block per head.
        pipe_.InitBuffer(reduction_, 10 * tileHeads_ * sizeof(float));

        if (data_.ropeType == 0) {
            pipe_.InitBuffer(indices_, GATHER_HEADS * DIM * sizeof(uint32_t));
            pipe_.InitBuffer(signs_, DIM * sizeof(float));
            auto indices = indices_.Get<uint32_t>();
            auto signs = signs_.Get<float>();
            for (uint32_t i = 0; i < GATHER_HEADS * DIM; ++i) {
                indices.SetValue(i, (i ^ 1U) * sizeof(float));
            }
            for (uint32_t i = 0; i < DIM; ++i) {
                signs.SetValue(i, i % 2 == 0 ? -1.0f : 1.0f);
            }
            const auto event = pipe_.FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(event);
            WaitFlag<HardEvent::S_V>(event);
        }

        if (data_.ropeWidth != data_.rotaryDim) {
            pipe_.InitBuffer(cacheIndices_, DIM * sizeof(uint32_t));
            auto cacheIndices = cacheIndices_.Get<uint32_t>();
            for (uint32_t i = 0; i < data_.rotaryDim; ++i) {
                const uint32_t source = data_.ropeType == 0 ? i / 2 : i % data_.ropeWidth;
                cacheIndices.SetValue(i, source * sizeof(float));
            }
            const auto event = pipe_.FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(event);
            WaitFlag<HardEvent::S_V>(event);
        }

        if (data_.normType == 3) {
            auto weights = weightQueue_.AllocTensor<T>();
            DataCopy(weights, qw_, DIM);
            DataCopy(weights[DIM], kw_, DIM);
            weightQueue_.EnQue(weights);
            weights = weightQueue_.DeQue<T>();
            Cast(params_.Get<float>(), weights, RoundMode::CAST_NONE, 2 * DIM);
            weightQueue_.FreeTensor(weights);
        }
        auto stats = reduction_.Get<float>();
        // Initialize padding read by Brcb when the final tile has fewer than eight heads.
        Duplicate(stats, 0.0f, tileHeads_);
        Duplicate(stats[tileHeads_], 1.0f, tileHeads_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process() {
        ProcessNorm<false>(query_, data_.qHeads, 0, 0);
        ProcessNorm<true>(key_, data_.kHeads, data_.qHeads, DIM);
    }

  private:
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t LANES = 64;
    static constexpr uint32_t GATHER_HEADS = 16;
    static constexpr uint32_t BANK_PADDING_FLOATS = 64; // 256 bytes on 910B.
    static constexpr uint8_t ROW_BLOCKS = DIM / 8;

    __aicore__ inline uint32_t HeadsInTile(int64_t task, int64_t heads, int64_t tilesPerToken) {
        const int64_t remaining = heads - (task % tilesPerToken) * tileHeads_;
        return remaining < tileHeads_ ? static_cast<uint32_t>(remaining) : tileHeads_;
    }

    __aicore__ inline void CopyInNorm(int64_t task, int64_t heads, int64_t firstHead, int64_t tilesPerToken) {
        const int64_t token = task / tilesPerToken;
        const int64_t head = (task % tilesPerToken) * tileHeads_;
        const uint32_t count = HeadsInTile(task, heads, tilesPerToken) * DIM;
        const int64_t ropeToken = data_.ropeBatch == 1 ? token % data_.seq : token;
        auto input = input_.AllocTensor<T>();
        DataCopy(input, packed_[(token * totalHeads_ + firstHead + head) * DIM], count);
        // Consecutive tasks can share a token; the FIFO preserves the parameter producer order.
        if (ropeToken != queuedRopeToken_) {
            DataCopy(input[tileElements_], cos_[ropeToken * data_.ropeWidth], data_.ropeWidth);
            DataCopy(input[tileElements_ + DIM], sin_[ropeToken * data_.ropeWidth], data_.ropeWidth);
            queuedRopeToken_ = ropeToken;
        }
        input_.EnQue(input);
    }

    template <bool copyValue>
    __aicore__ inline void ProcessNorm(
        const GlobalTensor<T> &destination, int64_t heads, int64_t firstHead, uint32_t weightOffset) {
        const int64_t tilesPerToken = (heads + tileHeads_ - 1) / tileHeads_;
        const int64_t tasks = data_.batch * data_.seq * tilesPerToken;
        const int64_t step = GetBlockNum();
        const int64_t begin = GetBlockIdx();
        if (begin >= tasks) {
            return;
        }
        CopyInNorm(begin, heads, firstHead, tilesPerToken);
        // Hk == Hv: K and V share the same task ownership and tail tile.
        if constexpr (copyValue) {
            CopyInValue(begin, tilesPerToken);
        }
        for (int64_t task = begin; task < tasks; task += step) {
            // Keep the next tile in flight while Vector consumes the current tile.
            if (task + step < tasks) {
                CopyInNorm(task + step, heads, firstHead, tilesPerToken);
                if constexpr (copyValue) {
                    CopyInValue(task + step, tilesPerToken);
                }
            }
            const uint32_t rows = HeadsInTile(task, heads, tilesPerToken);
            if constexpr (copyValue) {
                // Submit V DMA before K computation and its Vector-to-MTE3 dependency.
                auto value = valueQueue_.DeQue<T>();
                Store(value_, value, task, data_.vHeads, tilesPerToken, rows);
                valueQueue_.FreeTensor(value);
            }
            const int64_t token = task / tilesPerToken;
            Compute(rows, weightOffset, data_.ropeBatch == 1 ? token % data_.seq : token);
            auto output = output_.DeQue<T>();
            Store(destination, output, task, heads, tilesPerToken, rows);
            output_.FreeTensor(output);
        }
    }

    __aicore__ inline void Compute(uint32_t rows, uint32_t weightOffset, int64_t ropeToken) {
        auto input = input_.DeQue<T>();
        auto x = work_.Get<float>();
        auto tmp = x[tileElements_ + BANK_PADDING_FLOATS];
        auto sum = reduction_.Get<float>();
        auto ones = sum[tileHeads_];
        auto broadcast = sum[2 * tileHeads_];
        auto weight = params_.Get<float>()[weightOffset];
        auto cosine = params_.Get<float>()[2 * DIM];
        auto sine = params_.Get<float>()[3 * DIM + BANK_PADDING_FLOATS];
        const uint32_t count = rows * DIM;
        const uint8_t repeats = static_cast<uint8_t>(rows);

        Cast(x, input, RoundMode::CAST_NONE, count);
        if (ropeToken != computedRopeToken_) {
            if (data_.ropeWidth == data_.rotaryDim) {
                Cast(cosine, input[tileElements_], RoundMode::CAST_NONE, data_.rotaryDim);
                Cast(sine, input[tileElements_ + DIM], RoundMode::CAST_NONE, data_.rotaryDim);
            } else {
                // tmp is unused until the square stage; expansion never aliases its source.
                Cast(tmp, input[tileElements_], RoundMode::CAST_NONE, data_.ropeWidth);
                Cast(tmp[DIM], input[tileElements_ + DIM], RoundMode::CAST_NONE, data_.ropeWidth);
                PipeBarrier<PIPE_V>();
                Gather(cosine, tmp, cacheIndices_.Get<uint32_t>(), 0, data_.rotaryDim);
                Gather(sine, tmp[DIM], cacheIndices_.Get<uint32_t>(), 0, data_.rotaryDim);
            }
            PipeBarrier<PIPE_V>();
            if (data_.ropeType == 0) {
                Mul(sine, sine, signs_.Get<float>(), data_.rotaryDim);
            } else {
                Muls(sine, sine, -1.0f, data_.rotaryDim / 2);
            }
            computedRopeToken_ = ropeToken;
        }
        // The input is no longer read after these casts; allow MTE2 to reuse it now.
        input_.FreeTensor(input);
        PipeBarrier<PIPE_V>();
        Mul(tmp, x, x, count);
        PipeBarrier<PIPE_V>();
        // Fold 128 columns into 64, then reduce all heads in a single repeat instruction.
        Add(tmp, tmp, tmp[LANES], LANES, repeats, {1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, ROW_BLOCKS});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum, tmp, LANES, repeats, 1, 1, ROW_BLOCKS);
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, 1.0f / DIM, rows);
        PipeBarrier<PIPE_V>();
        Adds(sum, sum, data_.eps, rows);
        PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, rows);
        PipeBarrier<PIPE_V>();
        Div(sum, ones, sum, rows);
        PipeBarrier<PIPE_V>();
        Brcb(broadcast, sum, static_cast<uint8_t>((rows + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < DIM; column += LANES) {
            Mul(x[column], x[column], broadcast, LANES, repeats, {1, 1, 0, ROW_BLOCKS, ROW_BLOCKS, 1});
        }
        PipeBarrier<PIPE_V>();
        if (data_.normType == 3) {
            for (uint32_t column = 0; column < DIM; column += LANES) {
                Mul(x[column], x[column], weight[column], LANES, repeats, {1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if (data_.ropeType == 0) {
            Interleave(x, tmp, cosine, sine, rows);
        } else {
            Half(x, tmp, cosine, sine, repeats);
        }
        // Norm and RoPE remain FP32. Cast each complete row only at the final output boundary.
        auto output = output_.AllocTensor<T>();
        Cast(output, x, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        output_.EnQue(output);
    }

    __aicore__ inline void Half(const LocalTensor<float> &x, const LocalTensor<float> &tmp,
        const LocalTensor<float> &cosine, const LocalTensor<float> &sine, uint8_t repeats) {
        const uint32_t half = data_.rotaryDim / 2;
        const BinaryRepeatParams ropeParams{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, 0};
        Mul(tmp, x[half], sine, half, repeats, ropeParams);
        Mul(tmp[half], x, sine[half], half, repeats, ropeParams);
        PipeBarrier<PIPE_V>();
        Mul(x, x, cosine, half, repeats, ropeParams);
        Mul(x[half], x[half], cosine[half], half, repeats, ropeParams);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams addParams{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, ROW_BLOCKS};
        Add(x, x, tmp, half, repeats, addParams);
        Add(x[half], x[half], tmp[half], half, repeats, addParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Interleave(const LocalTensor<float> &x, const LocalTensor<float> &tmp,
        const LocalTensor<float> &cosine, const LocalTensor<float> &sine, uint32_t rows) {
        for (uint32_t head = 0; head < rows; head += GATHER_HEADS) {
            const uint32_t groupRows = rows - head < GATHER_HEADS ? rows - head : GATHER_HEADS;
            const uint32_t base = head * DIM;
            Gather(tmp[base], x[base], indices_.Get<uint32_t>(), 0, groupRows * DIM);
        }
        PipeBarrier<PIPE_V>();
        const uint8_t repeats = static_cast<uint8_t>(rows);
        const BinaryRepeatParams ropeParams{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, 0};
        const BinaryRepeatParams addParams{1, 1, 1, ROW_BLOCKS, ROW_BLOCKS, ROW_BLOCKS};
        // R=96 uses masks 64 and 32; the normalized suffix of x remains untouched.
        for (uint32_t column = 0; column < data_.rotaryDim; column += LANES) {
            const uint32_t mask = data_.rotaryDim - column < LANES ? data_.rotaryDim - column : LANES;
            Mul(tmp[column], tmp[column], sine[column], mask, repeats, ropeParams);
            Mul(x[column], x[column], cosine[column], mask, repeats, ropeParams);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < data_.rotaryDim; column += LANES) {
            const uint32_t mask = data_.rotaryDim - column < LANES ? data_.rotaryDim - column : LANES;
            Add(x[column], x[column], tmp[column], mask, repeats, addParams);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Store(const GlobalTensor<T> &destination, const LocalTensor<T> &output, int64_t task,
        int64_t heads, int64_t tilesPerToken, uint32_t rows) {
        const int64_t token = task / tilesPerToken;
        const int64_t head = (task % tilesPerToken) * tileHeads_;
        if (data_.outputBnsd == 0) {
            DataCopy(destination[(token * heads + head) * DIM], output, rows * DIM);
        } else {
            const int64_t batch = token / data_.seq;
            const int64_t seq = token % data_.seq;
            const int64_t offset = ((batch * heads + head) * data_.seq + seq) * DIM;
            DataCopyExtParams copy{static_cast<uint16_t>(rows), DIM * sizeof(T), 0,
                static_cast<uint32_t>((data_.seq - 1) * DIM * sizeof(T)), 0};
            DataCopyPad(destination[offset], output, copy);
        }
    }

    __aicore__ inline void CopyInValue(int64_t task, int64_t tilesPerToken) {
        const int64_t token = task / tilesPerToken;
        const int64_t head = (task % tilesPerToken) * tileHeads_;
        auto input = valueQueue_.AllocTensor<T>();
        DataCopy(input, packed_[(token * totalHeads_ + data_.qHeads + data_.kHeads + head) * DIM],
            HeadsInTile(task, data_.vHeads, tilesPerToken) * DIM);
        valueQueue_.EnQue(input);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> input_;
    TQue<QuePosition::VECOUT, 2> output_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> valueQueue_;
    TBuf<QuePosition::VECCALC> work_, params_, reduction_;
    TBuf<QuePosition::VECCALC> indices_, signs_, cacheIndices_;
    GlobalTensor<T> packed_, qw_, kw_, cos_, sin_, query_, key_, value_;
    EagleSplitNormRopeConcatTilingData data_;
    int64_t totalHeads_;
    int64_t queuedRopeToken_ = -1;
    int64_t computedRopeToken_ = -1;
    uint32_t tileHeads_, tileElements_;
};
} // namespace eagle_snrc
#endif
