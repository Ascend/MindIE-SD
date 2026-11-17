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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_ASCEND950_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_ASCEND950_H
#include "eagle_split_norm_rope_concat_config_950.h"
#include "eagle_split_norm_rope_concat_vector_950.h"
#include "eagle_split_norm_rope_concat_dma_950.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace eagle_snrc {
using namespace AscendC;

// Each core owns a contiguous token interval, split at batch boundaries.
// Q/K advance independently inside each shared coefficient window; V has its
// own tile and may advance beyond that window without consuming coefficients.
template <typename T> class EagleSplitNormRopeConcatAscend950 {
  public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qw, GM_ADDR kw, GM_ADDR cos, GM_ADDR sin, GM_ADDR query,
        GM_ADDR key, GM_ADDR value, const EagleSplitNormRopeConcatTilingData &tiling) {
        data_ = tiling;
        totalHeads_ = data_.qHeads + data_.kHeads + data_.vHeads;
        packed_.SetGlobalBuffer((__gm__ T *)qkv);
        qw_.SetGlobalBuffer((__gm__ T *)qw);
        kw_.SetGlobalBuffer((__gm__ T *)kw);
        cos_.SetGlobalBuffer((__gm__ T *)cos);
        sin_.SetGlobalBuffer((__gm__ T *)sin);
        query_.SetGlobalBuffer((__gm__ T *)query);
        key_.SetGlobalBuffer((__gm__ T *)key);
        value_.SetGlobalBuffer((__gm__ T *)value);
        const uint32_t qRows = data_.tokenTileQ * data_.qHeads;
        const uint32_t kRows = data_.tokenTileK * data_.kHeads;
        const uint32_t normBytes = (qRows > kRows ? qRows : kRows) * ROW_BYTES;
        pipe_.InitBuffer(input_, 2, normBytes);
        for (uint32_t slot = 0; slot < OUTPUT_SLOTS_950; ++slot) {
            pipe_.InitBuffer(output_[slot], 1, normBytes);
        }
        pipe_.InitBuffer(valueQueue_, 1, data_.tokenTileV * data_.vHeads * ROW_BYTES);
        pipe_.InitBuffer(weightQueue_, 1, 2 * ROW_BYTES);
        pipe_.InitBuffer(maps_, 3 * DIM * sizeof(uint32_t));
        // ReadCoefficients masks invalid columns but issues full row loads.
        const uint32_t guardBytes = ((DIM - data_.ropeWidth) * sizeof(T) + 31) / 32 * 32;
        const uint32_t cacheBytes = 2 * data_.ropeCacheTokens * data_.ropeWidth * sizeof(T) + guardBytes;
        for (uint32_t slot = 0; slot < ROPE_CACHE_SLOTS_950; ++slot) {
            pipe_.InitBuffer(ropeQueue_[slot], 1, cacheBytes);
        }
        outputSlot_ = 0;
        PrepareMaps();
        if (data_.normType == 3) {
            weights_ = weightQueue_.AllocTensor<T>();
            DataCopy(weights_, qw_, DIM);
            DataCopy(weights_[DIM], kw_, DIM);
            weightQueue_.EnQue(weights_);
            weights_ = weightQueue_.DeQue<T>();
        }
    }

    __aicore__ inline void Process() {
        const int64_t total = data_.batch * data_.seq;
        int64_t begin = total * GetBlockIdx() / GetBlockNum();
        const int64_t end = total * (GetBlockIdx() + 1) / GetBlockNum();
        while (begin < end) {
            const int64_t batchEnd = (begin / data_.seq + 1) * data_.seq;
            const int64_t stop = batchEnd < end ? batchEnd : end;
            ProcessBatch(begin, stop);
            begin = stop;
        }
        if (data_.normType == 3) {
            weightQueue_.FreeTensor(weights_);
        }
    }

  private:
    static constexpr uint32_t DIM = 128;
    static constexpr uint32_t ROW_BYTES = DIM * sizeof(T);
    struct Task {
        int64_t token;
        uint16_t tokens;
        bool key;
    };

    __aicore__ inline uint16_t Count(int64_t begin, int64_t end, uint32_t tile) {
        const int64_t remaining = end - begin;
        return static_cast<uint16_t>(remaining < tile ? remaining : tile);
    }

    __aicore__ inline Task TakeTask(int64_t &query, int64_t &key, bool &preferKey, int64_t end) {
        const bool useKey = key < end && (preferKey || query == end);
        auto &position = useKey ? key : query;
        Task x{position, Count(position, end, useKey ? data_.tokenTileK : data_.tokenTileQ), useKey};
        position += x.tokens;
        preferKey = !useKey;
        return x;
    }

    __aicore__ inline void CopyInRope(uint32_t slot, int64_t begin, uint16_t count) {
        auto cache = ropeQueue_[slot].AllocTensor<T>(); // Vector -> MTE2 on slot reuse.
        const int64_t first = data_.ropeBatch == 1 ? begin % data_.seq : begin;
        const uint32_t widthBytes = data_.ropeWidth * sizeof(T);
        // Keep the multi-segment DMA API: one submission per coefficient table,
        // packed GM rows and continuous cos/sin UB regions, both gaps zero.
        dma950::Load(cache, cos_[first * data_.ropeWidth], count, widthBytes, 0, 0);
        dma950::Load(
            cache[data_.ropeCacheTokens * data_.ropeWidth], sin_[first * data_.ropeWidth], count, widthBytes, 0, 0);
        ropeQueue_[slot].EnQue(cache);
    }

    __aicore__ inline void ProcessBatch(int64_t begin, int64_t end) {
        uint32_t slot = 0;
        int64_t valuePosition = begin;
        CopyInRope(slot, begin, Count(begin, end, data_.ropeCacheTokens));
        while (begin < end) {
            const int64_t windowEnd = begin + Count(begin, end, data_.ropeCacheTokens);
            const uint32_t nextSlot = (slot + 1) % ROPE_CACHE_SLOTS_950;
            // The next cache slot is distinct from the current Q/K readers.
            if (windowEnd < end) {
                CopyInRope(nextSlot, windowEnd, Count(windowEnd, end, data_.ropeCacheTokens));
            }
            auto cache = ropeQueue_[slot].DeQue<T>(); // MTE2 -> Vector for this window.
            ProcessWindow(begin, windowEnd, end, valuePosition, cache);
            // Both Q and K cursors reached windowEnd; Free/Alloc also protects
            // their outstanding VF reads before a later MTE2 refill.
            ropeQueue_[slot].FreeTensor(cache);
            begin = windowEnd;
            slot = nextSlot;
        }
    }

    __aicore__ inline void ProcessWindow(
        int64_t begin, int64_t end, int64_t batchEnd, int64_t &valuePosition, const LocalTensor<T> &cache) {
        int64_t query = begin, key = begin;
        bool preferKey = false;
        Task current = TakeTask(query, key, preferKey, end);
        CopyInNorm(current);
        while (current.tokens > 0) {
            const Task next = TakeTask(query, key, preferKey, end);
            if (next.tokens > 0) {
                CopyInNorm(next);
            }
            // V retains its independent tile, in-place layout transform and
            // MTE2 -> MTE3 queue. Its first load precedes this K computation.
            Task valueTask{valuePosition, 0, true};
            if (current.key && valuePosition < current.token + current.tokens) {
                valueTask.tokens = Count(valuePosition, batchEnd, data_.tokenTileV);
                CopyInValue(valueTask);
            }
            ComputeAndStore(current, begin, cache);
            if (valueTask.tokens > 0) {
                CopyOutValue(valueTask); // K's Store has already been submitted.
                valuePosition += valueTask.tokens;
            }
            // A smaller V tile may need more than one copy for this K tile.
            while (current.key && valuePosition < current.token + current.tokens) {
                valueTask = {valuePosition, Count(valuePosition, batchEnd, data_.tokenTileV), true};
                CopyInValue(valueTask);
                CopyOutValue(valueTask);
                valuePosition += valueTask.tokens;
            }
            current = next;
        }
    }

    __aicore__ inline void CopyInNorm(const Task &x) {
        auto input = input_.AllocTensor<T>();
        const int64_t heads = x.key ? data_.kHeads : data_.qHeads;
        const int64_t first = x.key ? data_.qHeads : 0;
        dma950::Load(input, packed_[(x.token * totalHeads_ + first) * DIM], x.tokens, heads * ROW_BYTES,
            static_cast<uint64_t>(totalHeads_ - heads) * ROW_BYTES, 0);
        input_.EnQue(input);
    }

    __aicore__ inline void PrepareMaps() {
        auto partner = maps_.Get<uint32_t>();
        auto cache = partner[DIM];
        auto signs = maps_.Get<float>()[2 * DIM];
        const uint32_t half = data_.rotaryDim / 2;
        for (uint32_t d = 0; d < DIM; ++d) {
            uint32_t p = d, c = 0;
            float sign = 1.0f;
            if (d < data_.rotaryDim) {
                if (data_.ropeType == 0) {
                    p = d ^ 1U;
                    sign = (d & 1U) == 0 ? -1.0f : 1.0f;
                    c = data_.ropeWidth == data_.rotaryDim ? d : d / 2;
                } else {
                    p = d < half ? d + half : d - half;
                    sign = d < half ? -1.0f : 1.0f;
                    c = data_.ropeWidth == data_.rotaryDim ? d : d % half;
                }
            }
            partner.SetValue(d, p);
            cache.SetValue(d, c);
            signs.SetValue(d, sign);
        }
        const auto event = pipe_.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(event);
        WaitFlag<HardEvent::S_V>(event);
    }

    template <reg950::RotarySpan span, bool affine>
    __aicore__ inline void ComputeFourRows(__ubuf__ T *out, __ubuf__ T *in, __ubuf__ T *weight, __ubuf__ T *cosine,
        __ubuf__ T *sine, __ubuf__ uint32_t *map, __ubuf__ float *sign, uint16_t rows, uint32_t stride) {
        if (data_.ropeType == 0) {
            asc_vf_call<reg950::ComputeFour<T, span, false, affine>>(out, in, weight, cosine, sine, map, map + DIM,
                sign, rows, data_.rotaryDim, data_.ropeWidth, data_.eps, DIM, stride);
        } else {
            asc_vf_call<reg950::ComputeFour<T, span, true, affine>>(out, in, weight, cosine, sine, map, map + DIM, sign,
                rows, data_.rotaryDim, data_.ropeWidth, data_.eps, DIM, stride);
        }
    }

    template <bool affine>
    __aicore__ inline void ComputeRows(__ubuf__ T *out, __ubuf__ T *in, __ubuf__ T *weight, __ubuf__ T *cosine,
        __ubuf__ T *sine, __ubuf__ uint32_t *map, __ubuf__ float *sign, uint16_t rows, uint32_t stride) {
        using Span = reg950::RotarySpan;
        if (data_.rotaryDim < reg950::LANES) {
            ComputeFourRows<Span::LowPartial, affine>(out, in, weight, cosine, sine, map, sign, rows, stride);
        } else if (data_.rotaryDim == reg950::LANES) {
            ComputeFourRows<Span::LowFull, affine>(out, in, weight, cosine, sine, map, sign, rows, stride);
        } else if (data_.rotaryDim < DIM) {
            ComputeFourRows<Span::BothPartial, affine>(out, in, weight, cosine, sine, map, sign, rows, stride);
        } else {
            ComputeFourRows<Span::BothFull, affine>(out, in, weight, cosine, sine, map, sign, rows, stride);
        }
    }

    __aicore__ inline void ComputeAndStore(const Task &x, int64_t cacheStart, const LocalTensor<T> &cache) {
        auto input = input_.DeQue<T>();
        auto &queue = output_[outputSlot_];
        auto output = queue.AllocTensor<T>(); // Prior MTE3 read completes before reuse.
        auto *in = (__ubuf__ T *)input.GetPhyAddr();
        auto *out = (__ubuf__ T *)output.GetPhyAddr();
        auto *map = (__ubuf__ uint32_t *)maps_.Get<uint32_t>().GetPhyAddr();
        auto *sign = (__ubuf__ float *)(map + 2 * DIM);
        const uint16_t rows = static_cast<uint16_t>(x.key ? data_.kHeads : data_.qHeads);
        const uint32_t stride = data_.outputBnsd ? x.tokens * DIM : DIM;
        auto *coefficients = (__ubuf__ T *)cache.GetPhyAddr() + (x.token - cacheStart) * data_.ropeWidth;
        for (uint16_t t = 0; t < x.tokens; ++t) {
            const uint32_t inputBase = static_cast<uint32_t>(t) * rows * DIM;
            const uint32_t outputBase = data_.outputBnsd ? t * DIM : inputBase;
            auto *cosine = coefficients + static_cast<uint32_t>(t) * data_.ropeWidth;
            auto *sine = cosine + data_.ropeCacheTokens * data_.ropeWidth;
            if (data_.normType == 3) {
                auto *weight = (__ubuf__ T *)weights_.GetPhyAddr() + (x.key ? DIM : 0);
                ComputeRows<true>(out + outputBase, in + inputBase, weight, cosine, sine, map, sign, rows, stride);
            } else {
                ComputeRows<false>(out + outputBase, in + inputBase, nullptr, cosine, sine, map, sign, rows, stride);
            }
        }
        input_.FreeTensor(input);
        queue.EnQue(output);
        output = queue.DeQue<T>(); // Vector -> MTE3.
        Store(x.key ? key_ : query_, output, x, rows);
        queue.FreeTensor(output);
        outputSlot_ = (outputSlot_ + 1) % OUTPUT_SLOTS_950;
    }

    __aicore__ inline void Store(const GlobalTensor<T> &dst, const LocalTensor<T> &src, const Task &x, uint16_t heads) {
        const int64_t batch = x.token / data_.seq, sequence = x.token % data_.seq;
        if (data_.outputBnsd) {
            const int64_t base = (batch * heads * data_.seq + sequence) * DIM;
            dma950::Store(dst[base], src, heads, x.tokens * ROW_BYTES, 0,
                static_cast<uint64_t>(data_.seq - x.tokens) * ROW_BYTES);
        } else {
            dma950::Store(dst[x.token * heads * DIM], src, x.tokens, heads * ROW_BYTES, 0, 0);
        }
    }

    __aicore__ inline void CopyInValue(const Task &x) {
        const uint16_t n = x.tokens, rows = static_cast<uint16_t>(data_.vHeads);
        const int64_t first = data_.qHeads + data_.kHeads;
        auto value = valueQueue_.AllocTensor<T>();
        if (!data_.outputBnsd) {
            dma950::Load(value, packed_[(x.token * totalHeads_ + first) * DIM], n, rows * ROW_BYTES,
                static_cast<uint64_t>(totalHeads_ - rows) * ROW_BYTES, 0);
        } else {
            for (uint16_t t = 0; t < n; ++t) {
                dma950::Load(value[t * DIM], packed_[((x.token + t) * totalHeads_ + first) * DIM], rows, ROW_BYTES, 0,
                    (n - 1) * ROW_BYTES / dma950::UB_BLOCK_BYTES);
            }
        }
        valueQueue_.EnQue(value);
    }

    __aicore__ inline void CopyOutValue(const Task &x) {
        auto value = valueQueue_.DeQue<T>(); // MTE2 -> MTE3, after the corresponding K store.
        Store(value_, value, x, static_cast<uint16_t>(data_.vHeads));
        valueQueue_.FreeTensor(value); // MTE3 -> MTE2 before next allocation.
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> input_;
    TQue<QuePosition::VECOUT, 1> output_[OUTPUT_SLOTS_950];
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> valueQueue_;
    TQue<QuePosition::VECIN, 1> weightQueue_;
    TQue<QuePosition::VECIN, 1> ropeQueue_[ROPE_CACHE_SLOTS_950];
    TBuf<QuePosition::VECCALC> maps_;
    LocalTensor<T> weights_;
    GlobalTensor<T> packed_, qw_, kw_, cos_, sin_, query_, key_, value_;
    EagleSplitNormRopeConcatTilingData data_;
    int64_t totalHeads_;
    uint32_t outputSlot_ = 0;
};
} // namespace eagle_snrc
#endif
#endif
