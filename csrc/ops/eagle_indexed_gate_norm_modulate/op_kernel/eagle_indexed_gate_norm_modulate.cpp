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
#include "kernel_operator.h"

using namespace AscendC;

template <typename Input, typename Output, typename Index, bool Tiled = false, uint32_t TILE_SIZE = 3072>
class KernelIndexedGateNormModulate {
  public:
    __aicore__ inline void Init(GM_ADDR residual, GM_ADDR gate, GM_ADDR branch, GM_ADDR weight, GM_ADDR shift,
        GM_ADDR scale, GM_ADDR indices, GM_ADDR residualOut, GM_ADDR modulatedOut,
        const EagleIndexedGateNormModulateTilingData &data) {
        rows_ = data.rows;
        indexedRows_ = data.indexedRows;
        hidden_ = data.hiddenSize;
        aligned_ = data.alignedSize;
        cores_ = data.usedCores;
        eps_ = data.eps;
        inverseHidden_ = data.inverseHidden;
        if constexpr (Tiled) {
            tileCount_ = (hidden_ + TILE_SIZE - 1) / TILE_SIZE;
            const uint32_t blocks = hidden_ / 128;
            tileBase_ = (blocks / tileCount_) * 128;
            extraTiles_ = blocks % tileCount_;
        }
        inputGm_[RESIDUAL].SetGlobalBuffer((__gm__ Input *)residual);
        inputGm_[GATE].SetGlobalBuffer((__gm__ Input *)gate);
        inputGm_[BRANCH].SetGlobalBuffer((__gm__ Input *)branch);
        inputGm_[WEIGHT].SetGlobalBuffer((__gm__ Input *)weight);
        inputGm_[SHIFT].SetGlobalBuffer((__gm__ Input *)shift);
        inputGm_[SCALE].SetGlobalBuffer((__gm__ Input *)scale);
        indicesGm_.SetGlobalBuffer((__gm__ Index *)indices);
        residualOutGm_.SetGlobalBuffer((__gm__ Output *)residualOut);
        modulatedOutGm_.SetGlobalBuffer((__gm__ Output *)modulatedOut);
        for (uint32_t i = 0; i < INPUT_COUNT; ++i) {
            if constexpr (Tiled) {
                if (i == WEIGHT) {
                    pipe_.InitBuffer(inputFloat_[i], aligned_ * sizeof(float));
                } else {
                    pipe_.InitBuffer(inputQueue_[i], 1, aligned_ * sizeof(Input));
                    if (i != SCALE && i != SHIFT) {
                        pipe_.InitBuffer(inputFloat_[i], TILE_SIZE * sizeof(float));
                    }
                }
            } else {
                pipe_.InitBuffer(inputQueue_[i], 1, aligned_ * sizeof(Input));
                pipe_.InitBuffer(inputFloat_[i], aligned_ * sizeof(float));
            }
        }
        pipe_.InitBuffer(updatedBuffer_, aligned_ * sizeof(float));
        pipe_.InitBuffer(workQueue_, 1, aligned_ * sizeof(float));
        if constexpr (Tiled) {
            pipe_.InitBuffer(weightOutputBuffer_, aligned_ * sizeof(Output));
        } else {
            pipe_.InitBuffer(residualOutQueue_, 1, aligned_ * sizeof(Output));
        }
        pipe_.InitBuffer(scalarBuffer_, 32);
        pipe_.InitBuffer(reduceBuffer_, Tiled ? 1024 : 256);
    }

    __aicore__ inline void Process() {
        const int64_t core = GetBlockIdx();
        const int64_t quotient = rows_ / cores_;
        const int64_t remainder = rows_ % cores_;
        const int64_t start = core * quotient + (core < remainder ? core : remainder);
        const int64_t count = quotient + (core < remainder ? 1 : 0);
        if (count == 0) {
            return;
        }
        // All rows on this core share the same read-only FP32 weight.
        auto weight = inputFloat_[WEIGHT].template Get<float>();
        if constexpr (Tiled) {
            LoadResidentWeight(weight);
            residualReady_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
            residualFree_ = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(residualFree_);
        } else {
            CopyInput(WEIGHT, 0);
            weight = ConvertInput(WEIGHT);
        }
        for (int64_t row = start; row < start + count; ++row) {
            // Scalar reads immutable indices through DataCache; no index DMA or UB slot.
            const int64_t index = static_cast<int64_t>(indicesGm_.GetValue(row));
            if (index < 0 || index >= indexedRows_) {
                // Invalid indices violate the API contract. Never dereference them in GM.
                WriteInvalidRow(row);
                continue;
            }
            if constexpr (Tiled) {
                ProcessWideRow(row, index, weight);
            } else {
                ProcessRow(row, index, weight);
            }
        }
        if constexpr (Tiled) {
            WaitFlag<HardEvent::MTE3_V>(residualFree_);
            pipe_.ReleaseEventID<HardEvent::V_MTE3>(residualReady_);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(residualFree_);
        }
    }

  private:
    enum InputSlot : uint32_t { RESIDUAL, GATE, BRANCH, WEIGHT, SHIFT, SCALE, INPUT_COUNT };

    __aicore__ inline void LoadResidentWeight(const LocalTensor<float> &weight) {
        auto raw = weightOutputBuffer_.template Get<Input>();
        DataCopyExtParams params{1, static_cast<uint32_t>(hidden_ * sizeof(Input)), 0, 0, 0};
        DataCopyPadExtParams<Input> padding{false, 0, 0, static_cast<Input>(0)};
        DataCopyPad(raw, inputGm_[WEIGHT], params, padding);
        const auto ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        Cast(weight, raw, RoundMode::CAST_NONE, hidden_);
        PipeBarrier<PIPE_V>(); // The raw weight region is now available for residual output.
    }

    __aicore__ inline void CopyInput(InputSlot slot, int64_t row, uint32_t offset = 0, uint32_t count = 0) {
        const uint32_t length = Tiled ? count : hidden_;
        auto local = inputQueue_[slot].template AllocTensor<Input>();
        DataCopyExtParams params{1, static_cast<uint32_t>(length * sizeof(Input)), 0, 0, 0};
        DataCopyPadExtParams<Input> padding{false, 0, 0, static_cast<Input>(0)};
        DataCopyPad(local, inputGm_[slot][row * hidden_ + offset], params, padding);
        inputQueue_[slot].EnQue(local);
        // Independent input queues may load concurrently; queue events protect buffer reuse.
    }

    __aicore__ inline LocalTensor<float> ConvertInput(InputSlot slot, uint32_t count = 0) {
        auto local = inputQueue_[slot].template DeQue<Input>(); // MTE2 -> V
        auto result = inputFloat_[slot].template Get<float>();
        Cast(result, local, RoundMode::CAST_NONE, Tiled ? count : hidden_);
        inputQueue_[slot].FreeTensor(local); // V -> MTE2 before reuse
        PipeBarrier<PIPE_V>();
        return result;
    }

    __aicore__ inline void StoreResidual(int64_t row, const LocalTensor<float> &updated) {
        if constexpr (Tiled) {
            WaitFlag<HardEvent::MTE3_V>(residualFree_);
            auto output = weightOutputBuffer_.template Get<Output>();
            Cast(output, updated, RoundMode::CAST_RINT, hidden_);
            SetFlag<HardEvent::V_MTE3>(residualReady_);
            WaitFlag<HardEvent::V_MTE3>(residualReady_);
            DataCopyExtParams params{1, static_cast<uint32_t>(hidden_ * sizeof(Output)), 0, 0, 0};
            DataCopyPad(residualOutGm_[row * hidden_], output, params);
            SetFlag<HardEvent::MTE3_V>(residualFree_);
            return;
        }
        auto output = residualOutQueue_.AllocTensor<Output>();
        Cast(output, updated, RoundMode::CAST_RINT, hidden_);
        residualOutQueue_.EnQue(output);
        output = residualOutQueue_.DeQue<Output>(); // V -> MTE3
        DataCopyExtParams params{1, static_cast<uint32_t>(hidden_ * sizeof(Output)), 0, 0, 0};
        DataCopyPad(residualOutGm_[row * hidden_], output, params);
        residualOutQueue_.FreeTensor(output); // MTE3 -> V before next row
    }

    __aicore__ inline void StoreModulated(int64_t row, const LocalTensor<float> &work) {
        // Same-base narrowing is safe: each repeat writes behind subsequent FP32 reads.
        auto output = work.ReinterpretCast<Output>();
        Cast(output, work, RoundMode::CAST_RINT, hidden_);
        workQueue_.EnQue(output);
        output = workQueue_.DeQue<Output>();
        DataCopyExtParams params{1, static_cast<uint32_t>(hidden_ * sizeof(Output)), 0, 0, 0};
        DataCopyPad(modulatedOutGm_[row * hidden_], output, params);
        workQueue_.FreeTensor(output);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessRow(int64_t row, int64_t index, const LocalTensor<float> &weight) {
        // Issue each following load before consuming the preceding input on V.
        CopyInput(GATE, index);
        CopyInput(BRANCH, row);
        auto gate = ConvertInput(GATE);
        CopyInput(RESIDUAL, row);
        auto branch = ConvertInput(BRANCH);
        auto updated = updatedBuffer_.Get<float>();
        Mul(updated, gate, branch, hidden_);
        PipeBarrier<PIPE_V>();
        auto residual = ConvertInput(RESIDUAL);
        Add(updated, updated, residual, hidden_);
        PipeBarrier<PIPE_V>();
        StoreResidual(row, updated);

        CopyInput(SCALE, index);
        CopyInput(SHIFT, index);
        auto scale = ConvertInput(SCALE);
        auto shift = ConvertInput(SHIFT);

        auto work = Normalize(updated, weight);
        Adds(scale, scale, 1.0f, hidden_);
        PipeBarrier<PIPE_V>();
        Mul(work, work, scale, hidden_);
        PipeBarrier<PIPE_V>();
        Add(work, work, shift, hidden_);
        PipeBarrier<PIPE_V>();
        StoreModulated(row, work);
    }

    __aicore__ inline void ProcessWideRow(int64_t row, int64_t index, const LocalTensor<float> &weight) {
        // Each raw input holds the entire row; FP32 computation uses balanced tiles.
        CopyInput(GATE, index, 0, hidden_);
        CopyInput(BRANCH, row, 0, hidden_);
        auto rawGate = inputQueue_[GATE].template DeQue<Input>();
        auto gate = inputFloat_[GATE].template Get<float>();
        auto branch = inputFloat_[BRANCH].template Get<float>();
        auto residual = inputFloat_[RESIDUAL].template Get<float>();
        auto updated = updatedBuffer_.Get<float>();
        LocalTensor<Input> rawBranch;
        LocalTensor<Input> rawResidual;
        uint32_t offset = 0;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint32_t count = TileLength(tile);
            Cast(gate, rawGate[offset], RoundMode::CAST_NONE, count);
            if (tile == 0) {
                CopyInput(RESIDUAL, row, 0, hidden_);
                rawBranch = inputQueue_[BRANCH].template DeQue<Input>();
            }
            Cast(branch, rawBranch[offset], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul(updated[offset], gate, branch, count);
            if (tile == 0) {
                rawResidual = inputQueue_[RESIDUAL].template DeQue<Input>();
            }
            Cast(residual, rawResidual[offset], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(updated[offset], updated[offset], residual, count);
            PipeBarrier<PIPE_V>();
            offset += count;
        }
        inputQueue_[GATE].FreeTensor(rawGate);
        inputQueue_[BRANCH].FreeTensor(rawBranch);
        inputQueue_[RESIDUAL].FreeTensor(rawResidual);
        StoreResidual(row, updated);
        auto work = Normalize(updated, weight);
        PipeBarrier<PIPE_V>();
        ModulateWideInput(index, work);
        StoreModulated(row, work);
    }

    __aicore__ inline void ModulateWideInput(int64_t index, const LocalTensor<float> &work) {
        // Updated is complete, so branch/residual
        // FP32 tiles can now hold scale/shift; the full-row weight remains resident.
        CopyInput(SCALE, index, 0, hidden_);
        CopyInput(SHIFT, index, 0, hidden_);
        auto rawScale = inputQueue_[SCALE].template DeQue<Input>();
        auto scale = inputFloat_[BRANCH].template Get<float>();
        auto shift = inputFloat_[RESIDUAL].template Get<float>();
        LocalTensor<Input> rawShift;
        uint32_t offset = 0;
        for (uint32_t tile = 0; tile < tileCount_; ++tile) {
            const uint32_t count = TileLength(tile);
            Cast(scale, rawScale[offset], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Adds(scale, scale, 1.0f, count);
            PipeBarrier<PIPE_V>();
            Mul(work[offset], work[offset], scale, count);
            PipeBarrier<PIPE_V>();
            if (tile == 0) {
                rawShift = inputQueue_[SHIFT].template DeQue<Input>();
            }
            Cast(shift, rawShift[offset], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(work[offset], work[offset], shift, count);
            // Finish reading both FP32 tiles before the next pair of Casts overwrites them.
            PipeBarrier<PIPE_V>();
            offset += count;
        }
        // All tiles have been consumed; release each entire low-precision input once.
        inputQueue_[SCALE].FreeTensor(rawScale);
        inputQueue_[SHIFT].FreeTensor(rawShift);
    }

    __aicore__ inline uint32_t TileLength(uint32_t tile) const {
        return tileBase_ + (tile < extraTiles_ ? 128 : 0) + (tile + 1 == tileCount_ ? hidden_ % 128 : 0);
    }

    __aicore__ inline LocalTensor<float> Normalize(
        const LocalTensor<float> &updated, const LocalTensor<float> &weight) {
        // AllocTensor waits for the previous row's output DMA before work is overwritten.
        auto work = workQueue_.AllocTensor<float>();
        auto statistic = scalarBuffer_.Get<float>();
        auto reduce = reduceBuffer_.Get<float>();
        const uint32_t fullRepeats = hidden_ / 64;
        const uint32_t tail = hidden_ % 64;
        const uint32_t reduceRepeats = (hidden_ + 63) / 64;
        // Brcb reads eight FP32 values; only statistic[0] holds the inverse RMS.
        Duplicate(statistic, 0.0f, 8);
        if constexpr (Tiled) {
            // Up to 112 partials; the 192-element scratch region is zero padded.
            Duplicate(reduce, 0.0f, 192);
        }
        if (tail != 0) {
            // Zero the last 64-element square tile, including lanes outside D.
            Duplicate(work[fullRepeats * 64], 0.0f, 64);
        }
        PipeBarrier<PIPE_V>();
        Mul(work, updated, updated, hidden_);
        PipeBarrier<PIPE_V>();
        // Normal-mode vcadd writes UB directly; the count-based ReduceSum uses Scalar.
        WholeReduceSum(reduce, work, 64, reduceRepeats, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        if constexpr (Tiled) {
            const uint32_t groups = (reduceRepeats + 63) / 64;
            // Separate destination avoids overlapping a later source repeat.
            WholeReduceSum(reduce[192], reduce, 64, groups, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(statistic, reduce[192], groups, 1, 1, 1, 8);
        } else {
            WholeReduceSum(statistic, reduce, reduceRepeats, 1, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        Muls(statistic, statistic, inverseHidden_, 1);
        PipeBarrier<PIPE_V>();
        Adds(statistic, statistic, eps_, 1);
        PipeBarrier<PIPE_V>();
        // A2 vrsqrt is approximate; use vector sqrt/div for FP16 output accuracy.
        Sqrt(statistic, statistic, 1);
        // Partial sums are dead after reduction; reuse their buffer for 1.0f.
        Duplicate(reduce, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        Div(statistic, reduce, statistic, 1);
        PipeBarrier<PIPE_V>();
        // Reuse the partial-sum buffer. Its first 32 B block is eight copies of inverse RMS.
        Brcb(reduce, statistic, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        // Reuse that same block across all blocks and repeats, without a Scalar UB read.
        const BinaryRepeatParams broadcastParams{1, 1, 0, 8, 8, 0};
        if (fullRepeats != 0) {
            Mul(work, updated, reduce, uint64_t(64), static_cast<uint8_t>(fullRepeats), broadcastParams);
        }
        if (tail != 0) {
            Mul(work[fullRepeats * 64], updated[fullRepeats * 64], reduce, static_cast<uint64_t>(tail), 1,
                broadcastParams);
        }
        PipeBarrier<PIPE_V>();
        Mul(work, work, weight, hidden_);
        return work;
    }

    __aicore__ inline void WriteInvalidRow(int64_t row) {
        union {
            uint32_t bits;
            float value;
        } quietNan;
        quietNan.bits = 0x7fc00000;
        auto updated = updatedBuffer_.Get<float>();
        Duplicate(updated, quietNan.value, hidden_);
        PipeBarrier<PIPE_V>();
        StoreResidual(row, updated);
        auto work = workQueue_.AllocTensor<float>();
        Duplicate(work, quietNan.value, hidden_);
        PipeBarrier<PIPE_V>();
        StoreModulated(row, work);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inputQueue_[INPUT_COUNT];
    TBuf<TPosition::VECCALC> inputFloat_[INPUT_COUNT];
    TBuf<TPosition::VECCALC> updatedBuffer_, scalarBuffer_, reduceBuffer_;
    TBuf<TPosition::VECIN> weightOutputBuffer_;
    TQue<TPosition::VECOUT, 1> workQueue_, residualOutQueue_;
    GlobalTensor<Input> inputGm_[INPUT_COUNT];
    GlobalTensor<Index> indicesGm_;
    GlobalTensor<Output> residualOutGm_, modulatedOutGm_;
    int64_t rows_, indexedRows_;
    uint32_t hidden_, aligned_, cores_;
    uint32_t tileCount_, tileBase_, extraTiles_;
    float eps_, inverseHidden_;
    event_t residualReady_, residualFree_;
};

extern "C" __global__ __aicore__ void eagle_indexed_gate_norm_modulate(GM_ADDR residual, GM_ADDR gate, GM_ADDR branch,
    GM_ADDR weight, GM_ADDR shift, GM_ADDR scale, GM_ADDR indices, GM_ADDR residualOut, GM_ADDR modulatedOut,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(data, tiling);
    if (TILING_KEY_IS(1)) {
        KernelIndexedGateNormModulate<DTYPE_RESIDUAL, DTYPE_RESIDUAL_OUT, DTYPE_INDICES, true, 2048> op;
        op.Init(residual, gate, branch, weight, shift, scale, indices, residualOut, modulatedOut, data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelIndexedGateNormModulate<DTYPE_RESIDUAL, DTYPE_RESIDUAL_OUT, DTYPE_INDICES, true, 1024> op;
        op.Init(residual, gate, branch, weight, shift, scale, indices, residualOut, modulatedOut, data);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelIndexedGateNormModulate<DTYPE_RESIDUAL, DTYPE_RESIDUAL_OUT, DTYPE_INDICES, true, 3072> op;
        op.Init(residual, gate, branch, weight, shift, scale, indices, residualOut, modulatedOut, data);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelIndexedGateNormModulate<DTYPE_RESIDUAL, DTYPE_RESIDUAL_OUT, DTYPE_INDICES> op;
        op.Init(residual, gate, branch, weight, shift, scale, indices, residualOut, modulatedOut, data);
        op.Process();
    }
}
