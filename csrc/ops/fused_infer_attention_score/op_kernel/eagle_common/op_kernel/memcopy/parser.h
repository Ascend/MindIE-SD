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
 * \file parser.h
 * \brief
 */
#ifndef PARSER_H
#define PARSER_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

using AscendC::GlobalTensor;

// ----------------------------------------------ActualSeqLensParser--------------------------------
enum class ActualSeqLensMode {
    BY_BATCH = 0,
    ACCUM = 1,
};

template <ActualSeqLensMode MODE, typename ACTLEN_T = uint64_t, bool WITH_ZERO_HEAD = false>
class ActualSeqLensParser {};

template <typename ACTLEN_T> class ActualSeqLensParser<ActualSeqLensMode::ACCUM, ACTLEN_T, false> {
  public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(
        GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal = 0) {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const {
        if (bIdx == 0) {
            return 0;
        }
        return actualSeqLengthsGm.GetValue(bIdx - 1);
    }

    __aicore__ inline uint64_t GetMxVscaleTBase(uint32_t bIdx) const {
        if (bIdx == 0) {
            return 0;
        }
        uint64_t vScaleTBaseOffset = 0;
        for (uint32_t idx = 0; idx < bIdx; idx++) {
            vScaleTBaseOffset += ((GetActualSeqLength(idx) + 63) >> 6);
        }
        return vScaleTBaseOffset;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const {
        if (bIdx == 0) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return (actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1));
    }

    __aicore__ inline uint64_t GetFullActualSeqLength(uint32_t bIdx) const { return GetActualSeqLength(bIdx); }

    __aicore__ inline uint64_t GetTSize() const { return actualSeqLengthsGm.GetValue(actualLenDims - 1); }

  private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims;
};

template <typename ACTLEN_T> class ActualSeqLensParser<ActualSeqLensMode::BY_BATCH, ACTLEN_T, false> {
  public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(
        GlobalTensor<ACTLEN_T> actualSeqLengthsGm, uint32_t actualLenDims, uint64_t defaultVal) {
        this->actualSeqLengthsGm = actualSeqLengthsGm;
        this->actualLenDims = actualLenDims;
        this->defaultVal = defaultVal;
    }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const {
        if (actualLenDims == 0) {
            return defaultVal;
        }
        if (actualLenDims == 1) {
            return actualSeqLengthsGm.GetValue(0);
        }
        return actualSeqLengthsGm.GetValue(bIdx);
    }

    __aicore__ inline uint32_t GetActualLenDims() const { return actualLenDims; }

  private:
    GlobalTensor<ACTLEN_T> actualSeqLengthsGm;
    uint32_t actualLenDims = 0;
    uint64_t defaultVal = 0;
};

// ----------------------------------------------BlockTableParser--------------------------------
class BlockTableParser {
  public:
    __aicore__ inline BlockTableParser() = default;

    __aicore__ inline void Init(GlobalTensor<int32_t> blockTableGm, uint32_t maxblockNumPerBatch) {
        this->blockTableGm = blockTableGm;
        this->maxblockNumPerBatch = maxblockNumPerBatch;
    }

    __aicore__ inline int32_t GetBlockIdx(uint32_t bIdx, uint32_t blockIdxInBatch) const {
        return blockTableGm.GetValue(bIdx * maxblockNumPerBatch + blockIdxInBatch);
    }

  private:
    GlobalTensor<int32_t> blockTableGm;
    uint32_t maxblockNumPerBatch;
};

// ----------------------------------------------WITH_ZERO_HEAD=true--------------------------------
template <ActualSeqLensMode MODE, typename ACTLEN_T> class ActualSeqLensParser<MODE, ACTLEN_T, true> {
  public:
    __aicore__ inline ActualSeqLensParser() = default;

    __aicore__ inline void Init(GlobalTensor<ACTLEN_T> cuSeqLensGm, GlobalTensor<ACTLEN_T> seqUsedGM,
        uint32_t cuSeqLensSize, uint32_t seqUsedSize, uint64_t defaultVal = 0) {
        this->cuSeqLensGm = cuSeqLensGm;
        this->seqUsedGM = seqUsedGM;
        this->cuSeqLensSize = cuSeqLensSize;
        this->seqUsedSize = seqUsedSize;
    }

    __aicore__ inline uint64_t GetTBase(uint32_t bIdx) const { return cuSeqLensGm.GetValue(bIdx); }

    __aicore__ inline uint64_t GetActualSeqLength(uint32_t bIdx) const {
        if (seqUsedSize == 0) {
            return cuSeqLensGm.GetValue(bIdx + 1) - cuSeqLensGm.GetValue(bIdx);
        }
        return seqUsedGM.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetFullActualSeqLength(uint32_t bIdx) const {
        return cuSeqLensGm.GetValue(bIdx + 1) - cuSeqLensGm.GetValue(bIdx);
    }

    __aicore__ inline uint64_t GetTSize() const { return cuSeqLensGm.GetValue(cuSeqLensSize); }

  private:
    GlobalTensor<ACTLEN_T> cuSeqLensGm;
    GlobalTensor<ACTLEN_T> seqUsedGM;
    uint32_t cuSeqLensSize;
    uint32_t seqUsedSize;
};

#endif
