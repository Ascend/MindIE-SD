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
 * \file axis.h
 * \brief
 */
#ifndef AXIS_H
#define AXIS_H

#include <type_traits>
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

struct AxisSlices;

struct Axis {
    uint32_t start;
    uint32_t sizeAct;

    __aicore__ inline Axis(uint32_t sizeAct_) : start(0), sizeAct(sizeAct_) {}
    __aicore__ inline Axis(uint32_t start_, uint32_t sizeAct_) : start(start_), sizeAct(sizeAct_) {}

    __aicore__ inline AxisSlices Split(uint32_t splitSize) const;

    template <uint32_t ALIGN_SIZE = BLOCK_CUBE> __aicore__ inline uint32_t AlignedSize() const {
        return (((ALIGN_SIZE) == 0) ? 0 : (((sizeAct) + (ALIGN_SIZE)-1) / (ALIGN_SIZE) * (ALIGN_SIZE)));
    }

    __aicore__ inline bool IsTailOf(const Axis &parent) const { return (start + sizeAct) >= parent.sizeAct; }
};

struct AxisSlices {
    struct Sentinel {
        uint32_t end_;
    };
    struct Iterator {
        Axis cur_;
        uint32_t tSizeAct_;
        uint32_t splitSize_;

        __aicore__ inline Iterator(const AxisSlices &slices)
            : cur_(Min(slices.splitSize_, slices.sizeAct_)), tSizeAct_(slices.sizeAct_), splitSize_(slices.splitSize_) {
        }

        __aicore__ inline Axis &operator*() { return cur_; }

        __aicore__ inline Iterator &operator++() {
            cur_.start += splitSize_;
            cur_.sizeAct = Min(splitSize_, tSizeAct_ - cur_.start);
            return *this;
        }

        __aicore__ inline bool operator!=(const Sentinel &end) const { return cur_.start < end.end_; }
    };

    uint32_t sizeAct_;
    uint32_t splitSize_;

    __aicore__ inline AxisSlices(uint32_t sizeAct, uint32_t splitSize) : sizeAct_(sizeAct), splitSize_(splitSize) {}

    __aicore__ inline Iterator begin() { return {*this}; }

    __aicore__ inline Sentinel end() { return Sentinel{sizeAct_}; }

    __aicore__ inline uint32_t size() const { return (sizeAct_ + (splitSize_ - 1)) / splitSize_; }
};

__aicore__ inline AxisSlices Axis::Split(uint32_t splitSize) const { return {sizeAct, splitSize}; }
#endif // AXIS_H
