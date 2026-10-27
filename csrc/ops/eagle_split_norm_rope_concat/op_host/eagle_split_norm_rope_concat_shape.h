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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_SHAPE_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_SHAPE_H
#include <cmath>
#include <cstdint>
#include <limits>
#include "exe_graph/runtime/shape.h"

namespace eagle_snrc {
struct ShapeInfo {
    int64_t batch;
    int64_t seq;
    int64_t heads[3];
    int64_t dim;
    int64_t rotaryDim;
    int64_t ropeBatch;
    int64_t ropeWidth;
};

inline bool ParseShape(const gert::Shape &packed, const gert::Shape &headShape, const int64_t *heads, int64_t rotaryDim,
    int64_t ropeType, float eps, ShapeInfo &info) {
    const size_t rank = packed.GetDimNum();
    if ((rank != 3 && rank != 4) || headShape.GetDimNum() != 1 || headShape.GetDim(0) != 3 || heads == nullptr ||
        (ropeType != 0 && ropeType != 1) || !std::isfinite(eps) || eps <= 0) {
        return false;
    }
    int64_t size = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t d = packed.GetDim(i);
        if (d <= 0 || d > std::numeric_limits<int32_t>::max() / size) {
            return false;
        }
        size *= d;
    }
    const int64_t totalHeads = packed.GetDim(rank - 2);
    info.batch = rank == 4 ? packed.GetDim(0) : 1;
    info.seq = packed.GetDim(rank - 3);
    info.dim = packed.GetDim(rank - 1);
    info.rotaryDim = rotaryDim == 0 ? info.dim : rotaryDim;
    if (info.dim != 128 || info.rotaryDim < 32 || info.rotaryDim > info.dim || info.rotaryDim % 32 != 0) {
        return false;
    }
    for (size_t i = 0; i < 3; ++i) {
        if (heads[i] <= 0 || heads[i] > totalHeads) {
            return false;
        }
        info.heads[i] = heads[i];
    }
    return heads[0] + heads[1] + heads[2] == totalHeads && heads[1] == heads[2] && heads[0] % heads[1] == 0;
}

inline bool CheckRope(const gert::Shape &cos, const gert::Shape &sin, int64_t rotaryDim, ShapeInfo &info) {
    const size_t rank = cos.GetDimNum();
    if (rank < 2 || rank > 4 || sin.GetDimNum() != rank) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (cos.GetDim(i) != sin.GetDim(i)) {
            return false;
        }
    }
    info.ropeBatch = rank == 2 ? 1 : cos.GetDim(0);
    info.ropeWidth = cos.GetDim(rank - 1);
    const int64_t seq = cos.GetDim(rank == 2 ? 0 : 1);
    return (info.ropeBatch == 1 || info.ropeBatch == info.batch) && seq == info.seq &&
        (info.ropeWidth == rotaryDim || info.ropeWidth == rotaryDim / 2) && (rank != 4 || cos.GetDim(2) == 1);
}
} // namespace eagle_snrc
#endif
