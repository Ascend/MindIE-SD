/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 */
#pragma once

#include <array>
#include <cstdint>
#include <cstring>

namespace EagleHadamardQuant {
struct Params {
    const char *error = nullptr;
    int64_t k = 0;
    int64_t blocks = 0;
    int64_t outputColumns = 0;
    int64_t scalePairs = 0;
    int format = 0;
};

inline bool MatchesDimension(int64_t actual, int64_t expected, bool allowUnknown) {
    return actual == expected || (allowUnknown && actual == -1);
}

// Shared metadata contract for ACLNN, Host shape/tiling, and the Torch plugin.
inline Params CheckParams(int64_t m, int64_t n, const int64_t *rotation, int64_t rank, int64_t blockSize,
    const char *dstType, bool isSignHNz, bool allowUnknown = false) {
    Params p;
    if (blockSize != 32 && blockSize != 64 && blockSize != 128 && blockSize != 256) {
        p.error = "block_size must be 32, 64, 128 or 256";
        return p;
    }
    const bool unknownM = allowUnknown && m == -1;
    const bool unknownN = allowUnknown && n == -1;
    if ((!unknownM && (m < 0 || m > UINT32_MAX)) || (!unknownN && (n <= 0 || n > UINT32_MAX || n % blockSize != 0))) {
        p.error = "seq and hd must fit uint32 and hd must be a positive multiple of block_size";
        return p;
    }
    if (dstType == nullptr || (std::strcmp(dstType, "mxfp4") != 0 && std::strcmp(dstType, "mxfp8") != 0)) {
        p.error = "dst_type must be mxfp4 or mxfp8 (E4M3)";
        return p;
    }
    if (isSignHNz) {
        if (rank != 5) {
            p.error = "is_sign_h_nz=True requires 5D NZ sign_h [B,N1,K1,K0,N0]";
            return p;
        }
        const auto tiles = blockSize / 16;
        if (!MatchesDimension(rotation[1], tiles, allowUnknown) ||
            !MatchesDimension(rotation[2], tiles, allowUnknown) || !MatchesDimension(rotation[3], 16, allowUnknown) ||
            !MatchesDimension(rotation[4], 16, allowUnknown)) {
            p.error = "NZ sign_h must have shape [B, block_size/16, block_size/16, 16, 16]";
            return p;
        }
    } else {
        if (rank != 2 && rank != 3) {
            p.error = "is_sign_h_nz=False requires ND sign_h [K,K] or [hd/block_size,K,K]";
            return p;
        }
        if (!MatchesDimension(rotation[rank - 2], blockSize, allowUnknown) ||
            !MatchesDimension(rotation[rank - 1], blockSize, allowUnknown)) {
            p.error = "block_size must match the ND sign_h matrix dimensions";
            return p;
        }
    }
    p.blocks = rank == 2 ? 1 : rotation[0];
    if ((p.blocks <= 0 && !(allowUnknown && p.blocks == -1)) ||
        (!unknownN && !(allowUnknown && p.blocks == -1) && p.blocks != n / blockSize &&
            !(p.blocks == 1 && (rank == 2 || isSignHNz)))) {
        p.error = "sign_h must contain hd/block_size rotations or one shared rotation (2D ND or NZ)";
        return p;
    }
    p.k = blockSize;
    p.format = std::strcmp(dstType, "mxfp4") == 0 ? 0 : 1;
    p.outputColumns = unknownN ? -1 : (p.format == 0 ? n / 2 : n);
    p.scalePairs = unknownN ? -1 : (n + 63) / 64;
    return p;
}
// CANN Shape and gert::Shape expose the same dimension accessors.
template <typename Shape>
inline Params CheckShape(const Shape &x, const Shape &rotation, int64_t blockSize, const char *dstType, bool isSignHNz,
    bool allowUnknown = false) {
    if (x.GetDimNum() != 2 || rotation.GetDimNum() > 5 || dstType == nullptr) {
        Params p;
        p.error = "x must be 2D, sign_h must have at most 5 dimensions, and dst_type must be provided";
        return p;
    }
    std::array<int64_t, 5> dims{};
    for (size_t i = 0; i < rotation.GetDimNum(); ++i) {
        dims[i] = rotation.GetDim(i);
    }
    return CheckParams(
        x.GetDim(0), x.GetDim(1), dims.data(), rotation.GetDimNum(), blockSize, dstType, isSignHNz, allowUnknown);
}
} // namespace EagleHadamardQuant
