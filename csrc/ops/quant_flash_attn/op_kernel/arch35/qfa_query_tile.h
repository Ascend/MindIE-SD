/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * SPDX-License-Identifier: MulanPSL-2.0
 */

#ifndef QFA_QUERY_TILE_H
#define QFA_QUERY_TILE_H

#include <cstdint>

#if defined(__CCE_AICORE__)
#define QFA_TILE_INLINE __aicore__ inline
#else
#define QFA_TILE_INLINE inline
#endif

namespace QFA_KERNEL {

struct QueryTile {
    uint64_t offset;
    uint32_t rowCount;
};

QFA_TILE_INLINE uint64_t QueryTileCount(uint64_t seqLen, uint32_t groupSize, uint32_t tileSize, bool isS1G) {
    if (isS1G) {
        return (seqLen * groupSize + tileSize - 1) / tileSize;
    }
    // BNSD scale DN2NZ copies cannot concatenate heads at an unaligned L1 address.
    return ((seqLen + tileSize - 1) / tileSize) * groupSize;
}

QFA_TILE_INLINE QueryTile GetQueryTile(
    uint64_t tileIdx, uint64_t seqLen, uint32_t groupSize, uint32_t tileSize, bool isS1G) {
    if (seqLen == 0 || groupSize == 0) {
        return {0, 0};
    }
    if (isS1G) {
        uint64_t offset = tileIdx * tileSize;
        uint64_t remaining = seqLen * groupSize - offset;
        return {offset, static_cast<uint32_t>(remaining < tileSize ? remaining : tileSize)};
    }
    uint64_t tilesPerHead = (seqLen + tileSize - 1) / tileSize;
    uint64_t seqOffset = (tileIdx % tilesPerHead) * tileSize;
    uint64_t offset = (tileIdx / tilesPerHead) * seqLen + seqOffset;
    uint64_t remaining = seqLen - seqOffset;
    return {offset, static_cast<uint32_t>(remaining < tileSize ? remaining : tileSize)};
}

} // namespace QFA_KERNEL

#undef QFA_TILE_INLINE
#endif
