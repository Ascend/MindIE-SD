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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_PLATFORM_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_PLATFORM_H

#include <algorithm>
#include <cstdint>
#include <utility>
#include "../op_kernel/eagle_split_norm_rope_concat_config_950.h"

namespace eagle_snrc {
constexpr uint32_t HEAD_TILE = 64;
constexpr uint64_t HEAD_DIM = 128;

// Prefer the platform architecture API without requiring its type or enumerators
// in SDKs that predate 3510. Both expressions participate in substitution.
template <typename Platform, typename Arch = decltype(std::declval<Platform &>().GetCurNpuArch())>
auto IsAscend950(Platform &platform, int) -> decltype(Arch::DAV_3510, bool()) {
    return platform.GetCurNpuArch() == Arch::DAV_3510;
}

template <typename Platform> bool IsAscend950(Platform &platform, long) {
    // Compatibility with the existing Eagle BSA/quant-BSA Host implementation:
    // SocVersion code 4 identifies 950 without referencing a new SDK enum name.
    constexpr uint32_t socVersion950 = 4;
    return static_cast<uint32_t>(platform.GetSocVersion()) == socVersion950;
}

struct UbBudget {
    uint64_t required;
    uint64_t available;
};

struct TilePlan950 {
    uint32_t targetRows = 0;
    uint32_t qTokens = 0;
    uint32_t kTokens = 0;
    uint32_t vTokens = 0;
    uint32_t cacheTokens = 0;
    uint64_t required = 0;
    uint64_t available = 0;
};

inline uint64_t Get950UbLimit(uint64_t queriedUb) {
    // Physical UB is at most 256 KiB. Reserve stack/API space before planning.
    constexpr uint64_t reserve = 8 * 1024;
    const uint64_t physical = std::min<uint64_t>(queriedUb, 256 * 1024);
    return physical > reserve ? (physical - reserve) / 32 * 32 : 0;
}

inline uint32_t Get950TokenTile(uint32_t target, uint32_t heads) {
    uint32_t tokens = std::max(1U, (2 * target + heads) / (2 * heads));
    while (100 * tokens * heads < 55 * target) {
        ++tokens;
    }
    return tokens;
}

inline bool Build950Plan(
    int64_t qHeads, int64_t kHeads, int64_t vHeads, uint32_t width, uint64_t queriedUb, TilePlan950 &plan) {
    if (qHeads < 1 || qHeads > MAX_HEADS_950 || kHeads < 1 || kHeads > MAX_HEADS_950 || vHeads < 1 ||
        vHeads > MAX_HEADS_950 || width == 0 || width > HEAD_DIM || width % 16 != 0) {
        return false;
    }
    const uint64_t limit = Get950UbLimit(queriedUb);
    constexpr uint64_t rowBytes = HEAD_DIM * sizeof(uint16_t);
    constexpr uint64_t fixed = 2 * rowBytes + 3 * HEAD_DIM * sizeof(uint32_t);
    // VF coefficient loads span 128 elements even for a compact row. The final
    // sin row needs a masked-read guard; cos may safely read into its sin region.
    const uint64_t guard = ((HEAD_DIM - width) * sizeof(uint16_t) + 31) / 32 * 32;
    for (uint32_t target = 128; target >= 98; --target) {
        const uint32_t q = Get950TokenTile(target, qHeads);
        const uint32_t k = Get950TokenTile(target, kHeads);
        const uint32_t v = Get950TokenTile(target, vHeads);
        // Two shared Q/K input slots and three rotating output slots must each
        // fit either branch. V transforms in place in its single bound queue.
        const uint64_t normSlot = std::max(q * qHeads, k * kHeads) * rowBytes;
        const uint64_t valueSlot = v * vHeads * rowBytes;
        const uint64_t base = (2 + OUTPUT_SLOTS_950) * normSlot + valueSlot + fixed + ROPE_CACHE_SLOTS_950 * guard;
        const uint64_t perToken = ROPE_CACHE_SLOTS_950 * 2 * width * sizeof(uint16_t);
        if (limit < base + perToken) {
            continue;
        }
        plan.targetRows = target;
        plan.qTokens = q;
        plan.kTokens = k;
        plan.vTokens = v;
        plan.cacheTokens = static_cast<uint32_t>((limit - base) / perToken);
        plan.required = base + perToken * plan.cacheTokens;
        plan.available = limit;
        return true;
    }
    return false;
}

inline UbBudget GetUbBudget(bool is950, uint64_t queriedUb, int64_t ropeType, bool compactCache) {
    if (is950) {
        // The head-dependent required size is supplied by Build950Plan.
        return {0, Get950UbLimit(queriedUb)};
    }
    // Keep the 910B allocation, bank gaps and reserve exactly as before.
    const uint64_t head128Ub = 20 * HEAD_TILE * HEAD_DIM + 28 * HEAD_DIM + 40 * HEAD_TILE + 2 * 256;
    const uint64_t required = head128Ub + (ropeType == 0 ? (16 * HEAD_DIM + HEAD_DIM) * sizeof(uint32_t) : 0) +
        (compactCache ? HEAD_DIM * sizeof(uint32_t) : 0);
    constexpr uint64_t reserved = 8 * 1024 + 256;
    const uint64_t physical = std::min<uint64_t>(queriedUb, 192 * 1024);
    return {required, physical > reserved ? physical - reserved : 0};
}

// ParseShape has already bounded the complete packed tensor to INT32_MAX.
inline int64_t GetHeadTaskCount(int64_t batch, int64_t seq, int64_t heads) {
    return batch * seq * ((heads + HEAD_TILE - 1) / HEAD_TILE);
}

inline int64_t Get950TaskCount(int64_t batch, int64_t seq) { return batch * seq; }

inline uint32_t GetBlockDim(uint32_t cores, int64_t tasks) {
    return static_cast<uint32_t>(std::min<int64_t>(cores, tasks));
}
} // namespace eagle_snrc
#endif
