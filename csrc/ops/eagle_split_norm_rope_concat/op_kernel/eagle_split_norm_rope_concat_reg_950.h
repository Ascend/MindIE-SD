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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_REG_950_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_REG_950_H

#include "kernel_operator.h"

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace eagle_snrc {
namespace reg950 {
using namespace AscendC;
namespace V = AscendC::Reg;

constexpr uint32_t DIM = 128;
constexpr uint32_t LANES = 64;

// DIST_UNPACK_B16 places 64 consecutive b16 elements in the low halves of
// 64 b32 lanes. ZERO selects these halves; it does NOT mean the lower 64
// elements of a normally loaded b16 register. Use this mapping for x, w, c, s.
template <typename T> __simd_callee__ inline void Load64(V::RegTensor<float> &dst, __ubuf__ T *src, V::MaskReg &mask) {
    static constexpr V::CastTrait widen = {
        V::RegLayout::ZERO, V::SatMode::UNKNOWN, V::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    V::RegTensor<T> packed;
    V::LoadAlign<T, V::LoadDist::DIST_UNPACK_B16>(packed, src);
    V::Cast<float, T, widen>(dst, packed, mask);
}

template <typename T> __simd_callee__ inline void Store64(__ubuf__ T *dst, V::RegTensor<float> &src, V::MaskReg &mask) {
    static constexpr V::CastTrait narrow = {
        V::RegLayout::ZERO, V::SatMode::NO_SAT, V::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    V::RegTensor<T> packed;
    V::Cast<T, float, narrow>(packed, src, mask);
    V::StoreAlign<T, V::StoreDist::DIST_PACK_B32>(dst, packed, mask);
}

// The index is an ELEMENT index in [0,127]. Gather itself only addresses one
// register (index modulo 64); source selection must therefore be explicit.
// In particular HALF/R96 lanes 16..47 use hi[0..31], not lo[0..31].
__simd_callee__ inline void Gather128(V::RegTensor<float> &dst, V::RegTensor<float> &lo, V::RegTensor<float> &hi,
    V::RegTensor<uint32_t> &index, V::MaskReg &fromHi) {
    V::RegTensor<float> lowSource, highSource;
    V::Gather(lowSource, lo, index);
    V::Gather(highSource, hi, index);
    V::Select(dst, highSource, lowSource, fromHi);
}

template <typename T>
__simd_callee__ inline void ExpandCache(V::RegTensor<float> &lo, V::RegTensor<float> &hi, __ubuf__ T *src,
    __ubuf__ uint32_t *indices, uint32_t width, V::MaskReg &all) {
    V::RegTensor<float> rawLo, rawHi;
    // Only W elements came from GM. Cast zeroes inactive FP32 register lanes;
    // the allocated UB padding is never used as a cache coefficient.
    uint32_t remaining = width;
    V::MaskReg validLo = V::UpdateMask<float>(remaining);
    V::MaskReg validHi = V::UpdateMask<float>(remaining);
    Load64(rawLo, src, validLo);
    Load64(rawHi, src + LANES, validHi);
    V::RegTensor<uint32_t> index;
    V::MaskReg fromHi;
    V::LoadAlign(index, indices);
    V::Compares<uint32_t, CMPMODE::GE>(fromHi, index, LANES, all);
    Gather128(lo, rawLo, rawHi, index, fromHi);
    V::LoadAlign(index, indices + LANES);
    V::Compares<uint32_t, CMPMODE::GE>(fromHi, index, LANES, all);
    Gather128(hi, rawLo, rawHi, index, fromHi);
}

} // namespace reg950
} // namespace eagle_snrc
#endif // __NPU_ARCH__ == 3510
#endif
