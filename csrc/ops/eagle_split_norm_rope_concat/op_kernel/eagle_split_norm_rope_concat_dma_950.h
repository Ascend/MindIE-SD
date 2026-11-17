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

#ifndef EAGLE_SPLIT_NORM_ROPE_CONCAT_DMA_950_H
#define EAGLE_SPLIT_NORM_ROPE_CONCAT_DMA_950_H
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
namespace eagle_snrc {
namespace dma950 {
using namespace AscendC;
constexpr uint32_t UB_BLOCK_BYTES = 32;
// CANN 9.1/950 DataCopyPad Normal: blockLen and GM gap are BYTES;
// UB gap is 32-byte blocks. Never narrow the GM gap to DataCopyParams u16.
// Both head rows and RoPE cache rows occupy whole UB blocks; no padding is needed.
template <typename T>
__aicore__ inline void Load(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint16_t count, uint32_t bytes,
    uint64_t gmGapBytes, uint32_t ubGapBlocks) {
    DataCopyExtParams p{};
    p.blockCount = count;
    p.blockLen = bytes;
    p.srcStride = gmGapBytes;
    p.dstStride = ubGapBlocks;
    DataCopyPadExtParams<T> padding{};
    padding.isPad = false;
    DataCopyPad<T, PaddingMode::Normal>(dst, src, p, padding);
}
template <typename T>
__aicore__ inline void Store(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint16_t count, uint32_t bytes,
    uint32_t ubGapBlocks, uint64_t gmGapBytes) {
    DataCopyExtParams p{};
    p.blockCount = count;
    p.blockLen = bytes;
    p.srcStride = ubGapBlocks;
    p.dstStride = gmGapBytes;
    DataCopyPad<T, PaddingMode::Normal>(dst, src, p);
}
}
} // namespace eagle_snrc::dma950
#endif
#endif
