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

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Sigmoid vector sequence adapted from cann/ops-nn, activation/sigmoid/op_kernel/arch35/sigmoid_bf16.h.
// https://gitcode.com/cann/ops-nn/blob/a5a8fbf3e02d7632ffa2c56b29e8755f1aa62720/LICENSE
#include "mx_quant.h"
#include "tiling_data.h"

namespace SigmoidMulMx {
template <typename T>
__aicore__ inline void GateProduct(
    __ubuf__ bfloat16_t *x, __ubuf__ bfloat16_t *gate, __ubuf__ T *product, uint32_t count, uint16_t chunks) {
    __VEC_SCOPE__ {
        static constexpr Reg::CastTrait widen = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait narrow = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        Reg::RegTensor<bfloat16_t> g, a, s;
        Reg::RegTensor<float> f, v, one;
        Reg::Duplicate(one, 1.0f);
        auto full = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
        for (uint16_t i = 0; i < chunks; ++i) {
            const uint32_t offset = i * VF_LEN_32;
            auto valid = Reg::UpdateMask<float>(count);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(g, gate + offset);
            Reg::LoadAlign<bfloat16_t, Reg::LoadDist::DIST_UNPACK_B16>(a, x + offset);
            Reg::Cast<float, bfloat16_t, widen>(f, g, valid);
            Reg::Muls(f, f, -1.0f, valid);
            Reg::Exp(f, f, valid);
            Reg::Adds(f, f, 1.0f, valid);
            Reg::Div(f, one, f, valid);
            // Zero inactive lanes before group max, including the unused half of an odd scale pair.
            if constexpr (sizeof(T) == sizeof(float)) {
                Reg::Cast<float, bfloat16_t, widen>(v, a, valid);
                Reg::Mul<float, Reg::MaskMergeMode::ZEROING>(f, f, v, valid);
                Reg::StoreAlign<float>(product + offset, f, full);
            } else {
                Reg::Cast<bfloat16_t, float, narrow>(s, f, valid);
                Reg::Mul<bfloat16_t, Reg::MaskMergeMode::ZEROING>(s, s, a, valid);
                Reg::StoreAlign<bfloat16_t, Reg::StoreDist::DIST_PACK_B32>(product + offset, s, full);
            }
        }
    }
}

template <typename T, bool CeilScale> class Kernel {
  public:
    __aicore__ inline void Run(GM_ADDR a, GM_ADDR g, GM_ADDR y, GM_ADDR scale, const SigmoidMulMxTilingData &tile) {
        x_.SetGlobalBuffer((__gm__ bfloat16_t *)a);
        g_.SetGlobalBuffer((__gm__ bfloat16_t *)g);
        y_.SetGlobalBuffer((__gm__ uint8_t *)y);
        scale_.SetGlobalBuffer((__gm__ uint8_t *)scale);
        capacity_ = tile.bufferElements;
        pipe_.InitBuffer(inputs_, 2, capacity_ * 2 * sizeof(bfloat16_t));
        pipe_.InitBuffer(outputs_, 2, capacity_ + tile.scaleCapacity);
        pipe_.InitBuffer(product_, capacity_ * sizeof(T));
        pipe_.InitBuffer(maxExp_, tile.scaleCapacity * (CeilScale ? sizeof(T) : sizeof(uint16_t)));
        pipe_.InitBuffer(recip_, tile.scaleCapacity * sizeof(uint16_t));
        const int64_t units = (tile.rows - 1) / tile.rowQuantum + 1;
        const int64_t cores = GetBlockNum(), core = GetBlockIdx();
        const int64_t base = units / cores, extra = units % cores;
        const int64_t first = (base * core + (core < extra ? core : extra)) * tile.rowQuantum;
        const int64_t limit = first + (base + (core < extra)) * tile.rowQuantum;
        const int64_t end = limit < tile.rows ? limit : tile.rows;
        const int64_t scaleStride = ((tile.cols - 1) / 64 + 1) * 2;
        for (int64_t row = first; row < end; row += tile.tileRows) {
            const uint32_t rows = end - row < tile.tileRows ? end - row : tile.tileRows;
            for (int64_t col = 0; col < tile.cols; col += tile.tileCols) {
                const uint32_t cols = tile.cols - col < tile.tileCols ? tile.cols - col : tile.tileCols;
                Process(row * tile.cols + col, row * scaleStride + col / 32, rows * cols);
            }
        }
    }

  private:
    __aicore__ inline void Process(int64_t offset, int64_t scaleOffset, uint32_t n) {
        auto in = inputs_.AllocTensor<bfloat16_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, 0};
        DataCopyPad(in, x_[offset], copy, pad);
        DataCopyPad(in[capacity_], g_[offset], copy, pad);
        inputs_.EnQue(in);
        in = inputs_.DeQue<bfloat16_t>();
        auto out = outputs_.AllocTensor<uint8_t>();
        auto *a = (__ubuf__ bfloat16_t *)in.GetPhyAddr();
        auto *p = (__ubuf__ T *)product_.Get<T>().GetPhyAddr();
        auto *maxExp = (__ubuf__ uint16_t *)maxExp_.Get<uint16_t>().GetPhyAddr();
        auto *recip = (__ubuf__ uint16_t *)recip_.Get<uint16_t>().GetPhyAddr();
        auto *quant = (__ubuf__ int8_t *)out.GetPhyAddr();
        auto *scale = (__ubuf__ uint16_t *)(quant + capacity_);
        uint16_t chunks = (n + 255) / 256;
        uint32_t numScales = ((n + 63) / 64) * 2;
        GateProduct(a, a + capacity_, p, n, chunks * 4);
        inputs_.FreeTensor(in);
        if constexpr (CeilScale && sizeof(T) == sizeof(float)) {
            auto *max32 = (__ubuf__ uint32_t *)maxExp_.Get<uint32_t>().GetPhyAddr();
            ComputeGroupMaxFp32(p, max32, chunks);
            ComputeScaleCeil(max32, scale, recip, (numScales + 63) / 64);
        } else {
            if constexpr (sizeof(T) == sizeof(float)) {
                ComputeGroupMaxFp32(p, maxExp, chunks);
            } else {
                ComputeGroupMaxBf16<CeilScale>(p, maxExp, chunks);
            }
            if constexpr (CeilScale) {
                ComputeScaleCeil(maxExp, scale, recip, (numScales + 63) / 64);
            } else {
                ComputeScaleOcp(maxExp, scale, recip, (numScales + 127) / 128, numScales);
            }
        }
        ComputeData(p, recip, quant, chunks);
        outputs_.EnQue(out);
        out = outputs_.DeQue<uint8_t>();
        copy.blockLen = n;
        DataCopyPad(y_[offset], out, copy);
        copy.blockLen = numScales;
        DataCopyPad(scale_[scaleOffset], out[capacity_], copy);
        outputs_.FreeTensor(out);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputs_;
    TQue<QuePosition::VECOUT, 1> outputs_;
    TBuf<TPosition::VECCALC> product_, maxExp_, recip_;
    GlobalTensor<bfloat16_t> x_, g_;
    GlobalTensor<uint8_t> y_, scale_;
    uint32_t capacity_;
};
} // namespace SigmoidMulMx

extern "C" __global__ __aicore__ void sigmoid_mul_dynamic_mx_quant(
    GM_ADDR x, GM_ADDR gate, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SigmoidMulMxTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(SigmoidMulMxTilingData, tile, tiling);
    const auto overflow = AscendC::GetCtrlSpr<60, 60>();
    AscendC::SetCtrlSpr<60, 60>(0);
    if (TILING_KEY_IS(0)) {
        SigmoidMulMx::Kernel<float, false> op;
        op.Run(x, gate, y, mxscale, tile);
    } else if (TILING_KEY_IS(1)) {
        SigmoidMulMx::Kernel<bfloat16_t, false> op;
        op.Run(x, gate, y, mxscale, tile);
    } else if (TILING_KEY_IS(2)) {
        SigmoidMulMx::Kernel<float, true> op;
        op.Run(x, gate, y, mxscale, tile);
    } else if (TILING_KEY_IS(3)) {
        SigmoidMulMx::Kernel<bfloat16_t, true> op;
        op.Run(x, gate, y, mxscale, tile);
    }
    AscendC::SetCtrlSpr<60, 60>(overflow);
}
