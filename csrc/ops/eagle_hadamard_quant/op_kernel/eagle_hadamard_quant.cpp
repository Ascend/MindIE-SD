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
#include "arch35/rotate_quant_tiling_data.h"
#include "arch35/kernel_rotate_quant.h"

template <typename Output, bool BlockDiagonal>
__aicore__ inline void RunEagle(GM_ADDR x, GM_ADDR rotation, GM_ADDR y, GM_ADDR scale, uint32_t m, uint32_t n,
    uint32_t k, uint32_t blocks, uint8_t layout, uint32_t rotationK) {
    AscendC::InitSocState();
    RotateQuantAptOpt::RotateQuantAptTilingData t{};
    t.M = m;
    t.N = n;
    t.K = t.nL1 = t.kL1 = t.baseN = k;
    t.B = blocks;
    t.nKRatio = n / k;
    // Each epilogue consumes at most 16384 values from the Fixpipe UB tile.
    t.mL1 = t.baseM = k == 256 ? 64 : 128;
    t.baseK = k < 16384 / k ? k : 16384 / k;
    const uint64_t rows = blocks == 1 ? static_cast<uint64_t>(m) * t.nKRatio : m;
    t.tailML1 = (rows - 1) % t.mL1 + 1;
    t.axis = -1;
    t.roundMode = 4; // rint
    t.rotationIsNz = layout;
    Blaze::Gemm::Kernel::RotateQuantKernel<bfloat16_t, Output, fp8_e8m0_t, false, BlockDiagonal>{}.Run(
        x, rotation, nullptr, y, scale, t, nullptr, rotationK);
}

template <typename Output>
__aicore__ inline void RunHadamard(
    GM_ADDR x, GM_ADDR signH, GM_ADDR y, GM_ADDR scale, const EagleHadamardQuantTilingData &data) {
    if (data.diagonal != 0) {
        constexpr uint32_t width = 128;
        RunEagle<Output, true>(x, signH, y, scale, data.m, data.n, width, data.n / width, data.layout, data.k);
    } else {
        RunEagle<Output, false>(x, signH, y, scale, data.m, data.n, data.k, data.blocks, data.layout, data.k);
    }
}

extern "C" __global__ __aicore__ void eagle_hadamard_quant(
    GM_ADDR x, GM_ADDR sign_h, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA(data, tiling);
    if (data.m == 0) {
        return;
    }
    if (TILING_KEY_IS(1)) {
        RunHadamard<fp4x2_e2m1_t>(x, sign_h, y, scale, data);
    } else if (TILING_KEY_IS(2)) {
        RunHadamard<fp8_e4m3fn_t>(x, sign_h, y, scale, data);
    }
}
