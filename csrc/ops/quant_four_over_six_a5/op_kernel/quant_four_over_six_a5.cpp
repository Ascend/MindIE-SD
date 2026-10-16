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
 * \file quant_four_over_six_a5.cpp
 * \brief
 */

#include "arch35/quant_four_over_six_a5_tail_axis.h"
#include "arch35/quant_four_over_six_a5_tilingdata.h"

// T9b: 仅保留 46 自适应路径。
// 量化轴为尾轴且 BlockSize 为 32 时，仅由 scaleAlg 决定路径；
// 46 自适应仅 scale_alg=2（tilingKey=12），dtype 由 DTYPE_X/DTYPE_Y 编译宏确定
// （def 已裁剪为 bf16 -> fp4x2_e2m1 单组合）。
#define TILING_KEY_TAIL_AXIS_SCALE_ALG_TWO 12

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace QuantFourOverSixA5;

namespace {
template <typename TX, typename TY>
__aicore__ inline void RunTailAxisBlock32Path(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(QuantFourOverSixA5TailAxisTilingData, tilingData, tiling);
    // 46 自适应（bf16 -> fp4x2_e2m1, scale_alg=2, dst_type_max=4.0）
    QuantFourOverSixA5::QuantFourOverSixA5TailAxis<TX, TY, 2> op;
    op.Init(x, y, mxScale, &tilingData);
    op.Process();
}
} // namespace

extern "C" __global__ __aicore__ void quant_four_over_six_a5(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    // T10 修复: 补 REGISTER_TILING_DEFAULT——T9b 删除 main/optimize tiling 结构时连带删除了
    // default 注册, 导致 TBE 单算子编译报 "must provide default tiling struct";
    // 46-only 后唯一结构即 TailAxisTilingData, default 与 FOR_TILINGKEY 映射同一结构(无冲突)。
    REGISTER_TILING_DEFAULT(QuantFourOverSixA5TailAxisTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 10 && TILING_KEY_VAR <= 12", QuantFourOverSixA5TailAxisTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    // 46 自适应路径：host 仅发出 tilingKey=12（尾轴+blocksize=32+scale_alg=2+dst_type_max=4）
    // 具体 dtype 组合在编译期由 DTYPE_X/DTYPE_Y 实例化完成。
    if (TILING_KEY_IS(TILING_KEY_TAIL_AXIS_SCALE_ALG_TWO)) {
        RunTailAxisBlock32Path<DTYPE_X, DTYPE_Y>(x, y, mxScale, tiling);
    }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
