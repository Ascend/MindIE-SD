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


/* !
 * \file quant_four_over_six_a5_tilingdata.h
 * \brief
 */

#ifndef OPS_NN_QUANT_FOUR_OVER_SIX_A5_H
#define OPS_NN_QUANT_FOUR_OVER_SIX_A5_H
#include <cstdint>

// T9b: 仅保留 46 自适应路径（dst_type_max=4, 尾轴, blocksize=32, scale_alg=2,
// bf16 -> fp4x2_e2m1）。main 模板（QuantFourOverSixA5TilingData）与优化模板
// （QuantFourOverSixA54OptimizeTilingData）随非尾轴路径一并删除。

struct QuantFourOverSixA5TailAxisTilingData {
    int64_t tilingKey{0};
    int64_t ubSize{0};
    int64_t roundMode{0};
    int64_t blockSize{0};
    int64_t totalCoreNum{0};
    int64_t usedCoreNum{0};
    int64_t rowTileNum{0};        // row 方向上的切核数
    int64_t colTileNum{0};        // col 方向上的切核数
    int64_t rowNum{0};            // 合轴之后 -2 轴大小
    int64_t colNum{0};            // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum{0}; // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen{0};        // 列方向尾块长度
    int64_t rowNormalBlockNum{0}; // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen{0};        // 行方向尾块长度
    int64_t maxUbBlockNum{0};     // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax{0.0f};
    float invDstTypeMax{0.0f};
};

#endif // OPS_NN_QUANT_FOUR_OVER_SIX_A5_WITH_DUAL_AXIS_H
