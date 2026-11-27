/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rotate_quant_tiling_data.h
 * \brief RotateQuant APT算子arch35 tiling数据结构定义
 */

#pragma once

#include "kernel_tiling/kernel_tiling.h"
#include "rotation_layout.h"

namespace RotateQuantAptOpt {
constexpr uint64_t STRUCT_ALIGNAS = 8;
#pragma pack(push, 8)
struct alignas(STRUCT_ALIGNAS) RotateQuantAptTilingData {
    // 矩阵维度参数
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t K = 0;
    uint32_t B = 0;
    uint32_t mL1 = 0;
    uint32_t nL1 = 0;
    uint32_t kL1 = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t baseK = 0;
    uint32_t nKRatio = 0; // N/K的比例（numBlocks）
    uint32_t tailML1 = 0; // 最后一个tile的blkM（余数或mL1）

    // dstTypeMax相关参数
    float dstTypeMax = 0.0;
    float invDstTypeMax = 0.0; // dstTypeMax的倒数

    // 属性参数
    int32_t axis = 0; // reduce轴
    int32_t roundMode = 0; // round模式（4:rint, 0:round, 1:floor）
    int32_t scaleAlg = 0; // scale算法
    uint8_t trans = 0; // 是否转置（0:false, 1:true）
    uint8_t hasAlpha = 0; // alpha是否有值（0:false, 1:true）
    uint8_t rotationIsNz = 0; // Occupies existing padding before groupMode.
    uint32_t groupMode = 0;
    uint32_t groupNum = 0;
    int32_t groupListType = 0;
    // 核配参数（tiling层决策，kernel/block层按此运行时分发）
    uint32_t aivNumRatio = 2; // AIC:AIV核配比（3510=2即1:2, DAV_9201/DAV_9202=1即1:1），
        // 驱动AIC↔AIV同步flag路由与fixpipe subBlockId路由
};
#pragma pack(pop)
} // namespace RotateQuantAptOpt
