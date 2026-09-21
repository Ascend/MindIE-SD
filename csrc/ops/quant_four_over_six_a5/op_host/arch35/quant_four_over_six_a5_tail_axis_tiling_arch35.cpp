/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file dynamic_mx_quant_tail_axis_tiling_arch35.cpp
 * \brief This tiling template only supports scenarios where the quantization axis is the last axis and the blockSize
 * is 32.
 */

#include "quant_four_over_six_a5_tiling_arch35.h"
#include <cmath>
#include <cstdlib>
#include "platform/platform_info.h"

template <typename T> inline T CeilDivQ(T a, T b) { return (a + b - 1) / b; }
inline bool IsFloatEqualQ(float a, float b) { return std::fabs(a - b) < 1e-6f; }
namespace optiling {
using namespace ge;
constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_TEN = 10;
constexpr int64_t SLICE_SIZE_256 = 256;
constexpr int64_t SLICE_SIZE_512 = 512;
constexpr int64_t DIGIT_EIGHT = 8;

constexpr int64_t BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE = 4;
constexpr int64_t BYTES_OF_OUTPUT_FP8_DB_BUFFER_TYPE = 2;
constexpr int64_t BYTES_OF_OUTPUT_FP4_DB_BUFFER_TYPE = 1;
constexpr int64_t BYTES_OF_MAX_VALUE_TYPE = 2;
constexpr int64_t BYTES_OF_SCALE_TYPE = 1;
constexpr int64_t BYTES_OF_INVERSE_SCALE_TYPE = 2;
constexpr int64_t DB_BUFFER = 2;
constexpr float LOAD_BALANCE_THRESHOLD = 0.75; // 负载均衡的阈值

ge::graphStatus QuantFourOverSixA5TailAxisTiling::SetTilingDataForTailAxis()
{
    tilingData_.tilingKey = tilingParam_.tilingKey;
    tilingData_.ubSize = tilingParam_.ubSize;
    tilingData_.roundMode = tilingParam_.roundMode;
    tilingData_.blockSize = tilingParam_.blockSize;
    tilingData_.totalCoreNum = tilingParam_.totalCoreNum;
    tilingData_.usedCoreNum = tilingParam_.usedCoreNum;
    tilingData_.rowTileNum = tilingParam_.rowTileNum;
    tilingData_.colTileNum = tilingParam_.colTileNum;
    tilingData_.rowNum = tilingParam_.rowNum;
    tilingData_.colNum = tilingParam_.colNum;
    tilingData_.colNormalBlockNum = tilingParam_.colNormalBlockNum;
    tilingData_.colTailLen = tilingParam_.colTailLen;
    tilingData_.rowNormalBlockNum = tilingParam_.rowNormalBlockNum;
    tilingData_.rowTailLen = tilingParam_.rowTailLen;
    tilingData_.maxUbBlockNum = tilingParam_.maxUbBlockNum;
    tilingData_.dstTypeMax = tilingParam_.dstTypeMax;
    tilingData_.invDstTypeMax = tilingParam_.invDstTypeMax;

    uint64_t tilingDataSize = sizeof(tilingData_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    auto rawTilingData = context_->GetRawTilingData();
    errno_t ret = memcpy_s(
        rawTilingData->GetData(), rawTilingData->GetCapacity(), reinterpret_cast<void*>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret = %d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);
    return ge::GRAPH_SUCCESS;
}

void QuantFourOverSixA5TailAxisTiling::PrintTilingDataForTailAxis()
{
    OP_LOGI(
        context_->GetNodeName(),
        "tilingData is tilingKey:%ld, ubSize:%ld, roundMode:%ld, blockSize:%ld, \
        totalCoreNum:%ld, usedCoreNum:%ld, rowTileNum:%ld, colTileNum:%ld, \
        rowNum:%ld, colNum:%ld, colNormalBlockNum:%ld, colTailLen:%ld, \
        rowNormalBlockNum:%ld, rowTailLen:%ld, maxUbBlockNum:%ld, dstTypeMax:%f, invDstTypeMax:%f.",
        tilingData_.tilingKey, tilingData_.ubSize, tilingData_.roundMode, tilingData_.blockSize,
        tilingData_.totalCoreNum, tilingData_.usedCoreNum, tilingData_.rowTileNum, tilingData_.colTileNum,
        tilingData_.rowNum, tilingData_.colNum, tilingData_.colNormalBlockNum, tilingData_.colTailLen,
        tilingData_.rowNormalBlockNum, tilingData_.rowTailLen, tilingData_.maxUbBlockNum, tilingData_.dstTypeMax,
        tilingData_.invDstTypeMax);
}

std::set<int64_t> QuantFourOverSixA5TailAxisTiling::FindSplitCombo(int64_t usedCoreNum) const
{
    std::set<int64_t> result;
    int64_t upbound = static_cast<int64_t>(std::ceil(std::sqrt(usedCoreNum) + 1));

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = usedCoreNum / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

ge::graphStatus QuantFourOverSixA5TailAxisTiling::AutoTiling()
{
    // 计算可用核数
    tilingParam_.usedCoreNum =
        std::min(tilingParam_.totalCoreNum, tilingParam_.rowBlockLoopNum * tilingParam_.colBlockLoopNum);
    tilingParam_.usedCoreNum = tilingParam_.usedCoreNum == 0 ? 1 : tilingParam_.usedCoreNum;

    // 仿真单核多tile场景: FORCE_CORE_NUM=1 强制单核执行, 跳过 AutoTiling
    const char* forceCoreStr = getenv("FORCE_CORE_NUM");
    if (forceCoreStr) {
        int64_t forceCoreNum = std::atol(forceCoreStr);
        if (forceCoreNum > 0) {
            tilingParam_.usedCoreNum = static_cast<uint16_t>(forceCoreNum);
        }
    }

    // 查找切分的组合
    std::set<int64_t> cutSet = FindSplitCombo(tilingParam_.usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    // 行方向切分，枚举m的取值
    for (int64_t m : cutSet) {
        if (m > tilingParam_.rowBlockLoopNum) {
            continue;
        }
        int64_t n = tilingParam_.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParam_.colBlockLoopNum) {
            continue;
        }

        int64_t rowNormalBlockNum = CeilDivQ(tilingParam_.rowBlockLoopNum, m);
        int64_t colNormalBlockNum = CeilDivQ(tilingParam_.colBlockLoopNum, n);
        int64_t delta = rowNormalBlockNum * colNormalBlockNum;
        if (m * n == static_cast<int64_t>(tilingParam_.usedCoreNum)) {
            if (tilingParam_.rowBlockLoopNum % m == 0 && tilingParam_.colBlockLoopNum % n == 0) {
                delta = 0;
            } else if (tilingParam_.rowBlockLoopNum % m == 0) {
                delta = delta - rowNormalBlockNum * (tilingParam_.colBlockLoopNum % colNormalBlockNum);
            } else if (tilingParam_.colBlockLoopNum % n == 0) {
                delta = delta - (tilingParam_.rowBlockLoopNum % rowNormalBlockNum) * colNormalBlockNum;
            } else {
                delta = delta - (tilingParam_.rowBlockLoopNum % rowNormalBlockNum) *
                                    (tilingParam_.colBlockLoopNum % colNormalBlockNum);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }
    // 检查数组是否为空
    if (allTiling.empty()) {
        OP_LOGE(context_->GetNodeName(), "Failed to find valid tiling split combination.");
        return ge::GRAPH_FAILED;
    }
    // 排序以选择最合适的切分
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int FirstIndex = 1; // 优先切 M
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[FirstIndex]) < std::make_pair(b[DeltaIndex], b[FirstIndex]);
    });

    tilingParam_.rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParam_.colTileNum = static_cast<uint16_t>(allTiling[0][1]);

    return ge::GRAPH_SUCCESS;
}

void QuantFourOverSixA5TailAxisTiling::CalcTilingKeyForTail()
{
    // blockSize == 32 的尾轴专用模板只区分 scaleAlg。
    // 10/11/12 分别对应 scaleAlg 0/1/2，dtype 由 kernel binary 的模板参数决定。
    tilingParam_.tilingKey = DIGIT_TEN + tilingParam_.scaleAlg;
}

void QuantFourOverSixA5TailAxisTiling::CalcAxisSize(const gert::Shape& xShape)
{
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParam_.rowNum *= xShape.GetDim(i);
    }
    tilingParam_.colNum = xShape.GetDim(xShape.GetDimNum() - 1);
}

ge::graphStatus QuantFourOverSixA5TailAxisTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "QuantFourOverSixA5TailAxisTiling DoTiling entering.");

    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    CalcAxisSize(xShape);

    int64_t sliceSize = SLICE_SIZE_512;
    if (tilingParam_.rowNum * CeilDivQ(tilingParam_.colNum, sliceSize) <= tilingParam_.totalCoreNum * LOAD_BALANCE_THRESHOLD) {
        sliceSize = SLICE_SIZE_256;
    }
    tilingParam_.rowBlockLoopNum = CeilDivQ(tilingParam_.rowNum, DIGIT_ONE);
    tilingParam_.colBlockLoopNum = CeilDivQ(tilingParam_.colNum, sliceSize);

    OP_CHECK_IF(
        AutoTiling() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "The auto tiling failed."), return ge::GRAPH_FAILED);

    tilingParam_.rowNormalBlockNum = CeilDivQ(tilingParam_.rowBlockLoopNum, tilingParam_.rowTileNum);
    tilingParam_.colNormalBlockNum = CeilDivQ(tilingParam_.colBlockLoopNum, tilingParam_.colTileNum);
    tilingParam_.rowTileNum = CeilDivQ(tilingParam_.rowBlockLoopNum, tilingParam_.rowNormalBlockNum);
    tilingParam_.colTileNum = CeilDivQ(tilingParam_.colBlockLoopNum, tilingParam_.colNormalBlockNum);
    tilingParam_.colNormalBlockNum = tilingParam_.colNormalBlockNum * sliceSize / SLICE_SIZE_256;
    tilingParam_.usedCoreNum = tilingParam_.rowTileNum * tilingParam_.colTileNum;
    tilingParam_.rowTailLen =
        tilingParam_.rowNum - (tilingParam_.rowNormalBlockNum * DIGIT_ONE * (tilingParam_.rowTileNum - DIGIT_ONE));
    tilingParam_.colTailLen =
        tilingParam_.colNum - (tilingParam_.colNormalBlockNum * SLICE_SIZE_256 * (tilingParam_.colTileNum - DIGIT_ONE));

    // 计算UB可以放下的block（1×256）数量
    // maxUbBlockNum * SLICE_SIZE_256 * (BYTES_OF_INPUT_TYPE * DB_BUFFER + BYTES_OF_OUTPUT_TYPE * DB_BUFFER) +
    // maxUbBlockNum * DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + BYTES_OF_SCALE_TYPE * DB_BUFFER +
    // BYTES_OF_INVERSE_SCALE_TYPE)
    // <= ubSize
    if (tilingParam_.dstType == ge::DT_FLOAT4_E2M1 || tilingParam_.dstType == ge::DT_FLOAT4_E1M2) { // Y FP4
        bool isD4adp = IsFloatEqualQ(tilingParam_.dstTypeMax, 4.0f);
        int64_t dbIn   = BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE;   // bf16×DB
        int64_t dbOut  = BYTES_OF_OUTPUT_FP4_DB_BUFFER_TYPE;     // FP4/2×DB
        int64_t dbScale = BYTES_OF_SCALE_TYPE * DB_BUFFER;        // E8M0×DB
        int64_t per256Overhead =
            SLICE_SIZE_256 * (dbIn + dbOut) +
            DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + dbScale + BYTES_OF_INVERSE_SCALE_TYPE);
        if (isD4adp) {
            // D4 extra per block: scale46(2×2) + recip extra(1×2) + seTmp(2×2) + y46(2×32) = 74 B/block
            per256Overhead += DIGIT_EIGHT * (2*2 + 1*2 + 2*2 + 2*32);
        }
        tilingParam_.maxUbBlockNum = tilingParam_.ubSize / per256Overhead;
        if (isD4adp) {
            OP_LOGI(context_->GetNodeName(),
                "[D4adp] dst_type_max=4.0 adaptive path ENABLED | ubSize=%ld B | per256Overhead=%ld B | "
                "maxUbBlockNum_1x256=%ld | maxUbBlockNum_1x32=%ld | "
                "colNormalBlockNum=%ld colTailLen=%ld | rowNormalBlockNum=%ld rowTailLen=%ld | "
                "usedCoreNum=%ld totalCoreNum=%ld",
                tilingParam_.ubSize, per256Overhead,
                tilingParam_.maxUbBlockNum, tilingParam_.maxUbBlockNum * DIGIT_EIGHT,
                tilingParam_.colNormalBlockNum, tilingParam_.colTailLen,
                tilingParam_.rowNormalBlockNum, tilingParam_.rowTailLen,
                tilingParam_.usedCoreNum, tilingParam_.totalCoreNum);
        }
    } else if (tilingParam_.dstType == ge::DT_FLOAT8_E5M2 || tilingParam_.dstType == ge::DT_FLOAT8_E4M3FN) { // Y FP8
        tilingParam_.maxUbBlockNum =
            tilingParam_.ubSize /
            (SLICE_SIZE_256 * (BYTES_OF_INPUT_UINT16_DB_BUFFER_TYPE + BYTES_OF_OUTPUT_FP8_DB_BUFFER_TYPE) +
             DIGIT_EIGHT * (BYTES_OF_MAX_VALUE_TYPE + BYTES_OF_SCALE_TYPE * DB_BUFFER + BYTES_OF_INVERSE_SCALE_TYPE));
    }
    tilingParam_.maxUbBlockNum *= DIGIT_EIGHT; // 转换成UB可以放下的block（1×32）数量

    CalcTilingKeyForTail();
    OP_CHECK_IF(
        SetTilingDataForTailAxis() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_, "DynamicMxQuantTailAxisSetTilingData set tiling data failed."), return ge::GRAPH_FAILED);

    context_->SetBlockDim(tilingData_.usedCoreNum);
    context_->SetTilingKey(tilingData_.tilingKey);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = tilingParam_.workspaceSize;

    PrintTilingDataForTailAxis();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
