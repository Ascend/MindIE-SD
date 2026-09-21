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
 * \file quant_four_over_six_a5_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_QUANT_FOUR_OVER_SIX_A5_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_QUANT_FOUR_OVER_SIX_A5_H
#include <cstdint>
#include <string>
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/quant_four_over_six_a5_tilingdata.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct QuantFourOverSixA5CompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct QuantFourOverSixA5TilingParam {
    int64_t totalCoreNum{0};
    int64_t ubSize{0};
    uint32_t vfLen{0};
    uint32_t workspaceSize{0};
    int64_t axis{0};
    int64_t roundMode{0};
    int64_t dstType{0};
    int64_t blockSize{0};
    int64_t scaleAlg{0};
    bool isTailAxis{false};
    bool isPad{false};
    int64_t tailBlockSize{0};
    int64_t blockSizeNumInAxis{0};
    int64_t preAxisSize{1};
    int64_t axisSize{1};
    int64_t quantAxisSize{1};
    int64_t postAxisSize{1};
    int64_t mxScaleSize{0};
    int64_t ubDim{0};
    int64_t ubFactor{0};
    int64_t uo{1};
    int64_t usedCoreNum{0};
    int64_t tailUbFactor{0};
    int64_t blockFactor{0};
    int64_t tailBlockFactor{0};
    int64_t tilingKey{0};
    // not tail axis tiling data
    int64_t mAlignSize{1};
    int64_t nAlignSize{1};
    int64_t nAlignNum{1};
    int64_t mAlignBlockCount{1};
    int64_t nAlignBlockCount{1};
    int64_t mAlignGroupCount{1};
    int64_t quantAxisIsOdd{0};
    int64_t totalGroupNum{0};
    int64_t groupPerCore{0};
    int64_t groupPerTail{0};
    int64_t groupPerUb{0};
    int64_t totalBlockNum{0};
    int64_t blockNumPerTask{0};
    int64_t totalTaskNum{0};
    int64_t loopNumPerHeadCore{0};
    int64_t loopNumPerTailCore{0};
    int64_t calcMode{0};
    int64_t ubRowLen{0};
    int64_t ubRowLenTail{0};
    int64_t ubRowCount{0};
    int64_t ubRowCountTail{0};
    uint32_t subNumForScale{0};
    uint32_t subNumForFP16Scale{0};
    int64_t blockCountPerBatch{0};
    int64_t scaleRowCountPerBatch{0};
    bool needPadPostAxis{false};
    bool isOptimize{false};
    // tail axis tiling data
    bool isBlockSize32{false};
    int64_t colTileNum{0};
    int64_t rowTileNum{0};
    int64_t rowNum{1};
    int64_t colNum{1};
    int64_t colNormalBlockNum{0};
    int64_t colTailLen{0};
    int64_t rowNormalBlockNum{0};
    int64_t rowTailLen{0};
    int64_t maxUbBlockNum{0};
    int64_t rowBlockLoopNum{0};
    int64_t colBlockLoopNum{0};

    float dstTypeMax{0.0};
    float invDstTypeMax{0.0};
};

enum class RoundModeList
{
    MODE_ROUND = 0,
    MODE_FLOOR = 1,
    MODE_CEIL = 2,
    MODE_TRUNC = 3,
    MODE_RINT = 4,
    MODE_HYBRID = 5,
    MODE_UNDEFINED = -1,
};

// T9b: 仅保留 46 自适应路径。非尾轴 optimize 模板（QuantFourOverSixA5OptimzieTiling）
// 与 main 模板逻辑已随清理删除，仅保留尾轴 block32 模板。
class QuantFourOverSixA5TailAxisTiling {
public:
    explicit QuantFourOverSixA5TailAxisTiling(gert::TilingContext* context, const QuantFourOverSixA5TilingParam& tilingParam)
        : context_(context), tilingParam_(tilingParam), tilingData_{}
    {}
    ~QuantFourOverSixA5TailAxisTiling()
    {}
    ge::graphStatus DoTiling();

private:
    ge::graphStatus SetTilingParams();
    void CalcTilingKeyForTail();
    void CalcAxisSize(const gert::Shape& xShape);
    ge::graphStatus AutoTiling();
    std::set<int64_t> FindSplitCombo(int64_t usedCoreNum) const;
    ge::graphStatus SetTilingDataForTailAxis();
    void PrintTilingDataForTailAxis();

private:
    gert::TilingContext* context_ = nullptr;
    QuantFourOverSixA5TailAxisTilingData tilingData_{};
    QuantFourOverSixA5TilingParam tilingParam_;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_QUANT_FOUR_OVER_SIX_A5_H
