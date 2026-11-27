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

#include "eagle_hadamard_quant_tiling.h"
#include "eagle_hadamard_quant_params.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "../op_kernel/arch35/rotation_layout.h"

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext *ctx) {
    if (ctx == nullptr || ctx->GetAttrs() == nullptr || ctx->GetPlatformInfo() == nullptr ||
        ctx->GetRequiredInputShape(0) == nullptr || ctx->GetRequiredInputShape(1) == nullptr ||
        ctx->GetInputDesc(0) == nullptr || ctx->GetInputDesc(1) == nullptr || ctx->GetOutputDesc(0) == nullptr ||
        ctx->GetOutputDesc(1) == nullptr || ctx->GetOutputShape(0) == nullptr || ctx->GetOutputShape(1) == nullptr ||
        ctx->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *blockSize = ctx->GetAttrs()->GetInt(0);
    const auto *dstType = ctx->GetAttrs()->GetStr(1);
    const auto *isNz = ctx->GetAttrs()->GetBool(2);
    if (blockSize == nullptr || dstType == nullptr || isNz == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &x = ctx->GetRequiredInputShape(0)->GetStorageShape();
    const auto &rotation = ctx->GetRequiredInputShape(1)->GetStorageShape();
    const auto p = EagleHadamardQuant::CheckShape(x, rotation, *blockSize, dstType, *isNz);
    if (p.error != nullptr) {
        OP_LOGE(ctx->GetNodeName(), "%s", p.error);
        return ge::GRAPH_FAILED;
    }
    const auto yType = p.format == 0 ? ge::DT_UINT8 : ge::DT_FLOAT8_E4M3FN;
    const auto &yShape = ctx->GetOutputShape(0)->GetStorageShape();
    const auto &scaleShape = ctx->GetOutputShape(1)->GetStorageShape();
    if (ctx->GetInputDesc(0)->GetDataType() != ge::DT_BF16 || ctx->GetInputDesc(1)->GetDataType() != ge::DT_BF16 ||
        ctx->GetOutputDesc(0)->GetDataType() != yType || ctx->GetOutputDesc(1)->GetDataType() != ge::DT_FLOAT8_E8M0 ||
        yShape.GetDimNum() != 2 || yShape.GetDim(0) != x.GetDim(0) || yShape.GetDim(1) != p.outputColumns ||
        scaleShape.GetDimNum() != 3 || scaleShape.GetDim(0) != x.GetDim(0) || scaleShape.GetDim(1) != p.scalePairs ||
        scaleShape.GetDim(2) != 2) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    const auto cores = platform.GetCoreNumAic();
    if (cores == 0) {
        return ge::GRAPH_FAILED;
    }
    EagleHadamardQuantTilingData data;
    data.set_m(x.GetDim(0));
    data.set_n(x.GetDim(1));
    data.set_k(p.k);
    data.set_blocks(p.blocks);
    // Explicit ND carrier: [B,columnTile,rowTile,row,column].
    data.set_layout(*isNz ? RotateQuantAptOpt::ROTATION_L1_NZ : RotateQuantAptOpt::ROTATION_ND);
    // Group independent per-block rotations into 128-column tiles when the shape permits.
    data.set_diagonal(p.blocks > 1 && x.GetDim(1) % 128 == 0 && (p.k == 32 || p.k == 64));
    ctx->SetBlockDim(cores);
    ctx->SetTilingKey(p.format == 0 ? 1 : 2);
    auto *raw = ctx->GetRawTilingData();
    if (raw->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    auto *workspace = ctx->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
// This operator derives its tiling entirely from the runtime context.
struct EagleHadamardQuantCompileInfo {};

static ge::graphStatus TilingPrepare(gert::TilingParseContext *) { return ge::GRAPH_SUCCESS; }

IMPL_OP_OPTILING(EagleHadamardQuant).Tiling(Tiling).TilingParse<EagleHadamardQuantCompileInfo>(TilingPrepare);
} // namespace optiling
