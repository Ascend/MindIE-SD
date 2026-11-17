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

#include <algorithm>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "eagle_split_norm_rope_concat_shape.h"
#include "eagle_split_norm_rope_concat_tiling.h"

namespace optiling {
static ge::graphStatus TilingEagleSplitNormRopeConcat(gert::TilingContext *ctx) {
    if (ctx == nullptr || ctx->GetAttrs() == nullptr || ctx->GetPlatformInfo() == nullptr ||
        ctx->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < 6; ++i) {
        if (i == 2 || i == 3) {
            continue;
        }
        if (ctx->GetRequiredInputShape(i) == nullptr || ctx->GetRequiredInputDesc(i) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto *attrs = ctx->GetAttrs();
    const auto *norm = attrs->GetAttrPointer<int64_t>(0);
    const auto *rope = attrs->GetAttrPointer<int64_t>(1);
    const auto *rotary = attrs->GetAttrPointer<int64_t>(2);
    const auto *eps = attrs->GetAttrPointer<float>(3);
    const auto *bnsd = attrs->GetAttrPointer<bool>(4);
    const auto *heads = ctx->GetInputTensor(1);
    if (norm == nullptr || rotary == nullptr || rope == nullptr || eps == nullptr || bnsd == nullptr ||
        heads == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (*norm != 2 && *norm != 3) {
        OP_LOGE(ctx->GetNodeName(), "norm_type must be 2 (RMS_NORM) or 3 (RMS_NORM_AFFINE); got %ld.", *norm);
        return ge::GRAPH_FAILED;
    }
    const auto dtype = ctx->GetRequiredInputDesc(0)->GetDataType();
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
        ctx->GetRequiredInputDesc(1)->GetDataType() != ge::DT_INT64) {
        return ge::GRAPH_FAILED;
    }
    eagle_snrc::ShapeInfo info{};
    if (!eagle_snrc::ParseShape(ctx->GetRequiredInputShape(0)->GetStorageShape(),
            ctx->GetRequiredInputShape(1)->GetStorageShape(), heads->GetData<int64_t>(), *rotary, *rope, *eps, info) ||
        !eagle_snrc::CheckRope(ctx->GetRequiredInputShape(4)->GetStorageShape(),
            ctx->GetRequiredInputShape(5)->GetStorageShape(), info.rotaryDim, info)) {
        OP_LOGE(ctx->GetNodeName(), "Invalid packed QKV, head counts or RoPE shape/attributes.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 2; i < 6; ++i) {
        const auto *desc = i < 4 ? ctx->GetOptionalInputDesc(i) : ctx->GetRequiredInputDesc(i);
        const auto *shapePtr = i < 4 ? ctx->GetOptionalInputShape(i) : ctx->GetRequiredInputShape(i);
        if (i < 4 && (desc == nullptr || shapePtr == nullptr)) {
            if (*norm == 3) {
                OP_LOGE(ctx->GetNodeName(), "Both norm weights are required for norm_type=3.");
                return ge::GRAPH_FAILED;
            }
            continue;
        }
        if (desc->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }
        const auto &shape = shapePtr->GetStorageShape();
        if (i < 4 && (shape.GetDimNum() != 1 || shape.GetDim(0) != info.dim)) {
            return ge::GRAPH_FAILED;
        }
    }
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    // Batch 64 heads for D=128. The BNSD DMA destination stride is uint32 bytes.
    constexpr uint32_t headTile = 64;
    // Include two 256-byte gaps in the FP32 work/parameter buffers for bank separation.
    const uint64_t head128Ub = 20 * headTile * 128 + 28 * 128 + 40 * headTile + 2 * 256;
    // INTERLEAVE reuses a 16-head byte-offset table and a 128-element FP32 sign vector.
    const uint64_t requiredUb = head128Ub + (*rope == 0 ? (16 * 128 + 128) * sizeof(uint32_t) : 0) +
        (info.ropeWidth != info.rotaryDim ? 128 * sizeof(uint32_t) : 0);
    constexpr uint64_t reservedUb = 8 * 1024 + 256;
    const uint64_t physicalUb = std::min<uint64_t>(ubBytes, 192 * 1024);
    const uint64_t availableUb = physicalUb > reservedUb ? physicalUb - reservedUb : 0;
    if (*bnsd && info.seq - 1 > std::numeric_limits<uint32_t>::max() / (128 * sizeof(uint16_t))) {
        OP_LOGE(ctx->GetNodeName(), "BNSD output DMA destination stride exceeds uint32 range.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0 || requiredUb > availableUb) {
        OP_LOGE(ctx->GetNodeName(), "Insufficient AIV cores or UB for the D=128 batched kernel.");
        return ge::GRAPH_FAILED;
    }
    EagleSplitNormRopeConcatTilingData data;
    data.set_batch(info.batch);
    data.set_seq(info.seq);
    data.set_qHeads(info.heads[0]);
    data.set_kHeads(info.heads[1]);
    data.set_vHeads(info.heads[2]);
    data.set_dim(info.dim);
    data.set_rotaryDim(info.rotaryDim);
    data.set_ropeWidth(info.ropeWidth);
    data.set_ropeType(*rope);
    data.set_normType(*norm);
    data.set_ropeBatch(info.ropeBatch);
    data.set_outputBnsd(*bnsd ? 1 : 0);
    data.set_eps(*eps);
    data.set_headTile(headTile);
    const int64_t tasks = info.batch * info.seq * ((info.heads[0] + headTile - 1) / headTile);
    ctx->SetBlockDim(std::min<int64_t>(cores, tasks));
    ctx->SetTilingKey(dtype == ge::DT_FLOAT16 ? 11 : 12);
    auto *workspace = ctx->GetWorkspaceSizes(1);
    if (workspace == nullptr || ctx->GetRawTilingData()->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(EagleSplitNormRopeConcat).Tiling(TilingEagleSplitNormRopeConcat).TilingInputsDataDependency({1});
} // namespace optiling
