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
#include "eagle_split_norm_rope_concat_platform.h"
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
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    const bool is950 = eagle_snrc::IsAscend950(platform, 0);
    eagle_snrc::ShapeInfo info{};
    if (!eagle_snrc::ParseShape(ctx->GetRequiredInputShape(0)->GetStorageShape(),
            ctx->GetRequiredInputShape(1)->GetStorageShape(), heads->GetData<int64_t>(), *rotary, *rope, *eps, info,
            is950) ||
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
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint32_t headTile = is950 ? eagle_snrc::MAX_HEADS_950 : eagle_snrc::GetHeadTile910(info.heads[0]);
    uint32_t outputQueues910 = is950 ? 0 : 1;
    auto ub = eagle_snrc::GetUbBudget(is950, ubBytes, *rope, info.ropeWidth != info.rotaryDim, headTile);
    if (!is950) {
        const auto dualOutput =
            eagle_snrc::GetUbBudget(false, ubBytes, *rope, info.ropeWidth != info.rotaryDim, headTile, 2);
        if (dualOutput.required <= dualOutput.available) {
            ub = dualOutput;
            outputQueues910 = 2;
        }
    }
    eagle_snrc::TilePlan950 plan{};
    if (is950) {
        if (!eagle_snrc::Build950Plan(info.heads[0], info.heads[1], info.heads[2], info.ropeWidth, ubBytes, plan)) {
            OP_LOGE(ctx->GetNodeName(),
                "Ascend950 requires each Q/K/V head count in [1,128] and UB for C=128..98 with two cache slots.");
            return ge::GRAPH_FAILED;
        }
        ub.required = plan.required;
    }
    // The 910B batched BNSD DMA uses a uint32 destination byte gap.
    if (!is950 && *bnsd &&
        static_cast<uint64_t>(info.seq - 1) > std::numeric_limits<uint32_t>::max() / (128 * sizeof(uint16_t))) {
        OP_LOGE(ctx->GetNodeName(), "BNSD output DMA destination stride exceeds uint32 range.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0 || ub.required > ub.available) {
        OP_LOGE(ctx->GetNodeName(),
            "Insufficient AIV cores or UB: is950=%u, cores=%u, query=%llu, required=%llu, available=%llu.",
            static_cast<uint32_t>(is950), cores, static_cast<unsigned long long>(ubBytes),
            static_cast<unsigned long long>(ub.required), static_cast<unsigned long long>(ub.available));
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
    data.set_tokenTileQ(plan.qTokens);
    data.set_tokenTileK(plan.kTokens);
    data.set_tokenTileV(plan.vTokens);
    data.set_ropeCacheTokens(plan.cacheTokens);
    data.set_targetRows(plan.targetRows);
    data.set_outputQueues910(outputQueues910);
    const uint32_t tokenTile910 = is950 ? 0 : outputQueues910;
    data.set_tokenTile910(tokenTile910);
    const int64_t tasks = is950
        ? eagle_snrc::Get950TaskCount(info.batch, info.seq)
        : eagle_snrc::GetHeadTaskCount(info.batch, info.seq, info.heads[0], headTile, tokenTile910);
    const uint32_t blockDim = eagle_snrc::GetBlockDim(cores, tasks);
    OP_LOGD(ctx->GetNodeName(),
        "is950=%u, UB query=%llu, required=%llu, available=%llu, AIV=%u, tasks=%lld, blockDim=%u.",
        static_cast<uint32_t>(is950), static_cast<unsigned long long>(ubBytes),
        static_cast<unsigned long long>(ub.required), static_cast<unsigned long long>(ub.available), cores,
        static_cast<long long>(tasks), blockDim);
    ctx->SetBlockDim(blockDim);
    ctx->SetTilingKey(dtype == ge::DT_FLOAT16 ? 11 : 12);
    auto *workspace = ctx->GetWorkspaceSizes(1);
    if (workspace == nullptr || ctx->GetRawTilingData()->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
    if (!is950) {
        OP_LOGD(ctx->GetNodeName(),
            "EAGLE910_TILE heads=%u tokenTile=%u outputQueues=%u buffersPerQueue=%u tokensPerBuffer=%u "
            "bnsdBlockBytes=%u UB=%llu/%llu.",
            headTile, tokenTile910, outputQueues910, 2 / tokenTile910, tokenTile910,
            tokenTile910 * 128 * static_cast<uint32_t>(sizeof(uint16_t)), static_cast<unsigned long long>(ub.required),
            static_cast<unsigned long long>(ub.available));
    }
    if (is950) {
        OP_LOGD(ctx->GetNodeName(), "EAGLE950_TILE C=%u Tq=%u Tk=%u Tv=%u cacheTokens=%u cacheSlots=2 UB=%llu/%llu.",
            plan.targetRows, plan.qTokens, plan.kTokens, plan.vTokens, plan.cacheTokens,
            static_cast<unsigned long long>(plan.required), static_cast<unsigned long long>(plan.available));
        // A successful Host tiling record; this is not device-object execution evidence.
        OP_LOGD(ctx->GetNodeName(),
            "EAGLE950_HOST_ROUTE b=%lld s=%lld q=%lld k=%lld v=%lld R=%u W=%u rope=%lld norm=%lld "
            "bnsd=%u dtype=%s route=Ascend950 END_EAGLE_ROUTE",
            static_cast<long long>(info.batch), static_cast<long long>(info.seq), static_cast<long long>(info.heads[0]),
            static_cast<long long>(info.heads[1]), static_cast<long long>(info.heads[2]),
            static_cast<unsigned>(info.rotaryDim), static_cast<unsigned>(info.ropeWidth), static_cast<long long>(*rope),
            static_cast<long long>(*norm), static_cast<unsigned>(*bnsd), dtype == ge::DT_BF16 ? "BF16" : "FP16");
    }
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(EagleSplitNormRopeConcat).Tiling(TilingEagleSplitNormRopeConcat).TilingInputsDataDependency({1});
} // namespace optiling
