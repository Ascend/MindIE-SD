/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
 * \file eagle_quant_batch_matmul_swiglu_tiling_base.cc
 * \brief
 */

#include "eagle_quant_batch_matmul_swiglu_tiling_base.h"
#include "quant_batch_matmul_info_factory.h"
#include "qbmm_compat.h"
#include "../deps/matmul/common/op_host/op_tiling/debug_tiling.h"
#include "platform/platform_infos_def.h"

using Ops::EagleQbmm::TilingPrepareForOpCache;

namespace {

constexpr size_t LAST_SECOND_DIM_INDEX = 2;
constexpr size_t LAST_BATCH_DIM_INDEX = 3;
constexpr size_t LAST_FIRST_DIM_INDEX = 1;
constexpr size_t MIN_DIM_NUM_ND = 2;
constexpr size_t MAX_DIM_NUM_ND = 6;
constexpr size_t A4W4_X2_LEN = 2;
constexpr int32_t IDX_K_LOW = 2;
constexpr int32_t IDX_K_HIGH = 3;
constexpr int32_t IDX_N_LOW = 4;
constexpr int32_t IDX_N_HIGH = 5;
constexpr int32_t IDX_B_LOW = 6;

constexpr uint64_t L2_REAL_SIZE = 168; // B4真实的L2Size大小
constexpr uint64_t L2_FAKE_SIZE = 96; // B4被上层修改后的L2Size大小
constexpr uint64_t GROUP_MKN_BIT_SIZE = 0xFFFF;
constexpr size_t SCALE_PERGROUP_MIN_DIM_NUM = 2;

static constexpr int8_t OUTPUT_INFER_FAIL = static_cast<int8_t>(-1);
static constexpr int8_t OUTPUT_INFER_SUCCESS = 1;
constexpr const char *DEFAULT_OP_NAME = "EagleQuantBatchMatmulSwiglu";

// EagleQuantBatchMatmulSwiglu input index, mc2 is not same
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t SCALE_INDEX = 2;
constexpr uint32_t OFFSET_INDEX = 3;
constexpr uint32_t BIAS_INDEX = 4;
constexpr uint32_t PERTOKEN_SCALE_INDEX = 5;

static optiling::EagleQbmmInfoFactory g_quantBatchMatmulInfoFactory;

const char *GetValidOpName(const gert::TilingContext *context, const char *defaultOpName) {
    const char *fallbackName = (defaultOpName != nullptr && defaultOpName[0] != '\0') ? defaultOpName : DEFAULT_OP_NAME;
    if (context == nullptr) {
        return fallbackName;
    }
    const char *nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    const char *nodeType = context->GetNodeType();
    if (nodeType != nullptr && nodeType[0] != '\0') {
        return nodeType;
    }
    return fallbackName;
}
} // namespace

namespace optiling {
using namespace ge;

const std::map<ge::DataType, std::vector<BasicQuantMode>> X2_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E4M3FN,
        {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE,
            BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT8_E5M2,
        {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE,
            BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E2M1, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_HIFLOAT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE, BasicQuantMode::PERBLOCK_MODE}},
    {ge::DT_INT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERCHANNEL_MODE}},

};

const std::map<ge::DataType, std::vector<BasicQuantMode>> X1_QUANT_MODE_MAP = {
    {ge::DT_FLOAT8_E4M3FN,
        {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE,
            BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT8_E5M2,
        {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE,
            BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_FLOAT4_E2M1, {BasicQuantMode::MX_PERGROUP_MODE}},
    {ge::DT_HIFLOAT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE, BasicQuantMode::PERBLOCK_MODE}},
    {ge::DT_INT8, {BasicQuantMode::PERTENSOR_MODE, BasicQuantMode::PERTOKEN_MODE}},

};

uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetX1Idx() const { return X1_INDEX; }
uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetX2Idx() const { return X2_INDEX; }
uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetScaleIdx() const { return SCALE_INDEX; }
uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetOffsetIdx() const { return OFFSET_INDEX; }
uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetBiasIdx() const { return BIAS_INDEX; }
uint32_t EagleQuantBatchMatmulSwigluTilingBase::GetPertokenIdx() const { return PERTOKEN_SCALE_INDEX; }

EagleQuantBatchMatmulSwigluTilingBase::EagleQuantBatchMatmulSwigluTilingBase(
    gert::TilingContext *context, bool isTilingOut)
    : TilingBaseClass(context), inputParams_(*(g_quantBatchMatmulInfoFactory.Get())), compileInfo_(),
      isTilingOut_(isTilingOut) {}

void ResetEagleQuantBatchMatmulSwigluInputParams() { g_quantBatchMatmulInfoFactory.Get()->Reset(); }

ge::graphStatus EagleQuantBatchMatmulSwigluTilingBase::GetShapeAttrsInfo() {
    inputParams_.opName = GetValidOpName(context_, GetDefaultOpName());
    OP_LOGE_IF(!SetPlatformInfoForTiling(), ge::GRAPH_FAILED, inputParams_.opName, "Set PlatformInfoFortiling fail");
    if (inputParams_.initFlag) {
        OP_LOGD(inputParams_.opName, "No need to get shape and attrs from tiling context again");
        return ge::GRAPH_SUCCESS;
    }

    OPS_LOG_D(inputParams_.opName, "TilingContext: %s", Ops::EagleQbmm::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(CheckContext() != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(inputParams_.opName, "Invalid context."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeInputs(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Fail to analyze context info."), return ge::GRAPH_PARAM_INVALID);
    auto transA_str = inputParams_.transA ? "true" : "false";
    auto transB_str = inputParams_.transB ? "true" : "false";
    auto hasBias_str = inputParams_.hasBias ? "true" : "false";
    OP_LOGD(inputParams_.opName, "Input params: MKN[%ld, %ld, %ld], transA[%s], transB[%s], bias[%s].",
        inputParams_.mSize, inputParams_.kSize, inputParams_.nSize, transA_str, transB_str, hasBias_str);

    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

const char *EagleQuantBatchMatmulSwigluTilingBase::GetDefaultOpName() const { return DEFAULT_OP_NAME; }

ge::graphStatus EagleQuantBatchMatmulSwigluTilingBase::CheckContext() {
    auto x1Shape = context_->GetInputShape(GetX1Idx());
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    auto x2Shape = context_->GetInputShape(GetX2Idx());
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    auto scaleShape = context_->GetInputShape(GetScaleIdx());
    auto scaleDesc = context_->GetInputDesc(GetScaleIdx());
    auto outputShape = context_->GetOutputShape(0);
    auto outputDesc = context_->GetOutputDesc(0);
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Function context_->GetAttrs() failed!"), return ge::GRAPH_FAILED);
    auto dtypeAttr = attrs->GetAttrPointer<int64_t>(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x1Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, x2Desc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, scaleDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dtypeAttr);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    OP_TILING_CHECK(context_->GetRawTilingData()->GetCapacity() < tilingDataSize_,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "context tiling data capacity %zu < actual tiling data size %zu.",
            context_->GetRawTilingData()->GetCapacity(), tilingDataSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool EagleQuantBatchMatmulSwigluTilingBase::AnalyzeAttrs() {
    const bool isQuantBatchMatmulV4 = (strcmp(context_->GetNodeType(), "QuantBatchMatmulV4") == 0);
    auto attrs = context_->GetAttrs();
    // if op calls EagleQuantBatchMatmulSwiglu with attrs, it could rewrite as GetXXXAttr()
    if (attrs) {
        size_t idx = 0;
        auto dtypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        if (isQuantBatchMatmulV4) {
            // skip v4 attr "compute_type"
            idx++;
        }
        auto transposeX1Ptr = attrs->GetAttrPointer<bool>(idx++);
        auto transposeX2Ptr = attrs->GetAttrPointer<bool>(idx++);
        const int64_t *groupSizePtr = nullptr;
        if (attrs->GetAttrNum() > idx) {
            groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        }
        OP_TILING_CHECK(!dtypePtr,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "There should be at least the required dtype attr."),
            return false);
        inputParams_.outDtype = *dtypePtr;
        inputParams_.transA = transposeX1Ptr ? *transposeX1Ptr : false;
        inputParams_.transB = transposeX2Ptr ? *transposeX2Ptr : false;
        inputParams_.groupSize = groupSizePtr ? static_cast<uint64_t>(*groupSizePtr) : 0ULL;
        if (inputParams_.groupSize != 0ULL) {
            inputParams_.groupSizeK = inputParams_.groupSize & GROUP_MKN_BIT_SIZE;
            inputParams_.groupSizeN =
                (inputParams_.groupSize >> 16U) & GROUP_MKN_BIT_SIZE; // 16 is the bit size of MKN group size
            inputParams_.groupSizeM =
                (inputParams_.groupSize >> 32U) & GROUP_MKN_BIT_SIZE; // groupSizeM start at 32 bit of groupSize
        }
    }

    EagleQuantBatchMatmulSwigluTrans trans = EagleQuantBatchMatmulSwigluTrans::NO_TRANS;
    SetTransAttr(trans);
    // qbmmv4 调用时，需要对compileInfo_初始化
    if (!compileInfo_.supportL0c2Out) {
        OP_TILING_CHECK(trans != EagleQuantBatchMatmulSwigluTrans::B_TRANS,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                "This platform only support output transpose_x1 false and transpose_x2 true, actual [%s, %s].",
                inputParams_.transA ? "true" : "false", inputParams_.transB ? "true" : "false"),
            return false);
    }
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::AnalyzeDtype() {
    inputParams_.aDtype = context_->GetInputDesc(GetX1Idx())->GetDataType();
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    inputParams_.bDtype = x2Desc->GetDataType();
    inputParams_.scaleDtype = context_->GetInputDesc(GetScaleIdx())->GetDataType();
    auto pertokenScaleDesc = context_->GetOptionalInputDesc(GetPertokenIdx());
    inputParams_.perTokenScaleDtype =
        pertokenScaleDesc != nullptr ? pertokenScaleDesc->GetDataType() : inputParams_.perTokenScaleDtype;
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    inputParams_.biasDtype = biasDesc != nullptr ? biasDesc->GetDataType() : ge::DT_INT32;
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    isUbQuant_ = inputParams_.cDtype == ge::DT_BF16 || pertokenScaleDesc != nullptr;
    SetFormat();

    OP_TILING_CHECK(!CheckDtype(), CUBE_INNER_ERR_REPORT(inputParams_.opName, "CheckDtype failed!"), return false);
    return true;
}

uint64_t EagleQuantBatchMatmulSwigluTilingBase::GetBatchSize(const gert::Shape &shape) const {
    uint64_t batch = 1;
    auto shapeLen = shape.GetDimNum();
    for (size_t i = LAST_BATCH_DIM_INDEX; i <= shape.GetDimNum(); i++) {
        batch = batch * shape.GetDim(shapeLen - i);
    }
    return batch;
}

bool EagleQuantBatchMatmulSwigluTilingBase::InferOutBatchDim(const gert::Shape &x1Shape, const gert::Shape &x2Shape) {
    inputParams_.batchC = 1U;
    auto x1DimNum = x1Shape.GetDimNum();
    auto x2DimNum = x2Shape.GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    const gert::Shape &shapeLong = x1DimNum > x2DimNum ? x1Shape : x2Shape;
    const gert::Shape &shapeShort = x1DimNum > x2DimNum ? x2Shape : x1Shape;
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    for (size_t i = 0; i < outDimNum - LAST_SECOND_DIM_INDEX; i++) {
        auto shortDim = i < validOffset ? 1 : shapeShort.GetDim(i - validOffset);
        auto longDim = shapeLong.GetDim(i);
        if (shortDim > 1 && longDim > 1 && shortDim != longDim) {
            return false;
        }
        inputParams_.batchC = inputParams_.batchC * static_cast<uint64_t>(std::max(shortDim, longDim));
    }
    return true;
}

int8_t EagleQuantBatchMatmulSwigluTilingBase::CheckFusionBatchA(const gert::Shape &x1Shape, const gert::Shape &x2Shape,
    const gert::StorageShape *biasShape, uint64_t fusedDimValue) const {
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    auto x1BatchLen = x1ShapeLen - 2;
    if (inputParams_.isPertoken) { // pertoken dim equals to original m dim, cannot fuse
        return 0;
    }
    if (inputParams_.hasBias && biasShape->GetStorageShape().GetDimNum() > 1) {
        // The bias can only be a 1-D or 3-D tensor. The batch dim can be fused only in the 3-D case.
        return 0;
    }
    if (static_cast<int64_t>(fusedDimValue) > static_cast<int64_t>(INT32_MAX)) { // exceed axis limit, cannot fuse
        if (inputParams_.aDtype == ge::DT_INT4) {
            return OUTPUT_INFER_FAIL;
        }
        return 0;
    }
    // only fusion when x2 shape length is 2
    if (x2ShapeLen == 2 && inputParams_.transA == false && x1BatchLen >= 1) {
        // perBlock量化模式限制，groupSizeM不能整除m轴时，不能合轴
        if (inputParams_.groupSizeM > 1 && x1Shape.GetDim(x1ShapeLen - MIN_DIM_NUM_ND) % inputParams_.groupSizeM != 0) {
            return 0;
        }
        OP_LOGD(inputParams_.opName, "CheckFusionBatchA success, start fusion batch A to m dimension.");
        return OUTPUT_INFER_SUCCESS;
    }
    return 0;
}

void EagleQuantBatchMatmulSwigluTilingBase::DoBatchFusion(uint64_t fusedDimValue) {
    inputParams_.mSize = fusedDimValue;
    inputParams_.batchC = 1UL;
    inputParams_.batchC1 = 1UL;
    inputParams_.batchC2 = 1UL;
    inputParams_.batchC3 = 1UL;
    inputParams_.batchC4 = 1UL;
    inputParams_.batchA = 1UL;
    inputParams_.batchA1 = 1UL;
    inputParams_.batchA2 = 1UL;
    inputParams_.batchA3 = 1UL;
    inputParams_.batchA4 = 1UL;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckShapeInRangeForMandtoryInputs(
    size_t x1ShapeLen, size_t x2ShapeLen) const {
    OP_TILING_CHECK(x1ShapeLen < MIN_DIM_NUM_ND || x2ShapeLen < MIN_DIM_NUM_ND,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "input x1 dimension and x2 dimension should be greater than 1, \
but x1 dimension: %zu, x2 dimension: %zu",
            x1ShapeLen, x2ShapeLen),
        return false);
    OP_TILING_CHECK(x1ShapeLen > MAX_DIM_NUM_ND || x2ShapeLen > MAX_DIM_NUM_ND,
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "x1 dimension and x2 dimension should be less than 7, \
but x1 dimension: %zu, x2 dimension: %zu",
            x1ShapeLen, x2ShapeLen),
        return false);

    if (inputParams_.aDtype == ge::DT_INT4 && !inputParams_.isPertoken) {
        OP_TILING_CHECK(x2ShapeLen != A4W4_X2_LEN,
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "If input dtype is int4 and without pertoken scale, \
the dimension of x2 must be 2."),
            return false);
    }
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckStorageShape4WeightNz(
    const gert::Shape &x2StorageShape, size_t &x2StorageDim) const {
    std::set<uint64_t> weightStorageLastDims = {WEIGHTNZ_STORAGE_LAST_DIM_32, WEIGHTNZ_STORAGE_LAST_DIM_64};
    OP_TILING_CHECK(
        weightStorageLastDims.find(x2StorageShape[x2StorageDim - LAST_FIRST_DIM_INDEX]) == weightStorageLastDims.end(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "When format of x2 is FRACTAL_NZ, the last first dim of x2 should be 32 for int8 \
                              and 64 for int4, actual is %ld.",
            x2StorageShape[x2StorageDim - LAST_FIRST_DIM_INDEX]),
        return false);
    OP_TILING_CHECK(x2StorageShape[x2StorageDim - LAST_SECOND_DIM_INDEX] != WEIGHTNZ_STORAGE_PENULTIMATE_DIM,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "When format of x2 is FRACTAL_NZ, the last second dim of x2 should be 16, actual is %ld.",
            x2StorageShape[x2StorageDim - LAST_SECOND_DIM_INDEX]),
        return false);
    auto x2StorageLastThirdDim = static_cast<uint64_t>(x2StorageShape[x2StorageDim - LAST_THIRD_DIM_INDEX]);
    auto x2StorageLastFourthDim = static_cast<uint64_t>(x2StorageShape[x2StorageDim - LAST_FOURTH_DIM_INDEX]);
    if (!inputParams_.transB) {
        OP_TILING_CHECK(
            !CheckStorageShape4WeightNzDimAlignment(
                inputParams_.kSize, inputParams_.nSize, x2StorageLastThirdDim, x2StorageLastFourthDim),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "Dim check does not pass, please read the error report above"),
            return false;);
    } else {
        OP_TILING_CHECK(
            !CheckStorageShape4WeightNzDimAlignment(
                inputParams_.nSize, inputParams_.kSize, x2StorageLastThirdDim, x2StorageLastFourthDim),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "Dim check does not pass, please read the error report above"),
            return false;);
    }
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckStorageShape4WeightNzDimAlignment(
    uint64_t innerDim, uint64_t outerDim, uint64_t x2StorageLastThirdDim, uint64_t x2StorageLastFourthDim) const {
    bool isNotLastThirdDimAligned = (x2StorageLastThirdDim != ops::CeilDiv(innerDim, WEIGHTNZ_STORAGE_PENULTIMATE_DIM));
    bool isNotLastFourthDimAligned = (x2StorageLastFourthDim != ops::CeilDiv(outerDim, WEIGHTNZ_STORAGE_LAST_DIM_32)) &&
        (x2StorageLastFourthDim != ops::CeilDiv(outerDim, WEIGHTNZ_STORAGE_LAST_DIM_64));
    OP_TILING_CHECK(isNotLastThirdDimAligned,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "When format of x2 is FRACTAL_NZ, the last third dim of x2 should be ceil(k/16) = %lu, actual is %lu.",
            ops::CeilDiv(innerDim, WEIGHTNZ_STORAGE_PENULTIMATE_DIM), x2StorageLastThirdDim),
        return false);
    OP_TILING_CHECK(isNotLastFourthDimAligned,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "When format of x2 is FRACTAL_NZ, the last fourth dim of x2 should be ceil(n/32) \
                = %lu for int8, or ceil(n/64) = %lu for int4, actual is %lu.",
            ops::CeilDiv(outerDim, WEIGHTNZ_STORAGE_LAST_DIM_32), ops::CeilDiv(outerDim, WEIGHTNZ_STORAGE_LAST_DIM_64),
            x2StorageLastFourthDim),
        return false);
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckShape4WeightNz() const {
    OP_TILING_CHECK(!compileInfo_.supportMmadS8S4 && (1 == inputParams_.kSize || 1 == inputParams_.nSize),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "When format of x2 is FRACTAL_NZ, n or k cannot be 1."),
        return false);

    const gert::Shape x2StorageShape = context_->GetInputShape(GetX2Idx())->GetStorageShape();
    auto x2StorageDim = x2StorageShape.GetDimNum();
    OP_TILING_CHECK(x2StorageDim > WEIGHTNZ_STORAGE_MAX_DIM || x2StorageDim < WEIGHTNZ_STORAGE_MIN_DIM,
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "When format of x2 is FRACTAL_NZ, the dim of storage shape should be 4-8, actual is %zu.", x2StorageDim),
        return false);
    if (!CheckStorageShape4WeightNz(x2StorageShape, x2StorageDim)) {
        return false;
    }
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::IsMicroScaling() const {
    return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
}

std::string EagleQuantBatchMatmulSwigluTilingBase::QuantModeToString(BasicQuantMode quantMode) const {
    std::string mode = "default";
    switch (quantMode) {
    case BasicQuantMode::PERTENSOR_MODE:
        mode = "pertensor";
        break;
    case BasicQuantMode::PERCHANNEL_MODE:
        mode = "perchannel";
        break;
    case BasicQuantMode::PERTOKEN_MODE:
        mode = "pertoken";
        break;
    case BasicQuantMode::MX_PERGROUP_MODE:
        mode = "mx_pergroup";
        break;
    case BasicQuantMode::PERBLOCK_MODE:
        mode = "perblock";
        break;
    default:
        break;
    }
    return mode;
}

bool EagleQuantBatchMatmulSwigluTilingBase::SetX2QuantMode(
    BasicQuantMode &x2QuantMode, bool isFp8Hif8Input, const gert::Shape &scaleShape) {
    auto scaleShapeLen = scaleShape.GetDimNum();
    if (IsMicroScaling()) {
        x2QuantMode = BasicQuantMode::MX_PERGROUP_MODE;
    } else if (isFp8Hif8Input && inputParams_.scaleDtype == ge::DT_FLOAT && scaleShapeLen > 1 &&
        (scaleShapeLen > SCALE_PERGROUP_MIN_DIM_NUM || scaleShape.GetDim(0) != 1 || inputParams_.groupSizeN > 1)) {
        x2QuantMode = BasicQuantMode::PERBLOCK_MODE;
    } else if (scaleShape.GetDim(scaleShapeLen - LAST_FIRST_DIM_INDEX) == 1) {
        x2QuantMode = BasicQuantMode::PERTENSOR_MODE;
    } else if (static_cast<uint64_t>(scaleShape.GetDim(scaleShapeLen - LAST_FIRST_DIM_INDEX)) == inputParams_.nSize) {
        x2QuantMode = BasicQuantMode::PERCHANNEL_MODE;
    } else {
        auto x2QuantModeList = X2_QUANT_MODE_MAP.find(inputParams_.bDtype);
        OP_TILING_CHECK(x2QuantModeList == X2_QUANT_MODE_MAP.end(),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "The dtype of x2 is invalid."), return false);
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Unexpected input, for the dtype of x2: %s, \
the shape of x2 and x2Scale(scale) not match any of quantification: %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
            QuantModeMapToString(x2QuantModeList->second).c_str());
        return false;
    }
    return true;
}
bool EagleQuantBatchMatmulSwigluTilingBase::SetX1QuantMode(
    BasicQuantMode &x1QuantMode, bool isFp8Hif8Input, const gert::StorageShape *pertokenShape) {
    if (pertokenShape == nullptr) {
        return true;
    }
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenLen = pertoken.GetDimNum();
    if (IsMicroScaling()) {
        x1QuantMode = BasicQuantMode::MX_PERGROUP_MODE;
    } else if (isFp8Hif8Input && inputParams_.perTokenScaleDtype == ge::DT_FLOAT && pertokenLen > 1) {
        x1QuantMode = BasicQuantMode::PERBLOCK_MODE;
    } else if (isFp8Hif8Input && pertokenLen == 1 && pertoken.GetDim(pertokenLen - LAST_FIRST_DIM_INDEX) == 1) {
        x1QuantMode = BasicQuantMode::PERTENSOR_MODE;
    } else if (pertokenLen == 1 &&
        static_cast<uint64_t>(pertoken.GetDim(pertokenLen - LAST_FIRST_DIM_INDEX)) == inputParams_.mSize) {
        x1QuantMode = BasicQuantMode::PERTOKEN_MODE;
    } else {
        auto x1QuantModeList = X1_QUANT_MODE_MAP.find(inputParams_.aDtype);
        OP_TILING_CHECK(x1QuantModeList == X1_QUANT_MODE_MAP.end(),
            CUBE_INNER_ERR_REPORT(inputParams_.opName, "The dtype of x1 is invalid."), return false);
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Unexpected input, for the dtype of x1: %s, \
the shape of x1 and x1Scale(pertokenScale) not match any of quantification: %s.",
            ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
            QuantModeMapToString(x1QuantModeList->second).c_str());
        return false;
    }
    return true;
}

std::string EagleQuantBatchMatmulSwigluTilingBase::QuantModeMapToString(
    const std::vector<BasicQuantMode> &quantModeList) const {
    std::stringstream ss;
    ss << "[";
    for (auto it = quantModeList.begin(); it != quantModeList.end(); it++) {
        ss << QuantModeToString(*it);
        if (it + 1 != quantModeList.end()) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

bool EagleQuantBatchMatmulSwigluTilingBase::SetQuantMode(
    const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape) {
    if (!compileInfo_.supportL12BtBf16) {
        inputParams_.isPerTensor = scaleShape.GetDim(0) == 1;
        return true;
    }
    BasicQuantMode x1QuantMode = BasicQuantMode::DEFAULT;
    BasicQuantMode x2QuantMode = BasicQuantMode::DEFAULT;
    bool isFp8Input = (inputParams_.aDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.aDtype == ge::DT_FLOAT8_E5M2);
    bool isFp8Hif8Input = inputParams_.aDtype == ge::DT_HIFLOAT8 || isFp8Input;
    if (!SetX1QuantMode(x1QuantMode, isFp8Hif8Input, pertokenShape) ||
        !SetX2QuantMode(x2QuantMode, isFp8Hif8Input, scaleShape)) {
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Get quantification of x1/x2 failed.");
        return false;
    }
    inputParams_.isPerChannel = x2QuantMode == BasicQuantMode::PERCHANNEL_MODE;
    inputParams_.isPerTensor = x2QuantMode == BasicQuantMode::PERTENSOR_MODE;
    inputParams_.isDoubleScale = x1QuantMode == BasicQuantMode::PERTENSOR_MODE;
    inputParams_.isPertoken = x1QuantMode == BasicQuantMode::PERTOKEN_MODE;
    inputParams_.isMxPerGroup = x1QuantMode == BasicQuantMode::MX_PERGROUP_MODE && x1QuantMode == x2QuantMode;
    inputParams_.isPerBlock = x1QuantMode == BasicQuantMode::PERBLOCK_MODE;
    OP_TILING_CHECK(!(inputParams_.isPerChannel || inputParams_.isDoubleScale || inputParams_.isPerTensor ||
                        inputParams_.isPertoken || inputParams_.isMxPerGroup || inputParams_.isPerBlock),
        CUBE_INNER_ERR_REPORT(inputParams_.opName,
            "Unexpected quantification, quantification of x1: %s, quantification of x2: %s",
            QuantModeToString(x1QuantMode).c_str(), QuantModeToString(x2QuantMode).c_str()),
        return false);
    return true;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
bool EagleQuantBatchMatmulSwigluTilingBase::AnalyzeInputs() {
    auto x1Shape = GetX1Shape(GetX1Idx());
    auto x2Shape = GetX2Shape(GetX2Idx());
    auto scaleShape = GetScaleShape(GetScaleIdx());
    auto pertokenShape = GetPertokenShape(GetPertokenIdx());
    inputParams_.isPertoken = pertokenShape != nullptr;
    auto biasShape = GetBiasShape(GetBiasIdx());
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = inputParams_.hasBias ? GetBatchSize(biasShape->GetStorageShape()) : 1;
    auto x1ShapeLen = x1Shape.GetDimNum();
    auto x2ShapeLen = x2Shape.GetDimNum();
    if (!isTilingOut_ && !CheckShapeInRangeForMandtoryInputs(x1ShapeLen, x2ShapeLen)) {
        return false;
    }
    auto x1Inner = x1Shape.GetDim(x1ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x1Outer = x1Shape.GetDim(x1ShapeLen - LAST_SECOND_DIM_INDEX);
    auto x2Inner = x2Shape.GetDim(x2ShapeLen - LAST_FIRST_DIM_INDEX);
    auto x2Outer = x2Shape.GetDim(x2ShapeLen - LAST_SECOND_DIM_INDEX);

    const std::vector<int64_t> dimValueOfMKN = {x1Inner, x1Outer, x2Inner, x2Outer};
    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? x1Inner : x1Outer);
    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? x1Outer : x1Inner);
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? x2Outer : x2Inner);
    const std::vector<gert::Shape *> mandtoryShape = {&x1Shape, &x2Shape, &scaleShape};
    if (!AnalyzeGroupInfo(scaleShape, pertokenShape)) {
        return false;
    }
    inputParams_.batchA = GetBatchSize(x1Shape);
    inputParams_.batchB = GetBatchSize(x2Shape);
    AnalyzeBatchInfo(context_->GetInputShape(0)->GetOriginShape(), context_->GetInputShape(1)->GetOriginShape());
    OP_TILING_CHECK(!InferOutBatchDim(x1Shape, x2Shape),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Batch dimension can not be broadcasted."), return false);
    if (!SetQuantMode(scaleShape, pertokenShape)) {
        return false;
    }
    if (inputParams_.bFormat == ge::FORMAT_FRACTAL_NZ && !CheckShape4WeightNz()) {
        return false;
    }
    if (!isTilingOut_ && !CheckShape(mandtoryShape, biasShape, pertokenShape, dimValueOfMKN)) {
        return false;
    }
    OP_TILING_CHECK(!CheckOutputShapeAvailable(),
        CUBE_INNER_ERR_REPORT(inputParams_.opName, "Multiple of output shape dims should be in boundary of INT64_MAX"),
        return false);
    uint64_t fusedDimValue = inputParams_.mSize * inputParams_.batchA;
    int8_t resultCheckFusionBatchA = CheckFusionBatchA(x1Shape, x2Shape, biasShape, fusedDimValue);
    OP_TILING_CHECK(resultCheckFusionBatchA == OUTPUT_INFER_FAIL,
        CUBE_INNER_ERR_REPORT(
            inputParams_.opName, "The fused M [%lu] exceed INT32_MAX [%d] in a4w4 case", fusedDimValue, INT32_MAX),
        return false);
    if (resultCheckFusionBatchA == OUTPUT_INFER_SUCCESS) {
        DoBatchFusion(fusedDimValue);
    }
    auto isPerTensorStr = inputParams_.isPerTensor ? "true" : "false";
    auto isPertokenStr = inputParams_.isPertoken ? "true" : "false";
    OP_LOGD(inputParams_.opName, "batchA: %lu, batchB: %lu, batchC: %lu, isPerTensor: %s, isPertoken: %s",
        inputParams_.batchA, inputParams_.batchB, inputParams_.batchC, isPerTensorStr, isPertokenStr);
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckOutputShapeAvailable() const {
    int64_t m = static_cast<int64_t>(inputParams_.mSize);
    int64_t n = static_cast<int64_t>(inputParams_.nSize);
    int64_t b = static_cast<int64_t>(inputParams_.batchC);
    int64_t mul = m * n;
    if (mul / m != n) {
        return false;
    }
    mul *= b;
    return mul / b == m * n;
}

void EagleQuantBatchMatmulSwigluTilingBase::AnalyzeBatchInfo(
    const gert::Shape &oriShapeA, const gert::Shape &oriShapeB) {
    int32_t numDimA = static_cast<int32_t>(oriShapeA.GetDimNum());
    int32_t numDimB = static_cast<int32_t>(oriShapeB.GetDimNum());
    inputParams_.batchA4 = numDimA > IDX_K_LOW ? oriShapeA.GetDim(numDimA - IDX_K_HIGH) : 1;
    inputParams_.batchA3 = numDimA > IDX_K_HIGH ? oriShapeA.GetDim(numDimA - IDX_N_LOW) : 1;
    inputParams_.batchA2 = numDimA > IDX_N_LOW ? oriShapeA.GetDim(numDimA - IDX_N_HIGH) : 1;
    inputParams_.batchA1 = numDimA > IDX_N_HIGH ? oriShapeA.GetDim(numDimA - IDX_B_LOW) : 1;
    inputParams_.batchB4 = numDimB > IDX_K_LOW ? oriShapeB.GetDim(numDimB - IDX_K_HIGH) : 1;
    inputParams_.batchB3 = numDimB > IDX_K_HIGH ? oriShapeB.GetDim(numDimB - IDX_N_LOW) : 1;
    inputParams_.batchB2 = numDimB > IDX_N_LOW ? oriShapeB.GetDim(numDimB - IDX_N_HIGH) : 1;
    inputParams_.batchB1 = numDimB > IDX_N_HIGH ? oriShapeB.GetDim(numDimB - IDX_B_LOW) : 1;
    auto outShape = context_->GetOutputShape(0)->GetStorageShape();
    int32_t numDimC = static_cast<int32_t>(outShape.GetDimNum());
    inputParams_.batchC4 = numDimC > IDX_K_LOW ? outShape.GetDim(numDimC - IDX_K_HIGH) : 1UL;
    inputParams_.batchC3 = numDimC > IDX_K_HIGH ? outShape.GetDim(numDimC - IDX_N_LOW) : 1UL;
    inputParams_.batchC2 = numDimC > IDX_N_LOW ? outShape.GetDim(numDimC - IDX_N_HIGH) : 1UL;
    inputParams_.batchC1 = numDimC > IDX_N_HIGH ? outShape.GetDim(numDimC - IDX_B_LOW) : 1UL;
}

bool EagleQuantBatchMatmulSwigluTilingBase::ReCalcGroupSize(
    uint64_t &groupSize, uint64_t inputSize, uint64_t scaleSize, const char *dimensionName) {
    if (groupSize == 0ULL) {
        std::string inputName = "x1";
        std::string scaleName = "pertoken_scale";
        if (strcmp(dimensionName, "n") == 0) {
            inputName = "x2";
            scaleName = "scale";
        }
        OP_TILING_CHECK(scaleSize == 0ULL,
            CUBE_INNER_ERR_REPORT(
                inputParams_.opName, "The %s dimension of %s is 0, invalid shape!", dimensionName, scaleName.c_str()),
            return false);
        OP_TILING_CHECK(inputSize % scaleSize != 0ULL,
            CUBE_INNER_ERR_REPORT(inputParams_.opName,
                "The group_size in %s dimension is 0 and the %s dimension of %s [%lu] is not divisible \
by the %s dimension of %s [%lu], the real group_size in %s dimension can not be inferred.",
                dimensionName, dimensionName, inputName.c_str(), inputSize, dimensionName, scaleName.c_str(), scaleSize,
                dimensionName),
            return false);
        groupSize = inputSize / scaleSize;
    }
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::AnalyzeMxGroupInfo(
    const gert::Shape &scaleShape, const gert::Shape &pertoken, size_t pertokenShapeLen, size_t scaleShapeLen) {
    const size_t mxScaleNonBatchDimNum = 3;
    const uint64_t microScalingAlign = 2;
    if (pertokenShapeLen < mxScaleNonBatchDimNum || scaleShapeLen < mxScaleNonBatchDimNum) {
        return true;
    }
    size_t pertokenBatchDimNum = pertokenShapeLen - mxScaleNonBatchDimNum;
    size_t scaleBatchDimNum = scaleShapeLen - mxScaleNonBatchDimNum;
    uint64_t mxPertokenMSize =
        static_cast<uint64_t>(pertoken.GetDim(pertokenBatchDimNum + (inputParams_.transA ? 1 : 0)));
    if (!ReCalcGroupSize(inputParams_.groupSizeM, inputParams_.mSize, mxPertokenMSize, "m")) {
        return false;
    }
    uint64_t mxPertokenKSize =
        static_cast<uint64_t>(pertoken.GetDim(pertokenBatchDimNum + (inputParams_.transA ? 0 : 1))) * microScalingAlign;
    if (!ReCalcGroupSize(inputParams_.groupSizeK, inputParams_.kSize, mxPertokenKSize, "k")) {
        return false;
    }
    uint64_t mxScaleNSize = static_cast<uint64_t>(scaleShape.GetDim(scaleBatchDimNum + (inputParams_.transB ? 0 : 1)));
    if (!ReCalcGroupSize(inputParams_.groupSizeN, inputParams_.nSize, mxScaleNSize, "n")) {
        return false;
    }
    OP_LOGD(inputParams_.opName, "Infer mx groupSize success. groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.",
        inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK);
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::AnalyzeGroupInfo(
    const gert::Shape &scaleShape, const gert::StorageShape *pertokenShape) {
    if (pertokenShape == nullptr) {
        return true;
    }
    auto &pertoken = pertokenShape->GetStorageShape();
    auto pertokenShapeLen = pertoken.GetDimNum();
    auto scaleShapeLen = scaleShape.GetDimNum();
    if (pertokenShapeLen < 2 ||
        scaleShapeLen < 2) { // scaleLen < 2 means invalid input, leave it to the checher behind.
        return true;
    }
    if (IsMicroScaling()) {
        return AnalyzeMxGroupInfo(scaleShape, pertoken, pertokenShapeLen, scaleShapeLen);
    }
    auto pertokenInner = pertoken.GetDim(1);
    auto pertokenOuter = pertoken.GetDim(0);
    uint64_t pertokenMSize = static_cast<uint64_t>(inputParams_.transA ? pertokenInner : pertokenOuter);
    if (!ReCalcGroupSize(inputParams_.groupSizeM, inputParams_.mSize, pertokenMSize, "m")) {
        return false;
    }
    uint64_t pertokenKSize = static_cast<uint64_t>(inputParams_.transA ? pertokenOuter : pertokenInner);
    if (!ReCalcGroupSize(inputParams_.groupSizeK, inputParams_.kSize, pertokenKSize, "k")) {
        return false;
    }
    auto scaleInner = scaleShape.GetDim(1);
    auto scaleOuter = scaleShape.GetDim(0);
    uint64_t scaleNSize = static_cast<uint64_t>(inputParams_.transB ? scaleOuter : scaleInner);
    if ((scaleInner == 1 && scaleOuter == inputParams_.nSize) ||
        (scaleInner == inputParams_.nSize && scaleOuter == 1) ||
        (inputParams_.nSize == 1 && (scaleInner == 1 || scaleOuter == 1))) {
        scaleNSize = inputParams_.nSize;
    }
    if (!ReCalcGroupSize(inputParams_.groupSizeN, inputParams_.nSize, scaleNSize, "n")) {
        return false;
    }
    OP_LOGD(inputParams_.opName, "Infer groupSize success. groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.",
        inputParams_.groupSizeM, inputParams_.groupSizeN, inputParams_.groupSizeK);
    return true;
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape &EagleQuantBatchMatmulSwigluTilingBase::GetScaleShape(const size_t index) {
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape EagleQuantBatchMatmulSwigluTilingBase::GetX1Shape(const size_t index) {
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::Shape EagleQuantBatchMatmulSwigluTilingBase::GetX2Shape(const size_t index) {
    return context_->GetInputShape(index)->GetOriginShape();
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *EagleQuantBatchMatmulSwigluTilingBase::GetPertokenShape(const size_t index) {
    return context_->GetOptionalInputShape(index);
}

// Notice: 修改此函数可能会影响mc2功能，使用isTilingOut_变量判断是否为mc2场景
const gert::StorageShape *EagleQuantBatchMatmulSwigluTilingBase::GetBiasShape(const size_t index) {
    return context_->GetOptionalInputShape(index);
}

void EagleQuantBatchMatmulSwigluTilingBase::SetFormat() {
    inputParams_.aFormat = ge::FORMAT_ND;
    inputParams_.bFormat = ge::FORMAT_ND;
    inputParams_.cFormat = ge::FORMAT_ND;
    auto x1Desc = context_->GetInputDesc(GetX1Idx());
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x1Desc->GetStorageFormat())) == ge::Format::FORMAT_FRACTAL_NZ) {
        inputParams_.aFormat = ge::FORMAT_FRACTAL_NZ;
    }
    auto x2Desc = context_->GetInputDesc(GetX2Idx());
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2Desc->GetStorageFormat())) == ge::Format::FORMAT_FRACTAL_NZ) {
        inputParams_.bFormat = ge::FORMAT_FRACTAL_NZ;
    }
}

void EagleQuantBatchMatmulSwigluTilingBase::SetTransAttr(EagleQuantBatchMatmulSwigluTrans &trans) const {
    if (!inputParams_.transA && inputParams_.transB) { // most for ND
        trans = EagleQuantBatchMatmulSwigluTrans::B_TRANS;
    } else if (!inputParams_.transA && !inputParams_.transB) { // most for weight NZ
        trans = EagleQuantBatchMatmulSwigluTrans::NO_TRANS;
    } else if (inputParams_.transA && !inputParams_.transB) {
        trans = EagleQuantBatchMatmulSwigluTrans::A_TRANS;
    } else {
        trans = EagleQuantBatchMatmulSwigluTrans::AB_TRANS;
    }
}

bool EagleQuantBatchMatmulSwigluTilingBase::IsCapable() { return true; }

void EagleQuantBatchMatmulSwigluTilingBase::InitCompileInfo() {
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.workspaceNum = ascendcPlatform.GetLibApiWorkSpaceSize();
    compileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfoPtr->GetPlatformRes("version", "Soc_version", compileInfo_.socVersionStr);
    compileInfo_.socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.ubSize = std::min<uint64_t>(compileInfo_.ubSize, 188160);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfo_.l0bSize);

    std::string platformRes;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", platformRes);
    compileInfo_.supportL0c2Out = !platformRes.empty();
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", platformRes);
    compileInfo_.supportL12BtBf16 = (platformRes.find("bf16") != std::string::npos);
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_mmad", platformRes);
    compileInfo_.supportMmadS8S4 = (platformRes.find("s8s4") != std::string::npos);
    compileInfo_.npuArch = compileInfo_.supportMmadS8S4 ? NpuArch::DAV_RESV : ascendcPlatform.GetCurNpuArch();
    TilingPrepareForOpCache(context_);
    compileInfoInit_ = true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::SetPlatformInfoForTiling() {
    // mc2和qbmm把compileInfo都赋值给compileInfo_，后续硬件信息可以直接从compileInfo_中获取
    if (!compileInfoInit_) {
        auto mmCompileInfo =
            reinterpret_cast<const EagleQuantBatchMatmulSwigluCompileInfo *>(context_->GetCompileInfo());
        if (mmCompileInfo != nullptr) {
            compileInfo_ = *mmCompileInfo;
        } else {
            // Generated ACLNN calls may omit TilingParse compile info. Reuse
            // the runtime-platform initialization also used by tiling-out calls.
            InitCompileInfo();
            OP_LOGE_IF(!compileInfoInit_, false, inputParams_.opName, "Runtime platform initialization failed");
        }
    }
    OP_LOGE_IF(compileInfo_.aicNum <= 0, false, inputParams_.opName, "coreNum <= 0");
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_LOGE_IF(platformInfoPtr == nullptr, false, inputParams_.opName, "platformInfoPtr is null");
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aicoreParams_.aicNum = compileInfo_.aicNum;
    if (ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::KIRINX90 &&
        ascendcPlatform.GetSocVersion() != platform_ascendc::SocVersion::KIRIN9030) {
        OP_LOGE_IF(compileInfo_.l2Size <= 0, false, inputParams_.opName, "l2Size <= 0");
    }
    // 纠正L2实际物理大小
    compileInfo_.l2Size = compileInfo_.l2Size == L2_FAKE_SIZE * MB_SIZE ? L2_REAL_SIZE * MB_SIZE : compileInfo_.l2Size;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;
    aicoreParams_.ubSize = std::min<uint64_t>(compileInfo_.ubSize, 188160);
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    aicoreParams_.blockDim = 0;
    return true;
}

bool EagleQuantBatchMatmulSwigluTilingBase::GetUbDequantExtreSpace() { return false; }

bool EagleQuantBatchMatmulSwigluTilingBase::CheckShape(const std::vector<gert::Shape *> &mandtoryShape,
    const gert::StorageShape *biasShape, const gert::StorageShape *pertokenShape,
    const std::vector<int64_t> &dimValueOfMKN) const {
    (void)mandtoryShape;
    (void)biasShape;
    (void)pertokenShape;
    (void)dimValueOfMKN;
    return false;
}

bool EagleQuantBatchMatmulSwigluTilingBase::CheckDtype() const { return false; }

ge::graphStatus EagleQuantBatchMatmulSwigluTilingBase::CalcUbTiling() { return ge::GRAPH_FAILED; }

uint64_t EagleQbmmInfo::GetMatmulApiMSize() const { return mSizePerNpu > 0U ? mSizePerNpu : mSize; }

uint64_t EagleQbmmInfo::GetTotalMatmulApiMSize(uint64_t baseM) const {
    if (mSizePerNpu > 0U) {
        return ops::CeilAlign(mSizePerNpu, baseM) * ops::CeilDiv(mSize, mSizePerNpu);
    } else {
        return mSize;
    }
}

uint64_t EagleQbmmInfo::GetTotalBaseMCnt(uint64_t baseM) const {
    return ops::CeilDiv(GetTotalMatmulApiMSize(baseM), baseM); // m方向需要的轮数
}

void EagleQbmmInfo::Reset() {
    initFlag = false;
    transA = false;
    transB = false;
    hasBias = false;
    mSize = 0UL;
    mSizePerNpu = 0UL;
    kSize = 0UL;
    nSize = 0UL;
    batchA = 0UL;
    batchA1 = 0UL;
    batchA2 = 0UL;
    batchA3 = 0UL;
    batchA4 = 0UL;
    batchB = 0UL;
    batchB1 = 0UL;
    batchB2 = 0UL;
    batchB3 = 0UL;
    batchB4 = 0UL;
    batchC = 0UL;
    batchC1 = 0UL;
    batchC2 = 0UL;
    batchC3 = 0UL;
    batchC4 = 0UL;
    batchBias = 0UL;
    aDtype = ge::DT_INT8;
    bDtype = ge::DT_INT8;
    cDtype = ge::DT_FLOAT16;
    biasDtype = ge::DT_INT32;
    scaleDtype = ge::DT_UINT64;
    perTokenScaleDtype = ge::DT_FLOAT;
    x2TableDtype = ge::DT_INT4;
    isPerTensor = false;
    isPerChannel = false;
    isPertoken = false;
    isDoubleScale = false;
    isMxPerGroup = false;
    isPerBlock = false;
    isPerBlockPerToken = false;
    isLut = false;
    outDtype = 0L;
    libApiWorkSpaceSize = 0U;
    bf16ExtreWorkSpaceSize = 0UL;
    groupSize = 0UL;
    groupSizeM = 0UL;
    groupSizeK = 0UL;
    groupSizeN = 0UL;
    x2TableKSize = 0UL;
    x2TableNSize = 0UL;
    opName = nullptr;
    aFormat = ge::FORMAT_ND;
    bFormat = ge::FORMAT_ND;
    cFormat = ge::FORMAT_ND;
}
} // namespace optiling
