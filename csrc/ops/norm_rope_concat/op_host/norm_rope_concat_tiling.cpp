/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
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
 * \file norm_rope_concat_tiling.cpp
 * \brief Tiling implementation for NormRopeConcat operator
 */

#include "norm_rope_concat_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"

using namespace ge;
using namespace nrc;
using namespace AscendC;
namespace optiling {
ge::graphStatus NormRopeConcatTiling::CheckInput(const InputTensorInfo &x)
{
    if (x.isRequired) {
        OP_CHECK_IF(x.desc == nullptr, OP_LOGE(context_.opName, "desc is nullptr"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(x.shape == nullptr, OP_LOGE(context_.opName, "shape is nullptr"), return ge::GRAPH_FAILED);
    }
    if (x.desc != nullptr) {
        OP_CHECK_IF(x.desc->GetDataType() != inputDtype_,
                    OP_LOGE(context_.opName, "DataType of tensor mismatch, expected %d, got %d",
                            static_cast<int>(inputDtype_), static_cast<int>(x.desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    }
    if (x.shape != nullptr) {
        auto shape = x.shape->GetStorageShape();
        switch (x.type) {
            case TensorType::INPUT_TENSOR:
            case TensorType::ENCODER_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != INPUT_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 4"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(BATCH_DIM) != batchSize_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be batchSize_(%ld)",
                                    shape.GetDim(BATCH_DIM), batchSize_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(HEAD_DIM) != headNum_,
                            OP_LOGE(context_.opName, "DIM2(%ld) Value of Shape should be headNum_(%ld)",
                                    shape.GetDim(HEAD_DIM), headNum_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(DIM_DIM) != headDim_,
                            OP_LOGE(context_.opName, "DIM3(%ld) Value of Shape should be headDim_(%ld)",
                                    shape.GetDim(DIM_DIM), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::NORM_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != 1, OP_LOGE(context_.opName, "Shape Dims of tensor should be 1"),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(0) != headDim_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should be headDim_(%ld)", shape.GetDim(0),
                                    headDim_),
                            return ge::GRAPH_FAILED);
                break;
            case TensorType::ROPE_TENSOR:
                OP_CHECK_IF(shape.GetDimNum() != ROPE_DIM_NUM,
                            OP_LOGE(context_.opName, "Shape Dims of tensor should be 2"), return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_SEQ_DIM) > maxSeq_,
                            OP_LOGE(context_.opName, "DIM0(%ld) Value of Shape should smaller than maxSeq_(%ld)",
                                    shape.GetDim(ROPE_SEQ_DIM), maxSeq_),
                            return ge::GRAPH_FAILED);
                OP_CHECK_IF(shape.GetDim(ROPE_DIM_DIM) != headDim_,
                            OP_LOGE(context_.opName, "DIM1(%ld) Value of Shape should be headDim_(%ld)",
                                    shape.GetDim(ROPE_DIM_DIM), headDim_),
                            return ge::GRAPH_FAILED);
                break;
            default:
                return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::CheckInput()
{
    OP_CHECK_IF((headDim_ % 2 != 0) || (headDim_ > 1024),
                OP_LOGE(context_.opName, "headDim of shape value should be even number and smaller than 1024 "),
                return ge::GRAPH_FAILED);

    for (size_t i = 0; i < context_.maxInputNum; ++i) {
        if (CheckInput(context_.inputs[i]) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    OP_CHECK_IF(totalQuerySeq_ != querySeq_ + encoderQuerySeq_,
                OP_LOGE(context_.opName, "totalQuerySeq_ != querySeq_ + encoderQuerySeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalKeySeq_ != keySeq_ + encoderKeySeq_,
                OP_LOGE(context_.opName, "totalKeySeq_ != keySeq_ + encoderKeySeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(totalValueSeq_ != valueSeq_ + encoderValueSeq_,
                OP_LOGE(context_.opName, "totalValueSeq_ != valueSeq_ + encoderValueSeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keySeq_ != valueSeq_, OP_LOGE(context_.opName, "keySeq_ != valueSeq_"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(encoderKeySeq_ != encoderValueSeq_, OP_LOGE(context_.opName, "encoderKeySeq_ != encoderValueSeq_"),
                return ge::GRAPH_FAILED);

    normDim_ = headDim_;
    OP_CHECK_IF(context_.eps == nullptr, OP_LOGE(context_.opName, "eps is nullptr"), return ge::GRAPH_FAILED);
    eps_ = *(context_.eps);
    scale_ = 1 / float(headDim_);
    int64_t normType = *context_.normType;
    OP_CHECK_IF(!IsNormTypeValid(normType), OP_LOGE(context_.opName, "normType is out of range"),
                return ge::GRAPH_FAILED);
    int64_t normAddedType = *context_.normAddedType;
    OP_CHECK_IF(!IsNormTypeValid(normAddedType), OP_LOGE(context_.opName, "normAddedType is out of range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::GetShapeInfo()
{
    size_t queryIndex = static_cast<size_t>(InputIndexForward::QUERY_INDEX);
    size_t keyIndex = static_cast<size_t>(InputIndexForward::KEY_INDEX);
    const gert::StorageShape *queryShape = context_.inputs[queryIndex].shape;
    const gert::StorageShape *keyShape = context_.inputs[keyIndex].shape;
    OP_CHECK_IF(queryShape == nullptr || queryShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE(context_.opName, "Shape of tensor query should be 4"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape == nullptr || keyShape->GetStorageShape().GetDimNum() != INPUT_DIM_NUM,
                OP_LOGE(context_.opName, "Shape of tensor key should be 4"), return ge::GRAPH_FAILED);

    batchSize_ = queryShape->GetStorageShape().GetDim(BATCH_DIM);
    headNum_ = queryShape->GetStorageShape().GetDim(HEAD_DIM);
    headDim_ = queryShape->GetStorageShape().GetDim(DIM_DIM);
    querySeq_ = queryShape->GetStorageShape().GetDim(SEQ_DIM);
    keySeq_ = keyShape->GetStorageShape().GetDim(SEQ_DIM);
    valueSeq_ = keySeq_;
    size_t valueIndex = static_cast<size_t>(InputIndexForward::VALUE_INDEX);
    const gert::StorageShape *valueShape = context_.inputs[valueIndex].shape;
    if (valueShape != nullptr && valueShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        valueSeq_ = valueShape->GetStorageShape().GetDim(SEQ_DIM);
    }

    size_t encoderQueryIndex = static_cast<size_t>(InputIndexForward::ENCODER_QUERY_INDEX);
    size_t encoderKeyIndex = static_cast<size_t>(InputIndexForward::ENCODER_KEY_INDEX);
    const gert::StorageShape *encoderQueryShape = context_.inputs[encoderQueryIndex].shape;
    const gert::StorageShape *encoderKeyShape = context_.inputs[encoderKeyIndex].shape;
    if (encoderQueryShape != nullptr && encoderQueryShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderQuerySeq_ = encoderQueryShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    if (encoderKeyShape != nullptr && encoderKeyShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderKeySeq_ = encoderKeyShape->GetStorageShape().GetDim(SEQ_DIM);
    }
    encoderValueSeq_ = encoderKeySeq_;
    size_t encoderValueIndex = static_cast<size_t>(InputIndexForward::ENCODER_VALUE_INDEX);
    const gert::StorageShape *encoderValueShape = context_.inputs[encoderValueIndex].shape;
    if (encoderValueShape != nullptr && encoderValueShape->GetStorageShape().GetDimNum() == INPUT_DIM_NUM) {
        encoderValueSeq_ = encoderValueShape->GetStorageShape().GetDim(SEQ_DIM);
    }

    size_t ropeSinIndex = static_cast<size_t>(InputIndexForward::ROPE_SIN_INDEX);
    const gert::StorageShape *ropeSinShape = context_.inputs[ropeSinIndex].shape;
    if (ropeSinShape != nullptr && ropeSinShape->GetStorageShape().GetDimNum() == ROPE_DIM_NUM) {
        ropeSeq_ = ropeSinShape->GetStorageShape().GetDim(ROPE_SEQ_DIM);
        ropeDim_ = ropeSinShape->GetStorageShape().GetDim(ROPE_DIM_DIM);
    }

    totalQuerySeq_ = querySeq_ + encoderQuerySeq_;
    totalKeySeq_ = keySeq_ + encoderKeySeq_;
    totalValueSeq_ = valueSeq_ + encoderValueSeq_;

    maxSeq_ = std::max({totalQuerySeq_, totalKeySeq_, totalValueSeq_});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeTilingKey()
{
    tilingKey_ =
        GET_TPL_TILING_KEY(static_cast<uint64_t>(*context_.normType), static_cast<uint64_t>(*context_.normAddedType),
                           static_cast<uint64_t>(*context_.ropeType), static_cast<uint64_t>(*context_.concatOrder),
                           static_cast<uint64_t>(*context_.isTraining));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeWorkSpace()
{
    size_t workspaceSize = compileInfo_.sysWorkspace;
    workspace_[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::FillTilingData()
{
    tilingData_.set_batch(batchSize_);
    tilingData_.set_querySeq(querySeq_);
    tilingData_.set_keySeq(keySeq_);
    tilingData_.set_valueSeq(valueSeq_);
    tilingData_.set_encoderQuerySeq(encoderQuerySeq_);
    tilingData_.set_encoderKeySeq(encoderKeySeq_);
    tilingData_.set_encoderValueSeq(encoderValueSeq_);
    tilingData_.set_totalQuerySeq(totalQuerySeq_);
    tilingData_.set_totalKeySeq(totalKeySeq_);
    tilingData_.set_totalValueSeq(totalValueSeq_);
    tilingData_.set_ropeActualSeq(ropeSeq_);
    tilingData_.set_splitHeadNum(splitHeadNum_);
    tilingData_.set_avgHeads(avgHeads_);
    tilingData_.set_tailHeads(tailHeads_);
    tilingData_.set_normDim(normDim_);
    tilingData_.set_ropeDim(ropeDim_);
    tilingData_.set_headDim(headDim_);
    tilingData_.set_headNum(headNum_);
    tilingData_.set_usedCore(usedCore_);
    tilingData_.set_eps(eps_);
    tilingData_.set_scale(scale_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::PrintTilingData(gert::TilingContext *ctx)
{
    OP_LOGD(ctx->GetNodeName(), "batch:               %ld.", tilingData_.get_batch());
    OP_LOGD(ctx->GetNodeName(), "querySeq:            %ld.", tilingData_.get_querySeq());
    OP_LOGD(ctx->GetNodeName(), "keySeq:              %ld.", tilingData_.get_keySeq());
    OP_LOGD(ctx->GetNodeName(), "valueSeq:            %ld.", tilingData_.get_valueSeq());
    OP_LOGD(ctx->GetNodeName(), "encoderQuerySeq:     %ld.", tilingData_.get_encoderQuerySeq());
    OP_LOGD(ctx->GetNodeName(), "encoderKeySeq:       %ld.", tilingData_.get_encoderKeySeq());
    OP_LOGD(ctx->GetNodeName(), "encoderValueSeq:     %ld.", tilingData_.get_encoderValueSeq());
    OP_LOGD(ctx->GetNodeName(), "totalQuerySeq:       %ld.", tilingData_.get_totalQuerySeq());
    OP_LOGD(ctx->GetNodeName(), "totalKeySeq:         %ld.", tilingData_.get_totalKeySeq());
    OP_LOGD(ctx->GetNodeName(), "totalValueSeq:       %ld.", tilingData_.get_totalValueSeq());
    OP_LOGD(ctx->GetNodeName(), "ropeActualSeq:       %ld.", tilingData_.get_ropeActualSeq());
    OP_LOGD(ctx->GetNodeName(), "splitHeadNum:        %ld.", tilingData_.get_splitHeadNum());
    OP_LOGD(ctx->GetNodeName(), "avgHeads:            %ld.", tilingData_.get_avgHeads());
    OP_LOGD(ctx->GetNodeName(), "tailHeads:           %ld.", tilingData_.get_tailHeads());
    OP_LOGD(ctx->GetNodeName(), "normDim:             %ld.", tilingData_.get_normDim());
    OP_LOGD(ctx->GetNodeName(), "ropeDim:             %ld.", tilingData_.get_ropeDim());
    OP_LOGD(ctx->GetNodeName(), "headDim:             %ld.", tilingData_.get_headDim());
    OP_LOGD(ctx->GetNodeName(), "headNum:             %ld.", tilingData_.get_headNum());
    OP_LOGD(ctx->GetNodeName(), "usedCore:            %ld.", tilingData_.get_usedCore());
    OP_LOGD(ctx->GetNodeName(), "eps:                 %f.", tilingData_.get_eps());
    OP_LOGD(ctx->GetNodeName(), "scale:               %f.", tilingData_.get_scale());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ConvertContextOne(gert::TilingContext *ctx, InputTensorInfo &x)
{
    size_t index = x.index;
    bool isRequired = x.isRequired;
    if (isRequired) {
        x.desc = ctx->GetInputDesc(index);
        x.shape = ctx->GetInputShape(index);
    } else {
        x.desc = ctx->GetOptionalInputDesc(index);
        x.shape = ctx->GetOptionalInputShape(index);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeCoreTilingStrategy()
{
    uint32_t maxSeq = std::max(querySeq_, keySeq_);
    uint32_t maxEncoderSeq = std::max(encoderQuerySeq_, encoderKeySeq_);
    uint32_t maxCoreSeq = std::max(maxSeq, maxEncoderSeq);
    usedCore_ = std::min(maxCoreSeq, compileInfo_.aivNum);

    numBlocks_ = usedCore_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::ComputeUBTilingStrategy()
{
    int64_t dataTypeSize = inputDtype_ == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    int64_t alignNum = BLOCK_SIZE / dataTypeSize;
    int64_t alignedNormDim_ = CeilAlign(normDim_, alignNum);
    int64_t alignedRopeDim_ = CeilAlign(ropeDim_, alignNum);
    int64_t normCoef = *(context_.normType) == static_cast<int64_t>(NormType::NONE) &&
                                *(context_.normAddedType) == static_cast<int64_t>(NormType::NONE) ?
                            0 :
                            1;
    int64_t ropeCoef = *(context_.ropeType) == static_cast<int64_t>(RopeType::NONE) ? 0 : 1;
    int64_t trainingCoef = *(context_.isTraining) ? 1 : 0;
    int64_t ropeUsedUbSize = SINGLE_BUFFER * alignedRopeDim_ * 2 * dataTypeSize + alignedRopeDim_ * sizeof(int32_t) +
                              alignedRopeDim_ * 2 * sizeof(float);
    int64_t normUsedUbSize = SINGLE_BUFFER * alignedNormDim_ * 2 * dataTypeSize + alignedNormDim_ * 2 * sizeof(float) +
                              MIN_SHARE_BUFFER;
    int64_t oneHeadUbSize =
        2 * alignedNormDim_ * sizeof(float) + 2 * DOUBLE_BUFFER * alignedNormDim_ * dataTypeSize +
        ropeCoef * ropeUsedUbSize +
        normCoef * (alignedNormDim_ * sizeof(float) + (trainingCoef * 2 * DOUBLE_BUFFER + 1) * sizeof(float));
    avgHeads_ = (compileInfo_.ubSize - ropeCoef * ropeUsedUbSize - normCoef * normUsedUbSize) / oneHeadUbSize;

    avgHeads_ = std::min(avgHeads_, static_cast<int64_t>(AVGHEAD_MAX_NUM));
    avgHeads_ = std::min(avgHeads_, headNum_);
    OP_CHECK_IF(avgHeads_ == 0, OP_LOGE(context_.opName, "input dim is too large"), return ge::GRAPH_FAILED);
    splitHeadNum_ = CeilDiv(headNum_, avgHeads_);
    tailHeads_ = headNum_ - (splitHeadNum_ - 1) * avgHeads_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::DoTiling(gert::TilingContext *ctx)
{
    auto platformInfoPtr = ctx->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    auto attrs = ctx->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(ctx, attrs);

    context_.opName = ctx->GetNodeName();
    for (size_t i = 0; i < context_.requiredInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = true;
        context_.inputs[i].type = FORWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }
    inputDtype_ = context_.inputs[0].desc->GetDataType();
    for (size_t i = context_.requiredInputNum; i < context_.maxInputNum; ++i) {
        context_.inputs[i].index = i;
        context_.inputs[i].isRequired = false;
        context_.inputs[i].type = FORWARD_INPUT_TYPE[i];
        ConvertContextOne(ctx, context_.inputs[i]);
    }
    context_.eps = attrs->GetAttrPointer<float>(static_cast<size_t>(AttrIndexForward::EPS_INDEX));
    context_.isTraining = attrs->GetAttrPointer<bool>(static_cast<size_t>(AttrIndexForward::IS_TRAINING_INDEX));
    context_.normType = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::NORM_TYPE_INDEX));
    context_.normAddedType = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::NORM_ADDED_TYPE_INDEX));
    context_.ropeType = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::ROPE_TYPE_INDEX));
    context_.concatOrder = attrs->GetAttrPointer<int64_t>(static_cast<size_t>(AttrIndexForward::CONCAT_ORDER_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.eps);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.isTraining);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.normType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.normAddedType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.ropeType);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, context_.concatOrder);

    size_t *workspaceSizes = ctx->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(ctx, workspaceSizes);
    workspace_ = workspaceSizes;

    if (GetShapeInfo() != ge::GRAPH_SUCCESS || CheckInput() != ge::GRAPH_SUCCESS ||
        ComputeTilingKey() != ge::GRAPH_SUCCESS || ComputeCoreTilingStrategy() != ge::GRAPH_SUCCESS ||
        ComputeUBTilingStrategy() != ge::GRAPH_SUCCESS || ComputeWorkSpace() != ge::GRAPH_SUCCESS ||
        FillTilingData() != ge::GRAPH_SUCCESS || PrintTilingData(ctx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ctx->SetTilingKey(tilingKey_);
    ctx->SetBlockDim(numBlocks_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NormRopeConcatTiling::SetTiling(gert::TilingContext *ctx)
{
    OP_CHECK_NULL_WITH_CONTEXT(ctx, ctx->GetRawTilingData());
    tilingData_.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
    ctx->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingNormRopeConcat(gert::TilingContext *ctx)
{
    OP_CHECK_IF(ctx == nullptr, OP_LOGE("NormRopeConcat", "context is nullptr"), return ge::GRAPH_FAILED);
    NormRopeConcatTiling tiling;
    if (tiling.DoTiling(ctx) == ge::SUCCESS) {
        tiling.SetTiling(ctx);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForNormRopeConcat(gert::TilingParseContext *context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto compileInfoPtr = context->GetCompiledInfo<NormRopeConcatCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    compileInfoPtr->sysWorkspace = ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NormRopeConcat)
    .Tiling(TilingNormRopeConcat)
    .TilingParse<NormRopeConcatCompileInfo>(TilingPrepareForNormRopeConcat);
} // namespace optiling

