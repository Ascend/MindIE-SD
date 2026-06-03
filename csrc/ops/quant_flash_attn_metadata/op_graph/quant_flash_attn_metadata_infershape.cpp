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
 * \file quant_flash_attn_metadata_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "../op_kernel_aicpu/quant_flash_attn_metadata.h"
#include "quant_flash_attn_metadata_proto.h"

using namespace ge;

namespace ops {
static ge::graphStatus InferShapeQuantFlashAttnMetadata(gert::InferShapeContext *context) {
    OP_LOGD(context->GetNodeName(), "InferShapeQuantFlashAttnMetadata");

    gert::Shape *oShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, oShape);
    oShape->SetDimNum(1);
    oShape->SetDim(0, optiling::QFA_META_SIZE);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDtypeQuantFlashAttnMetadata(gert::InferDataTypeContext *context) {
    OP_LOGD(context->GetNodeName(), "InferDtypeQuantFlashAttnMetadata");

    context->SetOutputDataType(0, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantFlashAttnMetadata)
    .InferShape(InferShapeQuantFlashAttnMetadata)
    .InferDataType(InferDtypeQuantFlashAttnMetadata);
} // namespace ops
