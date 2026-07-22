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
 * \file fused_infer_attention_score_tiling_register.cpp
 * \brief
 */

#include "fused_infer_attention_score_tiling.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
static ge::graphStatus TilingPrepareForFusedInferAttentionScore(gert::TilingParseContext * /* context */) {
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(EagleFusedInferAttentionScore)
    .TilingInputsDataDependency({ACTUAL_SEQ_Q_INDEX, ACTUAL_SEQ_KV_INDEX, QUERY_PADDING_SIZE_INDEX,
                                    KV_PADDING_SIZE_INDEX, ACTUAL_SHARED_PREFIX_LEN_INDEX},
        {gert::TilingPlacement::TILING_ON_HOST, gert::TilingPlacement::TILING_ON_AICPU})
    .Tiling(DoOpTilingEagleFusedInferAttentionScore)
    .TilingParse<FusedInferAttentionScoreCompileInfo>(
        TilingPrepareForFusedInferAttentionScore); // Register entrance functions to the framework

} // namespace optiling
