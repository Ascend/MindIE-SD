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
 * \file fused_infer_attention_score_tiling_v4.cpp
 * \brief
 */

#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "fused_infer_attention_score_tiling_v4.h"
#include "fused_infer_attention_score_tiling_impl.h"
#include "../fused_infer_attention_score_tiling_info_parser.h"
#include "../checkers/fia_checker.h"

#include "log/log.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
ge::graphStatus TilingFusedInferAttentionScoreV4(gert::TilingContext *context) {
    // Parse -> Check -> DoOpTiling
    FiaTilingInfo fiaInfo;
    FiaInfoParser fiaInfoParser(context);
    FusedInferAttentionScoreTilingImpl fiav4(context);
    if (fiaInfoParser.Parse(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    FIAChecker fiaChecker;
    fiaChecker.Init(fiaInfo);
    // Check函数只做校验，不能修改fiaInfo中的信息
    if (fiaChecker.Process(fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (FiaTilingRegistry::GetInstance().DoTilingImpl(context, &fiaInfo) == ge::GRAPH_SUCCESS) {
        return ge::GRAPH_SUCCESS;
    } else { // 假设，老的模板也注册，把else分支和下面的逻辑删掉
        OP_LOGD(context, "reconstruct template do not support, routing to old template.");
    }

    if (fiav4.DoOpTiling(context, fiaInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
