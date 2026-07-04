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
 * \file fused_infer_attention_score_tiling_v4.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V4_OP_IMPL_FUSEDINFERATTENTIONSCORE_V4_H_
#define AIR_CXX_RUNTIME_V4_OP_IMPL_FUSEDINFERATTENTIONSCORE_V4_H_
#include "register/tilingdata_base.h"
#include "../../../common/op_host/fia_tiling_base.h"
#include "../../../common/op_host/fia_tiling_info.h"

namespace optiling {
ge::graphStatus TilingFusedInferAttentionScoreV4(gert::TilingContext *context);

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V4_OP_IMPL_FUSEDINFERATTENTIONSCORE_V4_H_
