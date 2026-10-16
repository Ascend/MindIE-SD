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
 * \file fia_tiling_data_noquant_gqa.h
 * \brief
 */

#ifndef FIA_TILING_DATA_NOQUANT_GQA_H_
#define FIA_TILING_DATA_NOQUANT_GQA_H_

#include "../fia_tiling_data_public.h"

namespace optiling {

class NoQuantTilingArch35 {
  public:
    FiaBaseParams fiaBaseParams;
    FiaAttenMaskParams fiaAttenMaskParams;
    FiaPseParams fiaPseParams;
    FiaSystemPrefixParams fiaSystemPrefixParams;
    FiaPageAttentionParams fiaPageAttentionParams;
    FiaLeftPaddingParams fiaLeftPaddingParams;
    FiaPostQuantParams fiaPostQuantParams;
    FiaWorkspaceParams fiaWorkspaceParams;
    FiaS1OuterSplitCoreParams fiaS1OuterSplitCoreParams;
    FiaEmptyTensorParams fiaEmptyTensorParams;
};

class FusedInferAttentionScoreTilingData {
  public:
    NoQuantTilingArch35 baseTiling;
    FiaMetaData fiaMetaData;
};

} // namespace optiling
#endif
