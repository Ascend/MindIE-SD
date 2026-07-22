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
 * \file fia_tiling_data_fullquant.h
 * \brief
 */

#ifndef FIA_TILING_DATA_FULLQUANT_H_
#define FIA_TILING_DATA_FULLQUANT_H_

#include "../fia_tiling_data_public.h"

namespace optiling {

class FullQuantTiling {
  public:
    FiaBaseParams fiaBaseParams;
    FiaAttenMaskParams fiaAttenMaskParams;
    FiaPageAttentionParams fiaPageAttentionParams;
    FiaWorkspaceParams fiaWorkspaceParams;
    FiaS1OuterSplitCoreParams fiaS1OuterSplitCoreParams;
    FiaEmptyTensorParams fiaEmptyTensorParams;
};

class FusedInferAttentionScoreFullQuantTilingData {
  public:
    FullQuantTiling baseTiling;
    FiaMetaData fiaMetaData;
};

} // namespace optiling
#endif
