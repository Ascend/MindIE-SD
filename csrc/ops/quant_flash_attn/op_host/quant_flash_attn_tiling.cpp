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
 * \file quant_flash_attn_tiling.cpp
 * \brief QuantFlashAttn Tiling主入口
 */

#include <cmath>
#include "quant_flash_attn_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "quant_flash_attn_tiling_info.h"
#include "quant_flash_attn_tiling_info_parser.h"
#include "../../common/op_host/fia_tiling_templates_registry.h"

using namespace ge;

namespace optiling {

//参考训练算子公共info
struct QuantFlashAttnCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;
    uint64_t l2CacheSize;
    platform_ascendc::SocVersion socVersion;
    NpuArch npuArch;
};

ASCENDC_EXTERN_C ge::graphStatus TilingQuantFlashAttn(gert::TilingContext *context) {
    OP_LOGW(context, "QuantFlashAttn TilingQuantFlashAttn start.");

    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    QuantFlashAttnTilingInfo faInfo;
    QuantFlashAttnTilingInfoParser faInfoParser(context, faInfo);
    if (faInfoParser.Parse() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // FAChecker faChecker;
    // faChecker.Init(faInfo);
    // // Check函数只做校验，不能修改faInfo中的信息
    // if (faChecker.Process(faInfo) != ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_FAILED;
    // }

    // OP_LOGI(context, "QuantFlashAttn Tiling bSize:%d.", faInfo.bSize);
    return FiaTilingRegistry::GetInstance().DoTilingImpl(context, &faInfo);
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForQuantFlashAttn(gert::TilingParseContext *context) {
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<QuantFlashAttnCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2CacheSize);

    return ge::GRAPH_SUCCESS;
}

// 注册tiling函数：
IMPL_OP_OPTILING(QuantFlashAttn)
    .Tiling(TilingQuantFlashAttn)
    .TilingParse<QuantFlashAttnCompileInfo>(TilingPrepareForQuantFlashAttn);

} // namespace optiling
