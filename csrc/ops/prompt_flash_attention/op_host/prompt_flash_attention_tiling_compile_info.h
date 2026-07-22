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
 * \file prompt_flash_attention_tiling_compile_info.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H
#define PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H
#include <queue>
#include "exe_graph/runtime/tiling_context.h"
#include "op_host/data_copy_transpose_tiling_def.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {

struct PromptFlashAttentionCompileInfo {
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;
    size_t defaultSysWorkspaceSize;
    platform_ascendc::SocVersion socShortName;
};

}

#endif // PROMPT_FLASH_ATTENTION_TILING_STRUCT_COMPILE_INFO_H
