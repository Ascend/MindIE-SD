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
 * \file prompt_flash_attention_tiling_struct.h
 * \brief
 */
#ifndef PROMPT_FLASH_ATTENTION_TILING_STRUCT_H
#define PROMPT_FLASH_ATTENTION_TILING_STRUCT_H
namespace optiling {

enum class InputLayout {
    SH,
    BSH,
    BNSD,
    NSD,
    BSND,
    BNSD_BSND,
    TND,
    NTD,
    NTD_TND,
    NZ,
    BBH,
    BNBD,
    NONE,
};

enum class TilingMod {
    CVSAME = 0,
    CVDIFF,
    CVDIFF_BASE_API,
    CVDIFF_MLA,
};

enum class SplitCoreMode {
    SPLIT_NBS_VECTOR = 0,
    SPLIT_NBS_CUBE,
    SPLIT_ONEN_VECTOR,
    SPLIT_ONEN_CUBE,
    BALANCE_VECTOR,
    BALANCE_CUBE,
    SPLIT_S1OUT_CUBE,
};
} // namespace optiling

#endif // PROMPT_FLASH_ATTENTION_TILING_STRUCT_H
