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
 * \file quant_flash_attn_template_tiling_key.h
 * \brief
 */

#ifndef TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
#define TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_

#include "ascendc/host_api/tiling/template_argument.h"
#include "quant_flash_attn_tiling_data.h"

#define ASCENDC_TPL_5_BW 5
#define ASCENDC_TPL_10_BW 10
#define ASCENDC_TPL_3_BW 3

#define LAYOUT_ENUM_BSND 0
#define LAYOUT_ENUM_BNSD 1
#define LAYOUT_ENUM_TND 3
#define LAYOUT_ENUM_BNSD_BSND 4

#define KV_STORAGE_MODE_CONTINUE 0
#define KV_STORAGE_MODE_PA_BSND 1
#define KV_STORAGE_MODE_PA_BNSD 2

enum class QFA_LAYOUT : uint32_t {
    BSND = LAYOUT_ENUM_BSND,
    BNSD = LAYOUT_ENUM_BNSD,
    TND = LAYOUT_ENUM_TND,
};

template <typename QUANT_T, typename SCALE_T, typename OUT_T, const bool PAGE_ATTENTION = false,
    QFA_LAYOUT LAYOUT_T = QFA_LAYOUT::BSND, QFA_LAYOUT KV_LAYOUT_T = QFA_LAYOUT::BSND,
    QFA_LAYOUT OUT_LAYOUT_T = QFA_LAYOUT::BSND, const bool HAS_MASK = false, typename... Args>
struct QFAType {
    using quantType = QUANT_T;
    using scaleType = SCALE_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr QFA_LAYOUT qLayout = LAYOUT_T;
    static constexpr QFA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr QFA_LAYOUT outLayout = KV_LAYOUT_T;
    static constexpr bool hasMask = HAS_MASK;
};

ASCENDC_TPL_ARGS_DECL(QuantFlashAttn,
    //    bit 0-7 QueryLayout
    //    0: BSND
    //    1: BNSD
    //    3: TND
    //    4: BNSD_BSND
    ASCENDC_TPL_UINT_DECL(queryOutLayout, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND, LAYOUT_ENUM_BNSD,
        LAYOUT_ENUM_TND, LAYOUT_ENUM_BNSD_BSND),
    //    bit 8-15 kvLayout
    //    0: CONTINUE, kvLayout = qLayout
    //    1: PA_BBND
    //    2: PA_BNBD
    ASCENDC_TPL_UINT_DECL(kvStorageMode, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST, KV_STORAGE_MODE_CONTINUE,
        KV_STORAGE_MODE_PA_BSND, KV_STORAGE_MODE_PA_BNSD),
    //    bit 16 HasMask
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(hasMask, 0, 1), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(queryOutLayout, ASCENDC_TPL_UI_LIST, LAYOUT_ENUM_BSND,
                                         LAYOUT_ENUM_BNSD, LAYOUT_ENUM_TND, LAYOUT_ENUM_BNSD_BSND),
                    ASCENDC_TPL_UINT_SEL(kvStorageMode, ASCENDC_TPL_UI_LIST, KV_STORAGE_MODE_CONTINUE,
                        KV_STORAGE_MODE_PA_BSND, KV_STORAGE_MODE_PA_BNSD),
                    ASCENDC_TPL_BOOL_SEL(hasMask, 0, 1), ASCENDC_TPL_TILING_STRUCT_SEL(QuantFlashAttnTilingData)), );

#endif // TEMPLATE_TILING_KEY_QUANT_FLASH_ATTN_H_
