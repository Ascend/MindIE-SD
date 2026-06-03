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
 * \file quant_flash_attn_tiling.h
 * \brief QuantFlashAttn Tiling
 */

#ifndef QUANT_FLASH_ATTN_TILING_H_
#define QUANT_FLASH_ATTN_TILING_H_

#include <cstdint>
#include <register/op_impl_registry.h>

#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif

namespace optiling {

ASCENDC_EXTERN_C ge::graphStatus TilingQuantFlashAttn(gert::TilingContext *context);
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForQuantFlashAttn(gert::TilingParseContext *context);

} // namespace optiling

#endif // QUANT_FLASH_ATTN_TILING_H_
