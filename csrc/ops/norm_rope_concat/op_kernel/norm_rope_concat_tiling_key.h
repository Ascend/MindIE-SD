/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 *
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
 * \file norm_rope_concat_tiling_key.h
 * \brief Tiling key template arguments for NormRopeConcat kernel
 */

#ifndef _NORM_ROPE_CONCAT_TILING_KEY_H_
#define _NORM_ROPE_CONCAT_TILING_KEY_H_
#include "ascendc/host_api/tiling/template_argument.h"
ASCENDC_TPL_ARGS_DECL(NormRopeConcat, ASCENDC_TPL_UINT_DECL(NORM_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      ASCENDC_TPL_UINT_DECL(NORM_ADDED_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                      ASCENDC_TPL_UINT_DECL(ROPE_TYPE, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                      ASCENDC_TPL_UINT_DECL(CONCAT_ORDER, ASCENDC_TPL_1_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1),
                      ASCENDC_TPL_BOOL_DECL(IS_TRAINING, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(NORM_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                                     ASCENDC_TPL_UINT_SEL(NORM_ADDED_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 4),
                                     ASCENDC_TPL_UINT_SEL(ROPE_TYPE, ASCENDC_TPL_UI_RANGE, 1, 0, 2),
                                     ASCENDC_TPL_UINT_SEL(CONCAT_ORDER, ASCENDC_TPL_UI_RANGE, 1, 0, 1),
                                     ASCENDC_TPL_BOOL_SEL(IS_TRAINING, 0, 1)));
#endif // _NORM_ROPE_CONCAT_TILING_KEY_H_

