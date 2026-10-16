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
 * \file offset_calculator.h
 * \brief
 */
#ifndef OFFSET_CALCULATOR_H
#define OFFSET_CALCULATOR_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "memcopy/gm_layout.h"
#include "memcopy/parser.h"
#include "memcopy/offset_calculator_v2.h"
#include "memcopy/fa_gm_tensor.h"
#include "memcopy/fa_l1_tensor.h"
#include "memcopy/gm_coord.h"

#endif
