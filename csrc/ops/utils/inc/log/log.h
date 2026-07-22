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

#pragma once

#include "log/ops_log.h"

#ifndef OPS_UTILS_LOG_SUB_MOD_NAME
#define OPS_UTILS_LOG_SUB_MOD_NAME "UNKNOWN"
#endif

#ifndef OPS_UTILS_LOG_PACKAGE_TYPE
#define OPS_UTILS_LOG_PACKAGE_TYPE ""
#endif

#ifndef OP_LOGD
#define OP_LOGD(context, fmt, ...) OPS_LOG_D(context, fmt, ##__VA_ARGS__)
#endif

#ifndef OP_LOGI
#define OP_LOGI(context, fmt, ...) OPS_LOG_I(context, fmt, ##__VA_ARGS__)
#endif

#ifndef OP_LOGW
#define OP_LOGW(context, fmt, ...) OPS_LOG_W(context, fmt, ##__VA_ARGS__)
#endif

#ifndef OP_LOGE
#define OP_LOGE(context, fmt, ...) OPS_LOG_E_WITHOUT_REPORT(context, fmt, ##__VA_ARGS__)
#endif

#ifndef KERNEL_LOG_ERROR
#define KERNEL_LOG_ERROR(fmt, ...) OP_LOGE("aicpu", fmt, ##__VA_ARGS__)
#endif

#ifndef OP_CHECK_IF
#define OP_CHECK_IF(COND, LOG_FUNC, EXPR) OP_CHECK(COND, LOG_FUNC, EXPR)
#endif

#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    OP_CHECK((ptr) == nullptr, OP_LOGE(context, "%s is nullptr", #ptr), return ge::GRAPH_FAILED)
#endif
