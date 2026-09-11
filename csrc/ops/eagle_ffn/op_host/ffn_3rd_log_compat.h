/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ffn_3rd_log_compat.h
 * \brief vendored matmul tiling 的日志适配层：OP_LOG 系列、OPS_LOG 系列、OPS_ERR_IF
 *        直接复用仓内轻量日志（csrc/ops/utils/inc）。
 */
#ifndef FFN_3RD_LOG_COMPAT_H_
#define FFN_3RD_LOG_COMPAT_H_

#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#include "exe_graph/runtime/shape.h"
#include "graph/types.h"
#include "log/log.h"
#include "error/ops_error.h"

#ifndef CUBE_INNER_ERR_REPORT
#define CUBE_INNER_ERR_REPORT(opName, errMsg, ...) OP_LOGE(opName, errMsg, ##__VA_ARGS__)
#endif

#ifndef OP_TILING_CHECK
#define OP_TILING_CHECK(cond, log_func, expr) \
    do {                                      \
        if (cond) {                           \
            log_func;                         \
            expr;                             \
        }                                     \
    } while (0)
#endif

// vendored matmul tiling 日志格式化引用；定义在 3rd/common/src/tiling_util.cpp
namespace Ops {
namespace Base {
std::string ToString(const gert::Shape &shape);
std::string ToString(ge::DataType dtype);
} // namespace Base
} // namespace Ops

#endif // FFN_3RD_LOG_COMPAT_H_
