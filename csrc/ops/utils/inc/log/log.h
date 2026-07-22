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

#include <sstream>
#include <string>

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

#ifdef __cplusplus
namespace mindiesd_ops_log {
template <typename T> inline std::string ToLogString(const T &value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string ToLogString(const std::string &value) { return value; }

inline std::string ToLogString(const char *value) { return value == nullptr ? "null" : std::string(value); }
} // namespace mindiesd_ops_log

#ifndef OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
#define OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, paramName, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s is invalid. Reason: %s.", _paramName.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_WITH_INVALID_INPUT
#define OP_LOGE_WITH_INVALID_INPUT(opName, paramName) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        OP_LOGE(opName, "Parameter %s is invalid.", _paramName.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE
#define OP_LOGE_FOR_INVALID_SHAPE(opName, paramName, incorrectShape, correctShape) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectShape = ::mindiesd_ops_log::ToLogString(incorrectShape); \
        const auto _correctShape = ::mindiesd_ops_log::ToLogString(correctShape); \
        OP_LOGE(opName, "Parameter %s has incorrect shape %s. It should be %s.", _paramName.c_str(), \
            _incorrectShape.c_str(), _correctShape.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, paramName, incorrectShape, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectShape = ::mindiesd_ops_log::ToLogString(incorrectShape); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s has incorrect shape %s. Reason: %s.", _paramName.c_str(), \
            _incorrectShape.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName, paramNames, incorrectShapes, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectShapes = ::mindiesd_ops_log::ToLogString(incorrectShapes); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect shapes %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectShapes.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM
#define OP_LOGE_FOR_INVALID_SHAPEDIM(opName, paramName, incorrectDim, correctDim) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectDim = ::mindiesd_ops_log::ToLogString(incorrectDim); \
        const auto _correctDim = ::mindiesd_ops_log::ToLogString(correctDim); \
        OP_LOGE(opName, "Parameter %s has incorrect shape dim %s. It should be %s.", _paramName.c_str(), \
            _incorrectDim.c_str(), _correctDim.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, paramName, incorrectDim, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectDim = ::mindiesd_ops_log::ToLogString(incorrectDim); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s has incorrect shape dim %s. Reason: %s.", _paramName.c_str(), \
            _incorrectDim.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, paramNames, incorrectDims, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectDims = ::mindiesd_ops_log::ToLogString(incorrectDims); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect shape dims %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectDims.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE
#define OP_LOGE_FOR_INVALID_SHAPESIZE(opName, paramName, incorrectSize, correctSize) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectSize = ::mindiesd_ops_log::ToLogString(incorrectSize); \
        const auto _correctSize = ::mindiesd_ops_log::ToLogString(correctSize); \
        OP_LOGE(opName, "Parameter %s has incorrect shape size %s. It should be %s.", _paramName.c_str(), \
            _incorrectSize.c_str(), _correctSize.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName, paramName, incorrectSize, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectSize = ::mindiesd_ops_log::ToLogString(incorrectSize); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s has incorrect shape size %s. Reason: %s.", _paramName.c_str(), \
            _incorrectSize.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON
#define OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(opName, paramNames, incorrectSizes, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectSizes = ::mindiesd_ops_log::ToLogString(incorrectSizes); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect shape sizes %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectSizes.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_FORMAT
#define OP_LOGE_FOR_INVALID_FORMAT(opName, paramName, incorrectFormat, correctFormat) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectFormat = ::mindiesd_ops_log::ToLogString(incorrectFormat); \
        const auto _correctFormat = ::mindiesd_ops_log::ToLogString(correctFormat); \
        OP_LOGE(opName, "Parameter %s has incorrect format %s. It should be %s.", _paramName.c_str(), \
            _incorrectFormat.c_str(), _correctFormat.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON
#define OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName, paramNames, incorrectFormats, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectFormats = ::mindiesd_ops_log::ToLogString(incorrectFormats); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect formats %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectFormats.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE
#define OP_LOGE_FOR_INVALID_DTYPE(opName, paramName, incorrectDtype, correctDtype) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectDtype = ::mindiesd_ops_log::ToLogString(incorrectDtype); \
        const auto _correctDtype = ::mindiesd_ops_log::ToLogString(correctDtype); \
        OP_LOGE(opName, "Parameter %s has incorrect dtype %s. It should be %s.", _paramName.c_str(), \
            _incorrectDtype.c_str(), _correctDtype.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, paramName, incorrectDtype, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectDtype = ::mindiesd_ops_log::ToLogString(incorrectDtype); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s has incorrect dtype %s. Reason: %s.", _paramName.c_str(), \
            _incorrectDtype.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON
#define OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(opName, paramNames, incorrectDtypes, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectDtypes = ::mindiesd_ops_log::ToLogString(incorrectDtypes); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect dtypes %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectDtypes.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_TENSORNUM
#define OP_LOGE_FOR_INVALID_TENSORNUM(opName, paramName, incorrectNum, correctNum) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectNum = ::mindiesd_ops_log::ToLogString(incorrectNum); \
        const auto _correctNum = ::mindiesd_ops_log::ToLogString(correctNum); \
        OP_LOGE(opName, "Parameter %s has invalid tensor num %s. It should be %s.", _paramName.c_str(), \
            _incorrectNum.c_str(), _correctNum.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON
#define OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(opName, paramNames, incorrectNums, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectNums = ::mindiesd_ops_log::ToLogString(incorrectNums); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have invalid tensor nums %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectNums.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_LISTSIZE
#define OP_LOGE_FOR_INVALID_LISTSIZE(opName, paramName, incorrectSize, correctSize) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectSize = ::mindiesd_ops_log::ToLogString(incorrectSize); \
        const auto _correctSize = ::mindiesd_ops_log::ToLogString(correctSize); \
        OP_LOGE(opName, "Parameter %s has incorrect element num %s. It should be %s.", _paramName.c_str(), \
            _incorrectSize.c_str(), _correctSize.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE
#define OP_LOGE_FOR_INVALID_VALUE(opName, paramName, incorrectValue, correctValue) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectValue = ::mindiesd_ops_log::ToLogString(incorrectValue); \
        const auto _correctValue = ::mindiesd_ops_log::ToLogString(correctValue); \
        OP_LOGE(opName, "Parameter %s has incorrect value %s. It should be %s.", _paramName.c_str(), \
            _incorrectValue.c_str(), _correctValue.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUE_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, paramName, incorrectValue, reason) \
    do { \
        const auto _paramName = ::mindiesd_ops_log::ToLogString(paramName); \
        const auto _incorrectValue = ::mindiesd_ops_log::ToLogString(incorrectValue); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameter %s has incorrect value %s. Reason: %s.", _paramName.c_str(), \
            _incorrectValue.c_str(), _reason.c_str()); \
    } while (0)
#endif

#ifndef OP_LOGE_FOR_INVALID_VALUES_WITH_REASON
#define OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, paramNames, incorrectValues, reason) \
    do { \
        const auto _paramNames = ::mindiesd_ops_log::ToLogString(paramNames); \
        const auto _incorrectValues = ::mindiesd_ops_log::ToLogString(incorrectValues); \
        const auto _reason = ::mindiesd_ops_log::ToLogString(reason); \
        OP_LOGE(opName, "Parameters %s have incorrect values %s. Reason: %s.", _paramNames.c_str(), \
            _incorrectValues.c_str(), _reason.c_str()); \
    } while (0)
#endif
#endif // __cplusplus

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
