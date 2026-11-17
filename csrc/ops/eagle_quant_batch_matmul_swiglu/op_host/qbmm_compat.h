#pragma once

#include <sstream>
#include <string>
#include "log/log.h"
#include "log/error_code.h"

#ifndef OP_TILING_CHECK
#define OP_TILING_CHECK(condition, log, action) \
    do { \
        if (condition) { \
            log; \
            action; \
        } \
    } while (0)
#endif
#ifndef CUBE_INNER_ERR_REPORT
#define CUBE_INNER_ERR_REPORT(name, fmt, ...) OP_LOGE(name, fmt, ##__VA_ARGS__)
#endif
#ifndef OP_LOGE_WITHOUT_REPORT
#define OP_LOGE_WITHOUT_REPORT(name, fmt, ...) OP_LOGE(name, fmt, ##__VA_ARGS__)
#endif
#ifndef OP_LOGE_IF
#define OP_LOGE_IF(condition, result, name, fmt, ...) \
    do { \
        if (condition) { \
            OP_LOGE(name, fmt, ##__VA_ARGS__); \
            return result; \
        } \
    } while (0)
#endif
#ifndef OPS_CHECK_NULL_WITH_CONTEXT
#define OPS_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    OP_TILING_CHECK((pointer) == nullptr, OP_LOGE(context, "Null pointer: %s", #pointer), return ge::GRAPH_FAILED)
#endif

namespace eagle_qbmm_log {
template <typename Shape> inline std::string ToString(const Shape &shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i) {
            out << ',';
        }
        out << shape.GetDim(i);
    }
    out << ']';
    return out.str();
}
}
