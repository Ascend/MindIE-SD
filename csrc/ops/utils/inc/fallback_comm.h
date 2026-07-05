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
 * \file fallback_comm.h
 * \brief
 */

#ifndef INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
#define INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_

#include "aclnn/aclnn_base.h"
#include "exe_graph/runtime/op_execute_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/op_impl_registry.h"
#include "runtime/base.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {

aclDataType ToAclDataType(ge::DataType dtype);
} // namespace fallback

#ifdef __cplusplus
}
#endif

#endif // INC_EXTERNAL_GRAPH_FALLBACK_COMMON_H_
