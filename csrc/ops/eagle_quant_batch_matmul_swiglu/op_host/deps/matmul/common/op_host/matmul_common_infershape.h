/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_common_infershape.h
 * \brief
 */

#pragma once

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "qbmm_compat.h"

namespace Ops {
namespace EagleQbmm {
bool BroadcastBatchDim(const char *op_name, int64_t dim_a, int64_t dim_b, int64_t &dim_out);
ge::graphStatus InferShapeForBatchMatMul(gert::InferShapeContext *context, const int32_t attr_adj_idx,
    const size_t input_bias_index, const bool is_x2_packed = false, const char *fused_op_type = "");
ge::graphStatus InferShapeRangeForBatchMatMul(
    gert::InferShapeRangeContext *context, const int32_t attr_adj_idx, const size_t input_bias_index);
bool CheckIsUnknownDimNum(const gert::Shape &shape);
} // namespace NN
} // namespace Ops
