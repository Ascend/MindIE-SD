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
 * \file norm_rope_concat.cpp
 * \brief Kernel entry for NormRopeConcat fused operator
 */

#include "norm_rope_concat.h"
#include "norm_rope_concat_tiling_key.h"
using namespace nrc;

template <uint8_t NORM_TYPE, uint8_t ADDED_NORM_TYPE, uint8_t ROPE_TYPE, uint8_t CONCAT_ORDER, bool IS_TRAINING>
__global__ __aicore__ void
norm_rope_concat(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR encoder_query, GM_ADDR encoder_key,
                 GM_ADDR encoder_value, GM_ADDR norm_query_weight, GM_ADDR norm_query_bias, GM_ADDR norm_key_weight,
                 GM_ADDR norm_key_bias, GM_ADDR norm_added_query_weight, GM_ADDR norm_added_query_bias,
                 GM_ADDR norm_added_key_weight, GM_ADDR norm_added_key_bias, GM_ADDR rope_sin, GM_ADDR rope_cos,
                 GM_ADDR query_output, GM_ADDR key_output, GM_ADDR value_output, GM_ADDR norm_query_mean,
                 GM_ADDR norm_query_rstd, GM_ADDR norm_key_mean, GM_ADDR norm_key_rstd, GM_ADDR norm_added_query_mean,
                 GM_ADDR norm_added_query_rstd, GM_ADDR norm_added_key_mean, GM_ADDR norm_added_key_rstd,
                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    constexpr NormType normType = static_cast<NormType>(NORM_TYPE);
    constexpr NormType addedNormType = static_cast<NormType>(ADDED_NORM_TYPE);
    constexpr RopeType ropeType = static_cast<RopeType>(ROPE_TYPE);
    constexpr ConcatOrder concatOrder = static_cast<ConcatOrder>(CONCAT_ORDER);
    NormRopeConcat<normType, addedNormType, ropeType, concatOrder, IS_TRAINING> op(
        query, key, value, encoder_query, encoder_key, encoder_value, norm_query_weight, norm_query_bias,
        norm_key_weight, norm_key_bias, norm_added_query_weight, norm_added_query_bias, norm_added_key_weight,
        norm_added_key_bias, rope_sin, rope_cos, query_output, key_output, value_output, norm_query_mean,
        norm_query_rstd, norm_key_mean, norm_key_rstd, norm_added_query_mean, norm_added_query_rstd,
        norm_added_key_mean, norm_added_key_rstd);
    op.Init(&pipe, tilingData);
    op.Process();
}

