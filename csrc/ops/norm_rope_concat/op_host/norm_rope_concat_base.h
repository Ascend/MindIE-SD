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
 * \file norm_rope_concat_base.h
 * \brief Host-side base definitions for NormRopeConcat operator
 */
#ifndef _NORM_ROPE_CONCAT_HOST_BASE_H_
#define _NORM_ROPE_CONCAT_HOST_BASE_H_
#include <cstddef>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>

namespace nrc {
constexpr size_t INPUT_DIM_NUM = 4;  // B, S, H, D
constexpr size_t ROPE_DIM_NUM = 2;   // S, D
constexpr size_t OUTPUT_DIM_NUM = 4; // B, H, S, D
constexpr size_t BATCH_DIM = 0;
constexpr size_t SEQ_DIM = 1;
constexpr size_t HEAD_DIM = 2;
constexpr size_t DIM_DIM = 3;
constexpr size_t GRAD_SEQ_DIM = 2;
constexpr size_t GRAD_HEAD_DIM = 1;
constexpr size_t ROPE_SEQ_DIM = 0;
constexpr size_t ROPE_DIM_DIM = 1;
constexpr size_t T_SEQ_DIM = 2;
constexpr size_t T_HEAD_DIM = 1;
constexpr size_t TOTAL_NORM_NUM = 8;
constexpr size_t NUM_TWO = 2;
constexpr size_t NUM_THREE = 3;
constexpr size_t AVGHEAD_MAX_NUM = 4096;
constexpr size_t MIN_SHARE_BUFFER = 256;
constexpr size_t RES_UB_BUFFER = 1024;

enum class NormType : int64_t {
    NONE = 0,
    LAYER_NORM,
    LAYER_NORM_AFFINE,
    RMS_NORM,
    RMS_NORM_AFFINE,
    LAYER_NORM_ACROSS_HEADS,
    LAYER_NORM_AFFINE_ACROSS_HEADS,
    RMS_NORM_ACROSS_HEADS,
    RMS_NORM_AFFINE_ACROSS_HEADS,
    L2_NORM,
};

inline bool IsNormTypeValid(int64_t normType)
{
    return normType >= static_cast<int64_t>(NormType::NONE) &&
        normType <= static_cast<int64_t>(NormType::RMS_NORM_AFFINE);
}

enum class RopeType : int64_t {
    NONE = 0,
    INTERLEAVE,
    HALF,
};

inline bool IsRopeTypeValid(int64_t ropeType)
{
    return ropeType >= static_cast<int64_t>(RopeType::NONE) && ropeType <= static_cast<int64_t>(RopeType::HALF);
}

enum class ConcatOrder : int64_t {
    BEFORE_ENCODER = 0,
    AFTER_ENCODER,
};

inline bool IsConcatOrderValid(int64_t concatOrder)
{
    return concatOrder >= static_cast<int64_t>(ConcatOrder::BEFORE_ENCODER) &&
           concatOrder <= static_cast<int64_t>(ConcatOrder::AFTER_ENCODER);
}

enum class InputIndexForward : size_t {
    QUERY_INDEX = 0,
    KEY_INDEX,
    VALUE_INDEX,
    ENCODER_QUERY_INDEX,
    ENCODER_KEY_INDEX,
    ENCODER_VALUE_INDEX,
    NORM_QUERY_WEIGHT_INDEX,
    NORM_QUERY_BIAS_INDEX,
    NORM_KEY_WEIGHT_INDEX,
    NORM_KEY_BIAS_INDEX,
    NORM_ADDED_QUERY_WEIGHT_INDEX,
    NORM_ADDED_QUERY_BIAS_INDEX,
    NORM_ADDED_KEY_WEIGHT_INDEX,
    NORM_ADDED_KEY_BIAS_INDEX,
    ROPE_SIN_INDEX,
    ROPE_COS_INDEX
};

enum class OutputIndexForward : size_t {
    QUERY_OUTPUT_INDEX = 0,
    KEY_OUTPUT_INDEX,
    VALUE_OUTPUT_INDEX,
    NORM_QUERY_MEAN_INDEX,
    NORM_QUERY_RSTD_INDEX,
    NORM_KEY_MEAN_INDEX,
    NORM_KEY_RSTD_INDEX,
    NORM_ADDED_QUERY_MEAN_INDEX,
    NORM_ADDED_QUERY_RSTD_INDEX,
    NORM_ADDED_KEY_MEAN_INDEX,
    NORM_ADDED_KEY_RSTD_INDEX
};

enum class AttrIndexForward : size_t {
    NORM_TYPE_INDEX = 0,
    NORM_ADDED_TYPE_INDEX,
    ROPE_TYPE_INDEX,
    CONCAT_ORDER_INDEX,
    EPS_INDEX,
    IS_TRAINING_INDEX,
};

enum class InputIndexBackward : size_t {
    GRAD_QUERY_OUTPUT_INDEX = 0,
    GRAD_KEY_OUTPUT_INDEX,
    GRAD_VALUE_OUTPUT_INDEX,
    QUERY_INDEX,
    KEY_INDEX,
    ENCODER_QUERY_INDEX,
    ENCODER_KEY_INDEX,
    NORM_QUERY_WEIGHT_INDEX,
    NORM_QUERY_MEAN_INDEX,
    NORM_QUERY_RSTD_INDEX,
    NORM_KEY_WEIGHT_INDEX,
    NORM_KEY_MEAN_INDEX,
    NORM_KEY_RSTD_INDEX,
    NORM_ADDED_QUERY_WEIGHT_INDEX,
    NORM_ADDED_QUERY_MEAN_INDEX,
    NORM_ADDED_QUERY_RSTD_INDEX,
    NORM_ADDED_KEY_WEIGHT_INDEX,
    NORM_ADDED_KEY_MEAN_INDEX,
    NORM_ADDED_KEY_RSTD_INDEX,
    ROPE_SIN_INDEX,
    ROPE_COS_INDEX
};

enum class OutputIndexBackward : size_t {
    GRAD_QUERY_INDEX = 0,
    GRAD_KEY_INDEX,
    GRAD_VALUE_INDEX,
    GRAD_ENCODER_QUERY_INDEX,
    GRAD_ENCODER_KEY_INDEX,
    GRAD_ENCODER_VALUE_INDEX,
    GRAD_NORM_QUERY_WEIGHT_INDEX,
    GRAD_NORM_QUERY_BIAS_INDEX,
    GRAD_NORM_KEY_WEIGHT_INDEX,
    GRAD_NORM_KEY_BIAS_INDEX,
    GRAD_NORM_ADDED_QUERY_WEIGHT_INDEX,
    GRAD_NORM_ADDED_QUERY_BIAS_INDEX,
    GRAD_NORM_ADDED_KEY_WEIGHT_INDEX,
    GRAD_NORM_ADDED_KEY_BIAS_INDEX
};

enum class AttrIndexBackward : size_t {
    NORM_TYPE_INDEX = 0,
    NORM_ADDED_TYPE_INDEX,
    ROPE_TYPE_INDEX,
    CONCAT_ORDER_INDEX
};
} // namespace nrc

#endif // _NORM_ROPE_CONCAT_HOST_BASE_H_

