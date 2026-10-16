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
 * \file norm_rope_concat_tiling.h
 * \brief Tiling data and class for NormRopeConcat operator
 */
#ifndef _NORM_ROPE_CONCAT_TILING_H_
#define _NORM_ROPE_CONCAT_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"
#include "norm_rope_concat_base.h"
#include "../op_kernel/norm_rope_concat_tiling_key.h"

#ifndef ASCENDC_EXTERN_C
#ifdef ASCENDC_OP_TEST
#define ASCENDC_EXTERN_C extern "C"
#else
#define ASCENDC_EXTERN_C
#endif
#endif

template <typename T>
inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T>
static T CeilDiv(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

namespace optiling {
struct NormRopeConcatCompileInfo {
    uint32_t aivNum;
    uint64_t ubSize;
    uint64_t sysWorkspace;
};

BEGIN_TILING_DATA_DEF(NormRopeConcatTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch)
TILING_DATA_FIELD_DEF(int64_t, querySeq)
TILING_DATA_FIELD_DEF(int64_t, keySeq)
TILING_DATA_FIELD_DEF(int64_t, valueSeq)
TILING_DATA_FIELD_DEF(int64_t, encoderQuerySeq)
TILING_DATA_FIELD_DEF(int64_t, encoderKeySeq)
TILING_DATA_FIELD_DEF(int64_t, encoderValueSeq)
TILING_DATA_FIELD_DEF(int64_t, totalQuerySeq)
TILING_DATA_FIELD_DEF(int64_t, totalKeySeq)
TILING_DATA_FIELD_DEF(int64_t, totalValueSeq)
TILING_DATA_FIELD_DEF(int64_t, ropeActualSeq)
TILING_DATA_FIELD_DEF(int64_t, splitHeadNum)
TILING_DATA_FIELD_DEF(int64_t, avgHeads)
TILING_DATA_FIELD_DEF(int64_t, tailHeads)
TILING_DATA_FIELD_DEF(int64_t, normDim)
TILING_DATA_FIELD_DEF(int64_t, ropeDim)
TILING_DATA_FIELD_DEF(int64_t, headNum)
TILING_DATA_FIELD_DEF(int64_t, headDim)
TILING_DATA_FIELD_DEF(int64_t, usedCore)
TILING_DATA_FIELD_DEF(float, eps)
TILING_DATA_FIELD_DEF(float, scale)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(NormRopeConcat, NormRopeConcatTilingData)

enum class TensorType {
    INPUT_TENSOR,
    ENCODER_TENSOR,
    NORM_TENSOR,
    ROPE_TENSOR,
};

struct InputTensorInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
    TensorType type;
    size_t index;
    bool isRequired;
};

constexpr size_t MAX_FORWARD_INPUT_NUM = 16;
constexpr size_t MAX_BACKWARD_INPUT_NUM = 21;
constexpr size_t REQUIRED_FORWARD_INPUT_NUM = 3;
constexpr size_t REQUIRED_BACKWARD_INPUT_NUM = 5;
static constexpr TensorType FORWARD_INPUT_TYPE[MAX_FORWARD_INPUT_NUM] = {
    TensorType::INPUT_TENSOR,   // query
    TensorType::INPUT_TENSOR,   // key
    TensorType::INPUT_TENSOR,   // value
    TensorType::ENCODER_TENSOR, // encoder_query
    TensorType::ENCODER_TENSOR, // encoder_key
    TensorType::ENCODER_TENSOR, // encoder_value
    TensorType::NORM_TENSOR,    // norm_query_weight
    TensorType::NORM_TENSOR,    // norm_query_bias
    TensorType::NORM_TENSOR,    // norm_key_weight
    TensorType::NORM_TENSOR,    // norm_key_bias
    TensorType::NORM_TENSOR,    // norm_added_query_weight
    TensorType::NORM_TENSOR,    // norm_added_query_bias
    TensorType::NORM_TENSOR,    // norm_added_key_weight
    TensorType::NORM_TENSOR,    // norm_added_key_bias
    TensorType::ROPE_TENSOR,    // rope_sin
    TensorType::ROPE_TENSOR,    // rope_cos
};

constexpr int64_t IS_TRAINING_TILING_KEY = 1;
constexpr int64_t CONCAT_ORDER_TILING_KEY = 10;
constexpr int64_t ROPE_TYPE_TILING_KEY = 100;
constexpr int64_t NORM_ADDED_TYPE_TILING_KEY = 100000;
constexpr int64_t NORM_TYPE_TILING_KEY = 100000000;

constexpr int64_t SINGLE_BUFFER = 1;
constexpr int64_t DOUBLE_BUFFER = 1;
constexpr int64_t BLOCK_SIZE = 32;

struct NormRopeConcatContext {
    const char *opName;

    InputTensorInfo inputs[MAX_FORWARD_INPUT_NUM];

    const size_t requiredInputNum{REQUIRED_FORWARD_INPUT_NUM};
    const size_t maxInputNum{MAX_FORWARD_INPUT_NUM};

    const int64_t *normType;
    const int64_t *normAddedType;
    const int64_t *ropeType;
    const int64_t *concatOrder;
    const float *eps;
    const bool *isTraining;
};


class NormRopeConcatTiling {
public:
    NormRopeConcatTiling() = default;
    ~NormRopeConcatTiling() = default;

    ge::graphStatus DoTiling(gert::TilingContext *ctx);
    ge::graphStatus SetTiling(gert::TilingContext *ctx);

private:
    ge::graphStatus GetShapeInfo();
    ge::graphStatus CheckInput();
    ge::graphStatus CheckInput(const InputTensorInfo &x);
    ge::graphStatus ComputeTilingKey();
    ge::graphStatus ComputeCoreTilingStrategy();
    ge::graphStatus ComputeUBTilingStrategy();
    ge::graphStatus ComputeWorkSpace();
    ge::graphStatus FillTilingData();
    ge::graphStatus PrintTilingData(gert::TilingContext *ctx);
    ge::graphStatus ConvertContextOne(gert::TilingContext *ctx, InputTensorInfo &x);

private:
    int64_t batchSize_{0};
    int64_t headNum_{0};
    int64_t headDim_{0};
    int64_t querySeq_{0};
    int64_t keySeq_{0};
    int64_t valueSeq_{0};
    int64_t encoderQuerySeq_{0};
    int64_t encoderKeySeq_{0};
    int64_t encoderValueSeq_{0};
    int64_t totalQuerySeq_{0};
    int64_t totalKeySeq_{0};
    int64_t totalValueSeq_{0};
    int64_t ropeSeq_{0};
    int64_t ropeDim_{0};
    int64_t normDim_{0};
    int64_t splitHeadNum_{1};
    int64_t usedCore_{0};
    int64_t maxSeq_{0};
    int64_t avgHeads_{0};
    int64_t tailHeads_{0};
    float scale_{1.0f};
    float eps_{1e-5f};

    uint32_t numBlocks_{0};
    int64_t tilingKey_{0};
    size_t *workspace_{nullptr};
    ge::DataType inputDtype_{ge::DT_FLOAT};

    NormRopeConcatContext context_;
    NormRopeConcatTilingData tilingData_;

    NormRopeConcatCompileInfo compileInfo_;
};

} // namespace optiling

#endif // _NORM_ROPE_CONCAT_TILING_H_

