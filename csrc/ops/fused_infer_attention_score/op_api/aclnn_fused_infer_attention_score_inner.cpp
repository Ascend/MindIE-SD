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

#include "fused_infer_attention_score_inner.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

const uint64_t INT4_NUMS_IN_INT32 = 8;

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue) {
    if (tensorListKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorListValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit.");
        return;
    }
    if ((*tensorListKey)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    if ((*tensorListValue)[0]->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kv's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensorList *>(tensorListKey);
    for (uint64_t i = 0; i < tempKey->Size(); i++) {
        if ((*tempKey)[i] != nullptr) {
            op::Shape viewShape = (*tempKey)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempKey)[i]->SetViewShape(viewShape);
            (*tempKey)[i]->SetDataType(DataType::DT_INT4);
        }
    }

    auto tempValue = const_cast<aclTensorList *>(tensorListValue);
    for (uint64_t i = 0; i < tempValue->Size(); i++) {
        if ((*tempValue)[i] != nullptr) {
            op::Shape viewShape = (*tempValue)[i]->GetViewShape();
            auto viewShapeDim = viewShape.GetDimNum();
            if (viewShapeDim >= 1) {
                viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
            }
            (*tempValue)[i]->SetViewShape(viewShape);
            (*tempValue)[i]->SetDataType(DataType::DT_INT4);
        }
    }

    OP_LOGD("The conversion of kv from int32 to int4 is completed.");
}

void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue) {
    if (tensorKey == nullptr) {
        OP_LOGD("TensorListKey is nullptr,TensorPreProcess exit.");
        return;
    }
    if (tensorValue == nullptr) {
        OP_LOGD("tensorListValue is nullptr,TensorPreProcess exit..");
        return;
    }
    if (tensorKey->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    if (tensorValue->GetDataType() != DataType::DT_INT32) {
        OP_LOGD("The conversion of kvPrefix's from OriginalShape is completed.");
        return;
    }
    auto tempKey = const_cast<aclTensor *>(tensorKey);
    op::Shape viewKeyShape = tempKey->GetViewShape();
    auto viewKeyShapeDim = viewKeyShape.GetDimNum();
    viewKeyShape[viewKeyShapeDim - 1] = viewKeyShape[viewKeyShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempKey->SetViewShape(viewKeyShape);
    tempKey->SetDataType(DataType::DT_INT4);

    auto tempValue = const_cast<aclTensor *>(tensorValue);
    op::Shape viewValueShape = tempValue->GetViewShape();
    auto viewValueShapeDim = viewValueShape.GetDimNum();
    viewValueShape[viewValueShapeDim - 1] = viewValueShape[viewValueShapeDim - 1] * INT4_NUMS_IN_INT32;
    tempValue->SetViewShape(viewValueShape);
    tempValue->SetDataType(DataType::DT_INT4);

    OP_LOGD("The conversion of kvPrefix from int32 to int4 is completed.");
}

aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outTensor) {
    OP_LOGD("start fake tensor");
    if (inArray != nullptr) {
        OP_LOGD("input array is not nullptr");
        int64_t size = static_cast<int64_t>(inArray->Size());
        std::vector<int64_t> shape = {size};
        outTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT64, nullptr, 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), nullptr);
        if (outTensor == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc tensor failed");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    OP_LOGD("end fake tensor");
    return ACLNN_SUCCESS;
}

void FusedInferAttentionScoreProcessSoftmaxLse(
    bool softmaxLseFlag, const aclTensor *softmaxLse, const aclTensor *&tempTensor, const aclTensor *&placeHolder) {
    if (softmaxLseFlag == false) {
        std::vector<int64_t> shape = {0};
        int64_t addr = 0xff;
        tempTensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, shape.data(), 0, ACL_FORMAT_ND,
            shape.data(), shape.size(), static_cast<void *>(&addr));
        placeHolder = tempTensor;
    } else {
        placeHolder = softmaxLse;
    }
}

aclnnStatus CheckKVContiguous(const aclTensorList *key, const aclTensorList *value) {
    if (!IsContiguous((*key)[0]) || !IsContiguous((*value)[0])) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
}
#endif
