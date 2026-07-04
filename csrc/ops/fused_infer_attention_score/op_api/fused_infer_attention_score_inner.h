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

#ifndef ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_FUSED_INFER_ATTENTION_SCORE_INNER_H_
#define ACLNN_API __attribute__((visibility("default")))

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

void TensorPreProcess(const aclTensorList *&tensorListKey, const aclTensorList *&tensorListValue);
void PrefixTensorPreProcess(const aclTensor *&tensorKey, const aclTensor *&tensorValue);
aclnnStatus FakeArray(const aclIntArray *inArray, aclTensor *&outArray);

void FusedInferAttentionScoreProcessSoftmaxLse(
    bool softmaxLseFlag, const aclTensor *softmaxLse, const aclTensor *&tempTensor, const aclTensor *&placeHolder);

aclnnStatus CheckKVContiguous(const aclTensorList *key, const aclTensorList *value);

// 新版本opbase存在TensorV2的新接口，用弱符号判断当前opbase是新版本还是旧版本，旧版本不支持传入非连续tensor
bool NnopbaseSupportTensorV2() __attribute__((weak));

#ifdef __cplusplus
}
#endif

#endif
