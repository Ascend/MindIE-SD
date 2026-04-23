/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef GE_OP_FLASH_ATTENTION_TIK_H
#define GE_OP_FLASH_ATTENTION_TIK_H
#include "graph/operator_reg.h"
namespace ge {

REG_OP(FlashAttentionTik)
    .INPUT(q, TensorType({DT_FLOAT16}))
    .INPUT(k, TensorType({DT_FLOAT16}))
    .INPUT(v, TensorType({DT_FLOAT16}))
    .OUTPUT(y, TensorType({DT_FLOAT16}))
    .OP_END_FACTORY_REG(FlashAttentionTik)
}
#endif
