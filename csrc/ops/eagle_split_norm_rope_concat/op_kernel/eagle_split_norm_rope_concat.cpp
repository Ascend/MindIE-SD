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

#include "kernel_operator.h"
#include "eagle_split_norm_rope_concat_head128.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void eagle_split_norm_rope_concat(GM_ADDR qkv, GM_ADDR head_counts,
    GM_ADDR norm_query_weight, GM_ADDR norm_key_weight, GM_ADDR rope_cos, GM_ADDR rope_sin, GM_ADDR query, GM_ADDR key,
    GM_ADDR value, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(data, tiling);
    // head_counts is value-dependent host metadata; its values are carried in tiling.
    if (TILING_KEY_IS(11)) {
        eagle_snrc::EagleSplitNormRopeConcatHead128<half> op;
        op.Init(qkv, norm_query_weight, norm_key_weight, rope_cos, rope_sin, query, key, value, data);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        eagle_snrc::EagleSplitNormRopeConcatHead128<bfloat16_t> op;
        op.Init(qkv, norm_query_weight, norm_key_weight, rope_cos, rope_sin, query, key, value, data);
        op.Process();
    }
}
