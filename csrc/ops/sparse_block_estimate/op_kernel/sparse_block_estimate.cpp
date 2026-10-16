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

#include "kernel_operator.h"
#include "sparse_block_estimate.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
#include "arch35/sparse_block_estimate_kernel.h"
#endif

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
#define INVOKE_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA(tiling_data, tiling); \
        templateClass<__VA_ARGS__> op; \
        op.Init(q, k, sparse_mask, sparse_cnt_table, workspace, tiling_data); \
        op.InitBuffers(); \
        op.Process(); \
    } while (0)
#else
#define INVOKE_GENERAL_OP_IMPL(templateClass, ...) \
    do { \
        GET_TILING_DATA(tiling_data, tiling); \
        templateClass<__VA_ARGS__> op; \
        REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tiling_data.cubeTilingData); \
        op.Init(q, k, actual_seq_len, actual_seq_len_kv, sparse_mask, sparse_cnt_table, workspace, tiling_data); \
        op.InitBuffers(); \
        op.Process(); \
    } while (0)
#endif

extern "C" __global__ __aicore__ void sparse_block_estimate(GM_ADDR q, GM_ADDR k, GM_ADDR actual_seq_len,
    GM_ADDR actual_seq_len_kv, GM_ADDR sparse_mask, GM_ADDR sparse_cnt_table, GM_ADDR workspace, GM_ADDR tiling) {
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TILING_KEY_IS(2000000000000000000);
    TILING_KEY_IS(2000000000000000100);
#else
    TILING_KEY_IS(1000000000000000000);
    TILING_KEY_IS(1000000000000000010);
    TILING_KEY_IS(1000000000000000020);
    TILING_KEY_IS(1000000000000000001);
    TILING_KEY_IS(1000000000000000011);
    TILING_KEY_IS(1000000000000000021);

    TILING_KEY_IS(1000000000000000100);
    TILING_KEY_IS(1000000000000000110);
    TILING_KEY_IS(1000000000000000120);
    TILING_KEY_IS(1000000000000000101);
    TILING_KEY_IS(1000000000000000111);
    TILING_KEY_IS(1000000000000000121);
#endif

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310) && TILING_KEY_VAR == 2000000000000000000
    INVOKE_GENERAL_OP_IMPL(SparseEstimateArch35::SparseBlockEstimateKernel, half);
#elif defined(__CCE_AICORE__) && (__CCE_AICORE__ == 310) && TILING_KEY_VAR == 2000000000000000100
    INVOKE_GENERAL_OP_IMPL(SparseEstimateArch35::SparseBlockEstimateKernel, bfloat16_t);
#elif TILING_KEY_VAR == 1000000000000000000
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BNSD, half, false>);
#elif TILING_KEY_VAR == 1000000000000000010
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BSH, half, false>);
#elif TILING_KEY_VAR == 1000000000000000020
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::TND, half, false>);
#elif TILING_KEY_VAR == 1000000000000000001
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BNSD, half, true>);
#elif TILING_KEY_VAR == 1000000000000000011
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BSH, half, true>);
#elif TILING_KEY_VAR == 1000000000000000021
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::TND, half, true>);
#elif TILING_KEY_VAR == 1000000000000000100
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BNSD, bfloat16_t, false>);
#elif TILING_KEY_VAR == 1000000000000000110
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BSH, bfloat16_t, false>);
#elif TILING_KEY_VAR == 1000000000000000120
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::TND, bfloat16_t, false>);
#elif TILING_KEY_VAR == 1000000000000000101
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BNSD, bfloat16_t, true>);
#elif TILING_KEY_VAR == 1000000000000000111
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::BSH, bfloat16_t, true>);
#elif TILING_KEY_VAR == 1000000000000000121
    INVOKE_GENERAL_OP_IMPL(SparseBlockEstimate, INVOKE_TYPE<INPUT_LAYOUT::TND, bfloat16_t, true>);
#endif
}
