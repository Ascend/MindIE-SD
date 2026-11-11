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

#define __ENABLE_ASCENDC_PRINTF__

#include "ada_relu2_block_estimate.h"
#include "ada_relu2_block_estimate_kernel_interface.cpp"

extern "C" __global__ __aicore__ void ada_relu2_block_estimate(GM_ADDR query, GM_ADDR key, GM_ADDR actualSeqLengths,
    GM_ADDR actualSeqLengthsKV, GM_ADDR actualSeqLengthsGm, GM_ADDR actualSeqLengthsKVGm, GM_ADDR qScale,
    GM_ADDR kScale, GM_ADDR score, GM_ADDR rowSum, GM_ADDR workspace, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

#if ORIG_DTYPE_QUERY == DT_INT8
    TILING_KEY_IS(ADA_RELU2_QBNSD_KVBNSD_INT8_KEY);
#else
    TILING_KEY_IS(ADA_RELU2_QBNSD_KVBNSD_KEY);
#endif
#if TILING_KEY_VAR == ADA_RELU2_QBNSD_KVBNSD_KEY
    AscendC::InitSocState();
    AdaRelu2Regular<DTYPE_QUERY, DTYPE_KEY, float>(
        query, key, actualSeqLengthsGm, actualSeqLengthsKVGm, qScale, kScale, score, rowSum, workspace, tiling);
#elif TILING_KEY_VAR == ADA_RELU2_QBNSD_KVBNSD_INT8_KEY
    AscendC::InitSocState();
    AdaRelu2Int8Quant<DTYPE_QUERY, DTYPE_KEY, float>(
        query, key, actualSeqLengthsGm, actualSeqLengthsKVGm, qScale, kScale, score, rowSum, workspace, tiling);
#endif
}
