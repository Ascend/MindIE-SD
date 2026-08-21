/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "eagle_quant_block_sparse_attention_tilingkey.h"
#include "eagle_quant_block_sparse_attention_kernel_interface.cpp"

extern "C" __global__ __aicore__ void eagle_quant_block_sparse_attention(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                                            __gm__ uint8_t* blockSparseMask, __gm__ uint8_t* mask, __gm__ uint8_t* blockShape,
                                                            __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* blockTable,
                                                            __gm__ uint8_t* query_scale, __gm__ uint8_t* key_scale, __gm__ uint8_t* value_scale,
                                                            __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    if (TILING_KEY_VAR >= RFA_BASE_TILING) {
    __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // 读取tilingKey进行kernel分发
    __gm__ EagleQuantBlockSparseAttentionTilingData *tilingDataPtr = 
        reinterpret_cast<__gm__ EagleQuantBlockSparseAttentionTilingData *>(tiling);
    uint64_t tilingKey = tilingDataPtr->tilingKey;

#if (__CCE_AICORE__ == 310)
    TILING_KEY_IS(QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
    TILING_KEY_IS(QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
    TILING_KEY_IS(QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY);
 	TILING_KEY_IS(QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY);
    #if TILING_KEY_VAR == QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<
                int8_t, float8_e4m3_t, half, half, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND>(
                    query, key, value, mask, blockTable, query_scale, key_scale, value_scale, attentionOut,
                    actualSeqLengths, actualSeqLengthsKv, blockSparseMask, user, tiling);
    #elif TILING_KEY_VAR == QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<
                int8_t, float8_e4m3_t, half, half, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD>(
                    query, key, value, mask, blockTable, query_scale, key_scale, value_scale, attentionOut,
                    actualSeqLengths, actualSeqLengthsKv, blockSparseMask, user, tiling);
    #elif TILING_KEY_VAR == QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY
 	         BsaInferIntfRegular<
 	                 int8_t, float8_e4m3_t, bfloat16_t, half, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND>(
 	                     query, key, value, mask, blockTable, query_scale, key_scale, value_scale, attentionOut,
 	                     actualSeqLengths, actualSeqLengthsKv, blockSparseMask, user, tiling);
    #elif TILING_KEY_VAR == QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY
        BsaInferIntfRegular<
                int8_t, float8_e4m3_t, bfloat16_t, half, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD>(
                    query, key, value, mask, blockTable, query_scale, key_scale, value_scale, attentionOut,
                    actualSeqLengths, actualSeqLengthsKv, blockSparseMask, user, tiling);
 	 
    #endif
#endif
    }
}

