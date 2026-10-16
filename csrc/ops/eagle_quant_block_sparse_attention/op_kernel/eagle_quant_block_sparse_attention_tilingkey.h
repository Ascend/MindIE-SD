/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILINGKEY_H_
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILINGKEY_H_

/**
 * EagleQuantBlockSparseAttention TilingKey 定义
 *
 * TilingKey编码规则 (64-bit):
 * 位域: AAAABBBBCCCCDDDDEEEE
 * - [0-1]   Q Layout: 2=TND, 3=BNSD
 * - [2-4]   Mask Type: 0=NoMask, 3=CausalMask
 * - [5-7]   Softmax Precision: 0=Float, 1=Half
 * - [8-10]  PagedCache Flag: 0=NoCache, 1=WithCache
 * - [11-13] KV Layout: 00=TND, 20=BNSD
 * - [14-15] Data Type: 00=FP16, 22=BF16
 * - [16-18] Operator Category: 900=EagleQuantBlockSparseAttention
 */

#define RFA_BASE_TILING 9000000000000000

#if (__CCE_AICORE__ == 310)

#define QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY 9050010030444442
#define QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY 9050010050444443
#define QKINT8_VFP8E4M3_QTND_KVTND_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY 9050010030455552
#define QKINT8_VFP8E4M3_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_OBF16_NOMASK_KEY 9050010050455553

#endif
#endif // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILINGKEY_H_
