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
#include "eagle_block_sparse_attention_tilingkey.h"
#include "eagle_block_sparse_attention_kernel_interface.cpp"

extern "C" __global__ __aicore__ void eagle_block_sparse_attention(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockSparseMask,
    __gm__ uint8_t *mask, __gm__ uint8_t *blockShape, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKv, __gm__ uint8_t *blockTable, __gm__ uint8_t *qDequantScale,
    __gm__ uint8_t *kDequantScale, __gm__ uint8_t *vDequantScale, __gm__ uint8_t *pQuantScale,
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    if (TILING_KEY_VAR >= BSA_BASE_TILING) {
        __gm__ uint8_t *user = AscendC::GetUserWorkspace(workspace);
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        // 读取tilingKey进行kernel分发
        __gm__ EagleBlockSparseAttentionTilingData *tilingDataPtr =
            reinterpret_cast<__gm__ EagleBlockSparseAttentionTilingData *>(tiling);
        uint64_t tilingKey = tilingDataPtr->tilingKey;

#if (__CCE_AICORE__ == 220)
        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING);
        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT);
        TILING_KEY_IS(QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_BSA_TILING_LSE_OUT);
        TILING_KEY_IS(QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT);
        TILING_KEY_IS(QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_BSA_TILING_LSE_OUT);
        TILING_KEY_IS(QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT);

#if TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING
        BlockSparse::EagleBlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<half, float, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_TND_TND_NOCACHE_HALFSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<half, half, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_TND_TND_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<half, float, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_BNSD_BNSD_NOCACHE_HALFSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<half, half, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_BNSD_BNSD_NOCACHE_FLOATSM_NOMASK_BSA_TILING_LSE_OUT
        BlockSparse::EagleBlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::OUT_ONLY, 1, 1>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#else
        BlockSparse::EagleBlockSparseAttentionInfer<half, half, Epilogue::LseMode::NONE, 0, 0>(
            query, key, value, blockSparseMask, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv,
            blockShape, user, softmaxLse, tiling);
#endif
#endif
#if (__CCE_AICORE__ == 310)
        // 非量化+混合精度softmax
        TILING_KEY_IS(QF16_KVF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF16_KVF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF16_KVF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        // 非量化+混合精度softmax+lse
        TILING_KEY_IS(QF16_KVF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF16_KVF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF16_KVF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QBF16_KVBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);
        // 全量化FP8+混合精度softmax
        TILING_KEY_IS(QF8_KVF8_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF8_KVF8_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF8_KVF8_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_KEY);
        // 全量化FP8+混合精度softmax+lse
        TILING_KEY_IS(QF8_KVF8_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF8_KVF8_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF8_KVF8_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);
        TILING_KEY_IS(QF8_KVF8_OBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY);

        // mxfp4 OCP 量化(mxfp4 OCP + mixed SM + BF16 out)
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OBF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OBF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OBF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        // mxfp4 OCP 量化(mxfp4 OCP + mixed SM + FP16 out)
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4OCP_KVMFP4OCP_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        // mxfp4 CX 量化(mxfp4 CX + mixed SM + BF16 out)
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OBF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OBF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OBF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        // mxfp4 CX 量化(mxfp4 CX + mixed SM + FP16 out)
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY);
        TILING_KEY_IS(QMFP4CX_KVMFP4CX_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY);

#if TILING_KEY_VAR == QF16_KVF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::BNS1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::BSND, BsaKernelArch35::Format::BSND,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::BSN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::BNS1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::BSND, BsaKernelArch35::Format::BSND,
                            Epilogue::LseMode::NONE, Epilogue::LseFormat::BSN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
// lse
#elif TILING_KEY_VAR == QF16_KVF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::TN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::TND, BsaKernelArch35::Format::TND,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::TN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::BNS1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::BNSD, BsaKernelArch35::Format::BNSD,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::BNS1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF16_KVF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<half, half, float, BsaKernelArch35::Format::BSND, BsaKernelArch35::Format::BSND,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::BSN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QBF16_KVBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferIntfRegular<bfloat16_t, bfloat16_t, float, BsaKernelArch35::Format::BSND, BsaKernelArch35::Format::BSND,
                            Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::BSN1>(
            query, key, value, mask, blockTable, attentionOut, actualSeqLengths, actualSeqLengthsKv, blockSparseMask,
            user, softmaxLse, tiling);
// 全量化FP8场景
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::TND,
                                   BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::BNSD,
                                   BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE, Epilogue::LseFormat::BNS1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::BSND,
                                   BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE, Epilogue::LseFormat::BSN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::TND,
                                   BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::BNSD,
                                   BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE, Epilogue::LseFormat::BNS1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::BSND,
                                   BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE, Epilogue::LseFormat::BSN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
// 全量化FP8+lse
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::TND,
                                   BsaKernelArch35::Format::TND, Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::TN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::BNSD,
                                   BsaKernelArch35::Format::BNSD, Epilogue::LseMode::OUT_ONLY,
                                   Epilogue::LseFormat::BNS1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, half, float, BsaKernelArch35::Format::BSND,
                                   BsaKernelArch35::Format::BSND, Epilogue::LseMode::OUT_ONLY,
                                   Epilogue::LseFormat::BSN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QTND_KVTND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::TND,
                                   BsaKernelArch35::Format::TND, Epilogue::LseMode::OUT_ONLY, Epilogue::LseFormat::TN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QBNSD_KVBNSD_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::BNSD,
                                   BsaKernelArch35::Format::BNSD, Epilogue::LseMode::OUT_ONLY,
                                   Epilogue::LseFormat::BNS1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QF8_KVF8_OBF16_QBSND_KVBSND_NOCACHE_SMBF16_REF32_NOMASK_LSE_OUT_KEY
        BsaInferInterfaceFullQuant<fp8_e4m3fn_t, bfloat16_t, float, BsaKernelArch35::Format::BSND,
                                   BsaKernelArch35::Format::BSND, Epilogue::LseMode::OUT_ONLY,
                                   Epilogue::LseFormat::BSN1>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
// mxfp4全量化+dn
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OBF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::TND,
                                        BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1,
                                        Epilogue::MXQuantMode::OCP, Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OBF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::BNSD,
                                        BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BNS1, Epilogue::MXQuantMode::OCP,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OBF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::BSND,
                                        BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BSN1, Epilogue::MXQuantMode::OCP,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
// mxfp4 OCP 量化(mxfp4 OCP + mixed SM + FP16 out)
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::TND,
                                        BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1,
                                        Epilogue::MXQuantMode::OCP, Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::BNSD,
                                        BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BNS1, Epilogue::MXQuantMode::OCP,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4OCP_KVMFP4OCP_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::BSND,
                                        BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BSN1, Epilogue::MXQuantMode::OCP,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
// mxfp4 CX 量化(mxfp4 CX + mixed SM + BF16 out)
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OBF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::TND,
                                        BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1,
                                        Epilogue::MXQuantMode::CX, Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OBF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::BNSD,
                                        BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BNS1, Epilogue::MXQuantMode::CX,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OBF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, bfloat16_t, BsaKernelArch35::Format::BSND,
                                        BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BSN1, Epilogue::MXQuantMode::CX,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
// mxfp4 CX 量化(mxfp4 CX + mixed SM + FP16 out)
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OF16_QTND_KVTND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::TND,
                                        BsaKernelArch35::Format::TND, Epilogue::LseMode::NONE, Epilogue::LseFormat::TN1,
                                        Epilogue::MXQuantMode::CX, Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OF16_QBNSD_KVBNSD_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::BNSD,
                                        BsaKernelArch35::Format::BNSD, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BNS1, Epilogue::MXQuantMode::CX,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#elif TILING_KEY_VAR == QMFP4CX_KVMFP4CX_OF16_QBSND_KVBSND_NOCACHE_SMF16_REF32_NOMASK_KEY
        BsaInferInterfaceMXFP4FullQuant<fp4x2_e2m1_t, half, float, half, BsaKernelArch35::Format::BSND,
                                        BsaKernelArch35::Format::BSND, Epilogue::LseMode::NONE,
                                        Epilogue::LseFormat::BSN1, Epilogue::MXQuantMode::CX,
                                        Gemm::Tile::CopyL0CToUBMode::SPLIT_N, true>(
            query, key, value, blockSparseMask, mask, blockTable, actualSeqLengths, actualSeqLengthsKv, qDequantScale,
            kDequantScale, vDequantScale, attentionOut, user, softmaxLse, tiling);
#endif
#endif
    }
}
