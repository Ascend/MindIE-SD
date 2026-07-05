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

/*!
 * \file fused_infer_attention_score_apt.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
// ifa must include before pfa
#define FIA_ENABLE_MLA
#include "fused_infer_attention_score_template_tiling_key.h"
#if __has_include("../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h")
#include "../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_entry_regbase.h"
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_entry_regbase.h"
#else
#include "../incre_flash_attention/arch35/incre_flash_attention_entry_regbase.h"
#include "../prompt_flash_attention/arch35/prompt_flash_attention_entry_regbase.h"
#endif
#include "fused_infer_attention_score_tilingkey.h"

#define FullQuantTiling 15
template <uint8_t inOutLayoutType, uint16_t config, uint8_t pseMode, uint8_t quantMode, bool hasAttenMask, bool hasRope,
    uint8_t KvLayoutType, bool isFd, bool emptyTensor, bool enableKVPrefix, bool enableS1OutSplit,
    bool isReconstructTemp>
__global__ __aicore__ void eagle_fused_infer_attention_score(__gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *value, __gm__ uint8_t *pse_shift, __gm__ uint8_t *attenMask, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *deq_scale1, __gm__ uint8_t *quant_scale1,
    __gm__ uint8_t *deq_scale2, __gm__ uint8_t *quant_scale2, __gm__ uint8_t *quant_offset2,
    __gm__ uint8_t *antiquantScale, __gm__ uint8_t *antiquantOffset, __gm__ uint8_t *blocktable,
    __gm__ uint8_t *queryPaddingSize, __gm__ uint8_t *kvPaddingSize, __gm__ uint8_t *keyAntiquantScale,
    __gm__ uint8_t *keyAntiquantOffset, __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *valueAntiquantOffset,
    __gm__ uint8_t *keySharedPrefix, __gm__ uint8_t *valueSharedPrefix, __gm__ uint8_t *actualSharedPrefixLen,
    __gm__ uint8_t *queryRope, __gm__ uint8_t *keyRope, __gm__ uint8_t *keyRopeAntiquantScale,
    __gm__ uint8_t *dequantScaleQuery, __gm__ uint8_t *learnableSink, __gm__ uint8_t *qStartIdx,
    __gm__ uint8_t *kvStartIdx, __gm__ uint8_t *attentionOut, __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling) {
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    REGISTER_TILING_DEFAULT(FusedInferAttentionScoreTilingData);
#endif
    if (quantMode >= FullQuantTiling) {
        //pfa 模板
        prompt_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope,
            KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit, isReconstructTemp>(query, key, value,
            pse_shift, attenMask, actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2,
            quant_scale2, quant_offset2, antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize,
            keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, keySharedPrefix,
            valueSharedPrefix, actualSharedPrefixLen, queryRope, keyRope, dequantScaleQuery, learnableSink,
            attentionOut, softmaxLse, workspace, tiling);
    } else {
        //ifa 模板
        incre_flash_attention_FIAS_regbase<inOutLayoutType, config, pseMode, quantMode, hasAttenMask, hasRope,
            KvLayoutType, isFd, emptyTensor, enableKVPrefix, enableS1OutSplit>(query, key, value, pse_shift, attenMask,
            actualSeqLengths, actualSeqLengthsKV, deq_scale1, quant_scale1, deq_scale2, quant_scale2, quant_offset2,
            antiquantScale, antiquantOffset, blocktable, queryPaddingSize, kvPaddingSize, keyAntiquantScale,
            keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset, keySharedPrefix, valueSharedPrefix,
            actualSharedPrefixLen, attentionOut, softmaxLse, workspace, tiling);
    }
}
