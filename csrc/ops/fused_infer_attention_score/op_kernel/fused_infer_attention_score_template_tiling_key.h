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
 * \file fused_infer_attention_score_template_tiling_key.h
 * \brief
 */

#ifndef FUSED_INFER_ATTENTION_TEMPLATE_TILING_KEY_H_
#define FUSED_INFER_ATTENTION_TEMPLATE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#if __has_include("../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_tiling_regbase.h")
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_template_tiling_key_enum.h"
#include "../../prompt_flash_attention/op_kernel/arch35/prompt_flash_attention_tiling_regbase.h"
#include "../../incre_flash_attention/op_kernel/arch35/incre_flash_attention_tiling_regbase.h"
#else
#include "../prompt_flash_attention/arch35/prompt_flash_attention_template_tiling_key_enum.h"
#include "../prompt_flash_attention/arch35/prompt_flash_attention_tiling_regbase.h"
#include "../incre_flash_attention/arch35/incre_flash_attention_tiling_regbase.h"
#endif

#ifndef ORIG_DTYPE_QUERY
#define ORIG_DTYPE_QUERY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_KEY
#define ORIG_DTYPE_KEY (DT_BF16)
#endif

#ifndef ORIG_DTYPE_ATTENTION_OUT
#define ORIG_DTYPE_ATTENTION_OUT (DT_BF16)
#endif

ASCENDC_TPL_ARGS_DECL(EagleFusedInferAttentionScore,
    //    bit 8-1 InOutLayoutType(InputLayoutType-OutputLayoutType)
    //    0: InOutLayoutType_BNSD_BNSD
    //    1: InOutLayoutType_BSH_BSH
    //    2: InOutLayoutType_TND_TND
    ASCENDC_TPL_UINT_DECL(InOutLayoutType, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 255),
    //    bit 18-9 Config(S1,S2,D,DV)
    //    0: Config_S1Aligned64_S2Aligned256_DAligned64_DVAligned64
    //    1: Config_S1Aligned64_S2Aligned256_DAligned128_DVAligned128
    //    2: Config_S1Aligned128_S2Aligned128_DAligned64_DVAligned64
    //    3: Config_S1Aligned128_S2Aligned128_DAligned128_DVAligned128
    //    4: Config_S1Aligned128_S2Aligned128_DAligned192_DVAligned128
    //    5: Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned128
    //    6: Config_S1Aligned128_S2Aligned128_DAligned256_DVAligned256
    //    7: Config_S1Aligned128_S2Aligned128_DAligned512_DVAligned512
    //    8: Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64
    //    9: Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512
    //   10: Config_S1Aligned64_S2Aligned64_DAligned256_DVAligned256
    //   11: Config_S1Aligned64_S2Aligned64_DAligned512_DVAligned512
    //   12: Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64
    //   13: Config_S1Aligned16_S2Aligned512_DAligned128_DVAligned128
    //   14: Config_S1Aligned16_S2Aligned256_DAligned256_DVAligned256
    //   15: Config_S1Aligned16_S2Aligned128_DAligned512_DVAligned512
    //   16: Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64
    //   17: Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128
    //   18: Config_S1Aligned64_S2Aligned256_DAligned256_DVAligned256
    ASCENDC_TPL_UINT_DECL(Config, ASCENDC_TPL_10_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 1023),
    //    bit 22-19 PseMode
    //    0: PSE_MODE_PSE_OUTER_MUL_ADD_TYPE
    //    1: PSE_MODE_PSE_OUTER_ADD_MUL_TYPE
    //    2: PSE_MODE_PSE_INNER_MUL_ADD_TYPE: Not support
    //    3: PSE_MODE_PSE_INNER_MUL_ADD_SQRT_TYPE: Not support
    //    4: PSE_MODE_PSE_INVALID_TYPE
    //    9: PSE_MODE_PSE_NONE_TYPE
    ASCENDC_TPL_UINT_DECL(PseMode, ASCENDC_TPL_4_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 15),
    //    bit 27-23 QuantMode
    //    0: AntiquantMode_PER_CHANNEL
    //    2: AntiquantMode_K_PER_CHANNEL_V_PER_TOKEN
    //    6: AntiquantMode_K_PER_TOKEN
    //   17: FULLQUANT_MODE_QKV_PERBLOCK
    //   18: FULLQUANT_MODE_Q_PER_TOKEN_HEAD_KV_PER_TENSOR
    //   19: FULLQUANT_MODE_QKV_MXFP8_PREFILL
    //   20: FULLQUANT_MODE_QKV_MXFP8_DECODE
    //   21: FULLQUANT_MODE_QK_PER_TOKEN_HEAD_V_PER_HEAD
    //   30: FullQuantMode
    //   31: NoQuantMode
    ASCENDC_TPL_UINT_DECL(QuantMode, ASCENDC_TPL_5_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 31),
    //    bit 28 HasAttenMask
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasAttenMask, false, true),
    //    bit 29 HasRope
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(HasRope, false, true),
    //    bit 31-30 KvLayoutType
    /*
        To ensure template compatibility and reduce total tilingkey, KvLayoutType is assigned different optional values for kernel templates before/after refactoring.
        1. For refactored template(e.g. GQA non quant), KvLayoutType is template parameterized, KvLayoutType_NO_PA, KvLayoutType_PA_BBH, KvLayoutType_PA_BNBD, KvLayoutType_PA_NZ are required.
        2. For unrefactored templates, KvLayoutType_NO_PA and KvLayoutType_ENABLE_PA are recommended to reduce tilingkey and keep compatibility, KvLayoutType has same meaning as the original isPa.
    */
    //    0: KvLayoutType_NO_PA
    //    1: KvLayoutType_ENABLE_PA
    //    1: KvLayoutType_PA_BBH
    //    2: KvLayoutType_PA_BNBD
    //    3: KvLayoutType_PA_NZ
    ASCENDC_TPL_UINT_DECL(KvLayoutType, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_RANGE, 1, 0, 3),
    //    bit 32 IsFd
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(IsFd, false, true),
    //    bit 33 EmptyTensor
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EmptyTensor, false, true),
    //    bit 34 EnableKVPrefix
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EnableKVPrefix, false, true),
    //    bit 35 EnableS1OutSplit
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(EnableS1OutSplit, false, true),
    //    bit 36 IsReconstructTemp
    //    0: false
    //    1: true
    ASCENDC_TPL_BOOL_DECL(IsReconstructTemp, false, true), );

ASCENDC_TPL_SEL(
#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && \
    ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16)
    // fp8 per-block
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD,
                             InOutLayoutType_BSH_BSH, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64,
            Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FULLQUANT_MODE_QKV_PERBLOCK),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), ASCENDC_TPL_BOOL_SEL(HasRope, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA), ASCENDC_TPL_BOOL_SEL(IsFd, 0),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false), ASCENDC_TPL_BOOL_SEL(IsReconstructTemp, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)),
#endif

#if (ORIG_DTYPE_QUERY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_KEY == DT_FLOAT8_E4M3FN && ORIG_DTYPE_ATTENTION_OUT == DT_BF16)
    // fp8 per-block
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, InOutLayoutType_BNSD_BNSD,
                             InOutLayoutType_BSH_BSH, InOutLayoutType_NTD_NTD),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, Config_S1Aligned128_S2Aligned256_DAligned64_DVAligned64,
            Config_S1Aligned128_S2Aligned256_DAligned128_DVAligned128),
        ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, PSE_MODE_PSE_NONE_TYPE),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, FULLQUANT_MODE_QKV_PERBLOCK),
        ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0), ASCENDC_TPL_BOOL_SEL(HasRope, 0),
        ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA), ASCENDC_TPL_BOOL_SEL(IsFd, 0),
        ASCENDC_TPL_BOOL_SEL(EmptyTensor, 0), ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, false),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false), ASCENDC_TPL_BOOL_SEL(IsReconstructTemp, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)),
#endif
    // 空tensor，必须要有一个使得列表不为空，否则会报无法推导模板参数的错误
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(InOutLayoutType, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(Config, ASCENDC_TPL_UI_LIST, 0), ASCENDC_TPL_UINT_SEL(PseMode, ASCENDC_TPL_UI_LIST, 0),
        ASCENDC_TPL_UINT_SEL(QuantMode, ASCENDC_TPL_UI_LIST, NoQuantMode), ASCENDC_TPL_BOOL_SEL(HasAttenMask, 0),
        ASCENDC_TPL_BOOL_SEL(HasRope, 0), ASCENDC_TPL_UINT_SEL(KvLayoutType, ASCENDC_TPL_UI_LIST, KvLayoutType_NO_PA),
        ASCENDC_TPL_BOOL_SEL(IsFd, 0), ASCENDC_TPL_BOOL_SEL(EmptyTensor, 1), ASCENDC_TPL_BOOL_SEL(EnableKVPrefix, 0),
        ASCENDC_TPL_BOOL_SEL(EnableS1OutSplit, false), ASCENDC_TPL_BOOL_SEL(IsReconstructTemp, false),
        ASCENDC_TPL_TILING_STRUCT_SEL(FlashAttentionScoreSimplifiedTilingData)), );
#endif
