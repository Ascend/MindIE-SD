/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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
 * \file eagle_quant_batch_matmul_swiglu.cpp
 * \brief
 */

#include "eagle_quant_batch_matmul_swiglu.h"
#include "eagle_quant_batch_matmul_swiglu_init_output.h"
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#include "eagle_quant_batch_matmul_swiglu_cube_basic.h"
#if (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_SCALE == DT_FLOAT)
#include "eagle_quant_batch_matmul_swiglu_bf16_basic.h"
#include "eagle_quant_batch_matmul_swiglu_bf16.h"
#include "eagle_quant_batch_matmul_swiglu_bf16_opt.h"
#include "eagle_quant_batch_matmul_swiglu_pertoken.h"
#include "eagle_quant_batch_matmul_swiglu_pertoken_basic.h"
#include "eagle_quant_batch_matmul_swiglu_pertoken_opt.h"
#endif
#endif
#include "kernel_operator.h"
#include "eagle_quant_batch_matmul_swiglu_tiling_key.h"
#include "eagle_quant_batch_matmul_swiglu_kernel_tiling_data.h"

// if run with ttk without bias, can't get DTYPE_BIAS macro
#undef DTYPE_BIAS
#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
// s8->s32
#define DTYPE_BIAS int32_t
#else
// fp8/hif8->fp32
#define DTYPE_BIAS float
#endif

#if (defined(ORIG_DTYPE_X1) && defined(ORIG_DTYPE_SCALE))
#define SUPPORT_PERBLOCK \
    (ORIG_DTYPE_SCALE == DT_FLOAT && \
        (ORIG_DTYPE_X1 == DT_HIFLOAT8 || ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN || ORIG_DTYPE_X1 == DT_FLOAT8_E5M2))
#else
#define SUPPORT_PERBLOCK false
#endif

#undef ORIG_DTYPE_PERTOKEN_SCALE
#undef DTYPE_PERTOKEN_SCALE
#define ORIG_DTYPE_PERTOKEN_SCALE DT_FLOAT
#define DTYPE_PERTOKEN_SCALE float

using namespace AscendC;
using namespace matmul;

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

#if defined(ORIG_DTYPE_X1) && defined(DT_INT8) && ORIG_DTYPE_X1 == DT_INT8
#define DTYPE_LOC_LOCAL int32_t
#else
#define DTYPE_LOC_LOCAL float
#endif

#define INVOKE_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_CUBE_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        EagleQuantBatchMatmulSwigluBaseKernel<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, \
            transposeX1, transposeX2, EagleQuantBatchMatmulSwigluUpdate> \
            op; \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling); \
        BmmDequantBf16<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling); \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        BmmDequantBf16Opt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> \
            op; \
        op.Init(x1, x2, bias, scale, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        const TCubeTiling *mmTiling = &(qBmmV3TilingData->matmulTiling); \
        BmmDequantPertoken<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, transposeX2> \
            op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, mmTiling); \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        BmmDequantPertokenOpt<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, DTYPE_SCALE, DTYPE_Y, transposeX1, \
            transposeX2> \
            op; \
        op.Init(x1, x2, bias, scale, pertokenScale, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        BmmDequantPertokenBasic<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, \
            transposeX2, EagleQuantBatchMatmulSwigluUpdate> \
            op; \
        op.Init(x1, x2, scale, bias, pertokenScale, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        const EagleQuantBatchMatmulSwigluTilingData *qBmmV3TilingData = &tilingData; \
        BmmBasicDequantBf16<DTYPE_X1, DTYPE_X2, DTYPE_SCALE, DTYPE_Y, FORMAT_X1, FORMAT_X2, transposeX1, transposeX2, \
            EagleQuantBatchMatmulSwigluUpdate> \
            op; \
        op.Init(x1, x2, scale, bias, y, user1, qBmmV3TilingData, &tPipe); \
        op.Process(); \
        tPipe.Destroy(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(transposeX1, transposeX2) \
    do { \
        GET_TILING_DATA(tilingData, tiling); \
        BmmDequantInitOutput<DTYPE_Y> clearOp; \
        clearOp.Init(y, user1, &tilingData, &tPipe); \
        clearOp.Process(); \
        tPipe.Destroy(); \
        TPipe tPipeOp; \
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, transposeX1, transposeX2, \
            BMM_DEQUANT_PRELOAD_CFG> \
            op; \
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp); \
        op.Process(true); \
    } while (0)

#define INVOKE_QMM_PERTOKEN_ARCH20(templateClass, swizzle, biasFlag, biasType) \
    do { \
        templateClass<swizzle, biasFlag, format_x1, format_x2, format_y, int8_t, float, biasType, half> op; \
        op.Init(x1, x2, bias, scale, pertokenScale, y, &tilingData); \
        op.Process(); \
    } while (0)

#define INVOKE_QUANT_BATCH_MATMUL_PERTOKEN_ARCH20_IMPL(templateClass) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(QuantMatmulPertokenTilingDataArch20, tilingData, tiling); \
        const bool isFp32Bias = (tilingData.biasDtype == static_cast<uint32_t>(DT_FLOAT)); \
        if (tilingData.swizzleDirect == 0 && tilingData.withBias == false) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 0, false, int32_t); \
        } else if (tilingData.swizzleDirect == 0 && tilingData.withBias == true && isFp32Bias) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 0, true, float); \
        } else if (tilingData.swizzleDirect == 0 && tilingData.withBias == true) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 0, true, int32_t); \
        } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == false) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 1, false, int32_t); \
        } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == true && isFp32Bias) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 1, true, float); \
        } else if (tilingData.swizzleDirect == 1 && tilingData.withBias == true) { \
            INVOKE_QMM_PERTOKEN_ARCH20(templateClass, 1, true, int32_t); \
        } \
    } while (0)

template <int TRANS, int KERNEL_TEMPLATE_TYPE, int PERTOKEN, int OPTIONATTR>
__global__ __aicore__ void eagle_quant_batch_matmul_swiglu(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR bias, GM_ADDR pertokenScale, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
    if (workSpace == nullptr) {
        return;
    }
    TPipe tPipe;
    GM_ADDR user1 = GetUserWorkspace(workSpace);
    if (user1 == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(EagleQuantBatchMatmulSwigluTilingData);
#if ORIG_DTYPE_X1 == DT_INT8 && ORIG_DTYPE_X2 == DT_INT8 && ORIG_DTYPE_SCALE == DT_FLOAT && \
    (ORIG_DTYPE_Y == DT_BF16 || ORIG_DTYPE_Y == DT_FLOAT16) && (defined(FORMAT_X1) && FORMAT_X1 == FORMAT_ND) && \
    (defined(FORMAT_X2) && (FORMAT_X2 == FORMAT_ND || FORMAT_X2 == FORMAT_FRACTAL_NZ))
    if constexpr (KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_AICAIV_1_2 &&
        (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS || TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS ||
            TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS || TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS)) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL((TRANS & 2) != 0, (TRANS & 1) != 0);
        return;
    }
#endif
// 6bit from high to low: needClean, pertoken, opt, basic, transX1, transX2
#if (ORIG_DTYPE_Y == DT_FLOAT16 || ORIG_DTYPE_Y == DT_INT8 || ORIG_DTYPE_Y == DT_INT32) // fp16, int8, int32
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_PPMATMUL &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 13", PpMatmulTilingData);
        GET_TILING_DATA_WITH_STRUCT(PpMatmulTilingData, tilingData, tiling);
        PpMatmul<0, false, true, false, int8_t, uint64_t, int32_t, half> kernel;
        SET_FLAG(MTE2, S, EVENT_ID0);
        WAIT_FLAG(MTE2, S, EVENT_ID0);
        kernel.Init(x1, x2, bias, scale, y, &(tilingData));
        kernel.Process();
    } else {
        if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
            KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
            OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
            GET_TILING_DATA(tilingData, tiling);
            BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
            op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
            op.Process();
        }
        if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
            KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
            PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
            OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NEED_ATOMICLEAN) { // false true
            GET_TILING_DATA(tilingData, tiling);
            BmmDequantInitOutput<DTYPE_Y> clearOp;
            clearOp.Init(y, user1, &tilingData, &tPipe);
            clearOp.Process();
            tPipe.Destroy();

            TPipe tPipeOp;
            BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
            op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipeOp);
            op.Process();
        }
    }
#endif
#endif
#if ORIG_DTYPE_SCALE == DT_FLOAT
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
    if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_PERTOKEN_ARCH20_IMPL(PpMatMulNS::QuantBatchMatMulPertokenArch20);
    }
#endif
#endif
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || \
    (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#if (ORIG_DTYPE_SCALE != DT_FLOAT || ORIG_DTYPE_Y == DT_INT32)
    if constexpr (
#if ORIG_DTYPE_X1 == DT_INT8
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        GET_TILING_DATA(tilingData, tiling);
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        GET_TILING_DATA(tilingData, tiling);
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true false
        GET_TILING_DATA(tilingData, tiling);
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, false> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true true
        GET_TILING_DATA(tilingData, tiling);
        BmmDequant<DTYPE_X1, DTYPE_X2, FORMAT_X1, FORMAT_X2, int32_t, uint64_t, DTYPE_Y, true, true> op;
        op.Init(x1, x2, bias, scale, y, user1, &tilingData, &tPipe);
        op.Process();
    } else if constexpr (
#endif
#endif
#if ORIG_DTYPE_Y == DT_INT32 && (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ) && \
    (defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NEED_ATOMICLEAN) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_SPLITK_IMPL(false, false);
    } else if constexpr (
#endif
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_CUBE_IMPL(false, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_CUBE_IMPL(false, true);
#if ORIG_DTYPE_X1 == DT_INT8
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_CUBE_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_EAGLE_QUANT_BATCH_MATMUL_SWIGLU_CUBE_IMPL(true, true);
#endif
#endif
    }
#else
    if constexpr (
#if ORIG_DTYPE_X1 == DT_INT8
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
    } else if constexpr (
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
    } else if constexpr (
#endif
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
    } else if constexpr (
#endif
#endif
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
#endif
    }
#endif
#endif
#else //  bf16
    if constexpr (
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ) && (defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_AICAIV_1_2) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if constexpr (
#endif
#if ORIG_DTYPE_X1 == DT_INT8
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, false);
    } else if constexpr (
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_IMPL(true, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BF16_OPT_IMPL(true, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_BASIC_BLOCK_IMPL(true, true);
    } else if constexpr (
#endif
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false pertoken begin
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, false);
    } else if constexpr (
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_TBE &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_IMPL(true, true);
    } else if constexpr (
#endif
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, false);
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // false true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(false, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  false
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_OPT &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) { // true  true
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_OPT_IMPL(true, true);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_A_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_ALL_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(true, true);
    } else if constexpr (
#endif
#endif
#if (defined(FORMAT_X1) && FORMAT_X1 != FORMAT_FRACTAL_NZ)
        TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_NOT_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, false);
    } else if constexpr (TRANS == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_B_TRANS &&
        KERNEL_TEMPLATE_TYPE == QUANT_BATCH_MATMUl_V3_KERNEL_TEMPLATE_TYPE_BASIC &&
        PERTOKEN == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_IS_PERTOKEN &&
        OPTIONATTR == EAGLE_QUANT_BATCH_MATMUL_SWIGLU_OPTION_ATTR_NONE) {
        INVOKE_QUANT_BATCH_MATMUL_DEQUANT_PERTOKEN_BASIC_IMPL(false, true);
#endif
    }
#endif
}
