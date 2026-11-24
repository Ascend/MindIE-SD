/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */

#ifndef EPILOGUE_BLOCK_VF_DN_SOFTMAX_QUANT_BLOCK_H
#define EPILOGUE_BLOCK_VF_DN_SOFTMAX_QUANT_BLOCK_H

namespace NpuArch::Epilogue::Block {

template <typename T2, typename T>
__simd_vf__ inline void DnSoftmaxQuantBlock_fused_2x512x32_vf(__ubuf__ T2 *dstUbStart, __ubuf__ T *srcUbStart,
    __ubuf__ T *maxUbStart, __ubuf__ T *inMaxUbStart, __ubuf__ float *expMaxUbStart, __ubuf__ float *expSumUbStart,
    __ubuf__ float *inSumUbStart, __ubuf__ T *scaleUb) {
#define MAX_0 REG_A0
#define MAX_1 REG_A1
#define MAX_2 REG_A2
#define MAX_3 REG_A3
#define ACC_0 REG_A0
#define ACC_1 REG_A1
#define ACC_2 REG_A2
#define ACC_3 REG_A3

#define MAX_B0 TMP_0
#define MAX_B1 TMP_1
#define IN_MAX_B0 IN_MAX
#define IN_MAX_B1 IN_MAX2
#define TMP_0_B0 REG_A0
#define TMP_1_B0 REG_A1
#define TMP_0_B1 REG_A2
#define TMP_1_B1 REG_A3
#define MAX_1_B0 TMP_0
#define MAX_1_B1 TMP_1
#define LOCAL_REDUCE_B0 SRC_0
#define LOCAL_REDUCE_B1 SRC_1
#define GLOBAL_MAX_B0 SRC_2
#define GLOBAL_MAX_B1 SRC_3

    using namespace AscendC::Reg;
    constexpr static CastTrait castTraitFp32Zero = {
        RegLayout::ZERO,
        SatMode::UNKNOWN,
        MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN,
    };
    constexpr T minValue = -65504.0f;
    constexpr uint16_t VL_ELE_B16 = 128;
    enum HalfRegSlot : uint16_t {
        REG_A0 = 0,
        REG_A1,
        REG_A2,
        REG_A3,
        SRC_0,
        SRC_1,
        SRC_2,
        SRC_3,
        SCALE,
        IN_MAX,
        IN_MAX2,
        GLOBAL_MAX,
        EXP_MAX_16,
        LOCAL_REDUCE,
        TMP_0,
        TMP_1,
        SUM_B0,
        SUM_B1,
        SUM_1_B0,
        SUM_1_B1,
        EXP_MAX_16_B0,
        EXP_MAX_16_B1,
        HALF_REG_COUNT
    };
    enum FloatRegSlot : uint16_t {
        EXP_MAX_32_B0 = 0,
        IN_EXP_SUM_B0,
        EXP_SUM_32_B0,
        EXP_MAX_32_B1,
        IN_EXP_SUM_B1,
        EXP_SUM_32_B1,
        FLOAT_REG_COUNT
    };

    RegTensor<float> floatReg[FLOAT_REG_COUNT];
    RegTensor<T> halfReg[HALF_REG_COUNT];
    MaskReg pregAll = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregAll32 = CreateMask<float, MaskPattern::ALL>();

#define PROCESS_CORE_GROUP() \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_0], src_rd_ptr1, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_1], src_rd_ptr1, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_2], src_rd_ptr1, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_3], src_rd_ptr1, VL_ELE_B16); \
    Mul(halfReg[SRC_0], halfReg[SRC_0], halfReg[SCALE], pregAll); \
    Mul(halfReg[SRC_1], halfReg[SRC_1], halfReg[SCALE], pregAll); \
    Mul(halfReg[SRC_2], halfReg[SRC_2], halfReg[SCALE], pregAll); \
    Mul(halfReg[SRC_3], halfReg[SRC_3], halfReg[SCALE], pregAll); \
    Max(halfReg[MAX_0], halfReg[MAX_0], halfReg[SRC_0], pregAll); \
    StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src_wr_ptr1, halfReg[SRC_0], VL_ELE_B16, pregAll); \
    Max(halfReg[MAX_1], halfReg[MAX_1], halfReg[SRC_1], pregAll); \
    StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src_wr_ptr1, halfReg[SRC_1], VL_ELE_B16, pregAll); \
    Max(halfReg[MAX_2], halfReg[MAX_2], halfReg[SRC_2], pregAll); \
    StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src_wr_ptr1, halfReg[SRC_2], VL_ELE_B16, pregAll); \
    Max(halfReg[MAX_3], halfReg[MAX_3], halfReg[SRC_3], pregAll); \
    StoreAlign<T, PostLiteral::POST_MODE_UPDATE>(src_wr_ptr1, halfReg[SRC_3], VL_ELE_B16, pregAll);

#define PROCESS_K_BLOCK(k) \
    DataCopy<T, LoadDist::DIST_BRC_B16>(halfReg[SCALE], scaleUb + (k)); \
    PROCESS_CORE_GROUP() \
    PROCESS_CORE_GROUP() \
    PROCESS_CORE_GROUP() \
    PROCESS_CORE_GROUP()

#define PROCESS_EXP_GROUP(inMaxReg) \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_0], src_rd_ptr2, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_1], src_rd_ptr2, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_2], src_rd_ptr2, VL_ELE_B16); \
    LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[SRC_3], src_rd_ptr2, VL_ELE_B16); \
    Sub(halfReg[SRC_0], halfReg[SRC_0], halfReg[inMaxReg], pregAll); \
    Sub(halfReg[SRC_1], halfReg[SRC_1], halfReg[inMaxReg], pregAll); \
    Sub(halfReg[SRC_2], halfReg[SRC_2], halfReg[inMaxReg], pregAll); \
    Sub(halfReg[SRC_3], halfReg[SRC_3], halfReg[inMaxReg], pregAll); \
    Exp(halfReg[SRC_0], halfReg[SRC_0], pregAll); \
    Exp(halfReg[SRC_1], halfReg[SRC_1], pregAll); \
    Exp(halfReg[SRC_2], halfReg[SRC_2], pregAll); \
    Exp(halfReg[SRC_3], halfReg[SRC_3], pregAll); \
    Add(halfReg[ACC_0], halfReg[ACC_0], halfReg[SRC_0], pregAll); \
    Add(halfReg[ACC_1], halfReg[ACC_1], halfReg[SRC_1], pregAll); \
    Add(halfReg[ACC_2], halfReg[ACC_2], halfReg[SRC_2], pregAll); \
    Add(halfReg[ACC_3], halfReg[ACC_3], halfReg[SRC_3], pregAll); \
    Muls(halfReg[SRC_0], halfReg[SRC_0], (T)448.0f, pregAll); \
    Muls(halfReg[SRC_1], halfReg[SRC_1], (T)448.0f, pregAll); \
    Muls(halfReg[SRC_2], halfReg[SRC_2], (T)448.0f, pregAll); \
    Muls(halfReg[SRC_3], halfReg[SRC_3], (T)448.0f, pregAll); \
    Maxs((RegTensor<int16_t> &)halfReg[SRC_0], (RegTensor<int16_t> &)halfReg[SRC_0], (int16_t)(8128 + 128), pregAll); \
    Maxs((RegTensor<int16_t> &)halfReg[SRC_1], (RegTensor<int16_t> &)halfReg[SRC_1], (int16_t)(8128 + 128), pregAll); \
    Maxs((RegTensor<int16_t> &)halfReg[SRC_2], (RegTensor<int16_t> &)halfReg[SRC_2], (int16_t)(8128 + 128), pregAll); \
    Maxs((RegTensor<int16_t> &)halfReg[SRC_3], (RegTensor<int16_t> &)halfReg[SRC_3], (int16_t)(8128 + 128), pregAll); \
    Adds((RegTensor<int16_t> &)halfReg[SRC_0], (RegTensor<int16_t> &)halfReg[SRC_0], (int16_t)(-8128), pregAll); \
    Adds((RegTensor<int16_t> &)halfReg[SRC_1], (RegTensor<int16_t> &)halfReg[SRC_1], (int16_t)(-8128), pregAll); \
    Adds((RegTensor<int16_t> &)halfReg[SRC_2], (RegTensor<int16_t> &)halfReg[SRC_2], (int16_t)(-8128), pregAll); \
    Adds((RegTensor<int16_t> &)halfReg[SRC_3], (RegTensor<int16_t> &)halfReg[SRC_3], (int16_t)(-8128), pregAll); \
    ShiftRights((RegTensor<int16_t> &)halfReg[SRC_0], (RegTensor<int16_t> &)halfReg[SRC_0], (int16_t)7, pregAll); \
    ShiftRights((RegTensor<int16_t> &)halfReg[SRC_1], (RegTensor<int16_t> &)halfReg[SRC_1], (int16_t)7, pregAll); \
    ShiftRights((RegTensor<int16_t> &)halfReg[SRC_2], (RegTensor<int16_t> &)halfReg[SRC_2], (int16_t)7, pregAll); \
    ShiftRights((RegTensor<int16_t> &)halfReg[SRC_3], (RegTensor<int16_t> &)halfReg[SRC_3], (int16_t)7, pregAll); \
    StoreAlign<T2, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>( \
        dst_wr_ptr2, (RegTensor<T2> &)halfReg[SRC_0], VL_ELE_B16, pregAll); \
    StoreAlign<T2, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>( \
        dst_wr_ptr2, (RegTensor<T2> &)halfReg[SRC_1], VL_ELE_B16, pregAll); \
    StoreAlign<T2, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>( \
        dst_wr_ptr2, (RegTensor<T2> &)halfReg[SRC_2], VL_ELE_B16, pregAll); \
    StoreAlign<T2, PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK_B16>( \
        dst_wr_ptr2, (RegTensor<T2> &)halfReg[SRC_3], VL_ELE_B16, pregAll);

#define PROCESS_MAX_BLOCK_PART1(blockIdx, outReg) \
    do { \
        __ubuf__ T *srcUb = srcUbStart + (blockIdx) * 32 * 512; \
        Duplicate<T, T>(halfReg[MAX_0], minValue); \
        Duplicate<T, T>(halfReg[MAX_1], minValue); \
        Duplicate<T, T>(halfReg[MAX_2], minValue); \
        Duplicate<T, T>(halfReg[MAX_3], minValue); \
        __ubuf__ T *src_rd_ptr1 = srcUb; \
        __ubuf__ T *src_wr_ptr1 = srcUb; \
        for (uint16_t k = 0; k < 8; ++k) { \
            PROCESS_K_BLOCK(k); \
        } \
        Max(halfReg[MAX_0], halfReg[MAX_0], halfReg[MAX_1], pregAll); \
        Max(halfReg[MAX_2], halfReg[MAX_2], halfReg[MAX_3], pregAll); \
        Max(halfReg[outReg], halfReg[MAX_0], halfReg[MAX_2], pregAll); \
    } while (0)

#define PROCESS_EXP_BLOCK_PART1(blockIdx, outReg, inMaxReg) \
    do { \
        __ubuf__ T *srcUb = srcUbStart + (blockIdx) * 32 * 512; \
        __ubuf__ T2 *dstUb = dstUbStart + (blockIdx) * 32 * 512; \
        Duplicate<T, T>(halfReg[ACC_0], (T)0.0f); \
        Duplicate<T, T>(halfReg[ACC_1], (T)0.0f); \
        Duplicate<T, T>(halfReg[ACC_2], (T)0.0f); \
        Duplicate<T, T>(halfReg[ACC_3], (T)0.0f); \
        __ubuf__ T *src_rd_ptr2 = srcUb; \
        __ubuf__ T2 *dst_wr_ptr2 = dstUb; \
        for (uint16_t k = 0; k < 32; ++k) { \
            PROCESS_EXP_GROUP(inMaxReg); \
        } \
        Add(halfReg[ACC_0], halfReg[ACC_0], halfReg[ACC_1], pregAll); \
        Add(halfReg[ACC_2], halfReg[ACC_2], halfReg[ACC_3], pregAll); \
        Add(halfReg[outReg], halfReg[ACC_0], halfReg[ACC_2], pregAll); \
    } while (0)

    PROCESS_MAX_BLOCK_PART1(0, MAX_B0);
    PROCESS_MAX_BLOCK_PART1(1, MAX_B1);

    LoadAlign<T>(halfReg[IN_MAX_B0], inMaxUbStart + 0 * 128);
    LoadAlign<T>(halfReg[IN_MAX_B1], inMaxUbStart + 1 * 128);
    Interleave(halfReg[TMP_0_B0], halfReg[TMP_1_B0], halfReg[MAX_B0], halfReg[MAX_B0]);
    Interleave(halfReg[TMP_0_B1], halfReg[TMP_1_B1], halfReg[MAX_B1], halfReg[MAX_B1]);
    Max(halfReg[MAX_1_B0], halfReg[TMP_0_B0], halfReg[TMP_1_B0], pregAll);
    Max(halfReg[MAX_1_B1], halfReg[TMP_0_B1], halfReg[TMP_1_B1], pregAll);
    Interleave(halfReg[TMP_0_B0], halfReg[TMP_1_B0], halfReg[MAX_1_B0], halfReg[MAX_1_B0]);
    Interleave(halfReg[TMP_0_B1], halfReg[TMP_1_B1], halfReg[MAX_1_B1], halfReg[MAX_1_B1]);
    Max(halfReg[LOCAL_REDUCE_B0], halfReg[TMP_0_B0], halfReg[TMP_1_B0], pregAll);
    Max(halfReg[LOCAL_REDUCE_B1], halfReg[TMP_0_B1], halfReg[TMP_1_B1], pregAll);
    Max(halfReg[GLOBAL_MAX_B0], halfReg[LOCAL_REDUCE_B0], halfReg[IN_MAX_B0], pregAll);
    Max(halfReg[GLOBAL_MAX_B1], halfReg[LOCAL_REDUCE_B1], halfReg[IN_MAX_B1], pregAll);
    StoreAlign<T>(maxUbStart + 0 * 128, halfReg[GLOBAL_MAX_B0], pregAll);
    StoreAlign<T>(maxUbStart + 1 * 128, halfReg[GLOBAL_MAX_B1], pregAll);

    Sub(halfReg[EXP_MAX_16_B0], halfReg[IN_MAX_B0], halfReg[GLOBAL_MAX_B0], pregAll);
    Sub(halfReg[EXP_MAX_16_B1], halfReg[IN_MAX_B1], halfReg[GLOBAL_MAX_B1], pregAll);

    Exp(halfReg[EXP_MAX_16_B0], halfReg[EXP_MAX_16_B0], pregAll);
    Exp(halfReg[EXP_MAX_16_B1], halfReg[EXP_MAX_16_B1], pregAll);

    Cast<float, T, castTraitFp32Zero>(floatReg[EXP_MAX_32_B0], halfReg[EXP_MAX_16_B0], pregAll);
    Cast<float, T, castTraitFp32Zero>(floatReg[EXP_MAX_32_B1], halfReg[EXP_MAX_16_B1], pregAll);

    DeInterleave(halfReg[TMP_0_B0], halfReg[TMP_1_B0], halfReg[GLOBAL_MAX_B0], halfReg[GLOBAL_MAX_B0]);
    DeInterleave(halfReg[TMP_0_B1], halfReg[TMP_1_B1], halfReg[GLOBAL_MAX_B1], halfReg[GLOBAL_MAX_B1]);

    DeInterleave(halfReg[IN_MAX_B0], halfReg[TMP_1_B0], halfReg[TMP_0_B0], halfReg[TMP_0_B0]);
    DeInterleave(halfReg[IN_MAX_B1], halfReg[TMP_1_B1], halfReg[TMP_0_B1], halfReg[TMP_0_B1]);

    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();

    PROCESS_EXP_BLOCK_PART1(0, SUM_B0, IN_MAX_B0);
    PROCESS_EXP_BLOCK_PART1(1, SUM_B1, IN_MAX_B1);

    Interleave(halfReg[TMP_0_B0], halfReg[TMP_1_B0], halfReg[SUM_B0], halfReg[SUM_B0]);
    Interleave(halfReg[TMP_0_B1], halfReg[TMP_1_B1], halfReg[SUM_B1], halfReg[SUM_B1]);
    Add(halfReg[SUM_1_B0], halfReg[TMP_0_B0], halfReg[TMP_1_B0], pregAll);
    Add(halfReg[SUM_1_B1], halfReg[TMP_0_B1], halfReg[TMP_1_B1], pregAll);
    Interleave(halfReg[TMP_0_B0], halfReg[TMP_1_B0], halfReg[SUM_1_B0], halfReg[SUM_1_B0]);
    Interleave(halfReg[TMP_0_B1], halfReg[TMP_1_B1], halfReg[SUM_1_B1], halfReg[SUM_1_B1]);
    Add(halfReg[LOCAL_REDUCE_B0], halfReg[TMP_0_B0], halfReg[TMP_1_B0], pregAll);
    Add(halfReg[LOCAL_REDUCE_B1], halfReg[TMP_0_B1], halfReg[TMP_1_B1], pregAll);

    Cast<float, T, castTraitFp32Zero>(floatReg[EXP_SUM_32_B0], halfReg[LOCAL_REDUCE_B0], pregAll);
    Cast<float, T, castTraitFp32Zero>(floatReg[EXP_SUM_32_B1], halfReg[LOCAL_REDUCE_B1], pregAll);

    LoadAlign<float>(floatReg[IN_EXP_SUM_B0], inSumUbStart + 0 * 64);
    LoadAlign<float>(floatReg[IN_EXP_SUM_B1], inSumUbStart + 1 * 64);

    Mul<float, MaskMergeMode::ZEROING>(
        floatReg[IN_EXP_SUM_B0], floatReg[EXP_MAX_32_B0], floatReg[IN_EXP_SUM_B0], pregAll32);
    Mul<float, MaskMergeMode::ZEROING>(
        floatReg[IN_EXP_SUM_B1], floatReg[EXP_MAX_32_B1], floatReg[IN_EXP_SUM_B1], pregAll32);

    Add<float, MaskMergeMode::ZEROING>(
        floatReg[IN_EXP_SUM_B0], floatReg[IN_EXP_SUM_B0], floatReg[EXP_SUM_32_B0], pregAll32);
    Add<float, MaskMergeMode::ZEROING>(
        floatReg[IN_EXP_SUM_B1], floatReg[IN_EXP_SUM_B1], floatReg[EXP_SUM_32_B1], pregAll32);

    StoreAlign<float>(expSumUbStart + 0 * 64, floatReg[IN_EXP_SUM_B0], pregAll32);
    StoreAlign<float>(expSumUbStart + 1 * 64, floatReg[IN_EXP_SUM_B1], pregAll32);

    StoreAlign<float>(expMaxUbStart + 0 * 64, floatReg[EXP_MAX_32_B0], pregAll32);
    StoreAlign<float>(expMaxUbStart + 1 * 64, floatReg[EXP_MAX_32_B1], pregAll32);

#undef PROCESS_CORE_GROUP
#undef PROCESS_K_BLOCK
#undef PROCESS_EXP_GROUP
#undef PROCESS_MAX_BLOCK_PART1
#undef PROCESS_EXP_BLOCK_PART1

#undef MAX_0
#undef MAX_1
#undef MAX_2
#undef MAX_3
#undef ACC_0
#undef ACC_1
#undef ACC_2
#undef ACC_3

#undef MAX_B0
#undef MAX_B1
#undef IN_MAX_B0
#undef IN_MAX_B1
#undef TMP_0_B0
#undef TMP_1_B0
#undef TMP_0_B1
#undef TMP_1_B1
#undef MAX_1_B0
#undef MAX_1_B1
#undef LOCAL_REDUCE_B0
#undef LOCAL_REDUCE_B1
#undef GLOBAL_MAX_B0
#undef GLOBAL_MAX_B1
}

} // namespace NpuArch::Epilogue::Block
#endif
