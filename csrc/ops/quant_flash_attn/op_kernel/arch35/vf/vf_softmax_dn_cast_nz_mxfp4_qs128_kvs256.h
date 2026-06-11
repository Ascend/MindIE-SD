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
 * \file vf_softmax_dn_cast_nz_mxfp4.h
 * \brief
 */

#ifndef VF_SOFTMAX_DN_CAST_NZ_MXFP4_QS128_KVS256_H_
#define VF_SOFTMAX_DN_CAST_NZ_MXFP4_QS128_KVS256_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace Mxfp4Api {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

template <bool clear_gmax, typename T, typename T2, bool hasAtten = false, uint16_t S2Base = 256, uint16_t S1Base = 128>
__simd_vf__ inline void softmax_with_group_max_qs128_kvs256_vf(__ubuf__ T2 *pDest, __ubuf__ T *s,
    __ubuf__ T *local_group_max, __ubuf__ T *global_max, __ubuf__ uint8_t *indexesUb, const T dScale) {
    // ====================== 寄存器定义 ======================
    RegTensor<half> src_c0, src_c1, src_c2, src_c3;
    RegTensor<half> src_n0, src_n1, src_n2, src_n3;
    RegTensor<half> curr_group_max;
    RegTensor<half> next_group_max;
    RegTensor<half> group_gmax;
    RegTensor<half> min_val_reg;
    RegTensor<uint8_t> idx_nd2nz;

    // 量化专用寄存器
    RegTensor<bfloat16_t> src_bf16_0, src_bf16_1, src_bf16_2, src_bf16_3;
    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;

    // ====================== 分块常量 ======================
    const uint16_t ROWS_PER_GROUP = 32;
    const uint16_t GROUP_COUNT = S2Base / ROWS_PER_GROUP;
    const uint16_t ROW_SUB_LOOP = 4;
    const uint16_t ITER_PER_GROUP = ROWS_PER_GROUP / ROW_SUB_LOOP;
    uint32_t MID_VALID_CNT = S1Base * (GROUP_COUNT - 1);

    // ====================== 掩码定义 ======================
    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);

    MaskReg preg_invalid_max;

    // ====================== 全局最大值初始化 ======================
    if constexpr (clear_gmax) {
        Duplicate(group_gmax, MIN_VALUE);
    } else {
        LoadAlign(group_gmax, global_max);
    }

    LoadAlign(idx_nd2nz, indexesUb);

    // ====================== 预计算：第一个分组的最大值 ======================
    Duplicate(curr_group_max, MIN_VALUE);
    for (uint16_t iter = 0; iter < ITER_PER_GROUP; ++iter) {
        LoadAlign(src_c0, s + (iter * S1Base * ROW_SUB_LOOP + 0 * S1Base) * 2);
        LoadAlign(src_c1, s + (iter * S1Base * ROW_SUB_LOOP + 1 * S1Base) * 2);
        LoadAlign(src_c2, s + (iter * S1Base * ROW_SUB_LOOP + 2 * S1Base) * 2);
        LoadAlign(src_c3, s + (iter * S1Base * ROW_SUB_LOOP + 3 * S1Base) * 2);

        Max(src_c0, src_c0, src_c1, preg_all_16bit);
        Max(src_c2, src_c2, src_c3, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c0, preg_all_16bit);
        Max(curr_group_max, curr_group_max, src_c2, preg_all_16bit);
    }

    Muls(curr_group_max, curr_group_max, dScale, preg_all_16bit);
    Muls(curr_group_max, curr_group_max, INV_LN2, preg_all_16bit);
    Truncate<T, RoundMode::CAST_FLOOR>(curr_group_max, curr_group_max, preg_all_16bit);
    Max(group_gmax, group_gmax, curr_group_max, preg_all_16bit);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max, curr_group_max, preg_all_16bit);
    Adds(curr_group_max, curr_group_max, NEG_TWO_VALE, preg_all_16bit);
    Muls(curr_group_max, curr_group_max, LN2, preg_all_16bit);

    // ====================== 核心：双块流水分组循环 ======================
    for (uint16_t i = 0; i < GROUP_COUNT; i++) {
        MaskReg preg_valid_max = UpdateMask<half>(MID_VALID_CNT);
        MaskNot(preg_invalid_max, preg_valid_max, preg_all_16bit);

        // 初始化下一个块最大值
        Duplicate(next_group_max, MIN_VALUE);

        // ========== 第一个内循环：处理偶数子块 j = 0,2,4,6 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            uint16_t rowOffset_next = (i + 1) * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * S1Base + 0 * S1Base) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * S1Base + 1 * S1Base) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * S1Base + 2 * S1Base) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * S1Base + 3 * S1Base) * 2);
            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + j * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * S1Base + 0 * S1Base) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * S1Base + 1 * S1Base) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * S1Base + 2 * S1Base) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * S1Base + 3 * S1Base) * 2);

            Muls(src_c0, src_c0, dScale, preg_all_16bit);
            Muls(src_c1, src_c1, dScale, preg_all_16bit);
            Muls(src_c2, src_c2, dScale, preg_all_16bit);
            Muls(src_c3, src_c3, dScale, preg_all_16bit);

            Sub(src_c0, src_c0, curr_group_max, preg_all_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_all_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_all_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_all_16bit);

            Exp(src_c0, src_c0, preg_all_16bit);
            Exp(src_c1, src_c1, preg_all_16bit);
            Exp(src_c2, src_c2, preg_all_16bit);
            Exp(src_c3, src_c3, preg_all_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_all_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_all_16bit);

            // 数据合并
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_1,
                preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_3,
                preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_2,
                preg_all_8bit);
            Gather((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, idx_nd2nz);

            // 写入内存（偶数块：无+128偏移）
            StoreAlign(((__ubuf__ uint8_t *&)pDest) + i * 2048 + j * 256, (RegTensor<uint8_t> &)quant_0,
                preg_vl128 // 低128字节掩码
            );

            StoreAlign(((__ubuf__ uint8_t *&)pDest) + i * 2048 + j * 256 + 16256, (RegTensor<uint8_t> &)quant_0,
                preg_vl128_not // 高128字节掩码
            );
        }

        // ========== 第二个内循环：处理奇数子块 j+1 = 1,3,5,7 ==========
        for (uint16_t j = 0; j < ITER_PER_GROUP; j += 2) {
            // 预计算：下一个块 (i+1) 最大值
            uint16_t rowOffset_next = (i + 1) * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_n0, s + (rowOffset_next * S1Base + 0 * S1Base) * 2);
            LoadAlign(src_n1, s + (rowOffset_next * S1Base + 1 * S1Base) * 2);
            LoadAlign(src_n2, s + (rowOffset_next * S1Base + 2 * S1Base) * 2);
            LoadAlign(src_n3, s + (rowOffset_next * S1Base + 3 * S1Base) * 2);

            Max(src_n0, src_n0, src_n1, preg_valid_max);
            Max(src_n2, src_n2, src_n3, preg_valid_max);
            Max(next_group_max, next_group_max, src_n0, preg_valid_max);
            Max(next_group_max, next_group_max, src_n2, preg_valid_max);

            // 当前块计算：Sub + Exp
            uint16_t rowOffset_cur = i * ROWS_PER_GROUP + (j + 1) * ROW_SUB_LOOP;
            LoadAlign(src_c0, s + (rowOffset_cur * S1Base + 0 * S1Base) * 2);
            LoadAlign(src_c1, s + (rowOffset_cur * S1Base + 1 * S1Base) * 2);
            LoadAlign(src_c2, s + (rowOffset_cur * S1Base + 2 * S1Base) * 2);
            LoadAlign(src_c3, s + (rowOffset_cur * S1Base + 3 * S1Base) * 2);

            Muls(src_c0, src_c0, dScale, preg_all_16bit);
            Muls(src_c1, src_c1, dScale, preg_all_16bit);
            Muls(src_c2, src_c2, dScale, preg_all_16bit);
            Muls(src_c3, src_c3, dScale, preg_all_16bit);

            Sub(src_c0, src_c0, curr_group_max, preg_all_16bit);
            Sub(src_c1, src_c1, curr_group_max, preg_all_16bit);
            Sub(src_c2, src_c2, curr_group_max, preg_all_16bit);
            Sub(src_c3, src_c3, curr_group_max, preg_all_16bit);

            Exp(src_c0, src_c0, preg_all_16bit);
            Exp(src_c1, src_c1, preg_all_16bit);
            Exp(src_c2, src_c2, preg_all_16bit);
            Exp(src_c3, src_c3, preg_all_16bit);

            // 4bit 量化压缩
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_0, src_c0, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_1, src_c1, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_2, src_c2, preg_all_16bit);
            Cast<bfloat16_t, T, castTraitZero>(src_bf16_3, src_c3, preg_all_16bit);

            Cast<float4_e2m1x2_t, bfloat16_t, castTraitZero>(quant_0, src_bf16_0, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitOne>(quant_1, src_bf16_1, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitTwo>(quant_2, src_bf16_2, preg_all_16bit);
            Cast<float4_e2m1x2_t, bfloat16_t, castTraitThree>(quant_3, src_bf16_3, preg_all_16bit);

            // 数据合并
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_1,
                preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_3,
                preg_all_8bit);
            Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_2,
                preg_all_8bit);
            Gather((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, idx_nd2nz);

            // 奇数块：低128字节写入（基础地址 +128 偏移）
            StoreAlign(
                ((__ubuf__ uint8_t *&)pDest) + i * 2048 + j * 256 + 128, (RegTensor<uint8_t> &)quant_0, preg_vl128);

            // 奇数块：高128字节写入（远端地址）
            StoreAlign(((__ubuf__ uint8_t *&)pDest) + i * 2048 + j * 256 + 16384, (RegTensor<uint8_t> &)quant_0,
                preg_vl128_not);
        }

        // ====================== 全局/局部最大值更新 ======================
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, group_gmax, preg_invalid_max);

        // 下一块最大值归一化
        Muls(next_group_max, next_group_max, dScale, preg_valid_max);
        Muls(next_group_max, next_group_max, INV_LN2, preg_valid_max);
        Truncate<T, RoundMode::CAST_FLOOR>(next_group_max, next_group_max, preg_valid_max);
        Max(group_gmax, group_gmax, next_group_max, preg_valid_max);

        // 存储下一块最大值到 ulmax (i+1)*128 偏移
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(
            local_group_max + ((i + 1) * S1Base), next_group_max, preg_valid_max);

        // 更新当前块最大值，用于下一次循环
        Adds(next_group_max, next_group_max, NEG_TWO_VALE, preg_all_16bit);
        Muls(curr_group_max, next_group_max, LN2, preg_valid_max);
    }
}

template <bool clear_gmax, typename T, typename T2, bool hasAtten = false, uint16_t S2Base = 256, uint16_t S1Base = 128>
__aicore__ inline void softmaxWithGroupMaxQs128Kvs256CallVF(const LocalTensor<T2> &dstTensor,
    const LocalTensor<T> &srcTensor, const LocalTensor<T> &local_group_max, const LocalTensor<T> &global_max,
    const LocalTensor<uint8_t> &indexesBuf, const T scale) {
    __ubuf__ T2 *pDest = (__ubuf__ T2 *)dstTensor.GetPhyAddr();
    __ubuf__ T *input_x_local_UB = (__ubuf__ T *)srcTensor.GetPhyAddr();
    __ubuf__ T *localGroupMax = (__ubuf__ T *)local_group_max.GetPhyAddr();
    __ubuf__ T *globalMax = (__ubuf__ T *)global_max.GetPhyAddr();
    __ubuf__ uint8_t *indexesUb = (__ubuf__ uint8_t *)indexesBuf.GetPhyAddr();

    softmax_with_group_max_qs128_kvs256_vf<clear_gmax, T, T2, hasAtten, S2Base, S1Base>(
        pDest, input_x_local_UB, localGroupMax, globalMax, indexesUb, scale);
}

}
#endif // VF_SOFTMAX_DN_CAST_NZ_MXFP4_QS128_KVS256_H_
