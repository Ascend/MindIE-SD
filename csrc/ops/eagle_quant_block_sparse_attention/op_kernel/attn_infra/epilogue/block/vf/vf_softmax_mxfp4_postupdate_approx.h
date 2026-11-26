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
 * \file vf_softmax_mxfp4_postupdate_approx.h
 * \brief
 */

#ifndef VF_SOFTMAX_MXFP4_APPROX_H_
#define VF_SOFTMAX_MXFP4_APPROX_H_

#include "kernel_tensor.h"
#include "vf_common_def.h"

// C 约等于 7.1,lnC = 1.96
// input x M  x is   256 * 64 half tensor
//     x =  reshape 8 * 32 * 64
//     m = max(x, dim=1)
//     k = ceil(m / ln2)
//     y = Cexp(x - k * ln2)  = exp(x - (k*ln2-lnC)) , k*ln2-lnC may have precision issue
//     y = cast y to FP4 e2m1（伪量化）
//     K = max(k, dim=0)
//     mx = k - M if all K-M < clip else k - max(M, K) # 整体更新
//     mx = mx + 127
//     mx = clip(mx , 1, 255)
//     M = max(M, K) if any K-M >= clip else M
// output y, k, M, mx

namespace Mxfp4ApiApprox {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace Mxfp4Api;
// 定义拼装宏：将8个8位整型拼接为一个64位整型（小端序：前低后高）
#define PACK_U8_TO_U64(b0, b1, b2, b3, b4, b5, b6, b7) \
    ((uint64_t)(b0) | ((uint64_t)(b1) << 8) | ((uint64_t)(b2) << 16) | ((uint64_t)(b3) << 24) | \
        ((uint64_t)(b4) << 32) | ((uint64_t)(b5) << 40) | ((uint64_t)(b6) << 48) | ((uint64_t)(b7) << 56))

// def pseudo_quant_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
//     """将任意形状的浮点 tensor 伪量化到 FP4 e2m1 格网。

//     实现方式：对每个元素找到绝对值最近的可表示正数，再还原符号。
//     返回与输入同形状、同 dtype 的 tensor（量化后的浮点值）。
//     """
//     fp4_vals = _FP4_E2M1_POSITIVES.to(device=x.device, dtype=x.dtype)
//     sign = torch.sign(x)
//     abs_x = x.abs()
//     # [*shape, 8] 广播比较，找最近邻
//     diff = (abs_x.unsqueeze(-1) - fp4_vals).abs()
//     # 翻转最后一个维度，使得 argmin 遇到并列最小值时优先返回较大的索引（实现 Round-Half-Up）
//     idx = 7 - diff.flip(dims=[-1]).argmin(dim=-1) # ascendc 在中点默认向上面的偶数舍入
//     quantized = fp4_vals[idx]
//     # 0 的符号恢复为 +0（与硬件行为一致）
//     return quantized * sign.where(sign != 0, torch.ones_like(sign))

// half --> bf16 需要使用cast_trunc 消除 half --> fp16 -->fp4带来的 double rounding 误差，等效于 直接half ---> fp4
#define COMPUTE_EXP_APPROX(v_curr, v_next, v_max) \
    Sub(v_curr, v_curr, v_max, preg_all_16bit); \
    Sub(v_next, v_next, v_max, preg_all_16bit); \
    MulDstAdd(v_curr, v_k, v_b, preg_all_16bit); \
    MulDstAdd(v_next, v_k, v_b, preg_all_16bit); \
    Compares<T, AscendC::CMPMODE::GE>(preg_cmp0, v_curr, tau, preg_all_16bit); \
    Compares<T, AscendC::CMPMODE::GE>(preg_cmp1, v_next, tau, preg_all_16bit); \
    Maxs(v_curr, v_curr, (T)1.0f, preg_cmp0); \
    Maxs(v_next, v_next, (T)1.0f, preg_cmp1);

// 归约当前 Group 中所有 32 行的最大值（偶数行与奇数行合并）并计算缩放因子 k，同时更新当前列的全局最大值 K。
#define UPDATE_AND_STORE_MAX(v_max_n0, v_max_n1, group_idx) \
    Max(v_max_n0, v_max_n0, v_max_n1, preg_all_16bit); /* 合并偶数行(n0)和奇数行(n1)的局部 Max */ \
    Muls(v_max_n0, v_max_n0, INV_LN2, preg_all_16bit); \
    Truncate<T, RoundMode::CAST_CEIL>(vTmp0, v_max_n0, preg_all_16bit); /* k = ceil(m / ln2) */ \
    Max(v_cur_max, v_cur_max, vTmp0, preg_all_16bit); /* K = max(k, dim=0) 累积全局 Max */ \
    Muls(v_max_c0, vTmp0, LN2, preg_all_16bit); /* y = exp(x - k * ln2) 等价于 exp(x - (k*ln2)) */ \
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>( \
        (local_group_max) + ((group_idx) + 1) * VL_HALF, vTmp0, preg_all_16bit);

// 将 4 行 BF16 数据量化为 FP4 并拼装后存盘。
// 注意：硬件 Cast 会根据 castTrait 将 64 字节结果离散写到 256 字节向量中（如 traitZero 写 0,4,8...）。
// 随后通过 Gather 配合 v_idx_group，将离散的字节重新规整为连续的行数据（每行 32 字节）。
#define QUANT_AND_STORE_FP4(v_curr_half, v_next_half, v_curr_half_p1, v_next_half_p1) \
    Cast<int4x2_t, half, castTraitZeroTrunc>((RegTensor<int4x2_t> &)quant_0, v_curr_half, preg_all_16bit); \
    Cast<int4x2_t, half, castTraitOneTrunc>((RegTensor<int4x2_t> &)quant_1, v_next_half, preg_all_16bit); \
    Cast<int4x2_t, half, castTraitTwoTrunc>((RegTensor<int4x2_t> &)quant_2, v_curr_half_p1, preg_all_16bit); \
    Cast<int4x2_t, half, castTraitThreeTrunc>((RegTensor<int4x2_t> &)quant_3, v_next_half_p1, preg_all_16bit); \
    Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_1, preg_all_8bit); \
    Or((RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_2, (RegTensor<uint8_t> &)quant_3, preg_all_8bit); \
    Or((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_2, preg_all_8bit); \
    Gather((RegTensor<uint8_t> &)quant_0, (RegTensor<uint8_t> &)quant_0, \
        v_idx_group); /* 将打散的比特重排为连续的 4 行 (首尾各 128 字节) */ \
    StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>( \
        p_dest_low, (RegTensor<uint8_t> &)quant_0, 128, preg_vl128); /* 存前半 128 字节，刚好无缝拼接 4 行 */ \
    /* 使用 preg_vl128_not 写高位 128 字节时，硬件会自动加 128 偏移，因此基址计算减去 128 进行巧妙抵消 */ \
    StoreAlign<uint8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>( \
        p_dest_high, (RegTensor<uint8_t> &)quant_0, 128, preg_vl128_not);

template <typename T, typename T2, uint16_t VL_HALF = 128>
__simd_callee__ inline void softmax_fp4_callee(__ubuf__ T2 *pDest, __ubuf__ T *clip_ptr, __ubuf__ T *src,
    __ubuf__ T *local_group_max, __ubuf__ T *global_max_cur, __ubuf__ T *global_max_his,
    __ubuf__ uint8_t *idx_group_ptr, MaskReg &preg_diff, const uint16_t halfRows, const half clip, uint32_t qtail) {
    using namespace MicroAPI;

    RegTensor<half> v_max_c0, v_max_c1; // 当前组的 Max (交替使用以打破依赖)
    RegTensor<half> v_max_n0, v_max_n1; // 下一组的 Max (交替使用以打破依赖)
    RegTensor<half> v_cur_max, vTmp0, vTmp1, v_old_max, v_diff; // 全局最大

    RegTensor<half> v_data_c_curr, v_data_c_next; // 当前组数(当前 下一
    RegTensor<half> v_data_n_curr, v_data_n_next; // 下一组数(当前 下一

    RegTensor<half> v_data_c_curr_p1, v_data_c_next_p1;
    RegTensor<half> v_data_n_curr_p1, v_data_n_next_p1;

    RegTensor<half> v_k, v_b;

    RegTensor<uint8_t> v_idx_group;

    RegTensor<float4_e2m1x2_t> quant_0, quant_1, quant_2, quant_3;

    constexpr uint16_t ROWS_PER_GROUP = 32;
    const uint16_t HALF_GROUPS = halfRows / ROWS_PER_GROUP; // 256 / 32 = 8
    constexpr half LNC = 1.96; //C = 7.1
    constexpr half k = 2.603515625;
    constexpr half b = 7.8046875;
    constexpr half tau = -5.046875;

    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not, pregM, preg_cmp0, preg_cmp1;
    uint32_t m = qtail;
    pregM = UpdateMask<uint16_t>(m);
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);

    __ubuf__ uint8_t *p_dest_low = (__ubuf__ uint8_t *)pDest;
    __ubuf__ uint8_t *p_dest_high = p_dest_low + halfRows * 32 - 128;

    LoadAlign(v_old_max, global_max_his);
    LoadAlign(v_idx_group, idx_group_ptr);
    Duplicate(v_cur_max, MIN_VALUE);
    Duplicate(v_k, k);
    Duplicate(v_b, b);

    // ====================== 宏流水线序言 (Prologue) ======================
    // 预先计算0 组的 Max
    uint32_t offset_curr = 0;
    __ubuf__ T *src_c_ptr = src;
    __ubuf__ T *src_n_ptr;
    Duplicate(v_max_c0, MIN_VALUE);
    Duplicate(v_max_c1, MIN_VALUE);

    for (uint16_t i = 0; i < ROWS_PER_GROUP; i += 4) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr_p1, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next_p1, src_c_ptr, VL_HALF);

        Max(v_max_c0, v_max_c0, v_data_c_curr, preg_all_16bit);
        Max(v_max_c1, v_max_c1, v_data_c_next, preg_all_16bit);
        Max(v_max_c0, v_max_c0, v_data_c_curr_p1, preg_all_16bit);
        Max(v_max_c1, v_max_c1, v_data_c_next_p1, preg_all_16bit);
    }

    // 归约并存储 Group 0 的 Max 值，并为 Core Loop 准备寄存器。
    // 寄存器使用与目的说明：
    // 1. 入参 v_max_c0 和 v_max_c1: 分别保存了 Group 0 偶数行和奇数行的局部 Max。
    // 2. 出参复用 v_max_c0: 宏内部将合并后的 Group 0 最终 Max 重新写入 v_max_c0。
    //    目的：Core Loop 第 0 轮计算 Group 0 的 Exp 时，正好需要 v_max_c0 作为当前组的 Max 输入，实现了数据的无缝传递。
    // 3. group_idx = -1: 宏内部计算偏移为 (group_idx + 1) * VL_HALF = 0，正确存入 local_group_max 的第 0 个 slot。
    UPDATE_AND_STORE_MAX(v_max_c0, v_max_c1, -1)

    // ====================== 宏流水线核心 (Core Loop) ======================
    // 采用 4 步展开的软件流水线设计：在处理当前组（Group group）的 Exp 及量化计算的同时，
    // 提前加载并计算下一组（Group group+1）的局部最大值，以极大掩盖访存延迟。
    for (uint16_t group = 0; group < HALF_GROUPS - 1; ++group) {
        offset_curr = group * ROWS_PER_GROUP * VL_HALF;
        uint32_t offset_next = (group + 1) * ROWS_PER_GROUP * VL_HALF;
        src_c_ptr = src + offset_curr;
        src_n_ptr = src + offset_next;

        Duplicate(v_max_n0, MIN_VALUE);
        Duplicate(v_max_n1, MIN_VALUE);

        for (uint16_t i = 0; i < ROWS_PER_GROUP; i += 4) {
            // Load four rows from the next group for Max reduction.
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_n_curr, src_n_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_n_next, src_n_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_n_curr_p1, src_n_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_n_next_p1, src_n_ptr, VL_HALF);

            // Load the matching four rows from the current group for Exp and quantization.
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr, src_c_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next, src_c_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr_p1, src_c_ptr, VL_HALF);
            LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next_p1, src_c_ptr, VL_HALF);

            Max(v_max_n0, v_max_n0, v_data_n_curr, preg_all_16bit);
            Max(v_max_n1, v_max_n1, v_data_n_next, preg_all_16bit);
            Max(v_max_n0, v_max_n0, v_data_n_curr_p1, preg_all_16bit);
            Max(v_max_n1, v_max_n1, v_data_n_next_p1, preg_all_16bit);

            COMPUTE_EXP_APPROX(v_data_c_curr, v_data_c_next, v_max_c0)
            COMPUTE_EXP_APPROX(v_data_c_curr_p1, v_data_c_next_p1, v_max_c0)
            QUANT_AND_STORE_FP4(v_data_c_curr, v_data_c_next, v_data_c_curr_p1, v_data_c_next_p1)
        }
        UPDATE_AND_STORE_MAX(v_max_n0, v_max_n1, group)
    }

    //========== global max =====================
    // 此时 v_cur_max 前 64 个元素保存了前 256 行的局部最大值，后 64 个元素保存了后 256 行的局部最大值。
    // 利用 Interleave/DeInterleave 将这两半元素对齐比较，从而高效求出跨整个 512 行（64列）的全局最大值 K。
    Interleave(vTmp0, vTmp1, v_cur_max, v_cur_max);
    Max(vTmp0, vTmp0, vTmp1, preg_all_16bit); // 00 11 22, ..., 63.63 (两半的最大值)
    DeInterleave(v_cur_max, vTmp1, vTmp0, vTmp0); // 0...63; 0...63 (将全局最大值 K 写回原位)
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(global_max_cur, v_cur_max, preg_all_16bit);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    Sub(v_diff, v_cur_max, v_old_max, preg_all_16bit); // v_diff is max mxvals
    Reduce<AscendC::Reg::ReduceType::MAX>(v_diff, v_diff, pregM);
    Duplicate(v_diff, v_diff, preg_all_16bit);
    Compares<T, AscendC::CMPMODE::GE>(preg_diff, v_diff, clip, preg_all_16bit);
    StoreAlign<T, StoreDist::DIST_FIRST_ELEMENT_B16>(clip_ptr, v_diff, preg_all_16bit);

    // ====================== 宏流水线尾声 (Epilogue) ======================
    offset_curr = (HALF_GROUPS - 1) * ROWS_PER_GROUP * VL_HALF;
    src_c_ptr = src + offset_curr;

    for (uint16_t i = 0; i < ROWS_PER_GROUP; i += 4) {
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_curr_p1, src_c_ptr, VL_HALF);
        LoadAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(v_data_c_next_p1, src_c_ptr, VL_HALF);

        COMPUTE_EXP_APPROX(v_data_c_curr, v_data_c_next, v_max_c0)
        COMPUTE_EXP_APPROX(v_data_c_curr_p1, v_data_c_next_p1, v_max_c0)
        QUANT_AND_STORE_FP4(v_data_c_curr, v_data_c_next, v_data_c_curr_p1, v_data_c_next_p1)
    }
}

#undef COMPUTE_EXP_APPROX
#undef UPDATE_AND_STORE_MAX
#undef QUANT_AND_STORE_FP4

template <typename T, uint16_t VL_HALF = 128>
__simd_callee__ inline void softmax_mx_callee(__ubuf__ uint8_t *mxDest, __ubuf__ T *expdiff,
    __ubuf__ T *local_group_max, __ubuf__ T *global_max_cur, __ubuf__ T *global_max_his, MaskReg &preg_update,
    uint32_t forceUpdate, const uint16_t halfRows = 128) {
    using namespace MicroAPI;
    forceUpdate *= VL_HALF;
    RegTensor<T> v_max_c0, v_max_c1;
    RegTensor<T> v_old_max, vTmp0, v_expdiff, v_update;
    RegTensor<uint8_t> v_group_max_int8_0, v_group_max_int8_1;

    constexpr uint16_t ROWS_PER_GROUP = 32;
    const uint16_t HALF_GROUPS = halfRows / ROWS_PER_GROUP;
    constexpr uint32_t mxFracSize = 32;

    MaskReg preg_all_16bit = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8bit = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl64 = CreateMask<T, MaskPattern::VL64>();
    MaskReg preg_vl128_not;
    MaskReg preg_force = UpdateMask<uint16_t>(forceUpdate);
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8bit);
    MaskOr(preg_update, preg_update, preg_force, preg_all_16bit);

    LoadAlign(v_old_max, global_max_his); //small
    // 根据 isUpdate 全局标量判断是否需要更新历史最大值 M。
    // 出于硬件效率，若 Block 中任意一列触发了 K - M >= 100，则该 Block 的所有列均更新 M_new = max(M, K)。
    LoadAlign(v_update, global_max_cur);
    Max(v_update, v_old_max, v_update, preg_all_16bit); // large, 得到新的历史最大值 M_new
    Sub(v_expdiff, v_old_max, v_update,
        preg_all_16bit); // 计算 M_old - M_new，用于 FlashAttention 后续缩放先前输出 (expdiff)

    Select(vTmp0, v_update, v_old_max, preg_update);
    Adds(vTmp0, vTmp0, (T)-127, preg_all_16bit);

    uint32_t zRowNum = halfRows; //  = ROWS / 32 * 16;
    auto mxDest2 = mxDest + zRowNum / 2 - zRowNum * 4; // 4 = 64 / 16
    for (uint16_t pair = 0; pair < HALF_GROUPS / 2; ++pair) {
        LoadAlign(v_max_c0, local_group_max + pair * 2 * VL_HALF);
        LoadAlign(v_max_c1, local_group_max + (pair * 2 + 1) * VL_HALF);
        Sub(v_max_c0, v_max_c0, vTmp0, preg_all_16bit); // k - K
        Sub(v_max_c1, v_max_c1, vTmp0, preg_all_16bit);
        Cast<uint8_t, T, castTraitZero>(v_group_max_int8_0, v_max_c0, preg_all_16bit);
        Cast<uint8_t, T, castTraitOne>(v_group_max_int8_1, v_max_c1, preg_all_16bit);
        Or(v_group_max_int8_0, v_group_max_int8_0, v_group_max_int8_1, preg_all_8bit);
        // OCP MX4 标准：对 scale factor 做 +127 的 zero-point shift，并用 Maxs 限制下界为 1。
        Maxs(v_group_max_int8_0, v_group_max_int8_0, (uint8_t)1, preg_all_8bit); // [-126, CLIP]
        // mxDest 使用分形块（如 NZ Format）排布，32-byte (mxFracSize) 只是 block 的基址步长，无实际内存重叠。
        StoreAlign<uint8_t, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
            mxDest + pair * mxFracSize, v_group_max_int8_0, (uint32_t)HALF_GROUPS, preg_vl128);
        StoreAlign<uint8_t, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
            mxDest2 + pair * mxFracSize, v_group_max_int8_0, (uint32_t)HALF_GROUPS, preg_vl128_not);
    }

    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(global_max_his, v_update, preg_update);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM>(expdiff, v_expdiff, preg_update);
}

template <typename T, typename T2, uint16_t VL_HALF = 128>
__simd_vf__ inline void softmax_fused_vf(__ubuf__ T2 *pDest, __ubuf__ T *clip_ptr, __ubuf__ T *src,
    __ubuf__ T *localGroupMax, __ubuf__ T *globalMaxCur, __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr,
    __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff, uint16_t isFirst, uint32_t forceUpdate, uint16_t halfRows, half clip,
    uint32_t qtail) {
    using namespace MicroAPI;
    MaskReg preg_update; // ! isfirst == true, algo will always output all_zero mask
    __ubuf__ T *fp4GlobalMax = isFirst ? globalMaxHis : globalMaxCur;

    softmax_fp4_callee<T, T2, VL_HALF>(pDest, clip_ptr, src, localGroupMax, fp4GlobalMax, globalMaxHis, idxGroupPtr,
        preg_update, halfRows, clip, qtail);
    softmax_mx_callee<T, VL_HALF>(
        mxDest, expDiff, localGroupMax, globalMaxCur, globalMaxHis, preg_update, forceUpdate, halfRows);
}

// Fixed 64x256 rr-path fusion.  Four k_g vectors stay in registers until the
// global K/update decision is available; MX consumes them in two adjacent
// pairs.  localGroupMax remains a write-only compatibility output and is
// never loaded by this VF.
template <typename T, typename T2, uint16_t VL_HALF = 128>
__simd_vf__ inline void softmax_indexed_fused_64x256_vf(__ubuf__ T2 *pDest, __ubuf__ T *clipPtr, __ubuf__ T *src,
    //__ubuf__ T *localGroupMax, //  __ubuf__ T *globalMaxCur, is not needed when merge mx and fp4
    __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr, __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff,
    uint32_t forceUpdate, half clip, uint32_t qtail) {
    using namespace MicroAPI;
    constexpr uint16_t ROWS_PER_GROUP = 32U;
    constexpr uint16_t GROUPS = 4U;
    constexpr uint16_t HALF_ROWS = 128U;
    constexpr uint16_t QUADS_PER_GROUP = ROWS_PER_GROUP / 4U;
    constexpr uint32_t MX_FRAC_SIZE = 32U;
    constexpr half k = 2.603515625;
    constexpr half b = 7.8046875;
    constexpr half tau = -5.046875;

    enum HalfRegSlot : uint16_t {
        REDUCE_EVEN = 0,
        REDUCE_ODD,
        DATA_0,
        DATA_1,
        DATA_2,
        DATA_3,
        NEXT_0,
        NEXT_1,
        NEXT_2,
        NEXT_3,
        SCALE,
        GLOBAL_K,
        GLOBAL_TMP,
        HISTORY,
        DIFF,
        UPDATE,
        EXP_DIFF,
        MX_BASE,
        HALF_REG_COUNT
    };

    RegTensor<T> halfReg[HALF_REG_COUNT];
    RegTensor<T> kReg[GROUPS];
    RegTensor<half> v_k, v_b;
    RegTensor<float4_e2m1x2_t> fp4Reg[4];
    RegTensor<uint8_t> byteReg[3];

    MaskReg pregAll16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg pregAll8 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg pregVl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg pregVl128Not;
    MaskReg pregUpdate, preg_cmp0, preg_cmp1;
    MaskNot(pregVl128Not, pregVl128, pregAll8);
    uint32_t m = qtail;
    MaskReg pregTail = UpdateMask<uint16_t>(m);

    uint32_t forceUpdateMask = forceUpdate * VL_HALF;
    MaskReg pregForce = UpdateMask<uint16_t>(forceUpdateMask);

    __ubuf__ uint8_t *pDestLow = reinterpret_cast<__ubuf__ uint8_t *>(pDest);
    __ubuf__ uint8_t *pDestHigh = pDestLow + HALF_ROWS * 32U - 128U;

    LoadAlign(halfReg[HISTORY], globalMaxHis);
    LoadAlign(byteReg[2], idxGroupPtr);
    Duplicate(halfReg[GLOBAL_K], MIN_VALUE);
    Duplicate(v_k, k);
    Duplicate(v_b, b);

    __ubuf__ T *groupSrc = src;
    Duplicate(halfReg[REDUCE_EVEN], MIN_VALUE);
    Duplicate(halfReg[REDUCE_ODD], MIN_VALUE);
    for (uint16_t quad = 0; quad < QUADS_PER_GROUP; ++quad) {
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_0], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_1], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_2], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_3], groupSrc, VL_HALF);
        Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[DATA_0], pregAll16);
        Max(halfReg[REDUCE_ODD], halfReg[REDUCE_ODD], halfReg[DATA_1], pregAll16);
        Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[DATA_2], pregAll16);
        Max(halfReg[REDUCE_ODD], halfReg[REDUCE_ODD], halfReg[DATA_3], pregAll16);
    }
    Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[REDUCE_ODD], pregAll16);
    Muls(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], INV_LN2, pregAll16);
    Truncate<T, RoundMode::CAST_CEIL>(kReg[0], halfReg[REDUCE_EVEN], pregAll16);
    Max(halfReg[GLOBAL_K], halfReg[GLOBAL_K], kReg[0], pregAll16);
    // StoreAlign<T, StoreDist::DIST_NORM>(localGroupMax, kReg[0], pregAll16);
    Muls(halfReg[SCALE], kReg[0], LN2, pregAll16);

#define PROCESS_CORE_GROUP(CURRENT, NEXT) \
    do { \
        __ubuf__ T *currentSrc = src + (CURRENT) * ROWS_PER_GROUP * VL_HALF; \
        __ubuf__ T *nextSrc = src + (NEXT) * ROWS_PER_GROUP * VL_HALF; \
        Duplicate(halfReg[REDUCE_EVEN], MIN_VALUE); \
        Duplicate(halfReg[REDUCE_ODD], MIN_VALUE); \
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_0], nextSrc, VL_HALF); \
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_1], nextSrc, VL_HALF); \
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_2], nextSrc, VL_HALF); \
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_3], nextSrc, VL_HALF); \
        for (uint16_t quad = 0; quad < QUADS_PER_GROUP; ++quad) { \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_0], currentSrc, VL_HALF); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_1], currentSrc, VL_HALF); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_2], currentSrc, VL_HALF); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_3], currentSrc, VL_HALF); \
            Sub(halfReg[DATA_0], halfReg[DATA_0], halfReg[SCALE], pregAll16); \
            Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[NEXT_0], pregAll16); \
            MulDstAdd(halfReg[DATA_0], v_k, v_b, pregAll16); \
            Sub(halfReg[DATA_1], halfReg[DATA_1], halfReg[SCALE], pregAll16); \
            Max(halfReg[REDUCE_ODD], halfReg[REDUCE_ODD], halfReg[NEXT_1], pregAll16); \
            MulDstAdd(halfReg[DATA_1], v_k, v_b, pregAll16); \
            Sub(halfReg[DATA_2], halfReg[DATA_2], halfReg[SCALE], pregAll16); \
            Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[NEXT_2], pregAll16); \
            MulDstAdd(halfReg[DATA_2], v_k, v_b, pregAll16); \
            Sub(halfReg[DATA_3], halfReg[DATA_3], halfReg[SCALE], pregAll16); \
            Max(halfReg[REDUCE_ODD], halfReg[REDUCE_ODD], halfReg[NEXT_3], pregAll16); \
            MulDstAdd(halfReg[DATA_3], v_k, v_b, pregAll16); \
            Compares<T, AscendC::CMPMODE::GE>(preg_cmp0, halfReg[DATA_0], tau, pregAll16); \
            Compares<T, AscendC::CMPMODE::GE>(preg_cmp1, halfReg[DATA_1], tau, pregAll16); \
            Maxs(halfReg[DATA_0], halfReg[DATA_0], (T)1.0f, preg_cmp0); \
            Maxs(halfReg[DATA_1], halfReg[DATA_1], (T)1.0f, preg_cmp1); \
            Compares<T, AscendC::CMPMODE::GE>(preg_cmp0, halfReg[DATA_2], tau, pregAll16); \
            Compares<T, AscendC::CMPMODE::GE>(preg_cmp1, halfReg[DATA_3], tau, pregAll16); \
            Maxs(halfReg[DATA_2], halfReg[DATA_2], (T)1.0f, preg_cmp0); \
            Maxs(halfReg[DATA_3], halfReg[DATA_3], (T)1.0f, preg_cmp1); \
            Cast<int4x2_t, half, castTraitZeroTrunc>((RegTensor<int4x2_t> &)fp4Reg[0], halfReg[DATA_0], pregAll16); \
            Cast<int4x2_t, half, castTraitOneTrunc>((RegTensor<int4x2_t> &)fp4Reg[1], halfReg[DATA_1], pregAll16); \
            Cast<int4x2_t, half, castTraitTwoTrunc>((RegTensor<int4x2_t> &)fp4Reg[2], halfReg[DATA_2], pregAll16); \
            Cast<int4x2_t, half, castTraitThreeTrunc>((RegTensor<int4x2_t> &)fp4Reg[3], halfReg[DATA_3], pregAll16); \
            Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), \
                reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[1]), pregAll8); \
            Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]), \
                reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[3]), pregAll8); \
            Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), \
                reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]), pregAll8); \
            Gather(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), \
                reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), byteReg[2]); \
            StoreAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>( \
                pDestLow, reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), 128U, pregVl128); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_0], nextSrc, VL_HALF); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_1], nextSrc, VL_HALF); \
            StoreAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>( \
                pDestHigh, reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), 128U, pregVl128Not); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_2], nextSrc, VL_HALF); \
            LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[NEXT_3], nextSrc, VL_HALF); \
        } \
        Max(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], halfReg[REDUCE_ODD], pregAll16); \
        Muls(halfReg[REDUCE_EVEN], halfReg[REDUCE_EVEN], INV_LN2, pregAll16); \
        Truncate<T, RoundMode::CAST_CEIL>(kReg[(NEXT)], halfReg[REDUCE_EVEN], pregAll16); \
        Max(halfReg[GLOBAL_K], halfReg[GLOBAL_K], kReg[(NEXT)], pregAll16); \
        Muls(halfReg[SCALE], kReg[(NEXT)], LN2, pregAll16); \
    } while (0) // StoreAlign<T, StoreDist::DIST_NORM>(localGroupMax + (NEXT) * VL_HALF, kReg[(NEXT)], pregAll16);

    PROCESS_CORE_GROUP(0U, 1U);
    PROCESS_CORE_GROUP(1U, 2U);
    PROCESS_CORE_GROUP(2U, 3U);
#undef PROCESS_CORE_GROUP

    Interleave(halfReg[GLOBAL_TMP], halfReg[REDUCE_EVEN], halfReg[GLOBAL_K], halfReg[GLOBAL_K]);
    Max(halfReg[GLOBAL_TMP], halfReg[GLOBAL_TMP], halfReg[REDUCE_EVEN], pregAll16);
    DeInterleave(halfReg[GLOBAL_K], halfReg[REDUCE_EVEN], halfReg[GLOBAL_TMP], halfReg[GLOBAL_TMP]);

    Sub(halfReg[DIFF], halfReg[GLOBAL_K], halfReg[HISTORY], pregAll16);
    Reduce<AscendC::Reg::ReduceType::MAX>(halfReg[DIFF], halfReg[DIFF], pregTail);
    Duplicate(halfReg[DIFF], halfReg[DIFF], pregAll16);
    Compares<T, CMPMODE::GE>(pregUpdate, halfReg[DIFF], clip, pregAll16);
    StoreAlign<T, StoreDist::DIST_FIRST_ELEMENT_B16>(clipPtr, halfReg[DIFF], pregAll16);

    groupSrc = src + 3U * ROWS_PER_GROUP * VL_HALF;
    for (uint16_t quad = 0; quad < QUADS_PER_GROUP; ++quad) {
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_0], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_1], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_2], groupSrc, VL_HALF);
        LoadAlign<T, PostLiteral::POST_MODE_UPDATE>(halfReg[DATA_3], groupSrc, VL_HALF);
        Sub(halfReg[DATA_0], halfReg[DATA_0], halfReg[SCALE], pregAll16);
        Sub(halfReg[DATA_1], halfReg[DATA_1], halfReg[SCALE], pregAll16);
        MulDstAdd(halfReg[DATA_0], v_k, v_b, pregAll16);
        MulDstAdd(halfReg[DATA_1], v_k, v_b, pregAll16);
        Sub(halfReg[DATA_2], halfReg[DATA_2], halfReg[SCALE], pregAll16);
        Sub(halfReg[DATA_3], halfReg[DATA_3], halfReg[SCALE], pregAll16);
        MulDstAdd(halfReg[DATA_2], v_k, v_b, pregAll16);
        MulDstAdd(halfReg[DATA_3], v_k, v_b, pregAll16);
        Compares<T, AscendC::CMPMODE::GE>(preg_cmp0, halfReg[DATA_0], tau, pregAll16);
        Compares<T, AscendC::CMPMODE::GE>(preg_cmp1, halfReg[DATA_1], tau, pregAll16);
        Maxs(halfReg[DATA_0], halfReg[DATA_0], (T)1.0f, preg_cmp0);
        Maxs(halfReg[DATA_1], halfReg[DATA_1], (T)1.0f, preg_cmp1);
        Compares<T, AscendC::CMPMODE::GE>(preg_cmp0, halfReg[DATA_2], tau, pregAll16);
        Compares<T, AscendC::CMPMODE::GE>(preg_cmp1, halfReg[DATA_3], tau, pregAll16);
        Maxs(halfReg[DATA_2], halfReg[DATA_2], (T)1.0f, preg_cmp0);
        Maxs(halfReg[DATA_3], halfReg[DATA_3], (T)1.0f, preg_cmp1);
        Cast<int4x2_t, half, castTraitZeroTrunc>((RegTensor<int4x2_t> &)fp4Reg[0], halfReg[DATA_0], pregAll16);
        Cast<int4x2_t, half, castTraitOneTrunc>((RegTensor<int4x2_t> &)fp4Reg[1], halfReg[DATA_1], pregAll16);
        Cast<int4x2_t, half, castTraitTwoTrunc>((RegTensor<int4x2_t> &)fp4Reg[2], halfReg[DATA_2], pregAll16);
        Cast<int4x2_t, half, castTraitThreeTrunc>((RegTensor<int4x2_t> &)fp4Reg[3], halfReg[DATA_3], pregAll16);
        Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]),
            reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[1]), pregAll8);
        Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]),
            reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[3]), pregAll8);
        Or(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]),
            reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[2]), pregAll8);
        Gather(reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]),
            byteReg[2]);
        StoreAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(
            pDestLow, reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), 128U, pregVl128);
        StoreAlign<uint8_t, PostLiteral::POST_MODE_UPDATE>(
            pDestHigh, reinterpret_cast<RegTensor<uint8_t> &>(fp4Reg[0]), 128U, pregVl128Not);
    }

    MaskOr(pregUpdate, pregUpdate, pregForce, pregAll16);
    // curmax = globalk

    Max(halfReg[UPDATE], halfReg[GLOBAL_K], halfReg[HISTORY], pregAll16); // M_new = max(M, M_old)
    Select(halfReg[MX_BASE], halfReg[UPDATE], halfReg[HISTORY], pregUpdate); // if First, prgUpdate=1
    Adds(halfReg[MX_BASE], halfReg[MX_BASE], static_cast<T>(-127), pregAll16);

    Sub(halfReg[EXP_DIFF], halfReg[HISTORY], halfReg[UPDATE], pregAll16); // if update  M_old - M_new + 127

    __ubuf__ uint8_t *mxDestHigh = mxDest + HALF_ROWS / 2U - HALF_ROWS * 4U;

#define EMIT_MX_PAIR(EVEN, ODD, PAIR) \
    do { \
        Sub(kReg[(EVEN)], kReg[(EVEN)], halfReg[MX_BASE], pregAll16); \
        Sub(kReg[(ODD)], kReg[(ODD)], halfReg[MX_BASE], pregAll16); \
        Cast<uint8_t, T, castTraitZero>(byteReg[0], kReg[(EVEN)], pregAll16); \
        Cast<uint8_t, T, castTraitOne>(byteReg[1], kReg[(ODD)], pregAll16); \
        Or(byteReg[0], byteReg[0], byteReg[1], pregAll8); \
        Maxs(byteReg[0], byteReg[0], static_cast<uint8_t>(1), pregAll8); \
        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>( \
            mxDest + (PAIR) * MX_FRAC_SIZE, byteReg[0], GROUPS, pregVl128); \
        StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>( \
            mxDestHigh + (PAIR) * MX_FRAC_SIZE, byteReg[0], GROUPS, pregVl128Not); \
    } while (0)

    EMIT_MX_PAIR(0U, 1U, 0U);
    EMIT_MX_PAIR(2U, 3U, 1U);
#undef EMIT_MX_PAIR

    StoreAlign<T, StoreDist::DIST_NORM>(globalMaxHis, halfReg[UPDATE], pregUpdate); // if first also Store global His
    StoreAlign<T, StoreDist::DIST_NORM>(expDiff, halfReg[EXP_DIFF], pregUpdate);
}

// 预埋静态索引数组，专用于给 Gather 指令提供重排映射（打散恢复为连续 logits^T 排布）：
// 前 128 字节恢复 Row i, i+1, i+2, i+3，即从 0,4,8... / 1,5,9... 离散位置提取拼装。
// 后 128 字节恢复 Row i+256 到 i+259，即从 128,132,136... / 129,133,137... 提取。
__aicore__ inline void GetIdxGroup(const LocalTensor<uint8_t> &index) {
    constexpr uint64_t idx_uint64[32]{PACK_U8_TO_U64(0, 4, 8, 12, 16, 20, 24, 28),
        PACK_U8_TO_U64(32, 36, 40, 44, 48, 52, 56, 60), PACK_U8_TO_U64(64, 68, 72, 76, 80, 84, 88, 92),
        PACK_U8_TO_U64(96, 100, 104, 108, 112, 116, 120, 124),

        PACK_U8_TO_U64(1, 5, 9, 13, 17, 21, 25, 29), PACK_U8_TO_U64(33, 37, 41, 45, 49, 53, 57, 61),
        PACK_U8_TO_U64(65, 69, 73, 77, 81, 85, 89, 93), PACK_U8_TO_U64(97, 101, 105, 109, 113, 117, 121, 125),

        PACK_U8_TO_U64(2, 6, 10, 14, 18, 22, 26, 30), PACK_U8_TO_U64(34, 38, 42, 46, 50, 54, 58, 62),
        PACK_U8_TO_U64(66, 70, 74, 78, 82, 86, 90, 94), PACK_U8_TO_U64(98, 102, 106, 110, 114, 118, 122, 126),

        PACK_U8_TO_U64(3, 7, 11, 15, 19, 23, 27, 31), PACK_U8_TO_U64(35, 39, 43, 47, 51, 55, 59, 63),
        PACK_U8_TO_U64(67, 71, 75, 79, 83, 87, 91, 95), PACK_U8_TO_U64(99, 103, 107, 111, 115, 119, 123, 127),

        PACK_U8_TO_U64(128, 132, 136, 140, 144, 148, 152, 156), PACK_U8_TO_U64(160, 164, 168, 172, 176, 180, 184, 188),
        PACK_U8_TO_U64(192, 196, 200, 204, 208, 212, 216, 220), PACK_U8_TO_U64(224, 228, 232, 236, 240, 244, 248, 252),

        PACK_U8_TO_U64(129, 133, 137, 141, 145, 149, 153, 157), PACK_U8_TO_U64(161, 165, 169, 173, 177, 181, 185, 189),
        PACK_U8_TO_U64(193, 197, 201, 205, 209, 213, 217, 221), PACK_U8_TO_U64(225, 229, 233, 237, 241, 245, 249, 253),

        PACK_U8_TO_U64(130, 134, 138, 142, 146, 150, 154, 158), PACK_U8_TO_U64(162, 166, 170, 174, 178, 182, 186, 190),
        PACK_U8_TO_U64(194, 198, 202, 206, 210, 214, 218, 222), PACK_U8_TO_U64(226, 230, 234, 238, 242, 246, 250, 254),

        PACK_U8_TO_U64(131, 135, 139, 143, 147, 151, 155, 159), PACK_U8_TO_U64(163, 167, 171, 175, 179, 183, 187, 191),
        PACK_U8_TO_U64(195, 199, 203, 207, 211, 215, 219, 223), PACK_U8_TO_U64(227, 231, 235, 239, 243, 247, 251, 255)};

    const auto index_uint64 = index.template ReinterpretCast<uint64_t>();

#pragma unroll
    for (auto i = 0; i < 32; i++) {
        index_uint64.SetValue(i, idx_uint64[i]);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
}
#undef PACK_U8_TO_U64

// Deferred raw-pointer entry for the QMode2 three-slot pipeline.  The VF keeps
// the production Softmax/update semantics; only Scalar-side address materialization
// and the immediate V->S wait are removed from the hot issue path.
template <typename T, typename T2>
__aicore__ inline void SoftmaxCallVFDeferredRaw(__ubuf__ T2 *pDest, __ubuf__ T *src, __ubuf__ T *localGroupMax,
    __ubuf__ T *globalMaxCur, __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr, __ubuf__ T *tmp,
    __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff, const bool isFirst, uint16_t forceUpdate, const uint16_t n,
    const half clip, uint32_t qtail) {
    constexpr uint16_t VL_HALF = 128;
    forceUpdate = isFirst ? 1U : forceUpdate;
    softmax_fused_vf<T, T2, VL_HALF>(pDest, tmp, src, localGroupMax, globalMaxCur, globalMaxHis, idxGroupPtr, mxDest,
        expDiff, isFirst, forceUpdate, n / 2, static_cast<half>(clip), qtail);
}

template <typename T, typename T2>
__aicore__ inline void SoftmaxCallVF256TileRaw(__ubuf__ T2 *pDest, __ubuf__ T *src, __ubuf__ T *localGroupMax,
    __ubuf__ T *globalMaxCur, __ubuf__ T *globalMaxHis, __ubuf__ uint8_t *idxGroupPtr, __ubuf__ T *tmp,
    __ubuf__ uint8_t *mxDest, __ubuf__ T *expDiff, const bool isFirst, uint16_t forceUpdate, const uint16_t n,
    const half clip, uint32_t qtail) {
    forceUpdate = isFirst ? 1U : forceUpdate; // if first, his_max = -inf, forceupdate to use cur global K actually
    softmax_indexed_fused_64x256_vf<T, T2, 128U>(pDest, tmp, src, // localGroupMax,
        globalMaxHis, idxGroupPtr, mxDest, expDiff, forceUpdate, clip, qtail);
}

template <typename T, uint16_t VL_HALF = 128>
__simd_vf__ inline void pad_inf_vf(
    __ubuf__ T *src, const uint16_t nRound, const uint16_t n, const uint16_t nlt64 /*is n <=64*/) {
    using namespace MicroAPI;
    RegTensor<T> v_dup;
    MaskReg preg_vl64 = CreateMask<T, MaskPattern::VL64>();
    Duplicate(v_dup, (T)(-60000.0f));
    for (uint16_t t = 0; t < nlt64; t++) {
        for (uint16_t i = 0; i < 64; i++) {
            StoreAlign(src + i * VL_HALF + 64, v_dup, preg_vl64);
        }
        for (uint16_t i = n; i < 64; i++) {
            StoreAlign(src + i * VL_HALF, v_dup, preg_vl64);
        }
    }
    uint16_t ngt64 = nlt64 ^ 1;
    for (uint16_t t = 0; t < ngt64; t++) {
        for (uint16_t i = n - (nRound >> 1); i < (nRound >> 1); i++) {
            StoreAlign(src + i * VL_HALF + 64, v_dup, preg_vl64);
        }
    }
}

template <typename T> __aicore__ inline void SoftmaxPadCallVF(const LocalTensor<T> &srcTensor, uint16_t n) {
    uint16_t nRound = (n + 127) / 128 * 128;
    uint16_t nlt64 = n <= 64 ? 1 : 0;
    __ubuf__ T *src = (__ubuf__ T *)srcTensor.GetPhyAddr();
    pad_inf_vf(src, nRound, n, nlt64);
    PipeBarrier<PIPE_V>();
}

} // namespace Mxfp4ApiApprox
#endif // VF_SOFTMAX_MXFP4_APPROX_H_
