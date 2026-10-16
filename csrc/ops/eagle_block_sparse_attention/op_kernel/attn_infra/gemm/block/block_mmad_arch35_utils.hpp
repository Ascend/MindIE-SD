/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_ARCH35_UTILS_HPP
#define BLOCK_MMAD_ARCH35_UTILS_HPP

namespace NpuArch::Gemm::Block {

namespace MXFP4 {
static constexpr uint32_t KB_BYTE = 1024;

// L1 分配
// pL1Tensor: 16K, 128*256*sizeof(e2m1), BUFCNT=20
static constexpr uint32_t L1_P_BUF_CNT = 20;
static constexpr uint32_t L1_P_BUF_SIZE = 16 * KB_BYTE;
static constexpr uint32_t L1_P_BUF_OFFSET = 0;

// pDescaleL1Tensor: 1.25K, 128*2*(256/64+1)*sizeof(e8m0), BUFCNT=20
static constexpr uint32_t L1_P_SCALE_BUF_CNT = 20;
static constexpr uint32_t L1_P_SCALE_BUF_SIZE = 1280;
static constexpr uint32_t L1_P_SCALE_BUF_OFFSET = 320 * KB_BYTE;

// qL1Tensor: 8K, 128*D*sizeof(e2m1), BUFCNT=2
static constexpr uint32_t L1_Q_BUF_CNT = 2;
static constexpr uint32_t L1_Q_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L1_Q_BUF_OFFSET = 345 * KB_BYTE;

// qDescaleL1Tensor: 0.5K, 128*D/32*sizeof(e8m0), BUFCNT=2
static constexpr uint32_t L1_Q_DESCALE_BUF_CNT = 2;
static constexpr uint32_t L1_Q_DESCALE_BUF_SIZE = 512;
static constexpr uint32_t L1_Q_DESCALE_BUF_OFFSET = 361 * KB_BYTE;

// kvL1Tensor: 32K, 512*D*sizeof(e2m1), BUFCNT=4
// todo 先用2块，避免qk、pv同步冲突
static constexpr uint32_t L1_KV_BUF_CNT = 4;
static constexpr uint32_t L1_KV_BUF_SIZE = 32 * KB_BYTE;
static constexpr uint32_t L1_KV_BUF_OFFSET = 362 * KB_BYTE;

// kvDescaleL1Tensor: 2K, 512*D/32*sizeof(e8m0), BUFCNT=4
// todo 先用2块，避免qk、pv同步冲突
static constexpr uint32_t L1_KV_DESCALE_BUF_CNT = 4;
static constexpr uint32_t L1_KV_DESCALE_BUF_SIZE = 2 * KB_BYTE;
static constexpr uint32_t L1_KV_DESCALE_BUF_OFFSET = 490 * KB_BYTE;

// localGlobalMaxL1: 0.5K, 256*sizeof(half), BUFCNT=4
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_CNT = 4;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_SIZE = 512;
static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_OFFSET = 498 * KB_BYTE;

// [QK/PV 共享 KV 池] event ID：4 路 MTE1_MTE2（MTE2 gather 写 ↔ MTE1 L1→L0 LoadData 读）。
//   QK(K) 与 PV(V) 共用同一组 ID——角色无关门不问槽装 K/V/seed，只保证「MTE1 读完→MTE2 才能覆写」。
//   与 PV 的 PV_L0AB(M_MTE1, 4/5)/PV_L0C(M_FIX, 4) 数值复用，不同 HardEvent 类型独立 flag 存储(结论#7)。
static constexpr uint32_t KV_EVENT0 = 4;
static constexpr uint32_t KV_EVENT1 = 5;
static constexpr uint32_t KV_EVENT2 = 6;
static constexpr uint32_t KV_EVENT3 = 7;

// [QK/PV 共享调试] true=打印槽位地址别名(GetPhyAddr)+游标/门轨迹；生产前改 false(if constexpr 编译期消除)。
static constexpr bool KV_SHARE_DEBUG = true;
// 每个 block 前 N 个 tile 打详细轨迹(ctor 别名打印不受此限)，避免长用例刷屏。
static constexpr uint32_t KV_TRACE_TILES = 40;

// L0 分配
static constexpr uint32_t L0A_QK_BUF_CNT = 2;
static constexpr uint32_t L0A_QK_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L0A_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0A_PV_BUF_CNT = 2;
static constexpr uint32_t L0A_PV_BUF_SIZE = 18 * KB_BYTE;
static constexpr uint32_t L0A_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0B_QK_BUF_CNT = 2;
static constexpr uint32_t L0B_QK_BUF_SIZE = 8 * KB_BYTE;
static constexpr uint32_t L0B_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0B_PV_BUF_CNT = 2;
static constexpr uint32_t L0B_PV_BUF_SIZE = 18 * KB_BYTE;
static constexpr uint32_t L0B_PV_BUF_OFFSET = 16 * KB_BYTE;

static constexpr uint32_t L0C_QK_BUF_CNT = 2;
static constexpr uint32_t L0C_QK_BUF_SIZE = 64 * KB_BYTE;
static constexpr uint32_t L0C_QK_BUF_OFFSET = 0;

static constexpr uint32_t L0C_PV_BUF_CNT = 1;
static constexpr uint32_t L0C_PV_BUF_SIZE = 72 * KB_BYTE;
static constexpr uint32_t L0C_PV_BUF_OFFSET = 128 * KB_BYTE;

static constexpr uint32_t BLOCK_SIZE = 32;

static constexpr uint32_t V_SCALE_L0A_SIZE = (256 / 64) * ((128 + 16) / 16) * 32;

static constexpr uint32_t mBaseSize = 128; // [#9] M 方向基准 tile（对应 QFA mBaseSize），用于 pad 满块判断

static constexpr uint32_t CONST_32 = 32;
static constexpr uint32_t CONST_64 = 64;
static constexpr uint32_t CONST_128 = 128;

static constexpr uint32_t ROW_SUM_PAD_NUM = 16;
static constexpr uint32_t ROW_SUM_NUM = 128;

// 硬件 NZ 分形块维度
static constexpr uint32_t NZ_C0_ELEMS = 16;  // NZ 分形 C0 维元素个数
static constexpr uint32_t FP4_C0_ELEMS = 64; // fp4 数据类型每个 C0 块元素个数

// PV mmad M 维（含 rowsum pad）
static constexpr uint32_t PV_MMAD_M_DIM = ROW_SUM_NUM + ROW_SUM_PAD_NUM; // = 144

// S2 方向 base tile 尺寸
static constexpr uint32_t S2_BASE_TILE_SIZE = 256;

// InitL0BufferForReduceSum seed fill 模式
// V data = fp4 4.0 打包，V scale = e8m0 0.25 打包 → 有效值 4.0 × 0.25 = 1.0
static constexpr uint16_t SEED_V_DATA_FILL = 0x6666;
static constexpr uint16_t SEED_V_SCALE_FILL = 0x7D7D;
static constexpr uint16_t ZERO_FILL_PATTERN = 0x0000;

// GM SetGlobalBuffer 最大长度哨兵值
static constexpr uint32_t GM_MAX_BUFFER_LEN = 0x7FFFFFFFu;

} // namespace MXFP4

} // namespace NpuArch::Gemm::Block

#endif // BLOCK_MMAD_ARCH35_UTILS_HPP
