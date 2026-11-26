/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_MASK2IDX_ARCH35
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_MASK2IDX_ARCH35

#include "../../../attn_infra/base_defs.hpp"
#include "../../../attn_infra/arch/resource.hpp"
#include "../../../attn_infra/epilogue/dispatch_policy.hpp"
#include "../../../attn_infra/gemm_coord.hpp"
#include "../../../attn_infra/matrix_coord.hpp"
#include "./vf/vf_mask2idx.h"

namespace NpuArch::Epilogue::Block {

template <class ElementSparseMask_, class ElementSparseIdx_, class ElementSparseCount_>
class BlockEpilogue<EpilogueBsaMask2Idx, ElementSparseMask_, ElementSparseIdx_, ElementSparseCount_> {
  public:
    using DispatchPolicy = EpilogueBsaMask2Idx;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSparseMask = ElementSparseMask_;
    using ElementSparseIdx = ElementSparseIdx_;
    using ElementSparseCount = ElementSparseCount_;

    static constexpr uint32_t IO_STAGES = DispatchPolicy::IO_STAGES;
    static constexpr uint32_t MAX_Y_BLOCK_NUM_ALIGNED = 16384;
    static constexpr uint32_t VL_U32 = Mask2IdxApi::VL_U32;
    static constexpr uint32_t SEG_COUNT_UB_BLOCK = 128U;
    static constexpr uint32_t ACCUM_COUNT_UB_BLOCK = 128U;
    static_assert(IO_STAGES == 2, "mask2idx requires two pipeline buffers");

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource) {
        constexpr uint32_t MASK_UB_OFF = 0;
        constexpr uint32_t IDX_I32_UB_OFF = MASK_UB_OFF + 2U * MAX_Y_BLOCK_NUM_ALIGNED;
        constexpr uint32_t SEG_CNT_UB_OFF = IDX_I32_UB_OFF + 2U * 4U * MAX_Y_BLOCK_NUM_ALIGNED;
        constexpr uint32_t ACCUM_CNT_UB_OFF = SEG_CNT_UB_OFF + 2U * SEG_COUNT_UB_BLOCK;

        static_assert(
            ACCUM_CNT_UB_OFF + IO_STAGES * ACCUM_COUNT_UB_BLOCK <= ArchTag::UB_SIZE, "mask2idx buffers must fit in UB");

        for (uint32_t i = 0; i < IO_STAGES; i++) {
            maskUb8[i] =
                resource.ubBuf.template GetBufferByByte<ElementSparseMask>(MASK_UB_OFF + i * MAX_Y_BLOCK_NUM_ALIGNED);
            idxUbI32[i] = resource.ubBuf.template GetBufferByByte<ElementSparseIdx>(
                IDX_I32_UB_OFF + i * 4U * MAX_Y_BLOCK_NUM_ALIGNED);
            segCountUb[i] =
                resource.ubBuf.template GetBufferByByte<ElementSparseCount>(SEG_CNT_UB_OFF + i * SEG_COUNT_UB_BLOCK);
            countAccumUb[i] = resource.ubBuf.template GetBufferByByte<ElementSparseCount>(
                ACCUM_CNT_UB_OFF + i * ACCUM_COUNT_UB_BLOCK);
        }
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementSparseMask> gSparseMask,
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx, AscendC::GlobalTensor<ElementSparseCount> gSparseCount,
        uint32_t totalRowNumBlockMask, uint32_t yBlockNumAligned, uint32_t avgRowPerSubCore,
        uint32_t preActiveSubCoreNum) {
        uint32_t subCoreIdx = AscendC::GetBlockIdx();
        uint32_t curSubCoreRowOffset = subCoreIdx * avgRowPerSubCore;
        uint32_t actDealtRow =
            (subCoreIdx == preActiveSubCoreNum - 1) ? (totalRowNumBlockMask - curSubCoreRowOffset) : avgRowPerSubCore;

        if (subCoreIdx >= preActiveSubCoreNum) {
            return;
        }

        uint32_t segmentSize = MAX_Y_BLOCK_NUM_ALIGNED;
        uint32_t segmentLoop = CeilDiv(yBlockNumAligned, segmentSize);
        uint32_t rowStrideU8 = CeilDiv(segmentSize, VL_U32) * VL_U32;
        uint32_t R = 1;
        if (segmentSize <= 128U) {
            R = 32;
        } else if (segmentSize <= 256U) {
            R = 16;
        } else if (segmentSize <= 512U) {
            R = 8;
        } else {
            R = 4;
        }
        while (R > 1 && R * rowStrideU8 > MAX_Y_BLOCK_NUM_ALIGNED) {
            R >>= 1;
        }

        uint32_t batchLoop = CeilDiv(actDealtRow, R);
        uint32_t totalTasks = batchLoop * segmentLoop;

        if (totalTasks == 0) {
            return;
        }

        // 1. Prologue: 提前预取第 0 个任务数据到 buffer 0
        {
            uint32_t actRowCur0 = (batchLoop == 1) ? actDealtRow : R;
            uint32_t globalRowOffset0 = curSubCoreRowOffset;
            uint32_t segStart0 = 0;
            uint32_t segCols0 = (yBlockNumAligned < segmentSize) ? yBlockNumAligned : segmentSize;
            uint32_t segRowStrideU8_0 = CeilDiv(segCols0, VL_U32) * VL_U32;

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::DataCopyPad(maskUb8[0], gSparseMask[globalRowOffset0 * yBlockNumAligned + segStart0],
                AscendC::DataCopyExtParams(actRowCur0, segCols0 * sizeof(ElementSparseMask),
                    (yBlockNumAligned - segCols0) * sizeof(ElementSparseMask), 0, 0),
                AscendC::DataCopyPadExtParams<ElementSparseMask>(true, 0,
                    static_cast<uint8_t>((segRowStrideU8_0 - segCols0 > 32U) ? 32U : (segRowStrideU8_0 - segCols0)),
                    0));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        }

        // 2. 主流水循环：当前任务计算/写回与下一任务数据搬入异步重叠
        for (uint32_t task = 0; task < totalTasks; task++) {
            uint32_t pp = task % 2;
            uint32_t next_pp = 1 - pp;

            uint32_t b = task / segmentLoop;
            uint32_t seg = task % segmentLoop;
            uint32_t curBatchRowOffset = b * R;
            uint32_t actRowCur = (b == batchLoop - 1) ? (actDealtRow - curBatchRowOffset) : R;
            uint32_t globalRowOffset = curSubCoreRowOffset + curBatchRowOffset;
            uint32_t segStart = seg * segmentSize;
            uint32_t segCols =
                (yBlockNumAligned - segStart < segmentSize) ? (yBlockNumAligned - segStart) : segmentSize;
            uint32_t segRowStrideU8 = CeilDiv(segCols, VL_U32) * VL_U32;
            uint32_t segRowStrideI32Elems = segRowStrideU8;

            // 预取下一轮任务 (task + 1) 到 next_pp buffer
            if (task + 1 < totalTasks) {
                uint32_t next_task = task + 1;
                uint32_t next_b = next_task / segmentLoop;
                uint32_t next_seg = next_task % segmentLoop;
                uint32_t next_curBatchRowOffset = next_b * R;
                uint32_t next_actRowCur = (next_b == batchLoop - 1) ? (actDealtRow - next_curBatchRowOffset) : R;
                uint32_t next_globalRowOffset = curSubCoreRowOffset + next_curBatchRowOffset;
                uint32_t next_segStart = next_seg * segmentSize;
                uint32_t next_segCols =
                    (yBlockNumAligned - next_segStart < segmentSize) ? (yBlockNumAligned - next_segStart) : segmentSize;
                uint32_t next_segRowStrideU8 = CeilDiv(next_segCols, VL_U32) * VL_U32;

                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(next_pp);
                AscendC::DataCopyPad(maskUb8[next_pp],
                    gSparseMask[next_globalRowOffset * yBlockNumAligned + next_segStart],
                    AscendC::DataCopyExtParams(next_actRowCur, next_segCols * sizeof(ElementSparseMask),
                        (yBlockNumAligned - next_segCols) * sizeof(ElementSparseMask), 0, 0),
                    AscendC::DataCopyPadExtParams<ElementSparseMask>(true, 0,
                        static_cast<uint8_t>(
                            (next_segRowStrideU8 - next_segCols > 32U) ? 32U : (next_segRowStrideU8 - next_segCols)),
                        0));
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(next_pp);
            }

            // 当前任务在 buffer pp 上执行计算
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(pp);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pp);

            Mask2IdxApi::Mask2IdxRowVF<int32_t>((__ubuf__ int32_t *)idxUbI32[pp].GetPhyAddr(),
                (__ubuf__ int32_t *)((segmentLoop == 1) ? countAccumUb[pp].GetPhyAddr() : segCountUb[pp].GetPhyAddr()),
                (__ubuf__ uint8_t *)maskUb8[pp].GetPhyAddr(), (__ubuf__ int32_t *)countAccumUb[pp].GetPhyAddr(),
                (__ubuf__ int32_t *)countAccumUb[1 - pp].GetPhyAddr(), segCols, segStart, actRowCur,
                segRowStrideI32Elems, segRowStrideU8, segmentLoop, seg);

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(pp);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pp);

            // 当前任务在 buffer pp 上写回 GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pp);

            if (segmentLoop == 1) {
                AscendC::DataCopyPad(gSparseIdx[globalRowOffset * yBlockNumAligned], idxUbI32[pp],
                    AscendC::DataCopyExtParams(static_cast<uint16_t>(actRowCur),
                        yBlockNumAligned * sizeof(ElementSparseIdx), (segRowStrideI32Elems - yBlockNumAligned) / 8U, 0,
                        0));
                AscendC::DataCopyPad(gSparseCount[globalRowOffset], countAccumUb[pp],
                    AscendC::DataCopyExtParams(1, actRowCur * sizeof(ElementSparseCount), 0, 0, 0));
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_S>(pp);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(pp);
                uint32_t segCount = static_cast<uint32_t>(segCountUb[pp].GetValue(0));
                uint32_t accumAfter = static_cast<uint32_t>(countAccumUb[pp].GetValue(0));
                uint32_t gmOffset = accumAfter - segCount;
                if (segCount > 0) {
                    AscendC::DataCopyPad(gSparseIdx[globalRowOffset * yBlockNumAligned + gmOffset], idxUbI32[pp],
                        AscendC::DataCopyExtParams(1, segCount * sizeof(ElementSparseIdx), 0, 0, 0));
                }

                if (seg == segmentLoop - 1) {
                    AscendC::DataCopyPad(gSparseCount[globalRowOffset], countAccumUb[pp],
                        AscendC::DataCopyExtParams(1, sizeof(ElementSparseCount), 0, 0, 0));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pp);
        }

        // 3. 收尾同步
    }

  private:
    AscendC::LocalTensor<ElementSparseMask> maskUb8[IO_STAGES];
    AscendC::LocalTensor<ElementSparseIdx> idxUbI32[IO_STAGES];
    AscendC::LocalTensor<ElementSparseCount> segCountUb[IO_STAGES];
    AscendC::LocalTensor<ElementSparseCount> countAccumUb[IO_STAGES];
};
} // namespace NpuArch::Epilogue::Block
#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_MASK2IDX_ARCH35
