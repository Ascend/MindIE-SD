/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
#define GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Block {

template <class ElementLocalGlobalMax_>
struct BlockMmadTla<CopyGlobalMaxL1ToUBBsa, void, void, ElementLocalGlobalMax_, void, void, void, void, void> {
public:
    using DispatchPolicy = CopyGlobalMaxL1ToUBBsa;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementLocalGlobalMax = ElementLocalGlobalMax_;

    static constexpr uint32_t L0_STAGES = DispatchPolicy::L0_STAGES;

    static constexpr uint32_t L1_SINGLE_GLOBAL_MAX_SIZE = 128;

    static constexpr uint8_t VEC0 = 0;
    static constexpr uint8_t VEC1 = 1;
    static constexpr uint32_t SUB_VEC_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = 32;

    __aicore__ inline BlockMmadTla(Arch::Resource<ArchTag> &resource)
    {
        // L1 区域顺序（与 QFA 一致）：P 数据 | P-scale | Q 数据 | Q-scale | KV(=V) 数据 | KV(=V)-scale
        // localGlobalMaxL1 = 256*4
        for (uint32_t i = 0; i < MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            localGlobalMaxL1[i] = resource.l1Buf.template GetBufferByByte<ElementLocalGlobalMax>(
                MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
    }
    __aicore__ inline ~BlockMmadTla() {}

    template <class TensorGMaxUb, class TileInfoT>
    __aicore__ inline void operator()(TensorGMaxUb &gMaxUbTensor, TileInfoT &delay3TileInfo)
    {
        // AscendC::PRINTF("CopyGMaxL1ToUb \n");
        uint32_t blockCount = static_cast<uint32_t>(tla::get<0>(gMaxUbTensor.shape()));
        uint32_t blockLenPerSubVec = static_cast<uint32_t>(tla::get<1>(gMaxUbTensor.shape())) / SUB_VEC_NUM /
                                     BLOCK_SIZE * sizeof(ElementLocalGlobalMax);

        // // [TODO] 此处也是假数据
        // AscendC::InitConstValueParams<ElementLocalGlobalMax> localGlobalMaxL1FillParams(1,
        // static_cast<uint16_t>(MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT  *
        // sizeof(ElementLocalGlobalMax) / BLOCK_SIZE ), 0, static_cast<half>(1.865));
        // AscendC::Fill(localGlobalMaxL1[0], localGlobalMaxL1FillParams);          // localGlobalMaxL1 数据
        // AscendC::PipeBarrier<PIPE_MTE2>();

        // AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(7);
        // AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(7);

        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax0 = localGlobalMaxL1[delay3TileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementLocalGlobalMax> localGlobalMax1 =
            localGlobalMaxL1[delay3TileInfo.tileMaxIdx][L1_SINGLE_GLOBAL_MAX_SIZE];
        AscendC::LocalTensor<ElementLocalGlobalMax> peerGlobalMax = gMaxUbTensor.data();

        AscendC::DataCopyParams intriParams;
        intriParams.blockCount = static_cast<uint16_t>(blockCount); // 连续数据块个数为1
        intriParams.blockLen =
            static_cast<uint16_t>(blockLenPerSubVec); // 连续数据块长度，单位为DataBlock，此处长度为128个half元素
        intriParams.srcGap = 0;                       // 源操作数做连续搬运
        intriParams.dstGap = 0;                       // 目的操作数连续排布
        AscendC::DataCopyL1ToUB<ElementLocalGlobalMax, VEC0>(peerGlobalMax, localGlobalMax0, intriParams);
        AscendC::DataCopyL1ToUB<ElementLocalGlobalMax, VEC1>(peerGlobalMax, localGlobalMax1, intriParams);
    }

protected:
    // MXFP4 专用：E8M0 scale 的 L1 buffer（V-scale 4 份，P-scale 20 份）+ MX 版 L1->L0 拷贝器
    AscendC::LocalTensor<half> localGlobalMaxL1[MXFP4::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_COPY_GLOBAL_MAX_L1_TO_UB_ARCH35_MXFP4_HPP
