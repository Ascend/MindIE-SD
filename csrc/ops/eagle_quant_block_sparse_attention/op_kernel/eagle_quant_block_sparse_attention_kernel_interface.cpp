/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file eagle_quant_block_sparse_attention_interface.cpp
 * \brief Block Sparse Attention Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 310)
#include "arch35/eagle_quant_block_sparse_attention_kernel_arch35_base.h"
#include "arch35/eagle_quant_block_sparse_attention_kernel_arch35_qmode1.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 310)

using namespace BsaKernelArch35;

template <class InQKDtype, class InVDtype, class OUTPODType, class SMDtype, class REDtype, Format qFormat,
    Format kvFormat, int quantMode = 1, std::enable_if_t<quantMode == 1, int> = 0>
__global__ __aicore__ void BsaInferIntfRegular(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR query_scale, GM_ADDR key_scale, GM_ADDR value_scale, GM_ADDR o, GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen, GM_ADDR blockSparseMask, GM_ADDR workspace, GM_ADDR tiling) {
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InQKDtype;
    using ElementK = InQKDtype;
    using ElementV = InVDtype;
    using ElementS = SMDtype;
    using ElementP = float8_e4m3_t;
    using ElementO = OUTPODType;
    using ElementOTmp = REDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    // TODO: support both nZ and zN layouts.
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<512>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<256>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
        LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false, Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
        ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsaQMode1;
    using PType = Gemm::GemmType<ElementP, layout::zN>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<512>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<256>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
            void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false, Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
        ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaQMode1RescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
        TileCopyRescaleO, Arch::PositionL0C>;

    using BsaRegularKernelArch35 = BsaRegularKernelArch35<BsaRegularKernelArch35QMode1, EpilogueMask2Idx, BlockMmadQK,
        EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat>;
    BsaKernelParamsArch35 params{q, k, v, mask, blockTables, query_scale, key_scale, value_scale, actualQseqlen,
        actualKvseqlen, blockSparseMask, o, workspace, tiling};
    BsaRegularKernelArch35 bsaRegularKernelArch35;
    bsaRegularKernelArch35(params);
}

#endif
