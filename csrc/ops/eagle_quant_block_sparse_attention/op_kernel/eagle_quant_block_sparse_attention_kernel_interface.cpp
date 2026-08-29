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
#if (__CCE_AICORE__ == 220)
#include "eagle_quant_block_sparse_attention_kernel_regular_arch32.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/eagle_quant_block_sparse_attention_kernel_arch35_regular.h"
#include "arch35/eagle_quant_block_sparse_attention_kernel_arch35_qmode1.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)
namespace BlockSparse {
template <
    typename InputDtype = half,
    typename SoftmaxDtype = float,
    Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
    uint32_t QueryLayout = 0,      // 0=TND, 1=BNSD
    uint32_t KvCacheLayout = 0>    // 0=TND, 1=BNSD
__global__ __aicore__ void EagleQuantBlockSparseAttentionInfer(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR blockSparseMask,
    GM_ADDR mask,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR actualQseqlen,
    GM_ADDR actualKvseqlen,
    GM_ADDR blockShape,
    GM_ADDR workspace,                // S matrix (QK^T result)
    GM_ADDR lse,
    GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA2;
    using ElementQ = InputDtype;
    using LayoutQ = layout::RowMajor;
    using ElementK = InputDtype;
    using LayoutK = layout::ColumnMajor;
    using ElementV = InputDtype;
    using LayoutV = layout::RowMajor;
    using ElementS = SoftmaxDtype;
    using LayoutS = layout::RowMajor;
    using ElementP = InputDtype;
    using LayoutP = layout::RowMajor;
    using ElementO = InputDtype;
    using LayoutO = layout::RowMajor;
    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = int8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = SoftmaxDtype;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = SoftmaxDtype;
    using LayoutUpdate = layout::RowMajor;

    // Use sparse-specific dispatch policies
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using DispatchPolicyQK = Gemm::MmadAtlasA2SFAIQK<false, false>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK,
                                            QType, KType, SType>;

    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<false, false>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV,
                                            PType, VType, OTmpType>;

    // Epilogue policies for sparse attention
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SoftmaxDtype>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax,
        PType, SType, MaskType>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, SoftmaxDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO,
                                                            OType, OTmpType, OUpdateType, LseType>;

    // Kernel instantiation
    using EagleQuantBlockSparseAttentionKernelType = EagleQuantBlockSparseAttentionKernel<BlockMmadQK, BlockMmadPV,
                                                                    EpilogueOnlineSoftmax,
                                                                    EpilogueRescaleO,
                                                                    false,           // PAGED_CACHE_FLAG
                                                                    QueryLayout,     // QUERY_LAYOUT
                                                                    KvCacheLayout>;  // KV_CACHE_LAYOUT
    EagleQuantBlockSparseAttentionKernelParams params{q, k, v, blockSparseMask, mask, blockTables, actualQseqlen, actualKvseqlen,
                                o, lse, workspace, tiling};

    // Call block sparse attention kernel
    EagleQuantBlockSparseAttentionKernelType blockSparseAttenInfer;
    blockSparseAttenInfer(params);
}
}
#endif
#if (__CCE_AICORE__ == 310)

// Lookup helper for ElementP type selection based on ElementQ and ElementV
template <typename QType, typename VType, typename DefaultType>
struct ElementPTypeLookup {
    using type = DefaultType;
};

template <typename DefaultType>
struct ElementPTypeLookup<int8_t, float8_e4m3_t, DefaultType> {
    using type = float8_e4m3_t;
};

template <typename QType, typename KType, typename VType, typename OUTPODType>
struct GetQuantMode {
    static constexpr int value = 0;
};

template <>
struct GetQuantMode<int8_t, int8_t, float8_e4m3_t, bfloat16_t> {
    static constexpr int value = 1;
};

template <>
struct GetQuantMode<int8_t, int8_t, float8_e4m3_t, half> {
    static constexpr int value = 1;
};


template <int quantMode>
struct QKScaleGranularitySelector {
    static constexpr NpuArch::Gemm::Tile::ScaleGranularity value = NpuArch::Gemm::Tile::ScaleGranularity::NO_QUANT;
};

template <>
struct QKScaleGranularitySelector<1> {
    static constexpr NpuArch::Gemm::Tile::ScaleGranularity value = NpuArch::Gemm::Tile::ScaleGranularity::PER_TENSOR;
};

template <int quantMode>
struct PVScaleGranularitySelector {
    static constexpr NpuArch::Gemm::Tile::ScaleGranularity value = NpuArch::Gemm::Tile::ScaleGranularity::NO_QUANT;
};

template <>
struct PVScaleGranularitySelector<1> {
    static constexpr NpuArch::Gemm::Tile::ScaleGranularity value = NpuArch::Gemm::Tile::ScaleGranularity::NO_QUANT;
};

using namespace BsaKernelArch35;

template <class InQKDtype, class InVDtype, class OUTPODType, class SMDtype, class REDtype, Format qFormat, Format kvFormat>
__global__ __aicore__ void BsaInferIntfRegular(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
    GM_ADDR query_scale, GM_ADDR key_scale, GM_ADDR value_scale,
    GM_ADDR o, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
    GM_ADDR blockSparseMask, GM_ADDR workspace,
    GM_ADDR tiling
) {
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InQKDtype;
    using ElementK = InQKDtype;
    using ElementV = InVDtype;
    using ElementS = SMDtype;
    using ElementP = typename ElementPTypeLookup<ElementQ, ElementV, OUTPODType>::type;
    using ElementO = OUTPODType;
    using ElementOTmp = REDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    //tODO后续增加一个selector，支持nz和zn
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx = Epilogue::Block::BlockEpilogue<
        DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<512>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<256>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS, LayoutS,
        void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false, QKScaleGranularitySelector<GetQuantMode<ElementQ, ElementK, ElementV, OUTPODType>::value>::value>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK, ElementS, void, TileCopyQK>;
   // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using PType = Gemm::GemmType<ElementP, layout::zN>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<512>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<256>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
        void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false, PVScaleGranularitySelector<GetQuantMode<ElementQ, ElementK, ElementV, OUTPODType>::value>::value>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV, ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<
        ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<
        DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS, TileCopyRescaleO, Arch::PositionL0C>;

    using BsaRegularKernelArch35 = BsaRegularKernelArch35<
        EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat>;
    BsaKernelParamsArch35 params{q, k, v, mask, blockTables, query_scale, key_scale, value_scale,
        actualQseqlen, actualKvseqlen, blockSparseMask, o, workspace, tiling};
    BsaRegularKernelArch35 bsaRegularKernelArch35;
    bsaRegularKernelArch35.template operator()<GetQuantMode<ElementQ, ElementK, ElementV, OUTPODType>::value>(params);
}
#endif
