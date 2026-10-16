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
 * \file eagle_block_sparse_attention_interface.cpp
 * \brief Block Sparse Attention Interface
 */
#include "kernel_operator.h"
#if (__CCE_AICORE__ == 220)
#include "eagle_block_sparse_attention_kernel_regular_arch22.h"
#endif
#if (__CCE_AICORE__ == 310)
#include "arch35/eagle_block_sparse_attention_kernel_arch35_regular.h"
#include "arch35/eagle_block_sparse_attention_kernel_arch35_full_quant.h"
#include "arch35/eagle_block_sparse_attention_kernel_arch35_mxfp4_full_quant.h"
#endif

using namespace NpuArch;

#if (__CCE_AICORE__ == 220)
namespace BlockSparse {
template <typename InputDtype = half, typename SoftmaxDtype = float,
          Epilogue::LseMode lseMode = Epilogue::LseMode::NONE,
          uint32_t QueryLayout = 0,   // 0=TND, 1=BNSD, 2=BSND
          uint32_t KvCacheLayout = 0> // 0=TND, 1=BNSD, 2=BSND
__global__ __aicore__ void EagleBlockSparseAttentionInfer(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR blockSparseMask,
                                                     GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
                                                     GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR blockShape,
                                                     GM_ADDR workspace, // S matrix (QK^T result)
                                                     GM_ADDR lse, GM_ADDR tiling)
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
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

#if BSA_ENABLE_V_DOUBLE_BUFFER
    using L1TileShapePV = GemmShape<128, 128, PV_K_TILE>;
#else
    using L1TileShapePV = GemmShape<128, 128, 256>;
#endif
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using DispatchPolicyPV = Gemm::MmadAtlasA2SFAIPV<false, false>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    // Epilogue policies for sparse attention
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmax<lseMode, SoftmaxDtype>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleO<lseMode, SoftmaxDtype>;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    // Kernel instantiation
    using EagleBlockSparseAttentionKernelType =
        EagleBlockSparseAttentionKernel<BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO,
                                   false,          // PAGED_CACHE_FLAG
                                   QueryLayout,    // QUERY_LAYOUT
                                   KvCacheLayout>; // KV_CACHE_LAYOUT
    EagleBlockSparseAttentionKernelParams params{
        q, k, v, blockSparseMask, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};

    // Call block sparse attention kernel
    EagleBlockSparseAttentionKernelType blockSparseAttenInfer;
    blockSparseAttenInfer(params);
}
} // namespace BlockSparse
#endif
#if (__CCE_AICORE__ == 310)

using namespace BsaKernelArch35;

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat>
__global__ __aicore__ void BsaInferIntfRegular(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
                                               GM_ADDR o, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                               GM_ADDR blockSparseMask, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
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
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using BsaRegularKernelArch35 = BsaRegularKernelArch35<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax,
                                                          BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat>;
    BsaKernelParamsArch35 params{q, k,         v,   mask,  blockTables, actualQseqlen, actualKvseqlen, blockSparseMask,
                                 o, workspace, lse, tiling};
    BsaRegularKernelArch35 bsaRegularKernelArch35;
    bsaRegularKernelArch35(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat>
__global__ __aicore__ void BsaInferInterfaceFullQuant(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                      GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
                                                      GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv,
                                                      GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                      GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace,
                                                      GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;    // fp8_e4m3fn_t
    using ElementK = InDtype;    // fp8_e4m3fn_t
    using ElementV = InDtype;    // fp8_e4m3fn_t
    using ElementS = SMDtype;    // half/bfloat16_t
    using ElementP = InDtype;    // fp8_e4m3fn_t
    using ElementO = SMDtype;    // half/bfloat16_t
    using ElementOTmp = REDtype; // float
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
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
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad qk
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQK;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                                         Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsa;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和V的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad pv
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPV;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale o
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleO<lseMode, lseFormat>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;
    BsaFullQuantKernelParamsArch35 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    using BsaFullQuantKernelArch35 = BsaFullQuantKernelArch35<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax,
                                                              BlockMmadPV, EpilogueRescaleO, qFormat, kvFormat>;
    BsaFullQuantKernelArch35 bsaFullQuantKernelArch35;
    bsaFullQuantKernelArch35(params);
}

template <class InDtype, class SMDtype, class REDtype, class Otype, Format qFormat, Format kvFormat,
          Epilogue::LseMode lseMode, Epilogue::LseFormat lseFormat, Epilogue::MXQuantMode mxQuantMode,
          Gemm::Tile::CopyL0CToUBMode mm1L0C2UBMode, bool transposedMm1>
__global__ __aicore__ void BsaInferInterfaceMXFP4FullQuant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
    GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::AtlasA5;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;     // fp4x2_e2m1_t
    using ElementK = InDtype;     // fp4x2_e2m1_t
    using ElementV = InDtype;     // fp4x2_e2m1_t
    using ElementS = SMDtype;     // half
    using ElementP = InDtype;     // fp4x2_e2m1_t
    using ElementScale = uint8_t; // e8m0
    using ElementOTmp = REDtype;  // float
    using ElementDm = REDtype;
    using ElementO = Otype; // bfloat16_t / fp16
    using ElementGroupMax = SMDtype;
    using ElementLocalGlobalMax = SMDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
    using LayoutPScale = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    using LayoutLocalGlobalMax = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::MmadAtlasA5BsaQKMX<transposedMm1>;
    using TileCopyQK =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, layout::RowMajor, ElementK, layout::ColumnMajor, ElementS,
                                          LayoutS, void, mm1L0C2UBMode, false, Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueOnlineSoftmaxBsaMX<transposedMm1, mxQuantMode>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using PScaleType = Gemm::GemmType<ElementScale, LayoutPScale>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, PScaleType>;
    // compute PScale from group max: 输入 ElementGroupMax, 输出 PScaleType
    using DispatchPolicyComputePScale = Epilogue::EpilogueComputePScaleBsaMX<transposedMm1, mxQuantMode>;
    using EpilogueComputePScale =
        Epilogue::Block::BlockEpilogue<DispatchPolicyComputePScale, ElementGroupMax, ElementDm, PScaleType>;
    // copy global max from UB to L1: 输入 ElementLocalGlobalMax
    using DispatchPolicyCopyGlobalMaxUbToL1 = Epilogue::EpilogueCopyGlobalMaxUbToL1BsaMX;
    using EpilogueCopyGlobalMaxUbToL1 =
        Epilogue::Block::BlockEpilogue<DispatchPolicyCopyGlobalMaxUbToL1, ElementLocalGlobalMax, LayoutLocalGlobalMax>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::MmadAtlasA5BsaPVMX<transposedMm1>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M, false,
                                          Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // copy global max from L1 to UB: 输入 ElementLocalGlobalMax
    using BlockMmadCopyGlobalMaxL1ToUB = Gemm::Block::BlockMmadTla<Gemm::CopyGlobalMaxL1ToUBBsa, void, void,
                                                                   ElementLocalGlobalMax, void, void, void, void, void>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA5BsaRescaleOMX<lseMode, lseFormat, transposedMm1>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementDm,
                                                            TileCopyRescaleO, Arch::PositionL0C>;

    using BsaMXFP4FullQuantKernelArch35 =
        BsaMXFP4FullQuantKernelArch35<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                      EpilogueRescaleO, EpilogueComputePScale, EpilogueCopyGlobalMaxUbToL1,
                                      BlockMmadCopyGlobalMaxL1ToUB, qFormat, kvFormat, transposedMm1>;

    BsaFullQuantKernelParamsArch35 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    BsaMXFP4FullQuantKernelArch35 bsaMXFP4FullQuantKernelArch35;
    bsaMXFP4FullQuantKernelArch35(params);
}
#endif
