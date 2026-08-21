/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILING_H
#define EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "register/op_def_registry.h"

namespace optiling {
// mask 2 idx tile info
BEGIN_TILING_DATA_DEF(BsaMask2IdxTiling)
TILING_DATA_FIELD_DEF(uint32_t, xBlockNumAligned);
TILING_DATA_FIELD_DEF(uint32_t, yBlockNumAligned);
TILING_DATA_FIELD_DEF(uint32_t, avgRowPerSubCore);
TILING_DATA_FIELD_DEF(uint32_t, preActiveSubCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BsaMask2IdxTilingOp, BsaMask2IdxTiling)
// attention tile info
BEGIN_TILING_DATA_DEF(BsaBaseTiling)
TILING_DATA_FIELD_DEF(uint32_t, qBaseTile);
TILING_DATA_FIELD_DEF(uint32_t, kvBaseTile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BsaBaseTilingOp, BsaBaseTiling)
// matmul phase L1 tile info
BEGIN_TILING_DATA_DEF(BsaMmPhaseL1Tiling)
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm1L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileM);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileN);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKLeft);
TILING_DATA_FIELD_DEF(uint32_t, mm2L1TileKRight);
TILING_DATA_FIELD_DEF(uint32_t, qL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, kL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, vL1BufNum);
TILING_DATA_FIELD_DEF(uint32_t, pL1BufNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BsaMmPhaseL1TilingOp, BsaMmPhaseL1Tiling)
// EagleQuantBlockSparseAttention Tiling数据定义
BEGIN_TILING_DATA_DEF(EagleQuantBlockSparseAttentionTilingData)
// 基础参数
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, totalQBlocks);       // T: 所有batch中Q方向切块的总数
TILING_DATA_FIELD_DEF(uint32_t, firstQBlockNum);       // T: 所有batch中Q方向切块的总数

// 稀疏分块参数 (blockShape)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeX);  // block的x维度(Q方向)
TILING_DATA_FIELD_DEF(uint64_t, blockShapeY);  // block的y维度(KV方向)

// selectIdx相关参数
TILING_DATA_FIELD_DEF(uint32_t, maxKvBlockNum);      // 最大KV块数量（selectIdx的最后一维）
TILING_DATA_FIELD_DEF(uint32_t, maxQBlockNum);      // 最大KV块数量（selectIdx的最后一维）
TILING_DATA_FIELD_DEF(uint32_t, avgRowNumPerSubCore);
TILING_DATA_FIELD_DEF(uint32_t, preActivateSubCoreNum);


// query Layout: 0=TND, 1=BNSD
TILING_DATA_FIELD_DEF(uint32_t, queryLayout);

// KV Cache Layout: 0=TND, 1=BNSD
TILING_DATA_FIELD_DEF(uint32_t, kvCacheLayout);

// BNSD格式的最大序列长度（用于计算stride）
// 当actualSeqLengths为nullptr时，maxQSeqlen也用作统一的qseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen);  // BNSD格式Q的第三维（S维度），或统一的qseqlen值
// 当actualSeqLengthsKv为nullptr时，maxKvSeqlen也用作统一的kvseqlen值
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen);  // BNSD格式KV的第三维（S维度），或统一的kvseqlen值
TILING_DATA_FIELD_DEF(uint32_t, useUniformQSeqlen);  // 是否使用统一的qseqlen值（1=是，0=否）
TILING_DATA_FIELD_DEF(uint32_t, useUniformKvSeqlen);  // 是否使用统一的kvseqlen值（1=是，0=否）

// TilingKey for kernel dispatch (生成在tiling层)
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint64_t, selectNumIdxSize);
TILING_DATA_FIELD_DEF(uint64_t, selectIdxSize);
// Workspace大小
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);

TILING_DATA_FIELD_DEF_STRUCT(BsaMask2IdxTiling, BsaMask2IdxTileInfo);
TILING_DATA_FIELD_DEF_STRUCT(BsaBaseTiling, BsaBaseTileInfo);
TILING_DATA_FIELD_DEF_STRUCT(BsaMmPhaseL1Tiling, BsaMmPhaseL1TileInfo);
TILING_DATA_FIELD_DEF(uint32_t, sparsePatternMode);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EagleQuantBlockSparseAttention, EagleQuantBlockSparseAttentionTilingData)

// EagleQuantBlockSparseAttention编译信息
struct EagleQuantBlockSparseAttentionCompileInfo {
    uint32_t inputDataByte = 2;
    ge::DataType inputDataType;
    
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t sysWorkspaceSize = 0;
    platform_ascendc::SocVersion socVersion;
};

// 输入参数信息
struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct OptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

// KVCache Layout枚举
enum RFAKvCacheLayout : uint32_t {
    TND = 0,   // [T, N, D] format
    BNSD = 1   // [B, N, S, D] format
};

// Q Input Layout枚举
enum RFAQInputLayout : uint32_t {
    TND_Q = 0,  // [T, N, D] format
    BNSD_Q = 1  // [B, N, S, D] format
};

// inner prec 枚举
enum BsaInnerCalcPrec : uint32_t {
    ALL_HIGH = 0,
    ALL_LOW = 1,
    LOW_HIGH_MIXED = 4
};

// Tiling类
class BSATiling {
public:
    BSATiling() = default;
    ~BSATiling() = default;
    
    ge::graphStatus GetBsaTiling(gert::TilingContext *bsaContext,
                                  EagleQuantBlockSparseAttentionTilingData &tilingData);
    ge::graphStatus BsaSetTilingData(gert::TilingContext *context,
                                      EagleQuantBlockSparseAttentionTilingData &tilingData);

private:
    ge::graphStatus GetNpuInfo(gert::TilingContext *bsaContext);
    ge::graphStatus ParseAttrs(gert::TilingContext *bsaContext);
    ge::graphStatus GetInputLayout(gert::TilingContext *bsaContext);
    ge::graphStatus ParseRequiredTensors(gert::TilingContext *bsaContext);
    ge::graphStatus ParseOptionalTensors(gert::TilingContext *bsaContext);
    ge::graphStatus CheckQKVDtype(gert::TilingContext *bsaContext);
    ge::graphStatus CheckQKVDimVal(gert::TilingContext *bsaContext,
        uint32_t kHeads, uint32_t vHeads, uint32_t kHeadDim, uint32_t vHeadDim);
    ge::graphStatus ParseQKVInTND(gert::TilingContext *bsaContext);
    ge::graphStatus ParseQKVInBNSD(gert::TilingContext *bsaContext);
    ge::graphStatus ParseSeqlensInTND(gert::TilingContext *bsaContext);
    ge::graphStatus ParseSeqlensInBNSD(gert::TilingContext *bsaContext);
    ge::graphStatus ParseSeqlens(gert::TilingContext *bsaContext);
    ge::graphStatus ParseSparsePattern(gert::TilingContext *bsaContext);
    ge::graphStatus ParseAttenMask(gert::TilingContext *bsaContext);
    ge::graphStatus ParseBlockTable(gert::TilingContext *bsaContext);
    ge::graphStatus CheckSparsePattern(gert::TilingContext *bsaContext, const int64_t defaultShape);
    ge::graphStatus ValidateTNDSeqlenSum(gert::TilingContext *bsaContext);
    // 950 exclusive
    uint32_t GetCurQSTileNum950(int64_t curQSeqlen);
    void CalcBaseTileTilingParams950();
    void CalcSplitCoreTilingParams950();
    void CalcWorkspaceTilingParams950(gert::TilingContext *bsaContext);
    void CalcMatmulPhaseL1TileInfo950();
    // 910 exclusive
    ge::graphStatus CalculateTaskSplit(gert::TilingContext *bsaContext);
    ge::graphStatus CalculateWorkSpace(gert::TilingContext *bsaContext);
    // shared
    void CalculateBatchTaskSplit(int64_t qSeqlen, uint32_t groupSize,
        uint32_t &curTaskNum, uint32_t &curQBlockNum);
    ge::graphStatus FillTilingData(gert::TilingContext *bsaContext);
    uint64_t GenerateTilingKey(gert::TilingContext *bsaContext);
    
private:
    uint32_t batch_ = 0;
    uint32_t qSeqlen_ = 0;
    uint32_t kvSeqlen_ = 0;
    uint32_t numHeads_ = 0;
    uint32_t kvHeads_ = 0;
    uint32_t embeddingSize_ = 0;
    uint32_t blockSize_ = 128;
    int64_t blockShapeX_ = 0;  // block的x维度
    int64_t blockShapeY_ = 0;  // block的y维度
    float scaleValue_ = 0.0f;
    uint32_t maskType_ = 0;
    uint32_t sparsePatternMode_ = 0;
    uint32_t innerPrecise_ = 1;  // 0=float32 softmax, 1=fp16 softmax
    bool softmaxLseFlag_ = false;
    
    uint32_t totalQBlocks_ = 0;
    uint32_t maxKvBlockNum_ = 0;
    uint32_t maxQBlockNum_ = 0;
    uint32_t avgRowNumPerSubCore_ = 0;
    uint32_t preActivateSubCoreNum_ = 0;
    uint32_t firstQBlockNum_ = 0;
    uint32_t firstBatchTaskNum_ = 0;
    uint32_t totalTaskNum_ = 0;
    uint32_t maxNumBlocksPerBatch_ = 0;
    const int64_t *qSeqLenList_ = nullptr;
    const int64_t *kvSeqLenList_ = nullptr;
    const int64_t *blockShapeList = nullptr;
    bool useUniformQSeqlen_ = false;  // 是否使用统一的qseqlen值（使用maxQSeqlen_）
    bool useUniformKvSeqlen_ = false;  // 是否使用统一的kvseqlen值（使用maxKvSeqlen_）

    uint64_t mm1OutSize_ = 0;
    uint64_t smOnlineOutSize_ = 0;
    uint64_t mm2OutSize_ = 0;
    uint64_t updateSize_ = 0;
    uint64_t selectNumIdxSize_ = 0;
    uint64_t selectIdxSize_ = 0;
    
    RFAKvCacheLayout kvCacheLayout_ = RFAKvCacheLayout::TND;
    RFAQInputLayout qInputLayout_ = RFAQInputLayout::TND_Q;
    
    uint32_t blockDim_ = 20;
    uint32_t aivNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t socVer_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t workSpaceSize_ = 0;
    uint64_t libapiSize_ = 0;
    
    uint32_t maxQSeqlen_ = 0;  // BNSD格式Q的第三维（S维度）
    uint32_t maxKvSeqlen_ = 0;  // BNSD格式KV的第三维（S维度）
    int64_t totalTokensT_ = 0;  // TND格式Q的第一维（T维度，总token数）
    int64_t totalTokensKv_ = 0;  // TND格式KV的第一维（T维度，总token数

    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // L1 tile info
    // further splits the base tiles
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    
    ge::DataType dataType_ = ge::DT_FLOAT16;
    // 在 dataType_ 声明附近添加：
    ge::DataType vDataType_ = ge::DT_FLOAT16;  // 新增：记录V的独立dtype

    EagleQuantBlockSparseAttentionTilingData *tilingData_ = nullptr;
};

}  // namespace optiling

#endif  // EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_TILING_H

