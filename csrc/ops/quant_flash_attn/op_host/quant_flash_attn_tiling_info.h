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
 * \file quant_flash_attn_tiling_info.h
 * \brief
 */

#ifndef FA_TILING_INFO_H
#define FA_TILING_INFO_H

#include <vector>
#include "../../common/op_host/fia_tiling_base.h"
#include "../../common/op_host/fia_tiling_shape.h"

namespace optiling {

// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t VALUE_INDEX = 2;
constexpr uint32_t QUERY_DESCALE_INDEX = 3;
constexpr uint32_t KEY_DESCALE_INDEX = 4;
constexpr uint32_t VALUE_DESCALE_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 6;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 7;
constexpr uint32_t CU_SEQLENS_KV_INDEX = 8;
constexpr uint32_t SEQUSED_Q_INDEX = 9;
constexpr uint32_t SEQUSED_KV_INDEX = 10;
constexpr uint32_t SINKS_INDEX = 11;
constexpr uint32_t ATTN_MASK_INDEX = 12;
constexpr uint32_t METADATA_INDEX = 13;

// Attributes Index
constexpr uint32_t ATTR_QUERY_QUANT_MODE_INDEX = 0;
constexpr uint32_t ATTR_KEY_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_VALUE_QUANT_MODE_INDEX = 2;
constexpr uint32_t ATTR_QUERY_QUANT_BLOCK_SIZE_INDEX = 3;
constexpr uint32_t ATTR_KEY_QUANT_BLOCK_SIZE_INDEX = 4;
constexpr uint32_t ATTR_VALUE_QUANT_BLOCK_SIZE_INDEX = 5;
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 6; // scaleValue
constexpr uint32_t ATTR_MASK_MODE_INDEX = 7; // mask_mode
constexpr uint32_t ATTR_WIN_LEFT_INDEX = 8; // win_left (preToken)
constexpr uint32_t ATTR_WIN_RIGHT_INDEX = 9; // win_right (nextToken)
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 10; // max_seqlen_q
constexpr uint32_t ATTR_MAX_SEQLEN_KV_INDEX = 11; // max_seqlen_kv
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 12; // layout_q
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 13; // layout_kv
constexpr uint32_t ATTR_LAYOUT_OUT_INDEX = 14; // layout_out
constexpr uint32_t ATTR_SOFTMAX_PRECISION_INDEX = 15; // softmax_precision
constexpr uint32_t ATTR_RETURN_LSE_INDEX = 16; // return_softmax_lse

// Output Index
constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

// Params Name
const std::string QUERY_NAME = "q";
const std::string KEY_NAME = "k";
const std::string VALUE_NAME = "v";
const std::string Q_DESCALE_NAME = "q_descale";
const std::string K_DESCALE_NAME = "k_descale";
const std::string V_DESCALE_NAME = "v_descale";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string CU_SEQLENS_KV_NAME = "cu_seqlens_kv";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string SEQUSED_KV_NAME = "seqused_kv";
const std::string SINKS_NAME = "sinks";
const std::string ATTEN_MASK_NAME = "attn_mask";
const std::string METADATA_NAME = "metadata";
const std::string Q_QUANT_MODE_NAME = "q_quant_mode";
const std::string K_QUANT_MODE_NAME = "k_quant_mode";
const std::string V_QUANT_MODE_NAME = "v_quant_mode";
const std::string Q_DTYPE_NAME = "q_dtype";
const std::string K_DTYPE_NAME = "k_dtype";
const std::string V_DTYPE_NAME = "v_dtype";
const std::string QUANT_BLOCK_SIZE_QS_NAME = "quant_block_size_qs";
const std::string QUANT_BLOCK_SIZE_KS_NAME = "quant_block_size_ks";
const std::string QUANT_BLOCK_SIZE_VS_NAME = "quant_block_size_vs";
const std::string SOFTMAX_SCALE_NAME = "softmax_scale";
const std::string MASK_MODE_NAME = "mask_mode";
const std::string WIN_LEFT_NAME = "win_left";
const std::string WIN_RIGHT_NAME = "win_right";
const std::string MAX_SEQLEN_Q_NAME = "max_seqlen_q";
const std::string MAX_SEQLEN_KV_NAME = "max_seqlen_kv";
const std::string LAYOUT_Q_NAME = "layout_q";
const std::string LAYOUT_KV_NAME = "layout_kv";
const std::string LAYOUT_OUT_NAME = "layout_out";
const std::string SOFTMAX_PRECISION = "softmax_precision";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string ATTEN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

constexpr int64_t MASK_MODE_INT_MAX = 2147483647;

enum class MaskMode : int32_t { NO_MASK = 0, CAUSAL = 3, BAND = 4 };

enum class KvStorageMode : uint32_t { BATCH_CONTINUOUS = 0, PAGE_ATTENTION = 1 };

enum class QuantMode : uint32_t { GROUP_SCALING = 3, PER_BLOCK = 4 };

enum class QFA_DTYPE : uint32_t {
    FP8_E4M3 = 1,
    FP8_E8M0 = 2,
    HI_FLOAT8 = 3,
    FP4_E2M1 = 11,
    HI_FLOAT4 = 12,
};

struct QfaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l2Size = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint32_t cvRatio = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

struct FARequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct FAOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct FAParaInfo {
    FARequiredParaInfo query = {nullptr, nullptr};
    FARequiredParaInfo key = {nullptr, nullptr};
    FARequiredParaInfo value = {nullptr, nullptr};
    FARequiredParaInfo qDescale = {nullptr, nullptr};
    FARequiredParaInfo kDescale = {nullptr, nullptr};
    FARequiredParaInfo vDescale = {nullptr, nullptr};

    FAOptionalParaInfo blockTable = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    FAOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    FAOptionalParaInfo sequsedQ = {nullptr, nullptr};
    FAOptionalParaInfo sequsedKv = {nullptr, nullptr};
    FAOptionalParaInfo sinks = {nullptr, nullptr};
    FAOptionalParaInfo attnMask = {nullptr, nullptr};
    FAOptionalParaInfo metadata = {nullptr, nullptr};

    const int64_t *qQuantMode = nullptr;
    const int64_t *kQuantMode = nullptr;
    const int64_t *vQuantMode = nullptr;
    const int64_t *quantBlockSizeQs = nullptr;
    const int64_t *quantBlockSizeKs = nullptr;
    const int64_t *quantBlockSizeVs = nullptr;
    const float *softmaxScale = nullptr;
    const int64_t *maskMode = nullptr;
    const int64_t *winLeft = nullptr;
    const int64_t *winRight = nullptr;
    const int64_t *maxSeqlenQ = nullptr;
    const int64_t *maxSeqlenKV = nullptr;
    const char *layoutQ = nullptr;
    const char *layoutKV = nullptr;
    const char *layoutOut = nullptr;
    const int64_t *softmaxPrecision = nullptr;
    const int64_t *returnSoftMaxLse = nullptr;

    FARequiredParaInfo attnOut = {nullptr, nullptr};
    FARequiredParaInfo lseOut = {nullptr, nullptr};
};

class QuantFlashAttnTilingInfo : public TilingInfo {
  public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    FAParaInfo opParamInfo;

    // BaseParams
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t gSize = 0;
    uint32_t qkHeadDim = 0;
    uint32_t vHeadDim = 0;
    uint32_t queryTSize = 0;
    uint32_t keyTSize = 0;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    QuantMode qQuantMode = QuantMode::GROUP_SCALING;
    QuantMode kQuantMode = QuantMode::GROUP_SCALING;
    QuantMode vQuantMode = QuantMode::GROUP_SCALING;
    float softmaxScale = 0.0;

    // PageAttention
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;

    // mask 信息
    bool attnMaskFlag = false;
    uint32_t maskMode = 0;
    int64_t winLeft = -1;
    int64_t winRight = -1;

    // layout信息
    FiaLayout layoutQ;
    FiaLayout layoutKV;
    FiaLayout layoutOut;

    // seqLen信息
    int64_t maxSeqLenQ = 0;
    int64_t maxSeqLenKv = 0;
    uint32_t qSeqUsedSize = 0;
    uint32_t kvSeqUsedSize = 0;
    uint32_t qCuSeqLensSize = 0;
    uint32_t kvCuSeqLensSize = 0;

    // learnable sink 信息
    bool learnableSinkFlag = false;
    uint32_t returnSoftmaxLse = 0;
    uint32_t softmaxPresision = 1;
    uint32_t quantBlockSizeQs = 0;
    uint32_t quantBlockSizeKs = 0;
    uint32_t quantBlockSizeVs = 0;

    // DTYPE
    // ge::DT_FLOAT8_E8M0
    // ge::DT_FLOAT8_E4M3FN
    // ge::DT_FLOAT4_E2M1
    // ge::DT_HIFLOAT8
    ge::DataType inputQType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputKType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType inputVType = ge::DT_FLOAT8_E4M3FN;
    ge::DataType outputType = ge::DT_BF16;
};
} // namespace optiling
#endif // FA_TILING_INFO_H
