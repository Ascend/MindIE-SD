/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "eagle_quant_block_sparse_attention.h"

using namespace at;

namespace {
// V2 kernel supports BF16/FP16/INT8/FP8/MXFP4.
constexpr std::string_view EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_NAME = "aclnnEagleQuantBlockSparseAttention";

constexpr int64_t MASK_TYPE = 0; // no attention mask
constexpr int64_t PRE_TOKENS = 2147483647; // full context window
constexpr int64_t NEXT_TOKENS = 2147483647;

inline bool IsByteTensor(const at::Tensor &tensor) { return tensor.scalar_type() == at::ScalarType::Byte; }

inline bool IsMxfp4Dtype(const c10::optional<int64_t> &dtype) {
    return dtype.has_value() && dtype.value() == CANN_DTYPE_FLOAT4_E2M1;
}

inline void CheckQkvDtype(const char *name, const at::Tensor &tensor, const c10::optional<int64_t> &dtype) {
    if (!IsByteTensor(tensor)) {
        // Preserve dev's dtype hints for native storage and INT8 bitcast callers.
        TORCH_CHECK(!IsMxfp4Dtype(dtype), name, " must be torch.uint8 for MXFP4.");
        TORCH_CHECK(!dtype.has_value() || tensor.scalar_type() == at::kChar ||
                dtype.value() == static_cast<int64_t>(tensor.scalar_type()),
            "eagle_quant_block_sparse_attention: ", name, "_dtype is incompatible with tensor dtype.");
        return;
    }
    TORCH_CHECK(dtype.has_value(), name, "_dtype can not be None when ", name, " is torch.uint8.");
    TORCH_CHECK(IsByteTensor(tensor), name, " must be torch.uint8 when ", name, "_dtype is specified.");
    TORCH_CHECK(dtype.value() == CANN_DTYPE_FLOAT4_E2M1, name, "_dtype only supports torch_npu.float4_e2m1fn_x2.");
}

inline void CheckDequantScaleDtype(const char *name, const c10::optional<at::Tensor> &scale) {
    if (!scale.has_value()) {
        return;
    }
    TORCH_CHECK(
        scale->scalar_type() == at::kFloat, "eagle_quant_block_sparse_attention: ", name, " must be torch.float32.");
}

static TensorWrapper MakeScaleWrapper(const at::Tensor &scale) {
    if (scale.defined()) {
        TORCH_CHECK(
            scale.itemsize() == 1, "eagle_quant_block_sparse_attention: dtype scale must use one-byte E8M0 storage.");
    }
    return TensorWrapper{scale, ACL_DTYPE_FLOAT8_E8M0};
}

inline at::SmallVector<int64_t, 4> GetAttentionOutSize(
    const at::Tensor &query, const c10::optional<int64_t> &queryDtype) {
    at::SmallVector<int64_t, 4> outputSize;
    for (const int64_t size : query.sizes()) {
        outputSize.push_back(size);
    }
    if (IsMxfp4Dtype(queryDtype)) {
        outputSize.back() *= FP4_IN_INT8;
    }
    return outputSize;
}

inline at::ScalarType GetAttentionOutDtype(const at::Tensor &query, const c10::optional<at::Tensor> &qDequantScale,
    const c10::optional<at::ScalarType> &outputDtype) {
    if (outputDtype.has_value()) {
        TORCH_CHECK(outputDtype.value() == at::kHalf || outputDtype.value() == at::kBFloat16,
            "output_dtype only supports torch.float16 or torch.bfloat16.");
        return outputDtype.value();
    }
    return qDequantScale.has_value() ? at::kBFloat16 : query.scalar_type();
}
} // namespace

std::tuple<at::Tensor, at::Tensor> eagle_quant_block_sparse_attention_impl_npu(const at::Tensor &query,
    const at::Tensor &key, const at::Tensor &value, const c10::optional<at::Tensor> &block_sparse_mask,
    at::IntArrayRef block_shape, std::string q_input_layout, std::string kv_input_layout, int64_t num_key_value_heads,
    double scale_value, int64_t inner_precise, c10::OptionalIntArrayRef actual_seq_lengths,
    c10::OptionalIntArrayRef actual_seq_lengths_kv, int64_t softmax_lse_flag,
    const c10::optional<at::Tensor> &query_scale, const c10::optional<at::Tensor> &key_scale,
    const c10::optional<at::Tensor> &value_scale, const c10::optional<int64_t> &query_dtype,
    const c10::optional<int64_t> &key_dtype, const c10::optional<int64_t> &value_dtype,
    const c10::optional<at::ScalarType> &output_dtype, const c10::optional<at::Tensor> &query_dtype_scale,
    const c10::optional<at::Tensor> &key_dtype_scale, const c10::optional<at::Tensor> &value_dtype_scale) {
    TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
        "eagle_quant_block_sparse_attention: q_input_layout only supports 'TND' and 'BNSD', got ", q_input_layout);
    TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
        "eagle_quant_block_sparse_attention: kv_input_layout only supports 'TND' and 'BNSD', got ", kv_input_layout);
    TORCH_CHECK(q_input_layout == kv_input_layout,
        "eagle_quant_block_sparse_attention: q_input_layout and kv_input_layout must be consistent.");
    TORCH_CHECK(q_input_layout != "TND" || (actual_seq_lengths.has_value() && actual_seq_lengths_kv.has_value()),
        "eagle_quant_block_sparse_attention: actual_seq_lengths and actual_seq_lengths_kv are required for TND "
        "layout.");
    CheckQkvDtype("query", query, query_dtype);
    CheckQkvDtype("key", key, key_dtype);
    CheckQkvDtype("value", value, value_dtype);
    const bool isMxfp4 = IsMxfp4Dtype(query_dtype) || IsMxfp4Dtype(key_dtype) || IsMxfp4Dtype(value_dtype);
    TORCH_CHECK(!isMxfp4 || (IsMxfp4Dtype(query_dtype) && IsMxfp4Dtype(key_dtype) && IsMxfp4Dtype(value_dtype)),
        "eagle_quant_block_sparse_attention: MXFP4 requires query/key/value dtype to all be "
        "torch_npu.float4_e2m1fn_x2.");
    TORCH_CHECK(
        !isMxfp4 || (query_dtype_scale.has_value() && key_dtype_scale.has_value() && value_dtype_scale.has_value()),
        "eagle_quant_block_sparse_attention: MXFP4 requires query/key/value dtype scales.");
    TORCH_CHECK(!isMxfp4 || value_scale.has_value(), "eagle_quant_block_sparse_attention: MXFP4 requires value_scale.");
    if (isMxfp4) {
        CheckDequantScaleDtype("query_scale", query_scale);
        CheckDequantScaleDtype("key_scale", key_scale);
        CheckDequantScaleDtype("value_scale", value_scale);
    }

    const char *qLayoutPtr = q_input_layout.c_str();
    const char *kvLayoutPtr = kv_input_layout.c_str();

    // attenMaskOptional and blockTableOptional must be nullptr.
    c10::optional<at::Tensor> nulltensor = c10::nullopt;

    /* EXEC_NPU_CMD has ConvertType for c10::optional<at::IntArrayRef> only, not
        c10::OptionalIntArrayRef. Convert explicitly: nullopt -> nullptr (op tiling
        skips batch check), has_value() -> AclIntArray*. Do not use .value_or({})
        — empty array is interpreted as batch=0, conflicting with query batch dim. */
    c10::optional<at::IntArrayRef> optSeqLen =
        actual_seq_lengths.has_value() ? c10::optional<at::IntArrayRef>(actual_seq_lengths.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> optSeqLenKv = actual_seq_lengths_kv.has_value()
        ? c10::optional<at::IntArrayRef>(actual_seq_lengths_kv.value())
        : c10::nullopt;

    // blockSize=0: PagedAttention not supported.
    constexpr int64_t blockSize = 0;

    // Quantized input defaults to BF16 for compatibility; callers may request FP16 or BF16 explicitly.
    const auto &effectiveQueryScale = isMxfp4 ? query_dtype_scale : query_scale;
    auto outOptions = query.options().dtype(GetAttentionOutDtype(query, effectiveQueryScale, output_dtype));
    at::Tensor attentionOut = at_npu::native::empty_with_format(
        GetAttentionOutSize(query, query_dtype), outOptions, at_npu::native::get_npu_format(query));

    // TND: [T, N, 1], BNSD: [B, N, S, 1]
    at::Tensor softmaxLse;
    if (q_input_layout == "TND") {
        softmaxLse = at_npu::native::empty_with_format({query.size(0), query.size(1), 1},
            query.options().dtype(at::kFloat), at_npu::native::get_npu_format(query));
    } else {
        softmaxLse = at_npu::native::empty_with_format({query.size(0), query.size(1), query.size(2), 1},
            query.options().dtype(at::kFloat), at_npu::native::get_npu_format(query));
    }
    // Pass nullptr when flag=0 (op skips lse write).
    c10::optional<at::Tensor> softmaxLseOpt =
        (softmax_lse_flag != 0) ? c10::optional<at::Tensor>(softmaxLse) : c10::nullopt;

    const c10::optional<int64_t> noOverrideDtype = c10::nullopt;
    // Only MXFP4 changes the ACL dtype; qmode1 keeps dev's native/INT8 storage.
    auto queryWrapper = MakeTensorWrapper(query, isMxfp4 ? query_dtype : noOverrideDtype);
    auto keyWrapper = MakeTensorWrapper(key, isMxfp4 ? key_dtype : noOverrideDtype);
    auto valueWrapper = MakeTensorWrapper(value, isMxfp4 ? value_dtype : noOverrideDtype);
    const at::Tensor queryScaleTensor =
        isMxfp4 ? query_dtype_scale.value_or(at::Tensor()) : query_scale.value_or(at::Tensor());
    const at::Tensor keyScaleTensor =
        isMxfp4 ? key_dtype_scale.value_or(at::Tensor()) : key_scale.value_or(at::Tensor());
    const at::Tensor valueScaleTensor =
        isMxfp4 ? value_dtype_scale.value_or(at::Tensor()) : value_scale.value_or(at::Tensor());
    auto queryScaleWrapper =
        isMxfp4 ? MakeScaleWrapper(queryScaleTensor) : MakeTensorWrapper(queryScaleTensor, noOverrideDtype);
    auto keyScaleWrapper =
        isMxfp4 ? MakeScaleWrapper(keyScaleTensor) : MakeTensorWrapper(keyScaleTensor, noOverrideDtype);
    auto valueScaleWrapper =
        isMxfp4 ? MakeScaleWrapper(valueScaleTensor) : MakeTensorWrapper(valueScaleTensor, noOverrideDtype);

    const at::Tensor qLevel0ScaleTensor = isMxfp4 ? query_scale.value_or(at::Tensor()) : at::Tensor();
    const at::Tensor kLevel0ScaleTensor = isMxfp4 ? key_scale.value_or(at::Tensor()) : at::Tensor();
    const at::Tensor vLevel0ScaleTensor = isMxfp4 ? value_scale.value_or(at::Tensor()) : at::Tensor();
    auto qLevel0ScaleWrapper = MakeTensorWrapper(qLevel0ScaleTensor, noOverrideDtype);
    auto kLevel0ScaleWrapper = MakeTensorWrapper(kLevel0ScaleTensor, noOverrideDtype);
    auto vLevel0ScaleWrapper = MakeTensorWrapper(vLevel0ScaleTensor, noOverrideDtype);

    // V2 API: query/key/value_scale inserted after blockTableOptional.
    // BF16/FP16 path: pass nulltensor (nullptr) for all three scales.
    // Quantized paths pass their scale tensors; MXFP4 dtype scales are wrapped as FLOAT8_E8M0.
    EXEC_NPU_CMD<EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_NAME>(queryWrapper, keyWrapper, valueWrapper, block_sparse_mask,
        nulltensor, // attenMaskOptional (nullptr)
        block_shape,
        optSeqLen, // nullptr when not set
        optSeqLenKv, // nullptr when not set
        nulltensor, // blockTableOptional (nullptr)
        queryScaleWrapper, // qmode1 dequant scale or MXFP4 dtype scale
        keyScaleWrapper, // qmode1 dequant scale or MXFP4 dtype scale
        valueScaleWrapper, // qmode1 dequant scale or MXFP4 dtype scale
        qLevel0ScaleWrapper, // nullptr or FLOAT32 level0 scale
        kLevel0ScaleWrapper, // nullptr or FLOAT32 level0 scale
        vLevel0ScaleWrapper, // nullptr or FLOAT32 level0 scale
        qLayoutPtr, kvLayoutPtr, num_key_value_heads, MASK_TYPE, scale_value, inner_precise, blockSize, PRE_TOKENS,
        NEXT_TOKENS, softmax_lse_flag, attentionOut,
        softmaxLseOpt); // nullptr when flag=0

    return std::make_tuple(attentionOut, softmaxLse);
}
