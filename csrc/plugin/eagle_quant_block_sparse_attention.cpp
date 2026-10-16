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
// V2 kernel supports BF16/FP16/FP8 natively.
constexpr std::string_view EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_NAME = "aclnnEagleQuantBlockSparseAttention";

constexpr int64_t MASK_TYPE = 0; // no attention mask
constexpr int64_t PRE_TOKENS = 2147483647; // full context window
constexpr int64_t NEXT_TOKENS = 2147483647;

inline at::ScalarType ResolveOutputDtype(const at::Tensor &query, const c10::optional<at::Tensor> &query_scale,
    const c10::optional<at::ScalarType> &output_dtype)
{
    if (output_dtype.has_value()) {
        return output_dtype.value();
    }
    // Quant path default: BF16. Non-quant: match query.
    return query_scale.has_value() ? at::kBFloat16 : query.scalar_type();
}

// Validate optional *_dtype against tensor storage. INT8 (Char) storage is the
// quant bitcast path (source/golden: value.view(int8) + value_dtype=fp8); the
// logical FP8 dtype legitimately differs from int8 storage and its ScalarType
// enum is not portable (torch_npu.float8_e4m3fn maps to different slots across
// versions, e.g. prints as "UInt7"), so accept ANY hint on int8 tensors instead
// of enumerating FP8 enums. *_dtype never reaches the kernel — the kernel derives
// dtype from the tensor + tilingKey (DT_INT8 / DT_FLOAT8_E4M3FN V share one key).
// Do NOT rewrite ACL dtype via TensorWrapper here — custom ACL_DTYPE_FLOAT8_*
// values mismatch opdev and trigger "Key/Value datatype mismatch with query".
inline void CheckOptionalInputDtype(const char *name, const at::Tensor &tensor,
    const c10::optional<at::ScalarType> &dtype)
{
    if (!dtype.has_value() || dtype.value() == tensor.scalar_type() || tensor.scalar_type() == at::kChar) {
        return;
    }
    TORCH_CHECK(false, "eagle_quant_block_sparse_attention: ", name, "_dtype (", dtype.value(),
        ") is incompatible with tensor dtype (", tensor.scalar_type(), ")");
}
} // namespace

std::tuple<at::Tensor, at::Tensor> eagle_quant_block_sparse_attention_impl_npu(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const c10::optional<at::Tensor> &block_sparse_mask, at::IntArrayRef block_shape,
    std::string q_input_layout, std::string kv_input_layout, int64_t num_key_value_heads, double scale_value,
    int64_t inner_precise, c10::OptionalIntArrayRef actual_seq_lengths, c10::OptionalIntArrayRef actual_seq_lengths_kv,
    int64_t softmax_lse_flag, const c10::optional<at::Tensor> &query_scale,
    const c10::optional<at::Tensor> &key_scale, const c10::optional<at::Tensor> &value_scale,
    const c10::optional<at::ScalarType> &query_dtype, const c10::optional<at::ScalarType> &key_dtype,
    const c10::optional<at::ScalarType> &value_dtype, const c10::optional<at::ScalarType> &output_dtype) {
    TORCH_CHECK(q_input_layout == "TND" || q_input_layout == "BNSD",
        "eagle_quant_block_sparse_attention: q_input_layout only supports 'TND' and 'BNSD', got ", q_input_layout);
    TORCH_CHECK(kv_input_layout == "TND" || kv_input_layout == "BNSD",
        "eagle_quant_block_sparse_attention: kv_input_layout only supports 'TND' and 'BNSD', got ", kv_input_layout);
    TORCH_CHECK(q_input_layout == kv_input_layout,
        "eagle_quant_block_sparse_attention: q_input_layout and kv_input_layout must be consistent.");
    TORCH_CHECK(q_input_layout != "TND" || (actual_seq_lengths.has_value() && actual_seq_lengths_kv.has_value()),
        "eagle_quant_block_sparse_attention: actual_seq_lengths and actual_seq_lengths_kv are required for TND layout.");

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

    auto outOptions = query.options().dtype(ResolveOutputDtype(query, query_scale, output_dtype));
    at::Tensor attentionOut =
        at_npu::native::empty_with_format(query.sizes(), outOptions, at_npu::native::get_npu_format(query));

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

    CheckOptionalInputDtype("query", query, query_dtype);
    CheckOptionalInputDtype("key", key, key_dtype);
    CheckOptionalInputDtype("value", value, value_dtype);

    // Pass storage tensors as-is (same as pre-interface-fix path that passed
    // CheckDataType). query/key/value_dtype are API-compatible validators for
    // bitcast callers; output_dtype selects attentionOut allocation dtype.
    // query/key/value_scale inserted after blockTableOptional.
    // BF16/FP16 path: pass nulltensor (nullptr) for all three scales.
    // Quant path: pass FLOAT32 scale tensors for Q/K/V (OpDef valueScale is DT_FLOAT).
    EXEC_NPU_CMD<EAGLE_QUANT_BLOCK_SPARSE_ATTENTION_NAME>(query, key, value, block_sparse_mask,
        nulltensor, // attenMaskOptional (nullptr)
        block_shape,
        optSeqLen, // nullptr when not set
        optSeqLenKv, // nullptr when not set
        nulltensor, // blockTableOptional (nullptr)
        query_scale, // nullptr for BF16/FP16, FLOAT32 for quant
        key_scale, // nullptr for BF16/FP16, FLOAT32 for quant
        value_scale, // nullptr for BF16/FP16, FLOAT32 for quant
        qLayoutPtr, kvLayoutPtr, num_key_value_heads, MASK_TYPE, scale_value, inner_precise, blockSize, PRE_TOKENS,
        NEXT_TOKENS, softmax_lse_flag, attentionOut,
        softmaxLseOpt); // nullptr when flag=0

    return std::make_tuple(attentionOut, softmaxLse);
}
