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

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "quant_flash_attn.h"

using namespace at;

constexpr std::string_view QUANT_FLASH_ATTN_NAME = "aclnnQuantFlashAttn";

inline bool IsByteTensor(const at::Tensor &tensor) { return tensor.scalar_type() == at::ScalarType::Byte; }

inline void CheckQkvDtype(const char *name, const at::Tensor &tensor, const c10::optional<int64_t> &dtype) {
    if (!IsByteTensor(tensor)) {
        return;
    }
    TORCH_CHECK(dtype.has_value(), name, "_dtype can not be None when ", name, " is torch.uint8.");
    TORCH_CHECK(dtype.value() == CANN_DTYPE_FLOAT4_E2M1 || dtype.value() == CANN_DTYPE_HIFLOAT8, name,
        "_dtype must be torch_npu.float4_e2m1fn_x2 or torch_npu.hifloat8.");
}

inline void CheckDescaleDtype(const char *name, const at::Tensor &tensor, const c10::optional<int64_t> &dtype) {
    if (!IsByteTensor(tensor)) {
        return;
    }
    TORCH_CHECK(dtype.has_value(), name, "_dtype can not be None when ", name, " is torch.uint8.");
    TORCH_CHECK(dtype.value() == CANN_DTYPE_FLOAT8_E8M0, name, "_dtype must be torch_npu.float8_e8m0fnu.");
}

inline at::SmallVector<int64_t, 8> GetQfaAttentionOutSize(const at::Tensor &query, const at::Tensor &value,
    const c10::optional<int64_t> &q_dtype, const std::string &layout_q, const std::string &layout_out) {
    int64_t tSize = 0;
    int64_t nSize = 0;
    int64_t dSize = 0;
    int64_t sSize = 0;
    int64_t bSize = 0;
    if (layout_q == "TND") {
        tSize = query.size(0);
        nSize = query.size(1);
        dSize = value.size(2);
    } else if (layout_q == "BSND") {
        bSize = query.size(0);
        sSize = query.size(1);
        nSize = query.size(2);
        dSize = value.size(3);
    } else {
        bSize = query.size(0);
        nSize = query.size(1);
        sSize = query.size(2);
        dSize = value.size(3);
    }

    int64_t qDtypeRatio = q_dtype.has_value() && q_dtype.value() == CANN_DTYPE_FLOAT4_E2M1 ? 2 : 1;
    if (layout_out == "TND") {
        return {tSize, nSize, qDtypeRatio * dSize};
    }
    if (layout_out == "BNSD") {
        return {bSize, nSize, sSize, qDtypeRatio * dSize};
    }
    return {bSize, sSize, nSize, qDtypeRatio * dSize};
}

inline at::SmallVector<int64_t, 4> GetQfaSoftmaxLseSize(
    const at::Tensor &query, const std::string &layout_q, int64_t return_softmax_lse) {
    if (return_softmax_lse == 0) {
        return {0};
    }
    if (query.dim() == 3) {
        return {query.size(1), query.size(0)};
    }
    if (layout_q == "BSND") {
        return {query.size(0), query.size(2), query.size(1)};
    }
    return {query.size(0), query.size(1), query.size(2)};
}

std::tuple<at::Tensor, at::Tensor> quant_flash_attn_impl_npu(const at::Tensor &query, const at::Tensor &key,
    const at::Tensor &value, const at::Tensor &q_descale, const at::Tensor &k_descale, const at::Tensor &v_descale,
    int64_t q_quant_mode, int64_t k_quant_mode, int64_t v_quant_mode, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &cu_seqlens_kv,
    const c10::optional<at::Tensor> &seqused_q, const c10::optional<at::Tensor> &seqused_kv,
    const c10::optional<at::Tensor> &sinks, const c10::optional<at::Tensor> &attn_mask,
    const c10::optional<at::Tensor> &metadata, const c10::optional<int64_t> &q_dtype,
    const c10::optional<int64_t> &k_dtype, const c10::optional<int64_t> &v_dtype,
    const c10::optional<int64_t> &q_descale_dtype, const c10::optional<int64_t> &k_descale_dtype,
    const c10::optional<int64_t> &v_descale_dtype, int64_t quant_block_size_qs, int64_t quant_block_size_ks,
    int64_t quant_block_size_vs, double softmax_scale, int64_t mask_mode, int64_t win_left, int64_t win_right,
    int64_t max_seqlen_q, int64_t max_seqlen_kv, std::string layout_q, std::string layout_kv, std::string layout_out,
    int64_t softmax_precision, int64_t return_softmax_lse) {
    CheckQkvDtype("q", query, q_dtype);
    CheckQkvDtype("k", key, k_dtype);
    CheckQkvDtype("v", value, v_dtype);
    CheckDescaleDtype("q_descale", q_descale, q_descale_dtype);
    CheckDescaleDtype("k_descale", k_descale, k_descale_dtype);
    CheckDescaleDtype("v_descale", v_descale, v_descale_dtype);

    const c10::string_view device = "npu";
    at::Device outputDevice = at::Device(std::string(device));
    auto outputOptions = query.options().device(outputDevice);
    auto attentionOutSize = GetQfaAttentionOutSize(query, value, q_dtype, layout_q, layout_out);
    auto softmaxLseSize = GetQfaSoftmaxLseSize(query, layout_q, return_softmax_lse);
    at::Tensor attn_out = at_npu::native::empty_with_format(
        attentionOutSize, outputOptions.dtype(at::kBFloat16), at_npu::native::get_npu_format(query));

    at::Tensor softmax_lse = at_npu::native::empty_with_format(
        softmaxLseSize, outputOptions.dtype(at::kFloat), at_npu::native::get_npu_format(query));

    const char *layoutQPtr = layout_q.c_str();
    const char *layoutKvPtr = layout_kv.c_str();
    const char *layoutOutPtr = layout_out.c_str();

    auto blockTableTensor = block_table.value_or(at::Tensor());
    auto cuSeqlensQTensor = cu_seqlens_q.value_or(at::Tensor());
    auto cuSeqlensKvTensor = cu_seqlens_kv.value_or(at::Tensor());
    auto sequsedQTensor = seqused_q.value_or(at::Tensor());
    auto sequsedKvTensor = seqused_kv.value_or(at::Tensor());
    auto sinksTensor = sinks.value_or(at::Tensor());
    auto attnMaskTensor = attn_mask.value_or(at::Tensor());
    auto metadataTensor = metadata.value_or(at::Tensor());

    auto queryWrapper = MakeTensorWrapper(query, q_dtype);
    auto keyWrapper = MakeTensorWrapper(key, k_dtype);
    auto valueWrapper = MakeTensorWrapper(value, v_dtype);
    auto qDescaleWrapper = MakeTensorWrapper(q_descale, q_descale_dtype);
    auto kDescaleWrapper = MakeTensorWrapper(k_descale, k_descale_dtype);
    auto vDescaleWrapper = MakeTensorWrapper(v_descale, v_descale_dtype);

    EXEC_NPU_CMD<QUANT_FLASH_ATTN_NAME>(queryWrapper, keyWrapper, valueWrapper, qDescaleWrapper, kDescaleWrapper,
        vDescaleWrapper, blockTableTensor, cuSeqlensQTensor, cuSeqlensKvTensor, sequsedQTensor, sequsedKvTensor,
        sinksTensor, attnMaskTensor, metadataTensor, q_quant_mode, k_quant_mode, v_quant_mode, quant_block_size_qs,
        quant_block_size_ks, quant_block_size_vs, softmax_scale, mask_mode, win_left, win_right, max_seqlen_q,
        max_seqlen_kv, layoutQPtr, layoutKvPtr, layoutOutPtr, softmax_precision, return_softmax_lse, attn_out,
        softmax_lse);

    return std::make_tuple(attn_out, softmax_lse);
}
