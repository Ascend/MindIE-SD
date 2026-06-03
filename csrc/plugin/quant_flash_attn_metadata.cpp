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
#include "quant_flash_attn_metadata.h"

using namespace at;

constexpr std::string_view QUANT_FLASH_ATTN_METADATA_NAME = "aclnnQuantFlashAttnMetadata";

inline int64_t CalculateQfaMetadataBatchSize(const c10::optional<int64_t> &batch_size,
    const c10::optional<at::Tensor> &cu_seqlens_q, const c10::optional<at::Tensor> &seqused_q) {
    if (batch_size.has_value()) {
        return batch_size.value();
    }
    if (cu_seqlens_q.has_value() && cu_seqlens_q.value().defined() && cu_seqlens_q.value().size(0) > 0) {
        return cu_seqlens_q.value().size(0) - 1;
    }
    if (seqused_q.has_value() && seqused_q.value().defined()) {
        return seqused_q.value().size(0);
    }
    return 0;
}

at::Tensor quant_flash_attn_metadata_impl_npu(int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    int64_t q_quant_mode, int64_t k_quant_mode, int64_t v_quant_mode, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<int64_t> &batch_size,
    const c10::optional<int64_t> &max_seqlen_q, const c10::optional<int64_t> &max_seqlen_kv,
    const c10::optional<int64_t> &q_dtype, const c10::optional<int64_t> &k_dtype, const c10::optional<int64_t> &v_dtype,
    const c10::optional<int64_t> &mask_mode, const c10::optional<int64_t> &win_left,
    const c10::optional<int64_t> &win_right, const c10::optional<std::string> &layout_q,
    const c10::optional<std::string> &layout_kv, const c10::optional<std::string> &layout_out) {
    // CANN's Python wrapper allocates 4096 int32 elements. The AICPU metadata layout
    // can exceed the migrated op's 1024-element infer-shape constant for one section.
    constexpr int64_t QFA_META_SIZE = 4096;
    int64_t batchSize = CalculateQfaMetadataBatchSize(batch_size, cu_seqlens_q, seqused_q);
    int64_t maxSeqlenQ = max_seqlen_q.value_or(-1);
    int64_t maxSeqlenKv = max_seqlen_kv.value_or(-1);
    int64_t qDtype = q_dtype.value_or(0);
    int64_t kDtype = k_dtype.value_or(0);
    int64_t vDtype = v_dtype.value_or(0);
    int64_t maskMode = mask_mode.value_or(1);
    int64_t winLeft = win_left.value_or(-1);
    int64_t winRight = win_right.value_or(-1);
    std::string layoutQ = layout_q.value_or("BSND");
    std::string layoutKv = layout_kv.value_or("BSND");
    std::string layoutOut = layout_out.value_or("BSND");

    at::Tensor metadata = at_npu::native::empty_with_format(
        {QFA_META_SIZE}, at::TensorOptions(torch_npu::utils::get_npu_device_type()).dtype(c10::kInt), ACL_FORMAT_ND);

    const char *layoutQPtr = layoutQ.c_str();
    const char *layoutKvPtr = layoutKv.c_str();
    const char *layoutOutPtr = layoutOut.c_str();

    auto cuSeqlensQTensor = cu_seqlens_q.value_or(at::Tensor());
    auto cuSeqlensKvTensor = cu_seqlens_kv.value_or(at::Tensor());
    auto sequsedQTensor = seqused_q.value_or(at::Tensor());
    auto sequsedKvTensor = seqused_kv.value_or(at::Tensor());

    EXEC_NPU_CMD<QUANT_FLASH_ATTN_METADATA_NAME>(cuSeqlensQTensor, cuSeqlensKvTensor, sequsedQTensor, sequsedKvTensor,
        batchSize, maxSeqlenQ, maxSeqlenKv, num_heads_q, num_heads_kv, head_dim, q_quant_mode, k_quant_mode,
        v_quant_mode, qDtype, kDtype, vDtype, maskMode, winLeft, winRight, layoutQPtr, layoutKvPtr, layoutOutPtr,
        metadata);

    return metadata;
}
