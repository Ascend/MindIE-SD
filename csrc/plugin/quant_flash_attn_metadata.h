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

#ifndef QUANT_FLASH_ATTN_METADATA_MINDIE_SD_IMPL_H
#define QUANT_FLASH_ATTN_METADATA_MINDIE_SD_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>

at::Tensor quant_flash_attn_metadata_impl_npu(int64_t num_heads_q, int64_t num_heads_kv, int64_t head_dim,
    int64_t q_quant_mode, int64_t k_quant_mode, int64_t v_quant_mode, const c10::optional<at::Tensor> &cu_seqlens_q,
    const c10::optional<at::Tensor> &cu_seqlens_kv, const c10::optional<at::Tensor> &seqused_q,
    const c10::optional<at::Tensor> &seqused_kv, const c10::optional<int64_t> &batch_size,
    const c10::optional<int64_t> &max_seqlen_q, const c10::optional<int64_t> &max_seqlen_kv,
    const c10::optional<int64_t> &q_dtype, const c10::optional<int64_t> &k_dtype, const c10::optional<int64_t> &v_dtype,
    const c10::optional<int64_t> &mask_mode, const c10::optional<int64_t> &win_left,
    const c10::optional<int64_t> &win_right, const c10::optional<std::string> &layout_q,
    const c10::optional<std::string> &layout_kv, const c10::optional<std::string> &layout_out);

#endif // QUANT_FLASH_ATTN_METADATA_MINDIE_SD_IMPL_H
