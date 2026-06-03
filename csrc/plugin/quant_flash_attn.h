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

#ifndef QUANT_FLASH_ATTN_MINDIE_SD_IMPL_H
#define QUANT_FLASH_ATTN_MINDIE_SD_IMPL_H
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <string>
#include <tuple>

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
    int64_t softmax_precision, int64_t return_softmax_lse);

#endif // QUANT_FLASH_ATTN_MINDIE_SD_IMPL_H
