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

#include "rope.h"
#include "la.h"
#include "adalayernorm.h"
#include "la_preprocess.h"
#include "rainfusionattention.h"
#include "block_sparse_attention.h"
#include "sparse_block_estimate.h"


TORCH_LIBRARY(mindie, m)
{
    m.def("rope_mindie_sd(Tensor query, Tensor key, Tensor value, int mode) -> Tensor");
    m.def(
        "la_mindie_sd(Tensor query, Tensor key, Tensor value, \
        Tensor? atten_mask=None, Tensor? alibi_mask=None, Tensor? \
        drop_mask=None, float scale_value=1.0, int head_num=2, str input_layout='BNSD', \
        float keep_prob=1.0, int pre_tokens=2147483647, int next_tokens=1, \
        bool is_highPrecision=True)  -> (Tensor, Tensor)");
    m.def("adaln_mindie_sd(Tensor x, Tensor scale, Tensor shift, Tensor? weight=None, \
        Tensor? bias=None, float? epsilon=1e-5) \
        -> Tensor");
    m.def("la_preprocess_mindie_sd(Tensor query, Tensor key, Tensor value, int align_len=256) \
        -> (Tensor, Tensor, Tensor)");
    m.def("rainfusionattention_mindie_sd(Tensor query, Tensor key, Tensor value, Tensor select_idx, \
        Tensor select_num_idx, int[] blockshape, Tensor? attn_mask=None, int[]? actual_seq_qlen=None, \
        int[]? actual_seq_kvlen=None, Tensor? block_table=None, str q_input_layout='TND', str kv_input_layout='TND', \
        int head_num=1, int mask_type=0, float scale=1.0, \
        int inner_precise=1, int block_size=0) -> (Tensor, Tensor)");
    m.def("block_sparse_attention(Tensor query, Tensor key,  \
        Tensor value, Tensor sparse_mask, Tensor sparse_count_table,  \
        str input_layout='BNSD', int sparse_size=128, int num_heads=1, \
        int num_key_value_heads=1, float scale_value=1,  \
        bool causal=True, int inner_precise=1, int pre_tokens=214748647, int next_tokens=0, \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None)   \
        -> Tensor");
    m.def("sparse_block_estimate_mindie_sd(Tensor query, Tensor key,  \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None,  \
        str input_layout='BNSD', int stride=8, int sparse_size=128,  \
        int num_heads=1, int num_key_value_heads=1, float scale_value=1,  \
        float threshold=1, bool causal=True, bool keep_sink=True,  \
        bool keep_recent=True, float row_sparse=1) \
        -> (Tensor, Tensor)");
}


TORCH_LIBRARY_IMPL(mindie, PrivateUse1, m)
{
    m.impl("rope_mindie_sd", &rope_mindie_sd_impl_npu);
    m.impl("la_mindie_sd", &la_mindie_sd_impl_npu);
    m.impl("adaln_mindie_sd", &adaln_mindie_sd_impl_npu);
    m.impl("la_preprocess_mindie_sd", &la_preprocess_mindie_sd_impl_npu);
    m.impl("rainfusionattention_mindie_sd", &rainfusionattention_mindie_sd_impl_npu);
    m.impl("block_sparse_attention", &block_sparse_attention_impl_npu);
    m.impl("sparse_block_estimate_mindie_sd", &sparse_block_estimate_mindie_sd_impl_npu);
}