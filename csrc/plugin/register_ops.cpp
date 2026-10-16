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

#include "la.h"
#include "adalayernorm.h"
#include "la_preprocess.h"
#include "rainfusionattention.h"
#include "ada_block_sparse_attention.h"
#include "sparse_block_estimate.h"
#include "layernorm.h"
#include "block_sparse_attention.h"
#include "quant_flash_attn.h"
#include "quant_flash_attn_metadata.h"
#include "fused_infer_attention_score.h"
#include "norm_rope_concat.h"

TORCH_LIBRARY(mindiesd, m) {
    m.def("la(Tensor query, Tensor key, Tensor value, \
        Tensor? atten_mask=None, Tensor? alibi_mask=None, Tensor? \
        drop_mask=None, float scale_value=1.0, int head_num=2, str input_layout='BNSD', \
        float keep_prob=1.0, int pre_tokens=2147483647, int next_tokens=1, \
        bool is_highPrecision=True)  -> (Tensor, Tensor)");
    m.def("adaln(Tensor x, Tensor scale, Tensor shift, Tensor? weight=None, \
        Tensor? bias=None, float? epsilon=1e-5) \
        -> Tensor");
    m.def("adaln_v2(Tensor x, Tensor scale, Tensor shift, Tensor? weight=None, \
        Tensor? bias=None, float? epsilon=1e-5) \
        -> (Tensor, Tensor, Tensor)");
    m.def("la_preprocess(Tensor query, Tensor key, Tensor value, int align_len=256) \
        -> (Tensor, Tensor, Tensor)");
    m.def("rainfusionattention(Tensor query, Tensor key, Tensor value, Tensor select_idx, \
        Tensor select_num_idx, int[] blockshape, Tensor? attn_mask=None, int[]? actual_seq_qlen=None, \
        int[]? actual_seq_kvlen=None, Tensor? block_table=None, str q_input_layout='TND', str kv_input_layout='TND', \
        int head_num=1, int mask_type=0, float scale=1.0, \
        int inner_precise=1, int block_size=0) -> (Tensor, Tensor)");
    m.def("ada_block_sparse_attention(Tensor query, Tensor key,  \
        Tensor value, Tensor sparse_mask, Tensor sparse_count_table,  \
        str input_layout='BNSD', int sparse_size=128, int num_heads=1, \
        int num_key_value_heads=1, float scale_value=1,  \
        bool causal=True, int inner_precise=1, int pre_tokens=214748647, int next_tokens=0, \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None)   \
        -> Tensor");
    m.def("sparse_block_estimate(Tensor query, Tensor key,  \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None,  \
        str input_layout='BNSD', int stride=8, int sparse_size=128,  \
        int num_heads=1, int num_key_value_heads=1, float scale_value=1,  \
        float threshold=1, bool causal=True, bool keep_sink=True,  \
        bool keep_recent=True, float row_sparse=1) \
        -> (Tensor, Tensor)");
    m.def("layernorm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, \
        int impl_mode=0) -> (Tensor, Tensor, Tensor)");
    m.def("block_sparse_attention(Tensor query, Tensor key, Tensor value, \
        Tensor? block_sparse_mask=None, int[] block_shape=[128,128], \
        str q_input_layout='BNSD', str kv_input_layout='BNSD', \
        int num_key_value_heads=1, float scale_value=1.0, int inner_precise=0, \
        int[]? actual_seq_lengths=None, int[]? actual_seq_lengths_kv=None, \
        int softmax_lse_flag=0, \
        Tensor? q_dequant_scale=None, Tensor? k_dequant_scale=None, \
        Tensor? v_dequant_scale=None) -> (Tensor, Tensor)");
    m.def("quant_flash_attn(Tensor query, Tensor key, Tensor value, \
        Tensor q_descale, Tensor k_descale, Tensor v_descale, \
        int q_quant_mode, int k_quant_mode, int v_quant_mode, \
        Tensor? block_table=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, \
        Tensor? seqused_q=None, Tensor? seqused_kv=None, \
        Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, \
        int? q_dtype=None, int? k_dtype=None, int? v_dtype=None, \
        int? q_descale_dtype=None, int? k_descale_dtype=None, int? v_descale_dtype=None, \
        int quant_block_size_qs=128, int quant_block_size_ks=256, int quant_block_size_vs=256, \
        float softmax_scale=1.0, int mask_mode=1, int win_left=-1, int win_right=-1, \
        int max_seqlen_q=-1, int max_seqlen_kv=-1, \
        str layout_q='BSND', str layout_kv='BSND', str layout_out='BSND', \
        int softmax_precision=0, int return_softmax_lse=0) -> (Tensor, Tensor)");
    m.def("quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, \
        int q_quant_mode, int k_quant_mode, int v_quant_mode, *, \
        Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, \
        Tensor? seqused_q=None, Tensor? seqused_kv=None, \
        int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, \
        int? q_dtype=None, int? k_dtype=None, int? v_dtype=None, \
        int? mask_mode=None, int? win_left=None, int? win_right=None, \
        str? layout_q=None, str? layout_kv=None, str? layout_out=None) -> Tensor");
    m.def("fused_infer_attention_score_v2(Tensor query, Tensor key, Tensor value, *, \
        Tensor? query_rope=None, Tensor? key_rope=None, Tensor? pse_shift=None, Tensor? atten_mask=None, \
        int[]? actual_seq_qlen=None, int[]? actual_seq_kvlen=None, Tensor? block_table=None, \
        Tensor? dequant_scale1=None, Tensor? quant_scale1=None, Tensor? dequant_scale2=None, \
        Tensor? dequant_scale_query=None, Tensor? dequant_scale_key=None, Tensor? dequant_offset_key=None, \
        Tensor? dequant_scale_value=None, Tensor? dequant_offset_value=None, Tensor? dequant_scale_key_rope=None, \
        Tensor? quant_scale_out=None, Tensor? quant_offset_out=None, Tensor? learnable_sink=None, \
        int num_query_heads=1, int num_key_value_heads=0, float softmax_scale=1.0, \
        int pre_tokens=2147483647, int next_tokens=2147483647, str input_layout='BSH', int sparse_mode=0, \
        int block_size=0, int query_quant_mode=0, int key_quant_mode=0, int value_quant_mode=0, \
        int inner_precise=0, bool return_softmax_lse=False, int? query_dtype=None, int? key_dtype=None, \
        int? value_dtype=None, int? query_rope_dtype=None, int? key_rope_dtype=None, \
        int? key_shared_prefix_dtype=None, int? value_shared_prefix_dtype=None, \
        int? dequant_scale_query_dtype=None, int? dequant_scale_key_dtype=None, \
        int? dequant_scale_value_dtype=None, int? dequant_scale_key_rope_dtype=None, \
        ScalarType? out_dtype=None) -> (Tensor, Tensor)");
    m.def("norm_rope_concat(Tensor query, Tensor key, Tensor value, \
        Tensor? encoder_query=None, Tensor? encoder_key=None, Tensor? encoder_value=None, \
        Tensor? norm_query_weight=None, Tensor? norm_query_bias=None, \
        Tensor? norm_key_weight=None, Tensor? norm_key_bias=None, \
        Tensor? norm_added_query_weight=None, Tensor? norm_added_query_bias=None, \
        Tensor? norm_added_key_weight=None, Tensor? norm_added_key_bias=None, \
        Tensor? rope_sin=None, Tensor? rope_cos=None, \
        int norm_type=0, int norm_added_type=0, int rope_type=0, int concat_order=0, \
        float eps=1e-5, bool is_training=False) \
        -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mindiesd, PrivateUse1, m) {
    m.impl("la", &la_mindie_sd_impl_npu);
    m.impl("adaln", &adaln_mindie_sd_impl_npu);
    m.impl("adaln_v2", &adaln_v2_mindie_sd_impl_npu);
    m.impl("la_preprocess", &la_preprocess_mindie_sd_impl_npu);
    m.impl("rainfusionattention", &rainfusionattention_mindie_sd_impl_npu);
    m.impl("ada_block_sparse_attention", &ada_block_sparse_attention_impl_npu);
    m.impl("sparse_block_estimate", &sparse_block_estimate_mindie_sd_impl_npu);
    m.impl("layernorm", &layernorm_mindie_sd_impl_npu);
    m.impl("block_sparse_attention", &block_sparse_attention_impl_npu);
    m.impl("quant_flash_attn", &quant_flash_attn_impl_npu);
    m.impl("quant_flash_attn_metadata", &quant_flash_attn_metadata_impl_npu);
    m.impl("fused_infer_attention_score_v2", &fused_infer_attention_score_v2_impl_npu);
    m.impl("norm_rope_concat", &norm_rope_concat_mindie_sd_impl_npu);
}

TORCH_LIBRARY_IMPL(mindiesd, CatchAll, m) { m.impl("quant_flash_attn_metadata", &quant_flash_attn_metadata_impl_npu); }

