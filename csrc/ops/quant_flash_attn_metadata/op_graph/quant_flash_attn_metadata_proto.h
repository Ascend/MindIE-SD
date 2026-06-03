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
 * \file flash_attn_metadata_proto.h
 * \brief
 */
#ifndef QUANT_FLASH_ATTN_METADATA_PROTO_H
#define QUANT_FLASH_ATTN_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Function KvQuantSparseFlashAttentionMetadata.

* @par Inputs:
* @li actual_seq_lengths_query: A matrix tensor. The type support int32.
* Efective sequence length of query in different batches.
* @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
* Effective sequence length of key/value in different batches.

* @par Attributes:
* @li batch_size: An int. batch size of QKV Tensor.
* @li query_seq_size: An int. Q tensor sequence len.
* @li query_head_num: An int. head number of query.
* @li kv_seq_size: An int. KV tensor sequence len.
* @li kv_head_num: An int. KV tensor head number.
* @li head_dim: An int. QKV head dim.
* @li topk_size: An int. topK size.
* @li sparse_block_size: An int. The block size in the sparse phase. Default: 1.
* @li layout_query: A string. Specifies the layout of `query`, the value must be one of ["BSND", "TND"]. Default: "BSND".
* @li layout_kv: A string. Specifies the layout of `key/value`, the value must be one of ["BSND", "TND", "PA_BSND"]. Default: "BSND".
* @li sparse_mode: Sparse mode. Default: 3.
* - 0: default mask
* - 3: rightDownCausal make
* - 4: band mask
* @li attention_mode: An int. Attention mode, 0: GQA/MHA; 1: MLA-naive; 2: MLA-absorb. Default: 0.
* @li tile_size: An int. Tile size. Default: 128.
* @li rope_head_dim: An int. Rope head dim. Default: 64.

* @par Outputs:
* @li metadata: A matrix tensor. The type support int32.
* The output of attention structure.
*/
REG_OP(QuantFlashAttnMetadata)
    .OPTIONAL_INPUT(cu_seqlens_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(cu_seqlens_kv, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_q, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(seqused_kv, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, -1)
    .ATTR(max_seqlen_kv, Int, -1)
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_kv, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .ATTR(mask_mode, Int, 1)
    .ATTR(win_left, Int, -1)
    .ATTR(win_right, Int, -1)
    .ATTR(layout_q, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(layout_out, String, "BSND")
    .REQUIRED_ATTR(custom_soc_version, String)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .OP_END_FACTORY_REG(QuantFlashAttnMetadata)

} // namespace ge

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_METADATA_PROTO_H
