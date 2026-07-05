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

#ifndef FUSED_INFER_ATTENTION_SCORE_PARAM_H
#define FUSED_INFER_ATTENTION_SCORE_PARAM_H

#include <memory>
#include <sstream>
#include "op_host_csv_case_loader.h"

namespace FusedInferAttentionScoreUT {

struct FusedInferAttentionHostUtParamBase : public HostUtParamBase {
    int64_t num_heads;
    float scale;
    int64_t pre_tokens;
    int64_t next_tokens;
    std::string input_layout;
    int64_t num_key_value_heads;
    int64_t sparse_mode;
    int64_t inner_precise;
    int64_t block_size;
    int64_t antiquant_mode;
    bool softmax_lse_flag;
    int64_t key_antiquant_mode;
    int64_t value_antiquant_mode;
    int64_t query_quant_mode;
    int64_t pse_type;
    int64_t out_dtype;

    FusedInferAttentionHostUtParamBase(const csv_map &csvMap) : HostUtParamBase(csvMap) {
        this->num_heads = std::stoll(ReadMap(csvMap, "num_heads"));
        this->scale = std::stof(ReadMap(csvMap, "scale"));
        this->pre_tokens = std::stoll(ReadMap(csvMap, "pre_tokens"));
        this->next_tokens = std::stoll(ReadMap(csvMap, "next_tokens"));
        this->input_layout = ReadMap(csvMap, "input_layout");
        this->num_key_value_heads = std::stoll(ReadMap(csvMap, "num_key_value_heads"));
        this->sparse_mode = std::stoll(ReadMap(csvMap, "sparse_mode"));
        this->inner_precise = std::stoll(ReadMap(csvMap, "inner_precise"));
        this->block_size = std::stoll(ReadMap(csvMap, "block_size"));
        this->antiquant_mode = std::stoll(ReadMap(csvMap, "antiquant_mode"));
        this->softmax_lse_flag = std::stoi(ReadMap(csvMap, "softmax_lse_flag"));
        this->key_antiquant_mode = std::stoll(ReadMap(csvMap, "key_antiquant_mode"));
        this->value_antiquant_mode = std::stoll(ReadMap(csvMap, "value_antiquant_mode"));
        this->query_quant_mode = std::stoll(ReadMap(csvMap, "query_quant_mode"));
        this->pse_type = std::stoll(ReadMap(csvMap, "pse_type"));
        this->out_dtype = std::stoll(ReadMap(csvMap, "out_dtype"));
    }
};

struct FusedInferAttentionTilingUtParam : public FusedInferAttentionHostUtParamBase {
    gert::TilingContextPara::TensorDescription query = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription pse_shift = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription atten_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription actual_seq_lengths = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription actual_seq_lengths_kv = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dequant_scale1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_scale1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dequant_scale2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_scale2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_offset2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription antiquant_offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription query_padding_size = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kv_padding_size = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_antiquant_offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value_antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value_antiquant_offset = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_shared_prefix = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription value_shared_prefix = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription actual_shared_prefix_len = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription query_rope = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_rope = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription key_rope_antiquant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription dequant_scale_query = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription learnable_sink = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription q_start_idx = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription kv_start_idx = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription attention_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;

    std::shared_ptr<std::vector<int64_t>> actual_seq_lengths_data;
    std::shared_ptr<std::vector<int64_t>> actual_seq_lengths_kv_data;

    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    static void ApplyConstData(const csv_map &csvMap, const std::string &name,
        gert::TilingContextPara::TensorDescription &desc, std::shared_ptr<std::vector<int64_t>> &buf) {
        std::string dataStr = ReadMap(csvMap, name + "_data");
        if (!dataStr.empty()) {
            buf = std::make_shared<std::vector<int64_t>>();
            std::istringstream iss(dataStr);
            int64_t val;
            while (iss >> val) {
                buf->push_back(val);
            }
            desc.isConst_ = true;
            desc.constValue_ = buf->data();
        }
    }

    FusedInferAttentionTilingUtParam(const csv_map &csvMap) : FusedInferAttentionHostUtParamBase(csvMap) {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_shape", "query_dtype", "query_format", this->query));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "pse_shift_shape", "pse_shift_dtype", "pse_shift_format", this->pse_shift));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "atten_mask_shape", "atten_mask_dtype", "atten_mask_format", this->atten_mask));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_lengths_shape", "actual_seq_lengths_dtype",
            "actual_seq_lengths_format", this->actual_seq_lengths));
        ApplyConstData(csvMap, "actual_seq_lengths", this->actual_seq_lengths, this->actual_seq_lengths_data);
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_lengths_kv_shape",
            "actual_seq_lengths_kv_dtype", "actual_seq_lengths_kv_format", this->actual_seq_lengths_kv));
        ApplyConstData(csvMap, "actual_seq_lengths_kv", this->actual_seq_lengths_kv, this->actual_seq_lengths_kv_data);
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "dequant_scale1_shape", "dequant_scale1_dtype", "dequant_scale1_format", this->dequant_scale1));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale1_shape", "quant_scale1_dtype", "quant_scale1_format", this->quant_scale1));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "dequant_scale2_shape", "dequant_scale2_dtype", "dequant_scale2_format", this->dequant_scale2));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale2_shape", "quant_scale2_dtype", "quant_scale2_format", this->quant_scale2));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "quant_offset2_shape", "quant_offset2_dtype", "quant_offset2_format", this->quant_offset2));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "antiquant_scale_shape", "antiquant_scale_dtype", "antiquant_scale_format", this->antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype",
            "antiquant_offset_format", this->antiquant_offset));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "query_padding_size_shape", "query_padding_size_dtype",
            "query_padding_size_format", this->query_padding_size));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "kv_padding_size_shape", "kv_padding_size_dtype", "kv_padding_size_format", this->kv_padding_size));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_antiquant_scale_shape", "key_antiquant_scale_dtype",
            "key_antiquant_scale_format", this->key_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_antiquant_offset_shape", "key_antiquant_offset_dtype",
            "key_antiquant_offset_format", this->key_antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_antiquant_scale_shape",
            "value_antiquant_scale_dtype", "value_antiquant_scale_format", this->value_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_antiquant_offset_shape",
            "value_antiquant_offset_dtype", "value_antiquant_offset_format", this->value_antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shared_prefix_shape", "key_shared_prefix_dtype",
            "key_shared_prefix_format", this->key_shared_prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_shared_prefix_shape", "value_shared_prefix_dtype",
            "value_shared_prefix_format", this->value_shared_prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_shared_prefix_len_shape",
            "actual_shared_prefix_len_dtype", "actual_shared_prefix_len_format", this->actual_shared_prefix_len));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_rope_shape", "query_rope_dtype", "query_rope_format", this->query_rope));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_rope_shape", "key_rope_dtype", "key_rope_format", this->key_rope));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_rope_antiquant_scale_shape",
            "key_rope_antiquant_scale_dtype", "key_rope_antiquant_scale_format", this->key_rope_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "dequant_scale_query_shape", "dequant_scale_query_dtype",
            "dequant_scale_query_format", this->dequant_scale_query));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "learnable_sink_shape", "learnable_sink_dtype", "learnable_sink_format", this->learnable_sink));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_start_idx_shape", "q_start_idx_dtype", "q_start_idx_format", this->q_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_start_idx_shape", "kv_start_idx_dtype", "kv_start_idx_format", this->kv_start_idx));

        this->outputInstance.emplace_back(GetTensorGE(
            csvMap, "attention_out_shape", "attention_out_dtype", "attention_out_format", this->attention_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectTilingKey = stoull(ReadMap(csvMap, "expectTilingKey"));
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct FusedInferAttentionInferShapeUtParam : public FusedInferAttentionHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription query = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription pse_shift = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription atten_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription actual_seq_lengths = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription actual_seq_lengths_kv = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription quant_scale1 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription quant_scale2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription quant_offset2 = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription block_table = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription query_padding_size = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kv_padding_size = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value_antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value_antiquant_offset = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_shared_prefix = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription value_shared_prefix = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription actual_shared_prefix_len = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription query_rope = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_rope = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription key_rope_antiquant_scale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription dequant_scale_query = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription learnable_sink = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription q_start_idx = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription kv_start_idx = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription attention_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    FusedInferAttentionInferShapeUtParam(const csv_map &csvMap) : FusedInferAttentionHostUtParamBase(csvMap) {
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_shape", "query_dtype", "query_format", this->query));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shape", "key_dtype", "key_format", this->key));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "value_shape", "value_dtype", "value_format", this->value));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "pse_shift_shape", "pse_shift_dtype", "pse_shift_format", this->pse_shift));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "atten_mask_shape", "atten_mask_dtype", "atten_mask_format", this->atten_mask));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_lengths_shape", "actual_seq_lengths_dtype",
            "actual_seq_lengths_format", this->actual_seq_lengths));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_seq_lengths_kv_shape",
            "actual_seq_lengths_kv_dtype", "actual_seq_lengths_kv_format", this->actual_seq_lengths_kv));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "dequant_scale1_shape", "dequant_scale1_dtype", "dequant_scale1_format", this->dequant_scale1));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale1_shape", "quant_scale1_dtype", "quant_scale1_format", this->quant_scale1));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "dequant_scale2_shape", "dequant_scale2_dtype", "dequant_scale2_format", this->dequant_scale2));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale2_shape", "quant_scale2_dtype", "quant_scale2_format", this->quant_scale2));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "quant_offset2_shape", "quant_offset2_dtype", "quant_offset2_format", this->quant_offset2));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "antiquant_scale_shape", "antiquant_scale_dtype", "antiquant_scale_format", this->antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "antiquant_offset_shape", "antiquant_offset_dtype",
            "antiquant_offset_format", this->antiquant_offset));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "query_padding_size_shape", "query_padding_size_dtype",
            "query_padding_size_format", this->query_padding_size));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "kv_padding_size_shape", "kv_padding_size_dtype", "kv_padding_size_format", this->kv_padding_size));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_antiquant_scale_shape", "key_antiquant_scale_dtype",
            "key_antiquant_scale_format", this->key_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_antiquant_offset_shape", "key_antiquant_offset_dtype",
            "key_antiquant_offset_format", this->key_antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_antiquant_scale_shape",
            "value_antiquant_scale_dtype", "value_antiquant_scale_format", this->value_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_antiquant_offset_shape",
            "value_antiquant_offset_dtype", "value_antiquant_offset_format", this->value_antiquant_offset));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_shared_prefix_shape", "key_shared_prefix_dtype",
            "key_shared_prefix_format", this->key_shared_prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "value_shared_prefix_shape", "value_shared_prefix_dtype",
            "value_shared_prefix_format", this->value_shared_prefix));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "actual_shared_prefix_len_shape",
            "actual_shared_prefix_len_dtype", "actual_shared_prefix_len_format", this->actual_shared_prefix_len));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "query_rope_shape", "query_rope_dtype", "query_rope_format", this->query_rope));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "key_rope_shape", "key_rope_dtype", "key_rope_format", this->key_rope));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "key_rope_antiquant_scale_shape",
            "key_rope_antiquant_scale_dtype", "key_rope_antiquant_scale_format", this->key_rope_antiquant_scale));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "dequant_scale_query_shape", "dequant_scale_query_dtype",
            "dequant_scale_query_format", this->dequant_scale_query));
        this->inputInstance.emplace_back(GetTensorGE(
            csvMap, "learnable_sink_shape", "learnable_sink_dtype", "learnable_sink_format", this->learnable_sink));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_start_idx_shape", "q_start_idx_dtype", "q_start_idx_format", this->q_start_idx));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "kv_start_idx_shape", "kv_start_idx_dtype", "kv_start_idx_format", this->kv_start_idx));

        this->outputInstance.emplace_back(GetTensorGE(
            csvMap, "attention_out_shape", "attention_out_dtype", "attention_out_format", this->attention_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {
                GetShapeArr(ReadMap(csvMap, "attention_out_shape")), GetShapeArr(ReadMap(csvMap, "softmax_lse_shape"))};
        }
    }
};

struct FusedInferAttentionInferDTypeUtParam : public FusedInferAttentionHostUtParamBase {
    ge::DataType query = ge::DT_UNDEFINED;
    ge::DataType key = ge::DT_UNDEFINED;
    ge::DataType value = ge::DT_UNDEFINED;
    ge::DataType pse_shift = ge::DT_UNDEFINED;
    ge::DataType atten_mask = ge::DT_UNDEFINED;
    ge::DataType actual_seq_lengths = ge::DT_UNDEFINED;
    ge::DataType actual_seq_lengths_kv = ge::DT_UNDEFINED;
    ge::DataType dequant_scale1 = ge::DT_UNDEFINED;
    ge::DataType quant_scale1 = ge::DT_UNDEFINED;
    ge::DataType dequant_scale2 = ge::DT_UNDEFINED;
    ge::DataType quant_scale2 = ge::DT_UNDEFINED;
    ge::DataType quant_offset2 = ge::DT_UNDEFINED;
    ge::DataType antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType block_table = ge::DT_UNDEFINED;
    ge::DataType query_padding_size = ge::DT_UNDEFINED;
    ge::DataType kv_padding_size = ge::DT_UNDEFINED;
    ge::DataType key_antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType key_antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType value_antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType value_antiquant_offset = ge::DT_UNDEFINED;
    ge::DataType key_shared_prefix = ge::DT_UNDEFINED;
    ge::DataType value_shared_prefix = ge::DT_UNDEFINED;
    ge::DataType actual_shared_prefix_len = ge::DT_UNDEFINED;
    ge::DataType query_rope = ge::DT_UNDEFINED;
    ge::DataType key_rope = ge::DT_UNDEFINED;
    ge::DataType key_rope_antiquant_scale = ge::DT_UNDEFINED;
    ge::DataType dequant_scale_query = ge::DT_UNDEFINED;
    ge::DataType learnable_sink = ge::DT_UNDEFINED;
    ge::DataType q_start_idx = ge::DT_UNDEFINED;
    ge::DataType kv_start_idx = ge::DT_UNDEFINED;

    ge::DataType attention_out = ge::DT_UNDEFINED;
    ge::DataType softmax_lse = ge::DT_UNDEFINED;

    FusedInferAttentionInferDTypeUtParam(const csv_map &csvMap) : FusedInferAttentionHostUtParamBase(csvMap) {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "query_dtype", this->query));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "key_dtype", this->key));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "value_dtype", this->value));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "pse_shift_dtype", this->pse_shift));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "atten_mask_dtype", this->atten_mask));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "actual_seq_lengths_dtype", this->actual_seq_lengths));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "actual_seq_lengths_kv_dtype", this->actual_seq_lengths_kv));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale1_dtype", this->dequant_scale1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "quant_scale1_dtype", this->quant_scale1));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale2_dtype", this->dequant_scale2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "quant_scale2_dtype", this->quant_scale2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "quant_offset2_dtype", this->quant_offset2));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_scale_dtype", this->antiquant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "antiquant_offset_dtype", this->antiquant_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "block_table_dtype", this->block_table));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "query_padding_size_dtype", this->query_padding_size));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "kv_padding_size_dtype", this->kv_padding_size));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "key_antiquant_scale_dtype", this->key_antiquant_scale));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "key_antiquant_offset_dtype", this->key_antiquant_offset));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "value_antiquant_scale_dtype", this->value_antiquant_scale));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "value_antiquant_offset_dtype", this->value_antiquant_offset));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "value_antiquant_offset_dtype", this->value_antiquant_offset));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "key_shared_prefix_dtype", this->key_shared_prefix));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "value_shared_prefix_dtype", this->value_shared_prefix));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "actual_shared_prefix_len_dtype", this->actual_shared_prefix_len));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "query_rope_dtype", this->query_rope));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "key_rope_dtype", this->key_rope));
        this->inputInstance.emplace_back(
            GetDataTypeGE(csvMap, "key_rope_antiquant_scale_dtype", this->key_rope_antiquant_scale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "dequant_scale_query_dtype", this->dequant_scale_query));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "learnable_sink_dtype", this->learnable_sink));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_start_idx_dtype", this->q_start_idx));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "kv_start_idx_dtype", this->kv_start_idx));

        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "attention_out_dtype", this->attention_out));
        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "softmax_lse_dtype", this->softmax_lse));
    }
};

} // namespace FusedInferAttentionScoreUT

#endif // FUSED_INFER_ATTENTION_SCORE_PARAM_H
