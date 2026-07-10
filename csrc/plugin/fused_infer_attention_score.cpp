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

#include <array>
#include <map>
#include <string>
#include <string_view>
#include <torch/library.h>
#include <vector>

#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "fused_infer_attention_score.h"
#include "pytorch_npu_helper.h"

using namespace at;

constexpr std::string_view FUSED_INFER_ATTENTION_SCORE_NAME = "aclnnEagleFusedInferAttentionScoreV5";
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;
constexpr int64_t DIM_NUM_3 = 3;
constexpr int64_t DIM_NUM_4 = 4;
constexpr int64_t PA_BBH_DIMS = 3;
constexpr int64_t PA_BNBD_DIMS = 4;
constexpr int64_t PA_NZ_DIMS = 5;

struct FiaLayoutInfo {
    std::string queryLayout;
    std::string outputLayout;
    int64_t queryDim;
};

const std::map<std::string, FiaLayoutInfo> FIA_LAYOUT_MAP = {
    {"BSH", {"BSH", "BSH", DIM_NUM_3}},
    {"BSND", {"BSND", "BSND", DIM_NUM_4}},
    {"BNSD", {"BNSD", "BNSD", DIM_NUM_4}},
    {"TND", {"TND", "TND", DIM_NUM_3}},
    {"NTD", {"NTD", "NTD", DIM_NUM_3}},
    {"BNSD_BSND", {"BNSD", "BSND", DIM_NUM_4}},
    {"BSH_BNSD", {"BSH", "BNSD", DIM_NUM_3}},
    {"BSND_BNSD", {"BSND", "BNSD", DIM_NUM_4}},
    {"NTD_TND", {"NTD", "TND", DIM_NUM_3}},
    {"BSH_NBSD", {"BSH", "NBSD", DIM_NUM_3}},
    {"BSND_NBSD", {"BSND", "NBSD", DIM_NUM_4}},
    {"BNSD_NBSD", {"BNSD", "NBSD", DIM_NUM_4}},
    {"TND_NTD", {"TND", "NTD", DIM_NUM_3}},
    {"NSD", {"NSD", "NSD", DIM_NUM_3}},
};

std::pair<std::string, std::string> GetFiaLayouts(const at::Tensor &query, const std::string &inputLayout) {
    auto iter = FIA_LAYOUT_MAP.find(inputLayout);
    TORCH_CHECK(iter != FIA_LAYOUT_MAP.end(), "unsupported fused_infer_attention_score input_layout: ", inputLayout);
    TORCH_CHECK(query.dim() == iter->second.queryDim, "query dim does not match input_layout ", inputLayout);
    return {iter->second.queryLayout, iter->second.outputLayout};
}

std::tuple<int64_t, int64_t, int64_t, int64_t> GetQueryBnsd(
    const at::Tensor &query, const std::string &queryLayout, int64_t numHeads) {
    if (queryLayout == "BSH") {
        return {query.size(DIM_0), numHeads, query.size(DIM_1), query.size(DIM_2) / numHeads};
    }
    if (queryLayout == "BSND") {
        return {query.size(DIM_0), query.size(DIM_2), query.size(DIM_1), query.size(DIM_3)};
    }
    if (queryLayout == "BNSD") {
        return {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2), query.size(DIM_3)};
    }
    if (queryLayout == "NSD") {
        return {1, query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    }
    TORCH_CHECK(false, "layout is not supported as BNSD-like query layout: ", queryLayout);
}

std::tuple<int64_t, int64_t, int64_t> GetQueryTnd(const at::Tensor &query, const std::string &queryLayout) {
    if (queryLayout == "TND") {
        return {query.size(DIM_0), query.size(DIM_1), query.size(DIM_2)};
    }
    if (queryLayout == "NTD") {
        return {query.size(DIM_1), query.size(DIM_0), query.size(DIM_2)};
    }
    TORCH_CHECK(false, "layout is not supported as TND-like query layout: ", queryLayout);
}

int64_t GetValueD(const c10::optional<at::Tensor> &blockTable, const at::Tensor &query, const at::Tensor &value,
    const std::string &queryLayout, int64_t numKeyValueHeads) {
    if (blockTable.has_value() && blockTable.value().defined()) {
        if (value.dim() == PA_BBH_DIMS) {
            return value.size(DIM_2) / numKeyValueHeads;
        }
        if (value.dim() == PA_BNBD_DIMS) {
            return value.size(DIM_3);
        }
        if (value.dim() == PA_NZ_DIMS) {
            return value.size(DIM_2) * value.size(DIM_4);
        }
        TORCH_CHECK(false, "when page attention is enabled, value dim should be 3, 4, or 5, but got ", value.dim());
    }

    TORCH_CHECK(value.dim() == query.dim(), "when page attention is disabled, value dim should equal query dim.");
    if (queryLayout == "BSH") {
        return value.size(DIM_2) / numKeyValueHeads;
    }
    if (queryLayout == "BSND" || queryLayout == "BNSD") {
        return value.size(DIM_3);
    }
    if (queryLayout == "TND" || queryLayout == "NTD" || queryLayout == "NSD") {
        return value.size(DIM_2);
    }
    TORCH_CHECK(false, "unsupported query layout for value dim inference: ", queryLayout);
}

at::Tensor EmptyLikeFiaOutput(const at::Tensor &query, at::IntArrayRef sizes, at::ScalarType dtype) {
    const c10::string_view device = "npu";
    at::Device outputDevice = at::Device(std::string(device));
    auto outputOptions = query.options().device(outputDevice).dtype(dtype);
    return at_npu::native::empty_with_format(sizes, outputOptions, at_npu::native::get_npu_format(query));
}

at::Tensor InferAttentionOut(const at::Tensor &query, const std::string &queryLayout, const std::string &outputLayout,
    int64_t numHeads, int64_t valueD, at::ScalarType outputDtype) {
    if (outputLayout == "BSH") {
        auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
        int64_t outH = numHeads * valueD;
        outH = (outH == 0 || query.size(DIM_2) == 0) ? query.size(DIM_2) : outH;
        return EmptyLikeFiaOutput(query, {b, s, outH}, outputDtype);
    }
    if (outputLayout == "BSND") {
        auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {b, s, n, outD}, outputDtype);
    }
    if (outputLayout == "BNSD") {
        auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {b, n, s, outD}, outputDtype);
    }
    if (outputLayout == "NBSD") {
        auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {n, b, s, outD}, outputDtype);
    }
    if (outputLayout == "TND") {
        auto [t, n, d] = GetQueryTnd(query, queryLayout);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {t, n, outD}, outputDtype);
    }
    if (outputLayout == "NTD") {
        auto [t, n, d] = GetQueryTnd(query, queryLayout);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {n, t, outD}, outputDtype);
    }
    if (outputLayout == "NSD") {
        auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
        int64_t outD = (valueD == 0 || d == 0) ? d : valueD;
        return EmptyLikeFiaOutput(query, {n, s, outD}, outputDtype);
    }
    TORCH_CHECK(false, "unsupported fused_infer_attention_score output layout: ", outputLayout);
}

at::Tensor InferLseOut(const at::Tensor &query, const std::string &inputLayout, const std::string &queryLayout,
    int64_t numHeads, bool returnSoftmaxLse) {
    if (!returnSoftmaxLse) {
        return EmptyLikeFiaOutput(query, {0}, at::kFloat);
    }
    if (inputLayout == "TND" || inputLayout == "NTD" || inputLayout == "TND_NTD" || inputLayout == "NTD_TND") {
        auto [t, n, d] = GetQueryTnd(query, queryLayout);
        return EmptyLikeFiaOutput(query, {t, n, 1}, at::kFloat);
    }
    auto [b, n, s, d] = GetQueryBnsd(query, queryLayout, numHeads);
    return EmptyLikeFiaOutput(query, {b, n, s, 1}, at::kFloat);
}

at::ScalarType InferOutputDtype(const at::Tensor &query, const c10::optional<at::Tensor> &queryRope,
    const c10::optional<at::Tensor> &quantScaleOut, const c10::optional<at::ScalarType> &outDtype) {
    if (outDtype.has_value()) {
        return outDtype.value();
    }
    if (quantScaleOut.has_value() && quantScaleOut.value().defined()) {
        return at::kChar;
    }
    if (query.scalar_type() == at::kChar) {
        if (queryRope.has_value() && queryRope.value().defined()) {
            return queryRope.value().scalar_type();
        }
        return at::kHalf;
    }
    return query.scalar_type();
}

std::tuple<at::Tensor, at::Tensor> fused_infer_attention_score_v2_impl_npu(const at::Tensor &query,
    const at::Tensor &key, const at::Tensor &value, const c10::optional<at::Tensor> &query_rope,
    const c10::optional<at::Tensor> &key_rope, const c10::optional<at::Tensor> &pse_shift,
    const c10::optional<at::Tensor> &atten_mask, c10::OptionalIntArrayRef actual_seq_qlen,
    c10::OptionalIntArrayRef actual_seq_kvlen, const c10::optional<at::Tensor> &block_table,
    const c10::optional<at::Tensor> &dequant_scale1, const c10::optional<at::Tensor> &quant_scale1,
    const c10::optional<at::Tensor> &dequant_scale2, const c10::optional<at::Tensor> &dequant_scale_query,
    const c10::optional<at::Tensor> &dequant_scale_key, const c10::optional<at::Tensor> &dequant_offset_key,
    const c10::optional<at::Tensor> &dequant_scale_value, const c10::optional<at::Tensor> &dequant_offset_value,
    const c10::optional<at::Tensor> &dequant_scale_key_rope, const c10::optional<at::Tensor> &quant_scale_out,
    const c10::optional<at::Tensor> &quant_offset_out, const c10::optional<at::Tensor> &learnable_sink,
    int64_t num_query_heads, int64_t num_key_value_heads, double softmax_scale, int64_t pre_tokens, int64_t next_tokens,
    std::string input_layout, int64_t sparse_mode, int64_t block_size, int64_t query_quant_mode, int64_t key_quant_mode,
    int64_t value_quant_mode, int64_t inner_precise, bool return_softmax_lse, const c10::optional<int64_t> &query_dtype,
    const c10::optional<int64_t> &key_dtype, const c10::optional<int64_t> &value_dtype,
    const c10::optional<int64_t> &query_rope_dtype, const c10::optional<int64_t> &key_rope_dtype,
    const c10::optional<int64_t> &key_shared_prefix_dtype, const c10::optional<int64_t> &value_shared_prefix_dtype,
    const c10::optional<int64_t> &dequant_scale_query_dtype, const c10::optional<int64_t> &dequant_scale_key_dtype,
    const c10::optional<int64_t> &dequant_scale_value_dtype, const c10::optional<int64_t> &dequant_scale_key_rope_dtype,
    const c10::optional<at::ScalarType> &out_dtype) {
    TORCH_CHECK(num_query_heads > 0, "num_query_heads should be greater than 0, but got ", num_query_heads);
    num_key_value_heads = num_key_value_heads == 0 ? num_query_heads : num_key_value_heads;

    auto [queryLayout, outputLayout] = GetFiaLayouts(query, input_layout);
    int64_t valueD = GetValueD(block_table, query, value, queryLayout, num_key_value_heads);
    at::ScalarType outputDtype = InferOutputDtype(query, query_rope, quant_scale_out, out_dtype);
    at::Tensor attentionOut = InferAttentionOut(query, queryLayout, outputLayout, num_query_heads, valueD, outputDtype);
    at::Tensor softmaxLse = InferLseOut(query, input_layout, queryLayout, num_query_heads, return_softmax_lse);

    char *inputLayoutPtr = const_cast<char *>(input_layout.c_str());
    at::Tensor actualSharedPrefixLen;
    at::Tensor antiquantScale;
    at::Tensor antiquantOffset;
    at::Tensor queryPaddingSize;
    at::Tensor kvPaddingSize;
    at::Tensor keySharedPrefix;
    at::Tensor valueSharedPrefix;
    int64_t antiquantMode = 0;

    std::vector<at::Tensor> keyVector{key};
    std::vector<at::Tensor> valueVector{value};
    at::TensorList keyTensors(keyVector);
    at::TensorList valueTensors(valueVector);

    auto queryWrapper = MakeTensorWrapper(query, query_dtype);
    auto keyWrapper = MakeTensorListWrapper(keyTensors, key_dtype);
    auto valueWrapper = MakeTensorListWrapper(valueTensors, value_dtype);
    auto queryRopeWrapper = MakeOptionalTensorWrapper(query_rope, query_rope_dtype);
    auto keyRopeWrapper = MakeOptionalTensorWrapper(key_rope, key_rope_dtype);
    auto dequantScaleKeyWrapper = MakeOptionalTensorWrapper(dequant_scale_key, dequant_scale_key_dtype);
    auto dequantScaleValueWrapper = MakeOptionalTensorWrapper(dequant_scale_value, dequant_scale_value_dtype);
    auto dequantScaleKeyRopeWrapper = MakeOptionalTensorWrapper(dequant_scale_key_rope, dequant_scale_key_rope_dtype);
    auto dequantScaleQueryWrapper = MakeOptionalTensorWrapper(dequant_scale_query, dequant_scale_query_dtype);
    c10::optional<at::IntArrayRef> actualSeqQlen =
        actual_seq_qlen.has_value() ? c10::optional<at::IntArrayRef>(actual_seq_qlen.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> actualSeqKvlen =
        actual_seq_kvlen.has_value() ? c10::optional<at::IntArrayRef>(actual_seq_kvlen.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> qStartIdx = c10::nullopt;
    c10::optional<at::IntArrayRef> kvStartIdx = c10::nullopt;
    int64_t pseType = 0;

    EXEC_NPU_CMD<FUSED_INFER_ATTENTION_SCORE_NAME>(queryWrapper, keyWrapper, valueWrapper, pse_shift, atten_mask,
        actualSeqQlen, actualSeqKvlen, dequant_scale1, quant_scale1, dequant_scale2, quant_scale_out, quant_offset_out,
        antiquantScale, antiquantOffset, block_table, queryPaddingSize, kvPaddingSize, dequantScaleKeyWrapper,
        dequant_offset_key, dequantScaleValueWrapper, dequant_offset_value, keySharedPrefix, valueSharedPrefix,
        actualSharedPrefixLen, queryRopeWrapper, keyRopeWrapper, dequantScaleKeyRopeWrapper, dequantScaleQueryWrapper,
        learnable_sink, qStartIdx, kvStartIdx, num_query_heads, softmax_scale, pre_tokens, next_tokens, inputLayoutPtr,
        num_key_value_heads, sparse_mode, inner_precise, block_size, antiquantMode, return_softmax_lse,
        query_quant_mode, key_quant_mode, value_quant_mode, pseType, attentionOut, softmaxLse);

    (void)key_shared_prefix_dtype;
    (void)value_shared_prefix_dtype;
    return std::make_tuple(attentionOut, softmaxLse);
}
