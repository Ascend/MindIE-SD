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

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "pytorch_npu_helper.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "ada_relu2_block_estimate.h"

namespace {
constexpr int64_t EXPECTED_BNSD_DIM = 4;
constexpr std::string_view ADA_RELU2_BLOCK_ESTIMATE_NAME = "aclnnAdaRelu2BlockEstimate";
} // namespace

std::tuple<at::Tensor, at::Tensor> ada_relu2_block_estimate_mindie_sd_impl_npu(const at::Tensor &query,
    const at::Tensor &key, c10::OptionalIntArrayRef actualSeqLengths, c10::OptionalIntArrayRef actualSeqLengthsKV,
    std::string inputLayout, int64_t stride, int64_t sparseSize, int64_t numHeads, int64_t numKeyValueHeads,
    double scaleValue, double threshold, const c10::optional<at::Tensor> &qScale,
    const c10::optional<at::Tensor> &kScale) {
    TORCH_CHECK(stride > 0, "stride must be positive, but got ", stride);
    TORCH_CHECK(sparseSize > 0, "sparse_size must be positive, but got ", sparseSize);
    TORCH_CHECK(numHeads > 0, "num_heads must be positive, but got ", numHeads);
    TORCH_CHECK(numKeyValueHeads > 0, "num_key_value_heads must be positive, but got ", numKeyValueHeads);
    TORCH_CHECK(numHeads % numKeyValueHeads == 0, "num_heads must be divisible by num_key_value_heads");
    if (inputLayout != "BNSD") {
        TORCH_WARN(
            "ada_relu2_block_estimate currently only supports BNSD; layout ", inputLayout, " is not supported yet");
        TORCH_CHECK(false, "ada_relu2_block_estimate only supports BNSD currently");
    }

    TORCH_CHECK(query.dim() == EXPECTED_BNSD_DIM && key.dim() == EXPECTED_BNSD_DIM,
        "BNSD query and key must both be 4D tensors");
    const int64_t batchSize = query.size(0);
    const int64_t queryHeads = query.size(1);
    const int64_t keyHeads = key.size(1);
    const int64_t querySeqLen = query.size(2);
    const int64_t keySeqLen = key.size(2);

    TORCH_CHECK(queryHeads == numHeads, "query head dimension must match num_heads");
    TORCH_CHECK(keyHeads == numKeyValueHeads, "key head dimension must match num_key_value_heads");
    const int64_t queryBlocks = (querySeqLen + sparseSize - 1) / sparseSize;
    const int64_t keyBlocks = (keySeqLen + sparseSize - 1) / sparseSize;
    const int64_t scoredKeyBlocks = std::max(keyBlocks - 1, static_cast<int64_t>(0));
    const int64_t reduceSize = sparseSize / stride;
    const auto hasFullLength = [](const c10::OptionalIntArrayRef &lengths, int64_t physicalLength) {
        if (!lengths.has_value()) {
            return true;
        }
        return std::all_of(
            lengths->begin(), lengths->end(), [physicalLength](int64_t length) { return length == physicalLength; });
    };
    const bool hasFullQueryLength = hasFullLength(actualSeqLengths, querySeqLen);
    const bool hasFullKeyLength = hasFullLength(actualSeqLengthsKV, keySeqLen);
    const auto outputOptions = query.options().dtype(c10::ScalarType::Float);
    const std::vector<int64_t> scoreShape{batchSize, queryHeads, queryBlocks, scoredKeyBlocks, reduceSize};
    const std::vector<int64_t> rowSumShape{batchSize, queryHeads, queryBlocks, reduceSize};
    at::Tensor score;
    at::Tensor rowSum;
    if (hasFullQueryLength && hasFullKeyLength) {
        // The kernel covers every physical block, so zero initialization is unnecessary.
        score = at::empty(scoreShape, outputOptions);
        rowSum = at::empty(rowSumShape, outputOptions);
    } else {
        // Keep unwritten padding deterministic when actual sequence lengths truncate the physical tensors.
        score = at::zeros(scoreShape, outputOptions);
        rowSum = at::zeros(rowSumShape, outputOptions);
    }
    c10::optional<at::IntArrayRef> actualSeqLen =
        actualSeqLengths.has_value() ? c10::optional<at::IntArrayRef>(actualSeqLengths.value()) : c10::nullopt;
    c10::optional<at::IntArrayRef> actualSeqLenKV =
        actualSeqLengthsKV.has_value() ? c10::optional<at::IntArrayRef>(actualSeqLengthsKV.value()) : c10::nullopt;
    c10::optional<at::Tensor> actualSeqLenGm = actualSeqLengths.has_value()
        ? c10::optional<at::Tensor>(at::tensor(actualSeqLengths.value(), query.options().dtype(at::kLong)))
        : c10::nullopt;
    c10::optional<at::Tensor> actualSeqLenKVGm = actualSeqLengthsKV.has_value()
        ? c10::optional<at::Tensor>(at::tensor(actualSeqLengthsKV.value(), key.options().dtype(at::kLong)))
        : c10::nullopt;
    const char *inputLayoutPtr = inputLayout.c_str();

    EXEC_NPU_CMD<ADA_RELU2_BLOCK_ESTIMATE_NAME>(query, key, actualSeqLen, actualSeqLenKV, actualSeqLenGm,
        actualSeqLenKVGm, qScale, kScale, inputLayoutPtr, stride, sparseSize, numHeads, numKeyValueHeads, scaleValue,
        threshold, score, rowSum);
    return std::make_tuple(score, rowSum);
}
