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

#include "../../../../op_host/arch35/fused_infer_attention_score_tiling_impl.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace FusedInferAttentionScoreUT {
class KvHeadScheduleProbe : public optiling::FusedInferAttentionScoreTilingImpl {
  public:
    KvHeadScheduleProbe() : FusedInferAttentionScoreTilingImpl(nullptr) {}

    void Verify(uint32_t queryHeads, uint32_t kvHeads, uint32_t queryLength, int64_t kvLength, uint32_t cores,
        uint64_t l2Bytes, bool enabled) {
        optiling::FiaTilingInfo info;
        info.opName = "EagleFusedInferAttentionScore";
        info.fullQuantMode = optiling::FiaFullQuantMode::QKV_PER_BLOCK_FULL_QUANT;
        info.inputLayout = optiling::TilingKeyLayout::BNSD;
        info.bSize = 1;
        info.n1Size = queryHeads;
        info.n2Size = kvHeads;
        info.gSize = queryHeads / kvHeads;
        info.s1Size = queryLength;
        info.s2Size = kvLength;
        info.qkHeadDim = info.vHeadDim = 128;
        platformInfo_.aicNum = cores;
        platformInfo_.l2Size = l2Bytes;
        sOuterFactor_ = 64;
        ApplyKvHeadSchedule(info);
        if (tilingKeyInfo_.kvHeadSchedule != enabled) {
            throw std::runtime_error("Unexpected KV-head schedule selection");
        }
        if (!enabled) {
            return;
        }
        const uint64_t blocks = (queryLength + 127) / 128;
        const auto &data = faRunTilingAdapter_.multiCoreParamsRegbase;
        std::vector<uint8_t> visited(queryHeads * blocks, 0);
        for (uint32_t kv = 0; kv < kvHeads; ++kv) {
            for (uint32_t core = 0; core < cores; ++core) {
                const uint64_t begin = data.bnStartIdx[core] * blocks + data.sparseStartIdx[core];
                const uint64_t end = data.bnStartIdx[core + 1] * blocks + data.sparseStartIdx[core + 1];
                if (begin != info.gSize * blocks * core / cores || end != info.gSize * blocks * (core + 1) / cores) {
                    throw std::runtime_error("Incorrect group boundary");
                }
                for (uint64_t task = begin; task < end; ++task) {
                    const uint64_t index = kv * info.gSize * blocks + task;
                    if (index >= visited.size() || visited[index]++) {
                        throw std::runtime_error("Repeated or invalid Q task");
                    }
                }
            }
        }
        for (auto count : visited) {
            if (count != 1) {
                throw std::runtime_error("Missing Q task");
            }
        }
    }
};

void RunKvHeadScheduleCases() {
    constexpr uint64_t l2Bytes = 112ULL * 1024 * 1024;
    KvHeadScheduleProbe().Verify(40, 40, 75600, 75600, 32, l2Bytes, true);
    KvHeadScheduleProbe().Verify(40, 10, 75600, 75600, 32, l2Bytes, true);
    KvHeadScheduleProbe().Verify(32, 4, 2304, 30757, 32, l2Bytes, false);
    KvHeadScheduleProbe().Verify(40, 40, 128, 75600, 32, l2Bytes, false);
    KvHeadScheduleProbe().Verify(40, 40, 75600, 75600, 32, 0, false);
    KvHeadScheduleProbe().Verify(40, 40, 75600, 75600, 48, l2Bytes, false);
    KvHeadScheduleProbe().Verify(40, 40, 75600, 75600, 32, 32ULL * 1024 * 1024, false);
    std::cout << "KV-head schedule: 7 cases passed" << std::endl;
}

} // namespace FusedInferAttentionScoreUT

int main() {
    FusedInferAttentionScoreUT::RunKvHeadScheduleCases();
    return 0;
}
