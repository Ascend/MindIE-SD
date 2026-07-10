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

#ifdef FIA_ARCH35_STANDALONE_UT
#include <cstdlib>
#include <exception>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <gtest/gtest.h>
#endif
#include "../fused_infer_attention_score_param.h"
#include "../../../../op_host/fused_infer_attention_score_tiling_compile_info.h"
#include "tiling_case_executor.h"

namespace FusedInferAttentionScoreUT {

#ifndef FIA_ARCH35_STANDALONE_UT
class FusedInferAttentionScoreArch35TilingTest : public testing::TestWithParam<FusedInferAttentionTilingUtParam> {
  protected:
    static void SetUpTestCase() { std::cout << "FusedInferAttentionScore Arch35 TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FusedInferAttentionScore Arch35 TilingTest TearDown" << std::endl; }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FusedInferAttentionScoreArch35TilingTest);
#endif

static void RunFusedInferAttentionScoreArch35Case(const FusedInferAttentionTilingUtParam &param) {
    optiling::FusedInferAttentionScoreCompileInfo compileInfo = {};

    const std::string A5SocInfo = "{\n"
                                  "  \"hardware_info\": {\n"
                                  "    \"BT_SIZE\": 0,\n"
                                  "    \"load3d_constraints\": \"1\",\n"
                                  "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                  "    \"Intrinsic_data_move_l12ub\": true,\n"
                                  "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                  "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                  "    \"UB_SIZE\": 196608,\n"
                                  "    \"L2_SIZE\": 117440512,\n"
                                  "    \"L1_SIZE\": 524288,\n"
                                  "    \"L0A_SIZE\": 65536,\n"
                                  "    \"L0B_SIZE\": 65536,\n"
                                  "    \"L0C_SIZE\": 65536,\n"
                                  "    \"vector_core_cnt\": 64,\n"
                                  "    \"cube_core_cnt\": 32,\n"
                                  "    \"socVersion\": \"Ascend950\"\n"
                                  "  }\n"
                                  "}";

    gert::TilingContextPara tilingContextPara("FusedInferAttentionScore",
        {param.query, param.key, param.value, param.pse_shift, param.atten_mask, param.actual_seq_lengths,
            param.actual_seq_lengths_kv, param.dequant_scale1, param.quant_scale1, param.dequant_scale2,
            param.quant_scale2, param.quant_offset2, param.antiquant_scale, param.antiquant_offset, param.block_table,
            param.query_padding_size, param.kv_padding_size, param.key_antiquant_scale, param.key_antiquant_offset,
            param.value_antiquant_scale, param.value_antiquant_offset, param.key_shared_prefix,
            param.value_shared_prefix, param.actual_shared_prefix_len, param.query_rope, param.key_rope,
            param.key_rope_antiquant_scale, param.dequant_scale_query, param.learnable_sink, param.q_start_idx,
            param.kv_start_idx},
        {param.attention_out, param.softmax_lse},
        {
            {"num_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_heads)},
            {"scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.scale)},
            {"pre_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pre_tokens)},
            {"next_tokens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.next_tokens)},
            {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.input_layout)},
            {"num_key_value_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.num_key_value_heads)},
            {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.sparse_mode)},
            {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.inner_precise)},
            {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
            {"antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.antiquant_mode)},
            {"softmax_lse_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(param.softmax_lse_flag)},
            {"key_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.key_antiquant_mode)},
            {"value_antiquant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.value_antiquant_mode)},
            {"query_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.query_quant_mode)},
            {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.pse_type)},
            {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.out_dtype)},
        },
        &compileInfo, "Ascend950", 64, 196608, 16384, A5SocInfo);

    ExecuteTestCase(
        tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0, true);
}

#ifndef FIA_ARCH35_STANDALONE_UT
TEST_P(FusedInferAttentionScoreArch35TilingTest, param) { RunFusedInferAttentionScoreArch35Case(GetParam()); }

INSTANTIATE_TEST_SUITE_P(FusedInferAttentionScore, FusedInferAttentionScoreArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<FusedInferAttentionTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<FusedInferAttentionTilingUtParam>);
#else
static bool RunStandaloneCaseInSubprocess(const FusedInferAttentionTilingUtParam &param) {
    pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "[FAILED] " << param.case_name << ": fork failed" << std::endl;
        return false;
    }
    if (pid == 0) {
        try {
            if (std::getenv("FIA_UT_TRACE") != nullptr) {
                std::cerr << "[TRACE] run case: " << param.case_name << std::endl;
            }
            RunFusedInferAttentionScoreArch35Case(param);
            std::exit(0);
        } catch (const std::exception &ex) {
            std::cerr << "[FAILED] " << param.case_name << ": " << ex.what() << std::endl;
            std::exit(1);
        } catch (...) {
            std::cerr << "[FAILED] " << param.case_name << ": unknown exception" << std::endl;
            std::exit(1);
        }
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::cerr << "[FAILED] " << param.case_name << ": waitpid failed" << std::endl;
        return false;
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        std::cout << "[PASSED] " << param.case_name << std::endl;
        return true;
    }
    if (WIFSIGNALED(status)) {
        std::cerr << "[FAILED] " << param.case_name << ": signal " << WTERMSIG(status) << std::endl;
    } else {
        std::cerr << "[FAILED] " << param.case_name << ": exit " << WEXITSTATUS(status) << std::endl;
    }
    return false;
}

int RunStandalone(int argc, char **argv) {
    const std::string csvPath = argc > 1 ? argv[1] : ReplaceFileExtension2Csv(__FILE__);
    auto cases = GetCasesFromCsv<FusedInferAttentionTilingUtParam>(csvPath);
    if (cases.empty()) {
        std::cerr << "[FAILED] no cases loaded from " << csvPath << std::endl;
        return 1;
    }

    size_t failed = 0;
    for (const auto &param : cases) {
        if (!RunStandaloneCaseInSubprocess(param)) {
            ++failed;
        }
    }

    std::cout << "cases=" << cases.size() << ", failed=" << failed << std::endl;
    return failed == 0 ? 0 : 1;
}
#endif

} // namespace FusedInferAttentionScoreUT

#ifdef FIA_ARCH35_STANDALONE_UT
int main(int argc, char **argv) { return FusedInferAttentionScoreUT::RunStandalone(argc, argv); }
#endif
