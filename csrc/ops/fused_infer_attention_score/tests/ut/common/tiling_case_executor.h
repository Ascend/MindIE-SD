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

#ifndef OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
#define OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H

#include "tiling_context_faker.h"

using namespace std;

struct TilingInfo {
    int64_t tilingKey = -1;
    std::vector<int64_t> workspaceSizes;
    std::unique_ptr<uint8_t[]> tilingData;
    size_t tilingDataSize = 0;
    size_t blockNum = 0;
};

void ExecuteTestCase(const gert::TilingContextPara &tilingContextPara, ge::graphStatus expectResult = ge::GRAPH_FAILED,
    uint64_t expectTilingKey = 0, const string &expectTilingData = "", const std::vector<size_t> &expectWorkspaces = {},
    uint64_t tilingDataReservedLen = 0, bool useHashTilingData = false);

bool ExecuteTiling(const gert::TilingContextPara &tilingContextPara, TilingInfo &tilingInfo);

#endif // OPS_TRANSFORMER_DEV_TESTS_UT_COMMON_TILING_CASE_EXECUTOR_H
