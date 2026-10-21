# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.
"""Local FLOAT4 E2M1 tolerance tier; ULP is computed in the output dtype."""

BUDGET = {
    "atol": 0.5,
    "rtol": 0.5,
    "matched_ratio": 0.99,
    "fallback": 1.0,
    "ulp_factor": 32,
}
