# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Code-level comparison helpers for the qbmm_swiglu_mxquant tests."""

import torch

MATCH_RATIO = 0.9995
MAX_CODE_DIFF = 1


def unpack_fp4(buf):
    """Expand packed E2M1 bytes into one code per element."""
    buf = buf.contiguous()
    lo = (buf & 0x0F).to(torch.int16)
    hi = ((buf >> 4) & 0x0F).to(torch.int16)
    return torch.stack((lo, hi), dim=-1).reshape(buf.shape[0], buf.shape[1] * 2)


def code_diff(cand, ref, out_fp4):
    """Per-code absolute difference; FP4 outputs are compared unpacked."""
    if out_fp4:
        return (unpack_fp4(cand) - unpack_fp4(ref)).abs()
    return (cand.to(torch.int16) - ref.to(torch.int16)).abs()


def assert_codes_close(name, diff):
    """Require an exact-match ratio >= MATCH_RATIO and a max code diff <= 1.

    The ratio alone would accept a uniform off-by-one result, and the max diff
    alone would accept a single large outlier, so both checks are required.
    """
    max_diff = int(diff.max()) if diff.numel() else 0
    if max_diff > MAX_CODE_DIFF:
        raise AssertionError(f"{name} max code diff {max_diff} > {MAX_CODE_DIFF}")
    exact = float((diff == 0).float().mean()) if diff.numel() else 1.0
    if exact < MATCH_RATIO:
        raise AssertionError(f"{name} exact match ratio {exact:.6f} < {MATCH_RATIO}")
