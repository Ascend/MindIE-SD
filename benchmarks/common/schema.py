#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Op schema tables used by the report/compare tooling.

Single copy shared with the runtime op_defs layer so that adding or renaming an
op argument does not require touching two places. These must stay in sync with
`op_defs/*.py prepare_args`.
"""

# args that together identify one benchmark slot (a baseline key)
# fa/bsa carry batch_size/num_heads/head_dim/func so reports can show shape
# params and the tested Python kernel function; func is omitted when unset
# (default impl). gmm carries hidden_size/moe_inter/experts so varying them
# across runs cannot collide into one slot.
OP_SLOT_ARGS = {
    "fa": ("q_len", "kv_len", "batch_size", "num_heads", "head_dim", "dtype", "func"),
    "bsa": ("q_len", "kv_len", "batch_size", "num_heads", "head_dim", "sparsity", "dtype", "func"),
    "gmm": ("num_tokens", "hidden_size", "moe_inter", "experts", "top_k", "quant_algo"),
    "mm": ("M", "K", "N", "quant_algo"),
}

# op seq-scan axis used for line charts
OP_SEQ_AXIS = {
    "fa": "q_len",
    "bsa": "q_len",
    "gmm": "num_tokens",
    "mm": "M",
}

# op dtype/quant grouping key used to split sections/series
OP_SERIES_KEY = {
    "fa": "dtype",
    "bsa": "dtype",
    "gmm": "quant_algo",
    "mm": "quant_algo",
}

# metrics displayed in the HTML report per op. FA/BSA/MM are compute-bound and
# show MFU only; GMM additionally shows MBU (its grouped-GEMM read pattern is
# bandwidth-relevant).
OP_DISPLAY_METRICS = {
    "fa": ("MFU",),
    "bsa": ("MFU",),
    "gmm": ("MFU", "MBU"),
    "mm": ("MFU",),
}

# slot-level metrics carried by baselines and report snapshots
BASELINE_METRICS = ("MFU", "MBU", "latency(us)", "executed_path")

# metrics compared by the drift gate
COMPARE_METRICS = ("MFU", "MBU")

# (omit_key, default_key) pairs: omit a slot arg when it equals another arg's
# value. Keeps slot keys stable across workloads that do or do not set kv_len
# explicitly when it equals q_len (the op-level default).
SLOT_OMIT_WHEN_DEFAULT = {
    "fa": [("kv_len", "q_len")],
    "bsa": [("kv_len", "q_len")],
}
