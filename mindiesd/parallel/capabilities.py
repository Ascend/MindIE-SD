# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Device-free sparse consumer contracts shared by validation and execution."""

_BSA_PATHS = {
    ("rf_v2", "bf16"): ("cann", "rain_fusion"),
    ("rf_v2", "fp16"): ("cann", "rain_fusion"),
    ("rf_v3", "bf16"): ("eagle", "eagle"),
    ("rf_v3", "fp16"): ("eagle", "eagle"),
    ("rf_v3", "mix"): ("eagle", "eagle_quant"),
    ("rf_v3", "mxfp8"): ("cann", "mxfp8_native"),
}


def resolve_bsa_consumer(precision, backend, sparse_type="rf_v3"):
    """Resolve the backend and operator for one supported sparse path."""
    try:
        required, consumer = _BSA_PATHS[(sparse_type, precision)]
    except KeyError as error:
        raise ValueError(f"unsupported sparse BSA path {sparse_type!r} with precision={precision!r}") from error
    if backend is not None and backend != required:
        raise ValueError(f"{sparse_type} with precision={precision!r} requires bsa_backend={required!r}")
    return required, consumer
