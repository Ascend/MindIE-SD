#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Large-shape accuracy. Golden is NPU matmul + FP32 softmax + block sum."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_a5_correctness import main  # noqa: E402


if __name__ == "__main__":
    if "--preset" not in sys.argv and not any(arg.startswith("--preset=") for arg in sys.argv):
        sys.argv[1:1] = ["--preset", "target"]
    main()
