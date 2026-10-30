# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Representative Eagle matrix: every boundary plus crossed dtype/distribution axes."""

import torch

# Preserve all original shape/index/stride boundary classes verbatim.
BOUNDARY_CASES = (
    (0, 129, torch.int32, False),
    (1, 1, torch.int64, False),
    (3, 63, torch.int32, False),
    (3, 64, torch.int64, False),
    (81, 65, torch.int64, False),
    (3, 127, torch.int32, False),
    (3, 128, torch.int64, False),
    (81, 129, torch.int32, False),
    (3, 255, torch.int64, True),
    (3, 3968, torch.int32, False),
    (3, 3969, torch.int64, False),
    (3, 4095, torch.int32, False),
    (3, 4096, torch.int64, False),
    (81, 4097, torch.int32, False),
    (3, 5376, torch.int64, False),
    (3, 6656, torch.int32, True),
    (3, 6144, torch.int32, False),
    (3, 6145, torch.int64, False),
    (3, 6655, torch.int32, True),
    (81, 6656, torch.int64, False),
    (3, 6657, torch.int32, False),
    (3, 7167, torch.int32, True),
    (81, 7168, torch.int64, False),
)
DTYPE_PAIRS = (
    (torch.float16, torch.float16),
    (torch.float16, torch.bfloat16),
    (torch.bfloat16, torch.float16),
    (torch.bfloat16, torch.bfloat16),
)
# Cycle through all eight (input, output, distribution) triples over the shape
# boundaries; the first 16 rows repeat every triple on a different boundary.
ACCURACY_CASES = [
    ("uniform" if (i // 4) % 2 == 0 else "normal", *DTYPE_PAIRS[i % 4], *boundary)
    for i, boundary in enumerate(BOUNDARY_CASES)
]
# Empty outputs exercise every input/output dtype pairing. Scalar RMS and the
# maximum supported width additionally exercise both input dtypes.
ACCURACY_CASES += [
    ("normal", torch.float16, torch.bfloat16, *BOUNDARY_CASES[0]),
    ("uniform", torch.bfloat16, torch.float16, *BOUNDARY_CASES[0]),
    ("normal", torch.bfloat16, torch.bfloat16, *BOUNDARY_CASES[0]),
    ("normal", torch.bfloat16, torch.float16, *BOUNDARY_CASES[1]),
    ("normal", torch.float16, torch.bfloat16, *BOUNDARY_CASES[-1]),
]
DEFAULT_OUTPUT_CASES = (
    ("uniform", torch.float16, 1e-6),
    ("normal", torch.bfloat16, 1e-6),
    ("normal", torch.float16, 1e-3),
    ("uniform", torch.bfloat16, 1e-3),
)
DEFAULT_EPS_CASES = (
    (torch.float16, "uniform", False),
    (torch.float16, "normal", True),
    (torch.bfloat16, "uniform", True),
    (torch.bfloat16, "normal", False),
)
