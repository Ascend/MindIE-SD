#!/usr/bin/env python
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""Input contracts and scenario selection for the indexed gate kernel."""

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Case:
    name: str
    rows: int
    width: int
    gate_rows: int = 3
    distribution: str = 'normal'
    layout: str = 'contiguous'
    index_dtype: str = 'int64'
    fixed_index: int | None = None
    random_indices: bool = False

    def record(self):
        return asdict(self)


PRODUCTION = (
    Case('m1', 13640, 2048),
    Case('m2', 13640, 5376),
    Case('m3', 5672, 5376),
    Case('m4', 5672, 12000),
    Case('m2_uniform', 13640, 5376, distribution='uniform'),
    Case('m4_uniform', 5672, 12000, distribution='uniform'),
    Case('m3_k17', 5672, 5376, gate_rows=17, distribution='uniform'),
    Case('large_nh', 13640, 12000),
    Case('large_nh_uniform', 13640, 12000, distribution='uniform'),
    Case('large_nh_k17', 16384, 12000, gate_rows=17),
)
PERFORMANCE = PRODUCTION[:4] + (PRODUCTION[7], PRODUCTION[9])
EXTENDED = PRODUCTION[7:]
HIDDEN_20000 = (
    Case('large_h20000', 13640, 20000),
    Case('large_h20000_k17', 16384, 20000, gate_rows=17),
)
ALL_PRODUCTION = PRODUCTION + HIDDEN_20000
ALL_PERFORMANCE = PERFORMANCE + HIDDEN_20000
ORIGINAL = (
    Case('small_01', 1, 1, fixed_index=0),
    Case('small_02', 3, 31, index_dtype='int32'),
    Case('small_03', 3, 32),
    Case('small_04', 17, 33, index_dtype='int32'),
    Case('small_05', 97, 129),
    Case('small_06', 7, 128, layout='row_padded', index_dtype='int32'),
    Case('small_07', 1, 5376, fixed_index=0),
    Case('small_08', 1, 5376, fixed_index=2),
    Case('small_09', 3, 5376, index_dtype='int32'),
    Case('small_10', 17, 5376),
    Case('small_11', 7, 12000),
)
K_BOUNDARIES = (
    Case('k1_single', 1, 1, 1, fixed_index=0),
    Case('k1_repeat', 7, 129, 1, fixed_index=0),
    Case('k4_first', 1, 129, 4, fixed_index=0),
    Case('k4_last', 1, 5376, 4, fixed_index=3),
    Case('k4_random', 9, 33, 4),
    Case('k17_first', 1, 129, 17, fixed_index=0),
    Case('k17_interior', 1, 129, 17, fixed_index=3),
    Case('k17_last', 1, 5376, 17, fixed_index=16),
    Case('k17_random', 35, 12000, 17),
)
INVALID = (
    (
        'x_rank1',
        'x_rank3',
        'empty_rows',
        'empty_width',
        'empty_both',
        'output_shape',
        'other_shape',
        'gate_rank1',
        'gate_empty',
        'gate_width',
        'indices_rank2',
        'indices_length',
        'indices_dtype',
        'output_overlap',
        'all_cpu',
    )
    + tuple(
        f'{name}_{kind}' for name in ('output', 'x', 'gate', 'other') for kind in ('float16', 'float32', 'last_stride')
    )
    + tuple(f'{name}_cpu' for name in ('output', 'x', 'gate', 'other', 'indices'))
)


def small_cases():
    cases = [
        replace(c, name=c.name + '_' + d, distribution=d, random_indices=True)
        for c in ORIGINAL
        for d in ('uniform', 'normal')
    ]
    cases += [
        replace(
            c,
            name=f'{c.name}_{layout}_{index}_{d}',
            layout=layout,
            index_dtype=index,
            distribution=d,
            random_indices=True,
        )
        for c in K_BOUNDARIES
        for layout in ('contiguous', 'row_padded')
        for index in ('int32', 'int64')
        for d in ('uniform', 'normal')
    ]
    return cases


def tile_case(number, ub_size, layout, index_dtype, distribution):
    stage = ub_size // 2
    last_multi = (stage - 16) // 32 // 32 * 32
    max_columns = (stage - 8) // 16 // 32 * 32
    widths = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, last_multi, last_multi + 1, max_columns, max_columns + 1]
    shapes = [(1 if h == 1 else 5, h) for h in widths]
    row_tile = stage // (16 * 32 + 8)
    shapes += [(n, 8) for n in (row_tile - 1, row_tile, row_tile + 1)]
    n, h = shapes[number]
    return Case(
        f'tile_{number:02d}_{layout}_{index_dtype}_{distribution}',
        n,
        h,
        17,
        distribution,
        layout,
        index_dtype,
        random_indices=True,
    )


def production_case(name):
    return next(case for case in ALL_PRODUCTION if case.name == name)
