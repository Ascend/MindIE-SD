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
"""Small-shape CPU golden checks and public interface rejection tests."""

import os
from pathlib import Path
import traceback

import pytest

import indexed_gate_cases as cases
import indexed_gate_common as common


@pytest.fixture(scope='module', name='context')
def npu_context():
    device_id = int(os.environ['INDEXED_GATE_DEVICE'])
    output = Path(os.environ['INDEXED_GATE_OUTPUT'])
    output.mkdir(parents=True, exist_ok=False)
    dut, device = common.initialize(device_id)
    common.npu_snapshot(output / 'npu-before.log')
    yield dut, device, output
    common.npu_snapshot(output / 'npu-after.log')


def run_accuracy(context, case, group, repeat=False):
    dut, device, directory = context
    record = dict(case=case.record(), group=group, status='ERROR', source_sha256=common.sha(common.DUT))
    try:
        record.update(common.accuracy(dut, device, case))
        if repeat and record['status'] == 'PASS':
            _, tensors, _, _ = common.make_inputs(case, device)
            before = common.hashes(tensors)
            repeated = []
            for _ in range(2):
                tensors['output'].fill_(float('nan'))
                common.call(dut, tensors)
                common.torch.npu.synchronize()
                repeated.append(common.tensor_hash(tensors['output']))
            record.update(
                repeat_output_sha256=repeated,
                repeat_inputs_match=before == record['input_sha256'],
                repeat_inputs_unchanged=before == common.hashes(tensors),
                repeat_bitwise_equal=all(h == record['output_sha256'] for h in repeated),
            )
            assert (
                record['repeat_bitwise_equal'] and record['repeat_inputs_match'] and record['repeat_inputs_unchanged']
            )
        assert record['status'] == 'PASS', record
        return record
    except Exception:
        record.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        record['source_sha256_after'] = common.sha(common.DUT)
        common.save(directory / (case.name + '.json'), record)


@pytest.mark.parametrize('case', cases.small_cases(), ids=lambda c: c.name)
def test_accuracy(context, case):
    run_accuracy(context, case, 'original' if case.name.startswith('small_') else 'k_boundary')


@pytest.mark.parametrize('number', range(20))
@pytest.mark.parametrize('layout', ['contiguous', 'row_padded'])
@pytest.mark.parametrize('index_dtype', ['int32', 'int64'])
@pytest.mark.parametrize('distribution', ['uniform', 'normal'])
def test_tile_boundaries(context, number, layout, index_dtype, distribution):
    dut, device, directory = context
    vector, ub_size = dut._get_npu_resources(device.index)
    case = cases.tile_case(number, ub_size, layout, index_dtype, distribution)
    record = run_accuracy(context, case, 'tile_boundary', repeat=True)
    br, bc = dut._tile_shape(case.width, ub_size)
    columns = (case.width + bc - 1) // bc
    record.update(
        block_rows=br,
        block_cols=bc,
        ub_size=ub_size,
        tile_case=number,
        grid=min(vector, (case.rows + br - 1) // br * columns),
        column_tiles=columns,
    )
    try:
        assert bc % 32 == 0 and br * (16 * bc + 8) <= ub_size // 2
        if number == 13:
            assert (br, bc) == (2, case.width)
        elif number == 14:
            assert (br, bc) == (1, case.width + 31)
        elif number in (15, 16):
            assert br == 1 and columns == (1 if number == 15 else 2)
    except Exception:
        record.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        common.save(directory / (case.name + '.json'), record)


STORAGE_ALIAS_CASES = [
    (name, layout, index_dtype, distribution, 32)
    for name in ('x', 'gate', 'other', 'indices')
    for layout in ('same_start', 'partial_overlap', 'disjoint_views')
    for index_dtype in ('int32', 'int64')
    for distribution in ('uniform', 'normal')
] + [
    ('gate', 'same_start', dtype, distribution, 5376)
    for dtype in ('int32', 'int64')
    for distribution in ('uniform', 'normal')
]
CROSS_NPU_CASES = [
    (name, dtype, distribution)
    for name in ('output', 'x', 'gate', 'other', 'indices')
    for dtype in ('int32', 'int64')
    for distribution in ('uniform', 'normal')
]
INVALID_CASES = (
    [pytest.param(name, None, id=name) for name in cases.INVALID]
    + [
        pytest.param('storage_alias_' + '_'.join(map(str, case)), case, id='storage_alias_' + '_'.join(map(str, case)))
        for case in STORAGE_ALIAS_CASES
    ]
    + (
        [
            pytest.param('cross_npu_' + '_'.join(case), ('cross_npu', *case), id='cross_npu_' + '_'.join(case))
            for case in CROSS_NPU_CASES
        ]
        if os.environ.get('INDEXED_GATE_SECONDARY_DEVICE') is not None
        else []
    )
)


@pytest.mark.parametrize('name,alias_case', INVALID_CASES)
def test_invalid_arguments(context, name, alias_case):
    if alias_case is not None:
        if alias_case[0] == 'cross_npu':
            run_cross_npu(context, *alias_case[1:])
        else:
            run_output_storage_alias(context, *alias_case)
        return
    dut, device, directory = context
    torch = common.torch
    n, h = {'empty_rows': (0, 33), 'empty_width': (3, 0), 'empty_both': (0, 0)}.get(name, (3, 33))
    case = cases.Case(name, n, h, distribution='uniform')
    _, tensors, _, _ = common.make_inputs(case, device)
    if name == 'x_rank1':
        tensors['x'] = tensors['x'].flatten()
    elif name == 'x_rank3':
        tensors['x'] = tensors['x'].unsqueeze(0)
    elif name in ('output_shape', 'other_shape'):
        key = name.split('_')[0]
        tensors[key] = tensors[key][:-1]
    elif name == 'gate_rank1':
        tensors['gate'] = tensors['gate'][0]
    elif name == 'gate_empty':
        tensors['gate'] = tensors['gate'][:0]
    elif name == 'gate_width':
        tensors['gate'] = tensors['gate'][:, :-1]
    elif name == 'indices_rank2':
        tensors['indices'] = tensors['indices'][:, None]
    elif name == 'indices_length':
        tensors['indices'] = tensors['indices'][:-1]
    elif name == 'indices_dtype':
        tensors['indices'] = tensors['indices'].float()
    elif name == 'output_overlap':
        tensors['output'] = tensors['output'][:1].expand(3, 33)
    elif name == 'all_cpu':
        tensors = {key: value.cpu() for key, value in tensors.items()}
    elif not name.startswith('empty_'):
        key, kind = name.split('_', 1)
        if kind == 'last_stride':
            tensors[key] = torch.empty((tensors[key].shape[0], 66), dtype=torch.bfloat16, device=device)[:, ::2]
        elif kind == 'cpu':
            tensors[key] = tensors[key].cpu()
        else:
            tensors[key] = tensors[key].to(getattr(torch, kind))
    record = dict(
        case=case.record(),
        group='invalid',
        status='FAIL',
        source_sha256=common.sha(common.DUT),
        input_state=common.state(tensors),
        expected_error_type='ValueError',
    )
    try:
        check_rejection(dut, tensors, record)
    finally:
        record['source_sha256_after'] = common.sha(common.DUT)
        common.save(directory / ('invalid_' + name + '.json'), record)


def run_cross_npu(context, name, index_dtype, distribution):
    """Reject one tensor on a second explicitly selected, idle physical NPU."""
    dut, device, directory = context
    torch = common.torch
    secondary = int(os.environ['INDEXED_GATE_SECONDARY_DEVICE'])
    case_id = f'cross_npu_{name}_{index_dtype}_{distribution}'
    case = cases.Case(case_id, 3, 33, index_dtype=index_dtype, distribution=distribution)
    record = dict(
        case=case.record(),
        group='cross_npu',
        status='ERROR',
        moved_tensor=name,
        primary_device=device.index,
        secondary_device=secondary,
        expected_error='all inputs and output must be on the same device',
        source_sha256=common.sha(common.DUT),
    )
    try:
        assert secondary >= 0 and secondary != device.index
        _, tensors, _, generation = common.make_inputs(case, device)
        tensors[name] = tensors[name].to(torch.device('npu', secondary))
        torch.npu.synchronize(secondary)
        torch.npu.set_device(device)
        record.update(input_state=common.state(tensors), generation=generation)
        check_rejection(dut, tensors, record)
    except Exception:
        record.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        torch.npu.set_device(device)
        record['source_sha256_after'] = common.sha(common.DUT)
        common.save(directory / (case_id + '.json'), record)
        print(f"{case_id}: {record['status']}; {record.get('error', record.get('traceback', ''))}")


def check_rejection(dut, tensors, record, backing=None):
    """Call the real wrapper once and preserve evidence even when an assertion fails."""
    before = {key: common.tensor_hash(value) for key, value in tensors.items()}
    backing_before = common.tensor_hash(backing) if backing is not None else None
    record.update(tensor_sha256_before=before, backing_sha256_before=backing_before, calls=1)
    try:
        common.call(dut, tensors)
        common.torch.npu.synchronize()
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
    after = {key: common.tensor_hash(value) for key, value in tensors.items()}
    backing_after = common.tensor_hash(backing) if backing is not None else None
    record.update(
        tensor_sha256_after=after,
        backing_sha256_after=backing_after,
        inputs_unchanged=before == after,
        buffers_unchanged=before == after and backing_before == backing_after,
    )
    assert record.get('error_type') == 'ValueError', record
    if 'expected_error' in record:
        assert record['error'] == record['expected_error'], record
    assert record['buffers_unchanged'], record
    assert common.sha(common.DUT) == record['source_sha256']
    record['status'] = 'PASS'


def run_output_storage_alias(context, name, layout, index_dtype, distribution, width):
    """Reject output sharing any input storage, even disjoint logical views."""
    dut, device, directory = context
    torch = common.torch
    case_id = f'storage_alias_{name}_{layout}_{index_dtype}_{distribution}_{width}'
    case = cases.Case(case_id, 2, width, gate_rows=2, index_dtype=index_dtype, distribution=distribution)
    expected_error = f'output must not share storage with {name}'
    record = dict(
        case=case.record(),
        group='storage_alias',
        alias_input=name,
        layout=layout,
        expected_error=expected_error,
        status='ERROR',
        source_sha256=common.sha(common.DUT),
    )
    try:
        _, tensors, _, generation = common.make_inputs(case, device)
        # Four BF16 elements are eight bytes, preserving int32/int64 view alignment.
        elements = case.rows * width
        offset = {'same_start': 0, 'partial_overlap': 4, 'disjoint_views': elements}[layout]
        backing = torch.zeros(2 * elements + 16, dtype=torch.bfloat16, device=device)
        tensors['output'] = backing[:elements].view(case.rows, width)
        if name == 'indices':
            dtype = getattr(torch, index_dtype)
            start = offset * backing.element_size() // torch.empty((), dtype=dtype).element_size()
            alias = backing.view(dtype)[start : start + case.rows]
        else:
            alias = backing[offset : offset + elements].view(case.rows, width)
        alias.copy_(tensors[name])
        tensors[name] = alias
        # Exercise the original swapped-gate counterexample at the wider shape too.
        if name == 'gate' and width == 5376:
            tensors['x'].fill_(1)
            tensors['other'].fill_(1)
            alias[0].fill_(2)
            alias[1].fill_(3)
            tensors['indices'].copy_(torch.tensor([1, 0], dtype=getattr(torch, index_dtype), device='cpu'))
        output_start = tensors['output'].data_ptr()
        input_start = alias.data_ptr()
        output_end = output_start + tensors['output'].numel() * tensors['output'].element_size()
        input_end = input_start + alias.numel() * alias.element_size()
        shared = tensors['output'].untyped_storage().data_ptr() == alias.untyped_storage().data_ptr()
        intersects = max(output_start, input_start) < min(output_end, input_end)
        record.update(
            input_state=common.state(tensors),
            generation=generation,
            indices=tensors['indices'].cpu().tolist(),
            shared_storage=shared,
            logical_byte_ranges_intersect=intersects,
            output_byte_range=[output_start, output_end],
            input_byte_range=[input_start, input_end],
        )
        assert shared and intersects == (layout != 'disjoint_views')
        assert all(0 <= value < case.gate_rows for value in record['indices'])
        check_rejection(dut, tensors, record, backing)
    except Exception:
        record.update(status='ERROR', traceback=traceback.format_exc())
        raise
    finally:
        record['source_sha256_after'] = common.sha(common.DUT)
        common.save(directory / (case_id + '.json'), record)
        print(f"{case_id}: {record['status']}; {record.get('error', record.get('traceback', ''))}")
