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
"""Small-shape NPU operator tests with a CPU golden and interface checks."""

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import types

import pytest

from tests.layers.index_modulation._indexed_gate_kernel import indexed_gate_cases as cases
from tests.layers.index_modulation._indexed_gate_kernel import indexed_gate_common as common
from tests.ops.device_support import current_device_family, init_test_device


def capture_npu_snapshot(path):
    """Keep optional device diagnostics from masking operator test results."""
    try:
        common.npu_snapshot(path)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        diagnostic = f'NPU snapshot unavailable: {type(error).__name__}: {error}\n'
        print(diagnostic, file=sys.stderr, end='')
        try:
            path.write_text(diagnostic, encoding='utf-8')
        except OSError as write_error:
            print(f'Cannot save NPU snapshot diagnostic: {write_error}', file=sys.stderr)


@pytest.fixture(scope='module', name='context')
def npu_context(tmp_path_factory):
    import torch
    import torch_npu  # noqa: F401

    # Some older runtimes accept only lists even though the documented mask allows ranges.
    # Validate before set_device(), which can otherwise fail before secondary selection.
    visibility = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    if visibility is not None:
        try:
            _visible_runtime_ids(visibility, torch.npu.device_count())
        except RuntimeError as error:
            pytest.skip(f'indexed_gate requires a supported runtime visibility configuration: {error}')
    requested_device = os.environ.get('INDEXED_GATE_DEVICE')
    # Reuse the suite's visible logical device, even if earlier tests opened it.
    if requested_device is None:
        device = init_test_device()
    else:
        device_id = int(requested_device)
        suite_device = os.environ.get('MINDIE_TEST_DEVICE_ID')
        if suite_device is not None and device_id != int(suite_device):
            raise ValueError('INDEXED_GATE_DEVICE conflicts with MINDIE_TEST_DEVICE_ID')
        if device_id < 0 or not torch.npu.is_available() or device_id >= torch.npu.device_count():
            raise RuntimeError(f'NPU test resource unavailable: requested logical device {device_id}')
        torch.npu.set_device(device_id)
        device = torch.device('npu', device_id)
    # The kernel supports the 910 series (A2/A3) and 950 series (A5).
    family = current_device_family()
    if family not in {'A2', 'A3', 'A5'}:
        pytest.skip(f'indexed_gate does not support {family}')
    common.torch = torch
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(8)
        selection = dict(
            reason='unified suite logical device',
            visibility=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
        )
        requested_output = os.environ.get('INDEXED_GATE_OUTPUT')
        if requested_output is None:
            output = tmp_path_factory.mktemp('indexed_gate_small')
        else:
            output = Path(requested_output)
            output.mkdir(parents=True, exist_ok=False)
        common.save(output / 'device-selection.json', dict(device=device.index, **selection))
        # Load the real kernel after device setup, without importing unrelated native ops.
        package_name = 'indexed_gate_test_layers'
        package = types.ModuleType(package_name)
        package.__path__ = [str(common.DUT.parent)]
        sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(package_name + '.index_modulation', common.DUT)
        dut = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = dut
        spec.loader.exec_module(dut)
        capture_npu_snapshot(output / 'npu-before.log')
        print(f'Indexed gate small-shape evidence: {output}')
        try:
            yield dut, device, output
        finally:
            capture_npu_snapshot(output / 'npu-after.log')
    finally:
        torch.set_num_threads(previous_threads)


def _visible_runtime_ids(visibility, count):
    """Expand the runtime mask in order, including inclusive ranges such as 0-2."""
    if visibility is None:
        return list(range(count))
    runtime_ids = []
    for part in visibility.split(','):
        match = re.fullmatch(r'\s*(\d+)\s*(?:-\s*(\d+)\s*)?', part)
        if match is None:
            raise RuntimeError(f'Invalid ASCEND_RT_VISIBLE_DEVICES: {visibility!r}')
        first = int(match[1])
        last = int(match[2]) if match[2] is not None else first
        if last < first or last - first + 1 > count - len(runtime_ids):
            raise RuntimeError(f'Visible NPU range/order does not match device count {count}: {visibility!r}')
        runtime_ids.extend(range(first, last + 1))
    if len(runtime_ids) != count or len(set(runtime_ids)) != count:
        raise RuntimeError(f'Visible NPU IDs must be unique and match device count {count}: {visibility!r}')
    return runtime_ids


def _secondary_physical_ids(visibility, count):
    """Map visible ordinals through the driver Chip Logic ID to npu-smi NPU ID."""
    runtime_ids = _visible_runtime_ids(visibility, count)
    executable = shutil.which('npu-smi')
    if executable is None:
        raise RuntimeError('Cannot map secondary NPUs: npu-smi is unavailable')
    try:
        result = subprocess.run([executable, 'info', '-m'], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f'Cannot query secondary NPU mapping: {error}') from error
    if re.search(r'NPU\s+ID\s+Chip\s+ID\s+Chip\s+Logic\s+ID', result.stdout.replace('|', ' ')) is None:
        raise RuntimeError('Cannot identify npu-smi device mapping columns')
    mapping = {}
    for line in result.stdout.splitlines():
        match = re.match(r'^\s*(\d+)\s+\d+\s+(\d+)\s+\S', line.replace('|', ' '))
        if match is None:  # Ignore headers, borders, and MCU rows with no logic ID.
            continue
        physical_id, runtime_id = map(int, match.groups())
        if runtime_id in mapping:
            raise RuntimeError(f'Ambiguous npu-smi Chip Logic ID: {runtime_id}')
        mapping[runtime_id] = physical_id
    if any(runtime_id not in mapping for runtime_id in runtime_ids):
        raise RuntimeError(f'Visible NPU IDs are absent from npu-smi mapping: {runtime_ids}')
    physical_ids = [mapping[runtime_id] for runtime_id in runtime_ids]
    if len(set(physical_ids)) != count:
        raise RuntimeError('Secondary selection requires distinct physical NPUs')
    return physical_ids


@pytest.fixture(scope='module')
def secondary_device(context):
    """Use another idle NPU, or explicitly skip only the cross-device cases."""
    _, primary, directory = context
    requested_device = os.environ.get('INDEXED_GATE_SECONDARY_DEVICE')
    count = common.torch.npu.device_count()
    if requested_device is not None:
        device_id = int(requested_device)
        if device_id < 0 or device_id >= count or device_id == primary.index:
            pytest.fail('The secondary NPU must be a different visible logical device')
    elif count < 2:
        pytest.skip('Cross-device checks require a second visible NPU')

    # Mask entries are driver logical IDs, not necessarily npu-smi physical IDs.
    visibility = os.environ.get('ASCEND_RT_VISIBLE_DEVICES')
    try:
        physical_ids = _secondary_physical_ids(visibility, count)
    except RuntimeError as error:
        if requested_device is not None:
            pytest.fail(str(error))
        pytest.skip(f'Cross-device checks require a reliable NPU mapping: {error}')
    if requested_device is not None:
        physical_id, selection = common.select_device(physical_ids[device_id])
    else:
        for device_id, physical_id in enumerate(physical_ids):
            if device_id == primary.index:
                continue
            try:
                _, selection = common.select_device(physical_id)
            except RuntimeError:
                continue
            break
        else:
            pytest.skip('Cross-device checks require a second healthy idle NPU')
    common.save(
        directory / 'secondary-selection.json', dict(device=device_id, physical_device=physical_id, **selection)
    )
    return device_id


def run_accuracy(context, case, group, repeat=False):
    dut, device, directory = context
    record = dict(case=case.record(), group=group, status='ERROR', source_sha256=common.sha(common.DUT))
    try:
        record.update(common.accuracy(dut, device, case))
        if repeat and record['status'] == 'PASS':
            _, tensors, backings, _ = common.make_inputs(case, device)
            before = common.hashes(tensors)
            backing_before = common.hashes(backings)
            record.update(repeat_runs=[], repeat_inputs_match=before == record['input_sha256'])
            assert record['repeat_inputs_match'], record
            for iteration in range(2):
                tensors['output'].fill_(float('nan'))
                common.call(dut, tensors)
                common.torch.npu.synchronize()
                output_hash = common.tensor_hash(tensors['output'])
                result = dict(
                    iteration=iteration + 1,
                    output_sha256=output_hash,
                    inputs_unchanged=before == common.hashes(tensors),
                    input_padding_unchanged=backing_before == common.hashes(backings),
                    output_padding_unchanged=common.padding_ok(case, backings),
                    bitwise_equal=output_hash == record['output_sha256'],
                )
                record['repeat_runs'].append(result)
                record.update(
                    repeat_output_sha256=[run['output_sha256'] for run in record['repeat_runs']],
                    repeat_inputs_unchanged=all(run['inputs_unchanged'] for run in record['repeat_runs']),
                    repeat_bitwise_equal=all(run['bitwise_equal'] for run in record['repeat_runs']),
                )
                assert all(
                    result[key]
                    for key in (
                        'inputs_unchanged',
                        'input_padding_unchanged',
                        'output_padding_unchanged',
                        'bitwise_equal',
                    )
                ), record
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
    + [
        pytest.param('cross_npu_' + '_'.join(case), ('cross_npu', *case), id='cross_npu_' + '_'.join(case))
        for case in CROSS_NPU_CASES
    ]
)


@pytest.mark.parametrize('name,alias_case', INVALID_CASES)
def test_invalid_arguments(context, name, alias_case, request):
    if alias_case is not None:
        if alias_case[0] == 'cross_npu':
            run_cross_npu(context, request.getfixturevalue('secondary_device'), *alias_case[1:])
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


def run_cross_npu(context, secondary, name, index_dtype, distribution):
    """Reject one tensor on a second healthy, idle physical NPU."""
    dut, device, directory = context
    torch = common.torch
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
