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
"""Real DUT loading, reproducible BF16 inputs and numerical acceptance checks."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DUT = ROOT / 'mindiesd/layers/index_modulation.py'
DEPENDENCY = DUT.with_name('triton_utils.py')
SELECTOR = ROOT / 'tests/tools/select_npu_device.py'
SEED = 20261025


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_device(device_id=None, *, require_idle=True):
    """Reuse the repository selector, requiring OK health and optionally an idle card."""
    spec = importlib.util.spec_from_file_location('shared_npu_selector', SELECTOR)
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    cards = selector._parse_npu_smi_info(selector._run_npu_smi(['info']))
    if device_id is None:
        healthy_cards = {npu_id: info for npu_id, info in cards.items() if info['health'] == 'OK'}
        if not healthy_cards:
            raise RuntimeError('No healthy NPU is available')
        device_id, reason, notice = selector.select_npu(healthy_cards)
    else:
        reason, notice = 'explicit physical device', None
    info = cards.get(device_id)
    if not info or info['health'] != 'OK' or (require_idle and (info['processes'] or info['aicore'])):
        raise RuntimeError(f'Physical NPU {device_id} must be healthy and idle; actual={info}')
    return device_id, dict(cards=cards, reason=reason, notice=notice, selector_sha256=sha(SELECTOR))


def _json_safe(value):
    """Copy evidence with nonfinite floats replaced by JSON null values."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    temporary.replace(path)


def initialize(device_id):
    if device_id < 0 or os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        raise ValueError('Use a nonnegative physical device ID and unset ASCEND_RT_VISIBLE_DEVICES')
    global torch
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(device_id)
    torch.set_num_threads(8)
    # The real relative dependency is imported after set_device. No DUT rewrites or stubs.
    package_name = 'indexed_gate_test_layers'
    package = types.ModuleType(package_name)
    package.__path__ = [str(DUT.parent)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + '.index_modulation', DUT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, torch.device('npu', device_id)


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.int16)
    return hashlib.sha256(value.numpy().tobytes(order='C')).hexdigest()


def hashes(tensors):
    return {name: tensor_hash(value) for name, value in tensors.items() if name != 'output'}


def state(tensors):
    return {
        name: dict(
            shape=list(t.shape),
            dtype=str(t.dtype),
            device=str(t.device),
            stride=list(t.stride()),
            contiguous=t.is_contiguous(),
            storage_offset=t.storage_offset(),
        )
        for name, t in tensors.items()
    }


def make_inputs(case, device):
    generator = torch.Generator(device='cpu').manual_seed(SEED)
    parameter_rng = random.Random(SEED)
    cpu, parameters = {}, {}
    for name, rows in (('x', case.rows), ('gate', case.gate_rows), ('other', case.rows)):
        shape = (rows, case.width)
        if case.distribution == 'uniform':
            raw = torch.rand(shape, generator=generator) * 10 - 5
            parameters[name] = dict(distribution='uniform', low=-5, high=5, seed=SEED)
        else:
            mean, std = parameter_rng.uniform(-5, 5), parameter_rng.uniform(0.1, 2)
            raw = torch.randn(shape, generator=generator) * std + mean
            parameters[name] = dict(distribution='normal', mean=mean, std=std, clamp=None, seed=SEED)
        cpu[name] = raw.to(torch.bfloat16)
    index_type = getattr(torch, case.index_dtype)
    index_seed = SEED + 1
    index_generator = torch.Generator(device='cpu').manual_seed(index_seed)
    if case.fixed_index is not None:
        cpu['indices'] = torch.full((case.rows,), case.fixed_index, dtype=index_type)
        index_rule = 'fixed'
    elif case.random_indices:
        cpu['indices'] = torch.randint(0, case.gate_rows, (case.rows,), generator=index_generator, dtype=index_type)
        index_rule = 'uniform random integer in [0,K)'
    else:
        cpu['indices'] = (torch.arange(case.rows) % case.gate_rows).to(index_type)
        index_rule = 'row_id modulo K'
    parameters['indices'] = dict(
        rule=index_rule,
        seed=index_seed if case.random_indices and case.fixed_index is None else None,
        fixed=case.fixed_index,
        valid_range=[0, case.gate_rows - 1],
        observed_values=cpu['indices'].unique().tolist(),
        value_counts=torch.bincount(cpu['indices'].long(), minlength=case.gate_rows).tolist(),
        first_values=cpu['indices'][:32].tolist(),
    )
    cpu['output'] = torch.full((case.rows, case.width), float('nan'), dtype=torch.bfloat16)
    backings, device_tensors = {}, {}
    if case.layout == 'row_padded':
        for name, value in list(cpu.items()):
            if name == 'indices':
                backing = torch.full((case.rows * 2 + 2,), -7, dtype=index_type)
                slices = slice(1, 1 + case.rows * 2, 2)
            else:
                backing = torch.full((value.shape[0] + 2, case.width + 7), 13, dtype=torch.bfloat16)
                slices = (slice(1, -1), slice(3, 3 + case.width))
            backing[slices].copy_(value)
            cpu[name] = backing[slices]
            backings[name] = backing.to(device)
            device_tensors[name] = backings[name][slices]
    else:
        device_tensors = {name: value.to(device) for name, value in cpu.items()}
    torch.npu.synchronize()
    metadata = dict(
        case=case.record(),
        seed=SEED,
        parameters=parameters,
        cpu_input_sha256=hashes(cpu),
        initial_output_sha256=tensor_hash(cpu['output']),
        input_state=state(device_tensors),
        source_sha256=sha(DUT),
        dependency_sha256=sha(DEPENDENCY),
    )
    return cpu, device_tensors, backings, metadata


def call(dut, tensors):
    return dut.indexed_gate(tensors['output'], tensors['x'], tensors['gate'], tensors['other'], tensors['indices'])


def contract(returned, tensors):
    return dict(
        same_object=returned is tensors['output'],
        shape=returned.shape == tensors['x'].shape,
        dtype=returned.dtype == torch.bfloat16,
        device=returned.device == tensors['x'].device,
        stride=returned.stride() == tensors['output'].stride(),
    )


def compare(actual, golden):
    a, g = actual.detach().cpu().flatten(), golden.detach().cpu().flatten()
    assert a.shape == g.shape and a.dtype == g.dtype == torch.bfloat16
    dot = aa = gg = square_error = 0.0
    matches = bit_differences = 0
    max_abs = -1.0
    position = 0
    finite = True
    tol = 2**-7
    for start in range(0, a.numel(), 262144):
        av, gv = a[start : start + 262144].double(), g[start : start + 262144].double()
        finite = finite and bool(torch.isfinite(av).all() and torch.isfinite(gv).all())
        delta = (av - gv).abs()
        local = int(delta.argmax())
        if float(delta[local]) > max_abs:
            max_abs, position = float(delta[local]), start + local
        matches += int((delta <= tol + tol * gv.abs()).sum())
        bit_differences += int(
            (a[start : start + 262144].view(torch.int16) != g[start : start + 262144].view(torch.int16)).sum()
        )
        dot += float(torch.dot(av, gv))
        aa += float(torch.dot(av, av))
        gg += float(torch.dot(gv, gv))
        square_error += float(torch.dot(delta, delta))
    cosine = dot / math.sqrt(aa * gg) if aa and gg else float(aa == gg)
    ratio = math.sqrt(aa / gg) if gg else (1.0 if not aa else None)
    golden_at_max = float(g[position])
    exponent = math.floor(math.log2(abs(golden_at_max))) if golden_at_max else -126
    ulp = 2 ** (max(exponent, -126) - 7)
    limit = max(0.1, 32 * ulp)
    matched_ratio = matches / a.numel()
    return dict(
        finite=finite,
        output_dtype='torch.bfloat16',
        tolerance_dtype='torch.bfloat16',
        tolerance_reason='FP32 arithmetic followed by BF16 output rounding',
        atol=tol,
        rtol=tol,
        required_match_ratio=0.99,
        fallback=0.1,
        max_abs=max_abs,
        max_abs_limit=limit,
        max_error_coordinate=[position // actual.shape[-1], position % actual.shape[-1]],
        golden_at_max=golden_at_max,
        actual_at_max=float(a[position]),
        ulp=ulp,
        matched_ratio=matched_ratio,
        mismatch_count=a.numel() - matches,
        bit_mismatch_count=bit_differences,
        mixed_pass=finite and matched_ratio >= 0.99 and max_abs <= limit,
        cosine=cosine,
        norm_ratio=ratio,
        rmse=math.sqrt(square_error / a.numel()),
        similarity_pass=finite and cosine >= 0.99 and ratio is not None and 0.9 <= ratio <= 1.1,
    )


def padding_ok(case, backings):
    if not backings:
        return True
    output = backings['output'].cpu()
    mask = torch.ones(output.shape, dtype=torch.bool)
    mask[1:-1, 3 : 3 + case.width] = False
    return bool((output[mask] == 13).all())


def accuracy(dut, device, case, large=False):
    cpu, tensors, backings, record = make_inputs(case, device)
    record.update(
        started_at=now(),
        status='RUNNING',
        golden='NPU FP32' if large else 'CPU FP32',
        mixed_gate='soft diagnostic' if large else 'hard',
    )
    before = hashes(tensors)
    backing_before = hashes(backings)
    print(json.dumps(dict(stage='before_dut', **record)), flush=True)
    returned = call(dut, tensors)
    torch.npu.synchronize()
    actual = tensors['output'].cpu()
    ref = tensors if large else cpu
    golden = (
        ref['x'].float() + torch.index_select(ref['gate'].float(), 0, ref['indices'].long()) * ref['other'].float()
    ).to(torch.bfloat16)
    if large:
        torch.npu.synchronize()
    record.update(
        metrics=compare(actual, golden),
        output_contract=contract(returned, tensors),
        input_sha256=before,
        inputs_unchanged=before == hashes(tensors),
        cpu_to_npu_match=before == record['cpu_input_sha256'],
        input_padding_unchanged=backing_before == hashes(backings),
        output_padding_unchanged=padding_ok(case, backings),
        output_sha256=tensor_hash(actual),
        golden_sha256=tensor_hash(golden),
        source_sha256_after=sha(DUT),
        finished_at=now(),
    )
    passed = (
        record['metrics']['similarity_pass']
        and (large or record['metrics']['mixed_pass'])
        and all(record['output_contract'].values())
        and record['inputs_unchanged']
        and record['cpu_to_npu_match']
        and record['input_padding_unchanged']
        and record['output_padding_unchanged']
    )
    record['status'] = 'PASS' if passed else 'FAIL'
    print(json.dumps(record), flush=True)
    return record


def npu_snapshot(path):
    executable = shutil.which('npu-smi')
    if not executable:
        raise RuntimeError('npu-smi is unavailable')
    result = subprocess.run([executable, 'info'], capture_output=True, text=True, check=True)
    Path(path).write_text(result.stdout + result.stderr, encoding='utf-8')


def source_manifest():
    paths = [DUT, DEPENDENCY, SELECTOR] + sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh'))
    paths += sorted((HERE / 'tools').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


if __name__ == '__main__':
    print(select_device(require_idle=False)[0])
