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
"""Large BF16 scenarios: NPU FP32 golden, then fixed-input bitwise stress."""

import argparse
import json
from pathlib import Path
import traceback

import indexed_gate_cases as cases
import indexed_gate_common as common


def stress(dut, device, case, repeats, accepted, output):
    _, tensors, backings, record = common.make_inputs(case, device)
    torch = common.torch
    before = common.hashes(tensors)
    backing_before = common.hashes(backings)
    record.update(
        started_at=common.now(),
        status='RUNNING',
        baseline_calls=1,
        requested_repeats=repeats,
        completed_repeats=0,
        input_sha256=before,
    )
    if accepted is not None:
        assert accepted['status'] == 'PASS' and accepted['input_sha256'] == before
        assert accepted['source_sha256'] == record['source_sha256']
    common.call(dut, tensors)
    torch.npu.synchronize()
    baseline = tensors['output'].clone()
    record['baseline_sha256'] = common.tensor_hash(baseline)
    record['baseline_inputs_unchanged'] = before == common.hashes(tensors)
    assert record['baseline_inputs_unchanged'] and bool(torch.isfinite(baseline).all())
    if accepted is not None:
        assert record['baseline_sha256'] == accepted['output_sha256']
    for iteration in range(1, repeats + 1):
        returned = common.call(dut, tensors)
        torch.npu.synchronize()
        if not all(common.contract(returned, tensors).values()) or not torch.equal(
            tensors['output'].view(torch.int16), baseline.view(torch.int16)
        ):
            record.update(
                status='FAIL',
                failed_iteration=iteration,
                failure_metrics=common.compare(tensors['output'], baseline),
                inputs_unchanged=before == common.hashes(tensors),
            )
            torch.save(
                dict(baseline=baseline.cpu(), current=tensors['output'].cpu()), output / (case.name + '_failure.pt')
            )
            common.save(output / (case.name + '.json'), record)
            return record
        record['completed_repeats'] = iteration
        if iteration % 100 == 0:
            common.save(output / (case.name + '.json'), record)
            print(f'{case.name}: {iteration}/{repeats}', flush=True)
    record.update(
        final_sha256=common.tensor_hash(tensors['output']),
        inputs_unchanged=before == common.hashes(tensors),
        cpu_to_npu_match=before == record['cpu_input_sha256'],
        input_padding_unchanged=backing_before == common.hashes(backings),
        source_sha256_after=common.sha(common.DUT),
        finished_at=common.now(),
    )
    record['status'] = 'PASS' if record['inputs_unchanged'] and record['cpu_to_npu_match'] else 'FAIL'
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id', type=int)
    parser.add_argument(
        '--output', type=Path, default=common.HERE / 'artifacts' / ('accuracy_' + common.now().replace(':', ''))
    )
    parser.add_argument('--mode', choices=['all', 'accuracy', 'stress'], default='all')
    parser.add_argument('--stress-runs', type=int, default=1000)
    parser.add_argument('--accuracy-evidence', type=Path)
    parser.add_argument('--scenario', choices=[case.name for case in cases.ALL_PRODUCTION], action='append')
    parser.add_argument('--validate-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.device_id is not None and args.device_id < 0) or args.stress_runs <= 0:
        parser.error('device-id must be nonnegative; stress-runs must be positive')
    if args.validate_only:
        return 0
    if args.device_id is None:
        parser.error('Pass --device-id or use the shell entry for automatic selection')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dut, device = common.initialize(args.device_id)
    accepted, evidence_hash = {}, None
    if args.accuracy_evidence:
        previous = json.loads(args.accuracy_evidence.read_text())
        assert previous['status'] == 'PASS' and previous['device_id'] == args.device_id
        accepted = {item['case']['name']: item for item in previous['accuracy']}
        evidence_hash = common.sha(args.accuracy_evidence)
    record = dict(
        started_at=common.now(),
        mode=args.mode,
        status='RUNNING',
        device_id=args.device_id,
        source_sha256=common.sha(common.DUT),
        dependency_sha256=common.sha(common.DEPENDENCY),
        accuracy_evidence_sha256=evidence_hash,
        accuracy=[],
        stress=[],
    )
    destination = output / 'results.json'
    common.npu_snapshot(output / 'npu-before.log')
    for phase in ('accuracy', 'stress'):
        if args.mode not in ('all', phase):
            continue
        # Complete every golden before any baseline in all mode.
        if phase == 'stress' and record['accuracy']:
            accepted = {item['case']['name']: item for item in record['accuracy']}
        selected_cases = (
            [c for c in cases.ALL_PRODUCTION if c.name in args.scenario] if args.scenario else cases.PRODUCTION
        )
        for case in selected_cases:
            try:
                item = (
                    common.accuracy(dut, device, case, large=True)
                    if phase == 'accuracy'
                    else stress(dut, device, case, args.stress_runs, accepted.get(case.name), output)
                )
            except Exception:
                item = dict(case=case.record(), status='ERROR', traceback=traceback.format_exc())
            record[phase].append(item)
            common.save(output / (case.name + '_' + phase + '.json'), item)
            common.save(destination, record)
        if any(item['status'] != 'PASS' for item in record[phase]):
            break
    record.update(source_sha256_after=common.sha(common.DUT), finished_at=common.now())
    record['status'] = (
        'PASS'
        if all(item['status'] == 'PASS' for phase in ('accuracy', 'stress') for item in record[phase])
        else 'FAIL'
    )
    assert record['source_sha256'] == record['source_sha256_after']
    common.save(destination, record)
    common.npu_snapshot(output / 'npu-after.log')
    if record['status'] == 'PASS':
        for phase in ('accuracy', 'stress'):
            for item in record[phase]:
                print(
                    f"{phase} {item['case']['name']} {item['case']['distribution']} PASS"
                    + (f' repeats={args.stress_runs}' if phase == 'stress' else ''),
                    flush=True,
                )
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
