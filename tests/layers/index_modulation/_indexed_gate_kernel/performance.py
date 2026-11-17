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
"""Capture and validate complete msprof op batches without replacing failed rounds."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indexed_gate_cases as cases
import indexed_gate_common as common

TARGET = '_indexed_gate_kernel'


def require(value, message):
    if not value:
        raise ValueError(message)


def capture_record(folder, case, device, warmup, accepted):
    require((folder / 'exit-code.txt').read_text().strip() == '0', f'Nonzero msprof exit code: {folder}')
    command = json.loads((folder / 'command.json').read_text())
    for token in (
        f'--warm-up={warmup}',
        '--launch-count=1',
        '--kill=on',
        '--aic-metrics=BasicInfo',
        '--kernel-name=' + TARGET,
        '--msprof-mode',
    ):
        require(token in command, f'Missing command option {token}: {folder}')
    require(command[4:6] == ['msprof', 'op'], f'Unexpected collection tool: {folder}')
    require(command[command.index('--device-id') + 1] == str(device), 'Command device mismatch')
    require(command[command.index('--scenario') + 1] == case.name, 'Command scenario mismatch')
    log = re.sub(r'\x1b\[[0-9;]*m', '', (folder / 'msprof.log').read_text(encoding='utf-8'))
    before = [json.loads(line) for line in log.splitlines() if line.startswith('{')]
    before = [value for value in before if value.get('stage') == 'before_dut']
    require(len(before) == 1, f'Expected one before_dut record: {folder}')
    metadata = before[0]
    require(metadata['case'] == case.record() and metadata['wrapper_calls'] == 1, 'Capture case/call mismatch')
    require(metadata['source_sha256'] == accepted['source_sha256'] == common.sha(common.DUT), 'Capture DUT mismatch')
    require(
        metadata['dependency_sha256'] == accepted['dependency_sha256'] == common.sha(common.DEPENDENCY),
        'Capture dependency mismatch',
    )
    for field in ('cpu_input_sha256', 'initial_output_sha256', 'input_state', 'parameters', 'seed'):
        require(metadata[field] == accepted[field], f'Capture/golden {field} mismatch: {folder}')
    require(set(metadata['cpu_input_sha256']) == {'x', 'gate', 'other', 'indices'}, 'Missing input hashes')
    indices = metadata['parameters']['indices']
    require(
        indices['rule'] == 'row_id modulo K'
        and indices['seed'] is None
        and indices['fixed'] is None
        and indices['valid_range'] == [0, case.gate_rows - 1]
        and len(indices['value_counts']) == case.gate_rows
        and sum(indices['value_counts']) == case.rows,
        'Invalid cyclic indices metadata',
    )
    rng = random.Random(common.SEED)
    for name in ('x', 'gate', 'other'):
        expected = dict(
            distribution='normal', mean=rng.uniform(-5, 5), std=rng.uniform(0.1, 2), clamp=None, seed=common.SEED
        )
        require(metadata['parameters'][name] == expected, 'Normal parameters do not match seed')
    child = re.findall(r'Child process exited with status\s+(\d+)', log)
    require(not child or child == ['9'], f'Invalid profiler child status: {folder}: {child}')
    if child:
        require('<TerminateInAdvance>' in log and 'collection range' in log, 'Unexplained child status 9')
    warm = re.findall(r'<WarmUp>\s+Warm Up enabled\.\s+times:\s*(\d+)', log)
    require(warm == [str(warmup)], f'Warmup mismatch: {folder}: {warm}')
    csv_files = list(folder.rglob('*OpBasicInfo*.csv'))
    require(len(csv_files) == 1, f'Expected one OpBasicInfo CSV: {folder}')
    with csv_files[0].open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1, f'Expected one target launch: {folder}')
    row = rows[0]
    require(row['Op Name'] == TARGET and int(row['Device Id']) == device, 'CSV kernel/device mismatch')
    duration = float(row['Task Duration(us)'])
    require(math.isfinite(duration) and duration > 0 and int(row['Block Dim']) > 0, 'Invalid duration/grid')
    console = re.findall(r'Task\s+Duration\s*\(us\)\s*:\s*([0-9.eE+\-]+)', log)
    require(len(console) == 1 and abs(float(console[0]) - duration) <= 0.001, 'Console/CSV duration mismatch')
    for label, expected in (('Op Name', TARGET), ('Device Id', str(device)), ('Block Dim', row['Block Dim'])):
        observed = re.findall(r'^\s*' + re.escape(label) + r'\s*:\s*(\S+)\s*$', log, re.MULTILINE)
        require(observed == [expected], f'Console {label} mismatch: {folder}')
    return dict(
        status='PASS',
        duration_us=duration,
        csv=str(csv_files[0].relative_to(folder)),
        csv_sha256=common.sha(csv_files[0]),
        log_sha256=common.sha(folder / 'msprof.log'),
        cpu_input_sha256=metadata['cpu_input_sha256'],
        data_dir=str(folder),
        warnings=[line for line in log.splitlines() if '[WARN]' in line or 'Warning:' in line],
        child_status=child,
    )


def accepted_accuracy(path, device, selected_cases=None):
    value = json.loads(path.read_text())
    require(value['status'] == 'PASS' and value['device_id'] == device, 'Accuracy suite not accepted on this device')
    require(value['source_sha256'] == value['source_sha256_after'] == common.sha(common.DUT), 'Accuracy source changed')
    selected = {item['case']['name']: item for item in value['accuracy']}
    for case in cases.PERFORMANCE if selected_cases is None else selected_cases:
        item = selected[case.name]
        require(
            item['status'] == 'PASS' and item['case'] == case.record() and item['golden'] == 'NPU FP32',
            'Missing accepted large golden',
        )
        require(
            item['metrics']['similarity_pass']
            and item['inputs_unchanged']
            and item['cpu_to_npu_match']
            and item['input_sha256'] == item['cpu_input_sha256']
            and all(item['output_contract'].values()),
            'Invalid golden input/output contract',
        )
    return selected


def collect_batch(output, case, device, warmup, count, accepted, previous=None):
    require(count != 40 or previous is not None, '40 rounds require an accepted 20-round prefix')
    prefix = []
    reuse = {}
    if previous is not None:
        require(
            warmup == 100
            and count == 40
            and previous['warmup'] == 100
            and previous['requested_runs'] == 20
            and previous['status'] == 'PASS'
            and previous['spread_percent'] > 10
            and previous['case'] == case.record()
            and previous['device_id'] == device
            and len(previous['captures']) == 20
            and all(c['status'] == 'PASS' for c in previous['captures']),
            'Invalid batch prefix',
        )
        original = output / case.name / 'w100_n20' / 'batch.json'
        require(json.loads(original.read_text(encoding='utf-8')) == previous, 'Original batch evidence changed')
        prefix = list(previous['captures'])
        reuse = dict(
            reused_runs=20,
            additional_runs=20,
            reused_batch=str(original.relative_to(output)),
            reused_batch_sha256=common.sha(original),
        )
    folder = output / case.name / f'w{warmup}_n{count}'
    folder.mkdir(parents=True, exist_ok=False)
    batch = dict(
        case=case.record(),
        device_id=device,
        warmup=warmup,
        requested_runs=count,
        status='RUNNING',
        captures=prefix,
        started_at=common.now(),
        **reuse,
    )
    for number in range(len(prefix) + 1, count + 1):
        run = folder / f'run_{number:02d}'
        run.mkdir()
        command = [
            shutil.which('timeout'),
            '--signal=TERM',
            '--kill-after=30s',
            '600s',
            'msprof',
            'op',
            '--kernel-name=' + TARGET,
            '--aic-metrics=BasicInfo',
            f'--warm-up={warmup}',
            '--launch-count=1',
            '--kill=on',
            '--output=' + str(run),
            sys.executable,
            '-u',
            str(common.HERE / 'profile_indexed_gate.py'),
            '--device-id',
            str(device),
            '--scenario',
            case.name,
            '--msprof-mode',
        ]
        common.save(run / 'command.json', command)
        print(f'Capture {case.name} warmup={warmup} {number}/{count}: {run}', flush=True)
        with (run / 'msprof.log').open('w', encoding='utf-8') as log:
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
            ) as process:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                exit_code = process.wait()
        (run / 'exit-code.txt').write_text(str(exit_code) + '\n')
        try:
            capture = capture_record(run, case, device, warmup, accepted)
        except Exception as error:
            capture = dict(status='FAIL', data_dir=str(run), error=f'{type(error).__name__}: {error}')
        batch['captures'].append(capture)
        common.save(folder / 'batch.json', batch)
    batch['status'] = 'PASS' if all(c['status'] == 'PASS' for c in batch['captures']) else 'FAIL'
    if batch['status'] == 'PASS':
        values = [c['duration_us'] for c in batch['captures']]
        batch.update(
            min_us=min(values),
            median_us=statistics.median(values),
            max_us=max(values),
            spread_percent=(max(values) - min(values)) / min(values) * 100,
        )
    batch['finished_at'] = common.now()
    common.save(folder / 'batch.json', batch)
    return batch


def plain_table(headers, rows):
    widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    return '\n'.join(
        ''.join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip() for row in [headers] + rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id', type=int)
    parser.add_argument(
        '--output', type=Path, default=common.HERE / 'artifacts' / ('performance_' + common.now().replace(':', ''))
    )
    parser.add_argument('--accuracy-evidence', type=Path, required=True)
    parser.add_argument('--scenario', choices=[case.name for case in cases.ALL_PERFORMANCE], action='append')
    parser.add_argument('--validate-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device_id is not None and args.device_id < 0:
        parser.error('device-id must be nonnegative')
    if args.validate_only:
        return 0
    if args.device_id is None or not shutil.which('timeout') or not shutil.which('msprof'):
        parser.error('Use a nonnegative device-id with GNU timeout and msprof on PATH')
    selected_cases = (
        [c for c in cases.ALL_PERFORMANCE if c.name in args.scenario] if args.scenario else cases.PERFORMANCE
    )
    accepted = accepted_accuracy(args.accuracy_evidence, args.device_id, selected_cases)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = common.source_manifest()
    result = dict(
        status='RUNNING',
        source_sha256=common.sha(common.DUT),
        device_id=args.device_id,
        accuracy_evidence_sha256=common.sha(args.accuracy_evidence),
        batches=[],
        selected={},
        started_at=common.now(),
        script_sha256_before=before,
    )
    for case in selected_cases:
        selected = None
        for warmup, count in ((20, 20), (100, 20), (100, 40)):
            batch = collect_batch(
                output,
                case,
                args.device_id,
                warmup,
                count,
                accepted[case.name],
                previous=selected if count == 40 else None,
            )
            result['batches'].append(batch)
            common.save(output / 'performance.json', result)
            if batch['status'] != 'PASS':
                break
            selected = batch
            if batch['spread_percent'] <= 10:
                break
        if selected is not None and batch['status'] == 'PASS':
            result['selected'][case.name] = dict(
                warmup=selected['warmup'],
                runs=selected['requested_runs'],
                min_us=selected['min_us'],
                median_us=selected['median_us'],
                max_us=selected['max_us'],
                spread_percent=selected['spread_percent'],
            )
        common.save(output / 'performance.json', result)
    result.update(script_sha256_after=common.source_manifest(), finished_at=common.now())
    result['status'] = (
        'PASS'
        if before == result['script_sha256_after']
        and len(result['selected']) == len(selected_cases)
        and all(b['status'] == 'PASS' for b in result['batches'])
        else 'FAIL'
    )
    common.save(output / 'performance.json', result)
    if result['status'] != 'PASS':
        print('ERROR: incomplete/invalid performance batch. See performance.json.', flush=True)
        return 1
    rows = [
        [
            b['case']['name'],
            f"w{b['warmup']}_n{b['requested_runs']}/{i}",
            'Task Duration(us)',
            f"{c['duration_us']:.6f}",
            c['data_dir'],
        ]
        for b in result['batches']
        for i, c in enumerate(b['captures'], 1)
    ]
    final = plain_table(['Scenario', 'Test', 'Metric', 'Duration(us)', 'DataDir'], rows) + '\n\n'
    final += plain_table(
        ['Scenario', 'Minimum Task Duration(us)', 'Warmup', 'Runs'],
        [[name, f"{s['min_us']:.6f}", s['warmup'], s['runs']] for name, s in result['selected'].items()],
    )
    final += (
        '\n\nSpread >10% escalates from warmup20/20 rounds to warmup100/20, then appends20 for40 total. '
        'The cumulative40 batch reuses the original warmup100/20 rounds; different warmups are not pooled. '
        'Minima use only the selected complete batch. All valid samples are retained.\n'
        'Note: The minimum is the best performance observed under the current test conditions.\n'
    )
    (output / 'final_summary.txt').write_text(final, encoding='utf-8')
    print(final, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
