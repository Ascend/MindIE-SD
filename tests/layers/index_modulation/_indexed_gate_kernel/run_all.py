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
"""Run small accuracy, large NPU golden, stability and independent msprof captures."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

import indexed_gate_common as common
import indexed_gate_cases as cases


def describe(device_id, path):
    dut, device = common.initialize(device_id)
    import torch
    import torch_npu
    import triton

    vector, ub = dut._get_npu_resources(device.index)
    modules = {
        m.__name__: dict(version=getattr(m, '__version__', 'unknown'), path=m.__file__)
        for m in (torch, torch_npu, triton)
    }
    common.save(
        path,
        dict(
            python=sys.executable,
            version=sys.version,
            modules=modules,
            device_name=torch.npu.get_device_name(device),
            vector_count=vector,
            ub_size=ub,
            tiling={c.name: dut._tile_shape(c.width, ub) for c in cases.ALL_PRODUCTION},
            environment={
                k: v for k, v in os.environ.items() if k.startswith(('TRITON_', 'ASCEND_', 'LD_LIBRARY_PATH'))
            },
            source_sha256=common.sha(common.DUT),
            dependency_sha256=common.sha(common.DEPENDENCY),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device-id', type=int)
    parser.add_argument(
        '--output', type=Path, default=common.HERE / 'artifacts' / ('run_' + common.now().replace(':', ''))
    )
    parser.add_argument('--describe', type=Path, help=argparse.SUPPRESS)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        '--extended-only',
        action='store_true',
        help='Run all small cases and only N,H > 10K large accuracy/stress/performance cases',
    )
    scope.add_argument(
        '--hidden-20000-only',
        action='store_true',
        help='Run all small cases and H=20000, N>10K large accuracy/stress/performance cases',
    )
    args = parser.parse_args()
    if args.device_id is not None and args.device_id < 0:
        parser.error('device-id must be nonnegative')
    if args.describe:
        describe(args.device_id, args.describe)
        return 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for key in (
        'ASCEND_RT_VISIBLE_DEVICES',
        'TRITON_KERNEL_DUMP',
        'TRITON_DEBUG',
        'TRITON_ALWAYS_COMPILE',
        'TRITON_DUMP_DIR',
    ):
        os.environ.pop(key, None)
    os.environ['TRITON_CACHE_DIR'] = str(output / 'triton_cache')
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    device, selection = common.select_device(args.device_id)
    before = common.source_manifest()
    large_cases = (
        cases.HIDDEN_20000 if args.hidden_20000_only else cases.EXTENDED if args.extended_only else cases.PRODUCTION
    )
    perf_cases = [c for c in cases.ALL_PERFORMANCE if c in large_cases]
    record = dict(
        status='RUNNING',
        started_at=common.now(),
        device_id=device,
        selection=selection,
        script_sha256_before=before,
        stages=[],
        output=str(output),
        pid=os.getpid(),
        large_scenarios=[c.name for c in large_cases],
        performance_scenarios=[c.name for c in perf_cases],
    )
    common.save(output / 'suite.json', record)
    stopped = threading.Event()

    def monitor():
        while not stopped.is_set():
            try:
                stamp = common.now().replace(':', '')
                common.npu_snapshot(output / 'monitor' / (stamp + '.log'))
                processes = subprocess.run(['ps', '-eo', 'pid,ppid,args'], capture_output=True, text=True, check=True)
                (output / 'monitor' / (stamp + '.processes')).write_text(processes.stdout, encoding='utf-8')
            except Exception:
                (output / 'monitor-errors.log').open('a', encoding='utf-8').write(traceback.format_exc())
            stopped.wait(30)

    (output / 'monitor').mkdir()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    python = sys.executable
    os.environ['INDEXED_GATE_DEVICE'] = str(device)
    os.environ['INDEXED_GATE_OUTPUT'] = str(output / 'small')
    commands = [
        (
            'environment',
            [
                python,
                str(Path(__file__).resolve()),
                '--device-id',
                str(device),
                '--describe',
                str(output / 'environment.json'),
            ],
            180,
        ),
        (
            'small',
            [
                python,
                '-m',
                'pytest',
                '-q',
                '-s',
                '--confcutdir=' + str(common.HERE),
                '--junitxml=' + str(output / 'small.xml'),
                str(common.HERE / 'test_indexed_gate.py'),
            ],
            1800,
        ),
        (
            'accuracy',
            [
                'bash',
                str(common.HERE / 'run_indexed_gate_prod_accuracy.sh'),
                '--device-id',
                str(device),
                '--mode',
                'accuracy',
                '--output',
                str(output / 'accuracy'),
            ],
            1800,
        ),
        (
            'stress',
            [
                'bash',
                str(common.HERE / 'run_indexed_gate_prod_accuracy.sh'),
                '--device-id',
                str(device),
                '--mode',
                'stress',
                '--stress-runs',
                '1000',
                '--accuracy-evidence',
                str(output / 'accuracy/results.json'),
                '--output',
                str(output / 'stress'),
            ],
            3600,
        ),
        (
            'performance',
            [
                'bash',
                str(common.HERE / 'run_indexed_gate_msprof_op.sh'),
                '--device-id',
                str(device),
                '--accuracy-evidence',
                str(output / 'accuracy/results.json'),
                '--output',
                str(output / 'performance'),
            ],
            21600,
        ),
    ]
    if args.extended_only or args.hidden_20000_only:
        for name, command, _ in commands:
            selected_cases = (
                large_cases if name in ('accuracy', 'stress') else perf_cases if name == 'performance' else []
            )
            for case in selected_cases:
                command.extend(['--scenario', case.name])
    try:
        for name, command, timeout in commands:
            stage = dict(name=name, command=command, started_at=common.now(), status='RUNNING')
            record['stages'].append(stage)
            common.save(output / 'suite.json', record)
            print(f'STAGE {name} START device={device}', flush=True)
            with (output / (name + '.log')).open('w', encoding='utf-8') as log:
                completed = subprocess.run(
                    command, cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
                )
            stage.update(
                exit_code=completed.returncode,
                finished_at=common.now(),
                status='PASS' if completed.returncode == 0 else 'FAIL',
            )
            common.save(output / 'suite.json', record)
            print(f'STAGE {name} {stage["status"]}', flush=True)
            if completed.returncode:
                break
    except Exception:
        record.update(error=traceback.format_exc())
    finally:
        stopped.set()
        thread.join(timeout=15)
        record.update(script_sha256_after=common.source_manifest(), finished_at=common.now())
        record['status'] = (
            'PASS'
            if len(record['stages']) == len(commands)
            and all(s['status'] == 'PASS' for s in record['stages'])
            and before == record['script_sha256_after']
            and 'error' not in record
            else 'FAIL'
        )
        common.save(output / 'suite.json', record)
        print('SUITE ' + record['status'], flush=True)
    return 0 if record['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
