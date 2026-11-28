#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--target', action='append', help='Repeatable test file; default: repository pytest testpaths')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / 'tests/conftest.py').is_file():
        parser.error('--repo must be a MindIE-SD checkout')
    for target in args.target or []:
        path = (repo / target).resolve()
        invalid_syntax = any(part in target for part in ('\\', '::')) or '..' in Path(target).parts
        if (
            invalid_syntax
            or not target.startswith('tests/')
            or not target.endswith('.py')
            or not path.is_relative_to(repo / 'tests')
            or not path.is_file()
        ):
            parser.error(f'Target must be an existing relative tests/*.py file: {target}')
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='mindiesd-cpu-target-', dir=args.output.resolve()))
    root = output / 'mindiesd'
    task_name = 'MINDIE-SD@task_' + datetime.now().strftime('%Y%m%d%H%M%S')
    evidence = output / 'evidence'
    evidence.mkdir()
    print(f'Artifacts: {output}', flush=True)
    config = evidence / 'coveragerc'
    config.write_text(
        '[run]\nbranch = True\nrelative_files = False\n'
        'parallel = True\nconcurrency = thread,multiprocessing\n'
        f'sigterm = False\ninclude = {repo.as_posix()}/mindiesd/*\n'
        'disable_warnings = no-data-collected,module-not-python\n',
        encoding='utf-8',
    )
    env = dict(
        os.environ,
        MINDIE_TEST_MODE='CPU',
        TORCH_DEVICE_BACKEND_AUTOLOAD='0',
        ASCEND_RT_VISIBLE_DEVICES='-1',
        MINDIE_LOG_TO_FILE='false',
        MINDIE_LOG_TO_STDOUT='false',
        PYTHONUTF8='1',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',
    )
    for key in ('PYTEST_ADDOPTS', 'COVERAGE_PROCESS_START', 'COVERAGE_INVENTORY'):
        env.pop(key, None)

    git_executable = shutil.which('git')
    if not git_executable:
        parser.error('git must be available on PATH')

    def git(*argv):
        return subprocess.check_output([git_executable, *argv], cwd=repo).decode('utf-8').strip()

    def run(argv, log):
        log_path = evidence / log
        with log_path.open('w', encoding='utf-8') as stream:
            rc = subprocess.run(
                [sys.executable, *argv], cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
            ).returncode
        if rc:
            print(f'Command failed ({rc}); last 80 lines of {log_path}:', flush=True)
            print('\n'.join(log_path.read_text(encoding='utf-8', errors='replace').splitlines()[-80:]), flush=True)
        return rc

    commit = git('rev-parse', 'HEAD')
    status = git('status', '--short')
    hashes = {}
    for name in filter(None, git('ls-files', '-z', 'mindiesd', 'tests').split('\0')):
        source = repo / name
        if not source.is_file():
            continue
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        if name.startswith('mindiesd/') and name.endswith('.py'):
            dest = root / 'covstub' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
    (root / 'settingInfo.xml').write_text(
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<setting>\n    <type>3</type>\n    <stage>1</stage>\n'
        '    <executor>admin</executor>\n</setting>\n',
        encoding='utf-8',
    )
    # Let the repository configure and filter the CPU suite.
    plugin = evidence / 'cpu_target_inventory.py'
    inventory = evidence / 'targets.json'
    plugin.write_text(
        'import json, os\nfrom pathlib import Path\n'
        'def pytest_collection_finish(session):\n'
        '    if not session.config.option.collectonly:\n'
        '        return\n'
        '    files = sorted({item.path.relative_to(session.config.rootpath).as_posix() for item in session.items})\n'
        '    Path(os.environ["CPU_TARGET_INVENTORY"]).write_text(json.dumps(files))\n',
        encoding='utf-8',
    )
    env['CPU_TARGET_INVENTORY'] = str(inventory)
    env['PYTHONPATH'] = str(evidence) + os.pathsep + env.get('PYTHONPATH', '')
    collect_rc = run(
        [
            '-m',
            'pytest',
            *(args.target or []),
            '--test-mode=CPU',
            '--collect-only',
            '-q',
            '-p',
            'cpu_target_inventory',
        ],
        'collection.log',
    )
    if collect_rc:
        print(f'CPU collection failed ({collect_rc}); see {evidence / "collection.log"}')
        return collect_rc
    targets = args.target or json.loads(inventory.read_text(encoding='utf-8'))
    if not targets or len(targets) != len(set(targets)):
        parser.error('Targets must be nonempty and unique')
    names = []
    for target in targets:
        if (
            "\\" in target
            or not target.startswith('tests/')
            or not target.endswith('.py')
            or '..' in Path(target).parts
        ):
            parser.error(f'Target must be a relative tests/*.py file: {target}')
        name = target.removesuffix('.py').replace('/', '__')
        names.append(name)
    if len(names) != len(set(names)):
        parser.error('Multiple test files map to the same coverage directory name')
    runs, covdirs, suites = [], [], []
    test_rc = 0
    for index, (target, name) in enumerate(zip(targets, names), 1):
        covdata = root / task_name / name / 'covdata'
        covdata.mkdir(parents=True)
        covdirs.append(covdata)
        env['COVERAGE_FILE'] = str(covdata / 'coverage')
        junit = evidence / f'{name}.junit.xml'
        command = [
            '-m',
            'coverage',
            'run',
            f'--rcfile={config}',
            '-m',
            'pytest',
            target,
            '--test-mode=CPU',
            '-v',
            '--maxfail=0',
            '-p',
            'no:cov',
            f'--junitxml={junit}',
        ]
        rc = run(command, f'{name}.pytest.log')
        if rc:
            (covdata / 'FAILED').write_text('1\n', encoding='utf-8')
            test_rc = test_rc or rc
        runs.append({'target': target, 'directory': name, 'exit_code': rc, 'command': [sys.executable, *command]})
        if junit.exists():
            suites.extend(n.attrib for n in ET.parse(junit).iter('testsuite'))
        (evidence / 'target-results.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
        print(f'[{index}/{len(targets)}] {target}: exit {rc}', flush=True)
    raw_files = sorted(p for directory in covdirs for p in directory.glob('coverage.*'))
    checks = []
    for path in raw_files:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as conn:
            assert conn.execute('pragma integrity_check').fetchone()[0] == 'ok'
            assert conn.execute("select value from meta where key='has_arcs'").fetchone()[0] == '1'
            paths = [row[0] for row in conn.execute('select path from file')]
            assert all(Path(p).is_absolute() for p in paths)
            assert all(Path(p).is_relative_to(repo / 'mindiesd') for p in paths), 'Non-product files collected'
            contexts = [row[0] for row in conn.execute('select context from context')]
            assert all(c == '' for c in contexts), contexts
            checks.append(
                {
                    'file': path.relative_to(root).as_posix(),
                    'source_files': len(paths),
                    'arcs': conn.execute('select count(*) from arc').fetchone()[0],
                }
            )
    # Report from a separate combined file; retain original parallel files unchanged.
    env['COVERAGE_FILE'] = str(evidence / 'coverage')
    combine_rc = run(
        ['-m', 'coverage', 'combine', f'--rcfile={config}', '--keep', *(str(d) for d in covdirs)], 'combine.log'
    )
    report_rc = (
        run(['-m', 'coverage', 'json', f'--rcfile={config}', '-o', str(evidence / 'coverage.json')], 'report.log')
        if combine_rc == 0
        else combine_rc
    )
    changed = [
        name
        for name, digest in hashes.items()
        if not (repo / name).is_file() or hashlib.sha256((repo / name).read_bytes()).hexdigest() != digest
    ]
    executed_tests = sum(int(s.get('tests', 0)) - int(s.get('skipped', 0)) - int(s.get('errors', 0)) for s in suites)
    valid_data = executed_tests > 0 and any(c['arcs'] > 0 for c in checks)
    if not valid_data:
        print('No executed tests or no product coverage; refusing to publish an empty baseline.', flush=True)
    metadata = {
        'test_granularity': 'file',
        'targets': runs,
        'task': task_name,
        'commit': commit,
        'git_status_before': status,
        'git_status_after': git('status', '--short'),
        'test_exit_code': test_rc,
        'combine_exit_code': combine_rc,
        'report_exit_code': report_rc,
        'versions': {n: importlib.metadata.version(n) for n in ('coverage', 'pytest', 'torch')},
        'raw_data_checks': checks,
        'junit_suites': suites,
        'executed_tests': executed_tests,
        'source_sha256': hashes,
        'sources_changed_during_run': changed,
        'coverage_dir': f'mindiesd/{task_name}',
        'source_dir': 'mindiesd/covstub',
        'limitations': [
            'Repository CPU bootstrap; no real NPU/C++ kernel coverage',
            'multiprocessing configured; arbitrary subprocess startup not instrumented',
            'One mapping per file; default scope follows repository pytest testpaths',
            'Official coverage.py; suffix generated by installed coverage version',
            'Precision-test requires a registered MindIE-SD adapter',
        ],
    }
    (evidence / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    shutil.copy2(__file__, output / Path(__file__).name)
    rc = (
        test_rc
        or combine_rc
        or report_rc
        or int(any(not list(d.glob('coverage.*')) for d in covdirs) or bool(changed) or not valid_data)
    )
    if rc == 0:
        with tarfile.open(output / 'coverage.tar', 'w') as archive:
            archive.add(root, arcname='mindiesd')
        print(
            json.dumps(
                {'archive': str(output / 'coverage.tar'), 'suites': suites, 'raw_files': len(raw_files)}, indent=2
            )
        )
    else:
        print(f'Collection failed ({rc}); diagnostics retained, no delivery tar published.')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
