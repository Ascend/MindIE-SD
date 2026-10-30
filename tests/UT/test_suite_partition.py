# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

# pylint: disable=redefined-outer-name

import pytest

from tests.scripts import run_suites


@pytest.fixture
def suites(tmp_path, monkeypatch):
    root = tmp_path / 'tests'
    (root / 'ops').mkdir(parents=True)
    (root / 'test_frontend.py').touch()
    (root / 'ops/test_kernel.py').touch()
    monkeypatch.setattr(run_suites, 'TESTS', root)
    monkeypatch.setenv('MINDIE_TEST_RESULTS', str(tmp_path / 'results'))
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return 0

    monkeypatch.setattr(run_suites.subprocess, 'call', execute)
    return root, calls


def test_explicit_root_directory_runs_both_partitions(suites):
    root, calls = suites
    assert run_suites.main(['--all', str(root)]) == 0
    pytest_calls = [(command, kwargs) for command, kwargs in calls if 'pytest' in command]
    assert len(pytest_calls) == 2
    assert '--test-mode=CPU' in pytest_calls[0][0]
    assert str(root / 'test_frontend.py') in pytest_calls[0][0]
    assert str(root / 'ops/test_kernel.py') not in pytest_calls[0][0]
    assert '--test-mode=NPU' in pytest_calls[1][0]
    assert str(root / 'ops/test_kernel.py') in pytest_calls[1][0]
    assert str(root / 'test_frontend.py') not in pytest_calls[1][0]


def test_repeated_runs_use_disjoint_coverage_artifacts(suites):
    root, calls = suites
    for _ in range(2):
        assert run_suites.main(['--cpu_only', str(root)]) == 0
    data_paths = [kwargs['env']['COVERAGE_FILE'] for command, kwargs in calls if 'pytest' in command]
    assert len(set(data_paths)) == 2


def test_unavailable_npu_fails_before_operator_collection(suites, monkeypatch):
    root, calls = suites

    def fail_preflight(command, **kwargs):
        calls.append((command, kwargs))
        return 23

    monkeypatch.setattr(run_suites.subprocess, 'call', fail_preflight)
    assert run_suites.main(['--npu_only', str(root / 'ops')]) == 23
    assert not any('pytest' in command for command, _ in calls)


def test_cpu_entry_rejects_operator_selection(suites):
    root, _ = suites
    with pytest.raises(SystemExit) as error:
        run_suites.main(['--cpu_only', str(root / 'ops')])
    assert error.value.code == 2


@pytest.mark.parametrize('mode', ['--all', '--cpu_only'])
def test_explicit_full_directory_enforces_file_coverage(suites, monkeypatch, mode):
    root, calls = suites

    def coverage_failure(command, **kwargs):
        calls.append((command, kwargs))
        return 17 if any(str(arg).endswith('check_file_coverage.py') for arg in command) else 0

    monkeypatch.setattr(run_suites.subprocess, 'call', coverage_failure)
    assert run_suites.main([mode, str(root)]) == 17
