# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.

"""Test that a pytest collection error fails the run."""

from tests.scripts import run_suites


def test_collection_failure_propagates(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "test_missing_dependency.py").touch()
    monkeypatch.setattr(run_suites, "TESTS", root)
    monkeypatch.setenv("MINDIE_TEST_RESULTS", str(root / "results"))
    calls = []

    def failed_collection(command, **kwargs):
        calls.append((command, kwargs))
        return 2 if "pytest" in command else 0

    monkeypatch.setattr(run_suites.subprocess, "call", failed_collection)
    assert run_suites.main(["--cpu_only", str(root / "test_missing_dependency.py")]) == 2
    assert str(root / "test_missing_dependency.py") in calls[0][0]
    assert calls[0][1]["env"]["MINDIE_TEST_MODE"] == "CPU"
