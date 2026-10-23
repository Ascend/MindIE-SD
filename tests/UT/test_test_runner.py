# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2. See LICENSE for details.

"""Test that precision-mode import errors fail the run."""

import importlib.util
from pathlib import Path
import sys

import pytest


def test_precision_loader_propagates_import_failure(tmp_path, monkeypatch):
    runner_path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("mindiesd_test_runner", runner_path)
    runner = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("MINDIE_TEST_MODE", "ALL")
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec.loader.exec_module(runner)
    (tmp_path / "test_missing_dependency.py").touch()

    def fail_import(*_args):
        raise ModuleNotFoundError("missing test dependency")

    monkeypatch.setattr(runner, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError, match="missing test dependency"):
        runner.load_tests_from_names(str(tmp_path), "test_missing_dependency")
