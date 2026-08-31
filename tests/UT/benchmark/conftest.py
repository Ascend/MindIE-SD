#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Pytest bootstrap for the benchmark toolchain unit tests.

The offline tooling (benchmarks/common/, benchmarks/scripts/,
benchmarks/xpu-perf-plugin/op_defs/_common.py) is pure Python with no NPU /
xpu_perf dependency, so it can be tested on any machine. This conftest puts
the benchmarks/ root on sys.path so `common` and `op_defs._common` resolve,
and benchmarks/scripts on sys.path so `benchmark_report` imports without a
package.

Run:  python -m pytest tests/UT/benchmark -q
"""

import pathlib
import shutil
import sys
import uuid

import pytest

_BENCHMARKS_DIR = pathlib.Path(__file__).resolve().parents[3] / "benchmarks"
_SCRIPTS_DIR = _BENCHMARKS_DIR / "scripts"
_PLUGIN_DIR = _BENCHMARKS_DIR / "xpu-perf-plugin"

for _path in (_BENCHMARKS_DIR, _SCRIPTS_DIR, _PLUGIN_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def pytest_configure(config):
    # The sandboxed dev box denies writes to the default pytest cache dirs;
    # unregister the cacheprovider so no .pytest_cache is created and no
    # PytestCacheWarning is emitted. (addopts "-p no:cacheprovider" injected
    # here is too late: the plugin is already loaded at this hook.)
    cache = config.pluginmanager.get_plugin("cacheprovider")
    if cache is not None:
        config.pluginmanager.unregister(cache)


@pytest.fixture()
def tmp_path():
    """tmp_path replacement rooted in the workspace (repo-root tmp/, gitignored).

    The sandboxed dev box denies the system TEMP and pytest's pytest-of-*
    temp root; the repo-root tmp/ dir is known writable. Directories are
    removed after each test so repeated runs do not accumulate.
    """
    root = _BENCHMARKS_DIR.parent / "tmp" / "bench-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield path
    shutil.rmtree(path, ignore_errors=True)
