# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Load real parallel modules and stock functions without importing the NPU package."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[3]
PARALLEL_DIR = REPO_ROOT / "mindiesd/parallel"


def new_package(name, monkeypatch=None, directory=PARALLEL_DIR):
    package = ModuleType(name)
    package.__path__ = [str(directory)]
    register_module(name, package, monkeypatch)
    return package


def register_module(name, module, monkeypatch=None):
    if monkeypatch is None:
        sys.modules[name] = module
    else:
        monkeypatch.setitem(sys.modules, name, module)


def load_parallel_module(package, relative, monkeypatch=None):
    name = package.__name__ + "." + relative
    return load_module(name, PARALLEL_DIR / (relative + ".py"), monkeypatch, package, relative)


def load_module(name, path, monkeypatch, package=None, relative=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    register_module(name, module, monkeypatch)
    if package is not None:
        setattr(package, relative, module)
    spec.loader.exec_module(module)
    return module
