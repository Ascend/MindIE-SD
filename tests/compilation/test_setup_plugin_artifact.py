#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:

#     http://license.coscl.org.cn/MulanPSL2

# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import ast
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_find_usable_plugin_artifact():
    setup_tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in setup_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "find_usable_plugin_artifact"
    )
    namespace = {
        "logging": logging,
        "os": os,
        "subprocess": subprocess,
        "sys": sys,
    }
    exec(  # pylint: disable=exec-used
        compile(ast.Module(body=[function], type_ignores=[]), "setup.py", "exec"), namespace
    )
    return namespace["find_usable_plugin_artifact"]


find_usable_plugin_artifact = load_find_usable_plugin_artifact()


class TestFindUsablePluginArtifact(unittest.TestCase):
    def test_returns_loadable_build_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_dir = Path(tmp_dir) / "build"
            project_root = Path(tmp_dir) / "project"
            artifact = build_dir / "plugin_build" / "libPTAExtensionOPS.so"
            artifact.parent.mkdir(parents=True)
            artifact.touch()

            with mock.patch("subprocess.run") as run:
                run.return_value.returncode = 0
                result = find_usable_plugin_artifact(str(build_dir), str(project_root))

            self.assertEqual(result, str(artifact))
            run.assert_called_once()

    def test_returns_none_when_cached_artifact_cannot_be_loaded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            build_dir = Path(tmp_dir) / "build"
            project_root = Path(tmp_dir) / "project"
            artifact = project_root / "mindiesd" / "plugin" / "libPTAExtensionOPS.so"
            artifact.parent.mkdir(parents=True)
            artifact.touch()

            with mock.patch("subprocess.run") as run:
                run.return_value.returncode = 1
                run.return_value.stderr = "undefined symbol"
                run.return_value.stdout = ""
                result = find_usable_plugin_artifact(str(build_dir), str(project_root))

            self.assertIsNone(result)

    def test_returns_none_without_cached_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch("subprocess.run") as run:
            result = find_usable_plugin_artifact(tmp_dir, tmp_dir)

        self.assertIsNone(result)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
