#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "version.py"


class TestVersionMetadata(unittest.TestCase):
    def test_pyproject_uses_canonical_version_attr(self):
        pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('version = { attr = "version.__version__" }', pyproject_text)

    def test_version_uses_canonical_default(self):
        original = os.environ.pop("MINDIE_SD_VERSION_OVERRIDE", None)
        try:
            version_ns = runpy.run_path(str(VERSION_FILE))
            self.assertTrue(version_ns["__version__"])
            self.assertEqual(version_ns["__version__"], "3.0.0")
        finally:
            if original is not None:
                os.environ["MINDIE_SD_VERSION_OVERRIDE"] = original

    def test_version_honors_override(self):
        original = os.environ.get("MINDIE_SD_VERSION_OVERRIDE")
        os.environ["MINDIE_SD_VERSION_OVERRIDE"] = "9.9.9T20260415"
        try:
            version_ns = runpy.run_path(str(VERSION_FILE))
            self.assertEqual(version_ns["__version__"], "9.9.9post20260415")
        finally:
            if original is None:
                os.environ.pop("MINDIE_SD_VERSION_OVERRIDE", None)
            else:
                os.environ["MINDIE_SD_VERSION_OVERRIDE"] = original


if __name__ == "__main__":
    unittest.main()
