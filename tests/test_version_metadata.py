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

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestVersionMetadata(unittest.TestCase):
    def test_pyproject_uses_canonical_version_attr(self):
        pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('version = { attr = "mindiesd._version.__version__" }', pyproject_text)

    def test_version_file_exposes_literal_version(self):
        version_text = (ROOT / "mindiesd" / "_version.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', version_text, re.MULTILINE)

        if match is None:
            self.fail("version literal was not found in mindiesd/_version.py")
        self.assertEqual(match.group(1), "2.3.0")


if __name__ == "__main__":
    unittest.main()
