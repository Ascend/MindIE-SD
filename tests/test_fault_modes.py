#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import csv
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAULT_MODE_FILE = ROOT / "docs" / "zh" / "appendix" / "eplb_fault_mode_library.csv"
PACKAGE_FAULT_MODE_DIR = ROOT / "mindiesd" / "fault_modes"


class TestFaultModes(unittest.TestCase):
    def test_eplb_fault_mode_library_schema(self):
        with FAULT_MODE_FILE.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        self.assertEqual(len(rows), 11)
        self.assertEqual(len(rows[0]), 30)
        self.assertTrue(all(row["模式编号"].startswith("EPLB-") for row in rows))
        self.assertTrue(all(row["三级"] == "SD" for row in rows))

    def test_doc_fault_mode_library_only_contains_eplb(self):
        csv_files = sorted(FAULT_MODE_FILE.parent.glob("*.csv"))

        self.assertEqual(csv_files, [FAULT_MODE_FILE])

    def test_eplb_fault_mode_library_is_doc_reference_only(self):
        setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")
        pyproject_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        manifest_text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

        self.assertFalse(PACKAGE_FAULT_MODE_DIR.exists())
        self.assertNotIn("fault_modes/*.csv", setup_text)
        self.assertNotIn("fault_modes/*.csv", pyproject_text)
        self.assertNotIn("recursive-include mindiesd/fault_modes *.csv", manifest_text)


if __name__ == "__main__":
    unittest.main()
