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

"""Regression tests for the pre-commit gitleaks-offline-scan wrapper.

Issue #404: on a fresh local checkout the gitleaks hook hard-failed because
the ./gitleaks binary is not committed (CI installs it separately). The fix
routes the hook through pre-commit/gitleaks_scan.sh, which must:

  1. skip cleanly (exit 0, SKIP notice) when ./gitleaks is absent
  2. exec the binary with the exact pre-commit args when it is present
  3. stay wired into .pre-commit-config.yaml as the hook entry

The wrapper is run inside a private temp directory, so these tests are
hermetic: no NPU, no CANN, no gitleaks installation required.
"""

import os
import stat
import subprocess
import tempfile
import unittest

from tests.UT.test_gitleaks_scan import _repo_root


class TestGitleaksScanWrapper(unittest.TestCase):
    def setUp(self):
        self.wrapper = os.path.join(_repo_root(), "pre-commit", "gitleaks_scan.sh")
        self.assertTrue(os.path.isfile(self.wrapper), f"wrapper missing: {self.wrapper}")
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.temp_dir.cleanup)
        self.work_dir = self.temp_dir.name

    def _run_wrapper(self, *args):
        env = {**os.environ, "GITLEAKS_MOCK_ARGS": os.path.join(self.work_dir, "mock_args.txt")}
        return subprocess.run(
            ["bash", self.wrapper, *args],
            cwd=self.work_dir,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def test_skips_cleanly_when_binary_missing(self):
        result = self._run_wrapper("some_file.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SKIP", result.stdout)

    def test_execs_binary_with_pre_commit_args_when_present(self):
        mock_bin = os.path.join(self.work_dir, "gitleaks")
        with open(mock_bin, "w", encoding="utf-8") as handle:
            handle.write("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$GITLEAKS_MOCK_ARGS\"\n")
        os.chmod(mock_bin, os.stat(mock_bin).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        result = self._run_wrapper("file_a.py", "file_b.py")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        with open(os.path.join(self.work_dir, "mock_args.txt"), encoding="utf-8") as handle:
            args = [line for line in handle.read().splitlines() if line]
        self.assertEqual(
            args,
            ["protect", "--verbose", "--redact", "--config=pre-commit/.gitleaks.toml", "file_a.py", "file_b.py"],
        )

    def test_precommit_config_uses_wrapper_as_hook_entry(self):
        config_path = os.path.join(_repo_root(), ".pre-commit-config.yaml")
        with open(config_path, encoding="utf-8") as handle:
            config_text = handle.read()
        hook_block = None
        for block in config_text.split("- id: "):
            if block.startswith("gitleaks-offline-scan"):
                hook_block = "- id: " + block
                break
        self.assertIsNotNone(hook_block, "gitleaks-offline-scan hook not found in .pre-commit-config.yaml")
        self.assertIn("entry: pre-commit/gitleaks_scan.sh", hook_block)


if __name__ == "__main__":
    unittest.main()
