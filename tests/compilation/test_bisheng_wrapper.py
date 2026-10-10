# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Exercise the generated compiler wrapper without requiring CANN."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "build/build_ascendc_ops.sh"


class TestBishengWrapper(unittest.TestCase):
    def setUp(self):
        # addCleanup keeps the directory alive for each test and removes it on failure.
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.record = self.root / "arguments"
        self.env = dict(os.environ, ARGUMENT_RECORD=str(self.record), COMPILER_EXIT="0")

    def make_wrapper(self, verbose=False, unusual_paths=False):
        tool_dir = self.root / ("tools with spaces '$;" if unusual_paths else "tools")
        tool_dir.mkdir(exist_ok=True)
        compiler = tool_dir / "bisheng"
        compiler.write_text(
            '#!/bin/bash\nprintf "%s\\0" "$@" > "$ARGUMENT_RECORD"\nexit "$COMPILER_EXIT"\n', encoding="utf-8"
        )
        compiler.chmod(0o700)
        ccache = tool_dir / "ccache"
        ccache.write_text('#!/bin/bash\nexec "$@"\n', encoding="utf-8")
        ccache.chmod(0o700)
        build_dir = self.root / ("build with spaces" if unusual_paths else "build")
        self.env.update(BUILD_DIR=str(build_dir), BISHENG_REAL_PATH=str(compiler), VERBOSE=str(verbose).lower())

        # Load only the generator, so the build driver's entry point cannot run.
        source = SCRIPT.read_text(encoding="utf-8")
        generator = source.split("function gen_bisheng(){", 1)[1].split("function build_package(){", 1)[0]
        result = subprocess.run(
            ["/bin/bash", "-c", 'function gen_bisheng(){' + generator + '\ngen_bisheng "$1"', "test", str(ccache)],
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return build_dir / "gen_bisheng_dir/bisheng"

    def assert_arguments(self, wrapper, arguments, expected_exit=0):
        result = subprocess.run([str(wrapper), *arguments], env=self.env, capture_output=True, check=False)
        self.assertEqual(result.returncode, expected_exit, result.stderr)
        self.assertTrue(self.record.exists(), result.stderr)
        captured = self.record.read_bytes().split(b"\0")[:-1]
        self.assertEqual(captured, [arg.encode() for arg in arguments])

    def test_preserves_argument_boundaries(self):
        wrapper = self.make_wrapper()
        self.assert_arguments(wrapper, ["-DNAME=two words", "", "a'b", 'a"b', "*.cpp", "line1\nline2"])

    def test_metacharacters_are_literal_in_both_logging_modes(self):
        marker = self.root / "injected"
        arguments = [f"; touch {marker} #", f"$(touch {marker})", f"`touch {marker}`", "a && echo b", "a | echo b"]
        for verbose in (False, True):
            with self.subTest(verbose=verbose):
                wrapper = self.make_wrapper(verbose=verbose)
                self.assert_arguments(wrapper, arguments)
                self.assertFalse(marker.exists(), "compiler arguments must never execute as shell commands")

    def test_quotes_compiler_cache_and_build_paths(self):
        wrapper = self.make_wrapper(unusual_paths=True)
        self.assert_arguments(wrapper, ["input file.cpp", "-o", "output file.o"])

    def test_propagates_compiler_exit_code(self):
        wrapper = self.make_wrapper()
        self.env["COMPILER_EXIT"] = "23"
        self.assert_arguments(wrapper, ["input.cpp"], expected_exit=23)


if __name__ == "__main__":
    unittest.main()
