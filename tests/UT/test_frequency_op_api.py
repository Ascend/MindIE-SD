# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under Mulan PSL v2. See LICENSE for details.
"""Exercise the production frequency API loader against independent shared libraries."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("g++"), "requires Linux and g++")
class TestFrequencyOpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The class fixture must outlive setUpClass, including setup failures.
        cls.temporary = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        source = cls.directory / "loader.cpp"
        production = (ROOT / "csrc/plugin/find_op_path.cpp").read_text(encoding="utf-8")

        def definition_start(name):
            marker = production.index(name + "(")
            return production.rfind("\n", 0, marker) + 1

        names = production[definition_start("GetOpApiLibName") : definition_start("GetCustomOpApiLibPath")]
        loader = production[definition_start("GetOpApiLibHandler") : definition_start("LoadDefaultCustomOpApiHandler")]
        optional_header = (
            '#include "frequency_op_api.h"\n' if (ROOT / "csrc/plugin/frequency_op_api.h").is_file() else ""
        )
        source.write_text(
            "#include <dlfcn.h>\n#include <cstdlib>\n#include <string>\n"
            "void* TestDlopen(const char* name, int flags) {\n"
            '  std::string path = std::string(std::getenv("TEST_LIB_DIR")) + "/" + name;\n'
            "  return dlopen(path.c_str(), flags);\n}\n"
            "#define dlopen TestDlopen\n#define ASCEND_LOGW(...)\n"
            + optional_header
            + names
            + loader
            + "#include <iostream>\n"
            "int main() {\n"
            '  const char* names[] = {"aclnnFrequencyRegulator", '
            '"aclnnFrequencyRegulatorGetWorkspaceSize", "OtherApi"};\n'
            "  for (auto name : names) {\n"
            "    auto address = GetFuncFromDefaultLib(name);\n"
            "    std::cout << (address ? reinterpret_cast<int (*)()>(address)() : -1) << ' ';\n"
            "  }\n"
            "}\n",
            encoding="utf-8",
        )
        cls.executable = cls.directory / "loader"
        subprocess.run(
            ["g++", "-std=c++17", str(source), "-I", str(ROOT / "csrc/plugin"), "-ldl", "-o", str(cls.executable)],
            check=True,
            capture_output=True,
            text=True,
        )

    def run_libraries(self, oam_symbols, old_symbols):
        with tempfile.TemporaryDirectory(dir=self.directory) as directory:
            path = Path(directory)
            for name, symbols, result in (
                ("libopapi_oam.so", oam_symbols, 11),
                ("libopapi.so", old_symbols, 22),
            ):
                if symbols is None:
                    continue
                if name == "libopapi.so":
                    symbols = (*symbols, "OtherApi")
                source = path / (name + ".cpp")
                source.write_text(
                    "\n".join(f'extern "C" int {symbol}() {{ return {result}; }}' for symbol in symbols),
                    encoding="utf-8",
                )
                subprocess.run(
                    ["g++", "-shared", "-fPIC", str(source), "-o", str(path / name)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            environment = dict(os.environ, TEST_LIB_DIR=str(path))
            result = subprocess.run(
                [str(self.executable)],
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            return tuple(int(value) for value in result.stdout.split())

    def test_new_library_preferred(self):
        pair = ("aclnnFrequencyRegulator", "aclnnFrequencyRegulatorGetWorkspaceSize")
        self.assertEqual(self.run_libraries(pair, pair), (11, 11, 22))

    def test_new_library_without_legacy(self):
        pair = ("aclnnFrequencyRegulator", "aclnnFrequencyRegulatorGetWorkspaceSize")
        self.assertEqual(self.run_libraries(pair, None), (11, 11, -1))

    def test_legacy_fallback(self):
        pair = ("aclnnFrequencyRegulator", "aclnnFrequencyRegulatorGetWorkspaceSize")
        self.assertEqual(self.run_libraries(None, pair), (22, 22, 22))

    def test_incomplete_new_library_uses_legacy_pair(self):
        pair = ("aclnnFrequencyRegulator", "aclnnFrequencyRegulatorGetWorkspaceSize")
        for symbol in pair:
            with self.subTest(symbol=symbol):
                self.assertEqual(self.run_libraries((symbol,), pair), (22, 22, 22))

    def test_missing_or_incomplete_pairs_return_null(self):
        pair = ("aclnnFrequencyRegulator", "aclnnFrequencyRegulatorGetWorkspaceSize")
        for new, old in (((), ()), ((pair[0],), (pair[1],)), (None, ())):
            with self.subTest(new=new, old=old):
                self.assertEqual(self.run_libraries(new, old), (-1, -1, 22))


if __name__ == "__main__":
    unittest.main()
