# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# Licensed under Mulan PSL v2. See LICENSE for details.
"""Verify real shell archive and setuptools wheel metadata share one version."""

import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform.startswith("linux"), "release packaging requires Linux")
class TestReleaseArchiveVersions(unittest.TestCase):
    def test_tar_wheel_and_metadata_versions_agree(self):
        # Native compilation is irrelevant to version naming. A minimal real
        # setuptools build exercises the unmodified production archive script.
        cases = {
            "3.0.0": "3.0.0",
            "3.0.0.RC1.2": "3.0.0rc1.post2",
            "3.0.0.T": "3.0.0+t",
            "3.0.0.T20260415": "3.0.0+t20260415",
            "9.9.9T20260415": "9.9.9.post20260415",
            "3.0.0.RC01.02": "3.0.0rc1.post2",
            "3.0.0.RC1": "3.0.0rc1",
            "v3.0.0": "3.0.0",
        }
        for override, expected in cases.items():
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                (project / "build").mkdir()
                script = project / "build/build.sh"
                script.write_text(
                    (ROOT / "build/build.sh").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                shutil.copyfile(ROOT / "version.py", project / "version.py")
                (project / "requirements.txt").write_text("", encoding="utf-8")
                (project / "mindiesd").mkdir()
                (project / "mindiesd/__init__.py").write_text("", encoding="utf-8")
                (project / "setup.py").write_text(
                    "import runpy\nfrom setuptools import setup\n"
                    'setup(name="mindiesd", version=runpy.run_path("version.py")["__version__"], '
                    'packages=["mindiesd"])\n',
                    encoding="utf-8",
                )
                environment = dict(os.environ, MINDIE_SD_VERSION_OVERRIDE=override)
                result = subprocess.run(
                    ["bash", str(script)],
                    env=environment,
                    cwd=project,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                archives = list((project / "output").glob("*.tar.gz"))
                self.assertEqual(len(archives), 1)
                self.assertTrue(archives[0].name.startswith(f"Ascend-mindie-sd_{expected}_"))
                with tarfile.open(archives[0]) as archive:
                    wheels = [item for item in archive.getmembers() if item.name.endswith(".whl")]
                    self.assertEqual(len(wheels), 1)
                    self.assertIn(f"/Ascend-mindie-sd_{expected}_", wheels[0].name)
                    self.assertTrue(Path(wheels[0].name).name.startswith(f"mindiesd-{expected}-"))
                    with zipfile.ZipFile(io.BytesIO(archive.extractfile(wheels[0]).read())) as wheel:
                        metadata_file = next(name for name in wheel.namelist() if name.endswith("/METADATA"))
                        metadata = Parser().parsestr(wheel.read(metadata_file).decode("utf-8"))
                        self.assertEqual(metadata["Version"], expected)


if __name__ == "__main__":
    unittest.main()
