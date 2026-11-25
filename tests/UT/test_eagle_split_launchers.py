# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Exercise the production Bash launchers at their hardware and workload boundaries."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from tests.ops.eagle_split_norm_rope_concat.summarize_eagle_split_norm_rope_concat_layer import capture_device

SCRIPTS = Path(__file__).resolve().parents[1] / "ops" / "eagle_split_norm_rope_concat"


class TestCaptureDevice(unittest.TestCase):
    def test_sparse_a3_and_legacy_reports(self):
        self.assertEqual(
            capture_device({"device_id": 12, "device_binding": {"runtime_ordinal": 1, "chip_logic_id": 12}}), "npu:1"
        )
        self.assertEqual(capture_device({"device_id": 7}), "npu:7")
        self.assertEqual(capture_device({"device_id": 2, "device_binding": {"runtime_ordinal": 2}}), "npu:2")

    def test_wrong_a3_allocation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid device binding"):
            capture_device({"device_id": 12, "device_binding": {"runtime_ordinal": 1, "chip_logic_id": 8}})


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires a POSIX Bash environment")
class TestEagleSplitLaunchers(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.output = self.root / "output"
        self.trace = self.root / "trace"
        self.env = {
            **os.environ,
            "PATH": f"{self.root}:{os.environ['PATH']}",
            "PYTHON": "python3",
            "PYTHONPATH": str(self.root),
            "ASCEND_COMPUTE_UNIT": "ascend910_93",
            "ASCEND_RT_VISIBLE_DEVICES": "8,12",
            "TRACE": str(self.trace),
        }
        self.env.pop("EAGLE_TEST_DEVICE_RECORD", None)
        self.env.pop("FAIL_CASE", None)
        # The real a3_device.py reads the mocked runtime and SMI topology.
        (self.root / "torch_npu.py").write_text(
            "from types import SimpleNamespace\n"
            "_C = SimpleNamespace(_npu_get_soc_version=lambda: 250)\n"
            "npu = SimpleNamespace(device_count=lambda: 2, set_device=lambda ordinal: None, get_soc_version=lambda: 250)\n",
            encoding="utf-8",
        )
        # The two unmasked runtime ordinals are in the reverse order of the SMI table.
        (self.root / "sitecustomize.py").write_text(
            """import ctypes
from types import SimpleNamespace
load_library = ctypes.CDLL
def get_physical_id(ordinal, physical):
    ctypes.cast(physical, ctypes.POINTER(ctypes.c_int32))[0] = (42, 10)[ordinal]
    return 0
def load(name, *args, **kwargs):
    if name in ("libascendcl.so", "libacl_rt.so"):
        return SimpleNamespace(aclrtGetPhyDevIdByUserDevId=get_physical_id)
    return load_library(name, *args, **kwargs)
ctypes.CDLL = load
""",
            encoding="utf-8",
        )
        self.executable(
            "npu-smi",
            """#!/bin/sh
if [ "$2" = "-m" ]; then
    printf 'NPU ID Chip ID Chip Logic ID\n4 0 8\n6 0 12\n'
elif [ "$3" = "phyid-remap" ]; then
    case "$5" in
        42) logic=12; card=6 ;;
        10) logic=8; card=4 ;;
        *) exit 1 ;;
    esac
    printf 'Chip Physical ID: %s\nChip Logic ID: %s\nNPU ID: %s\nChip ID: 0\n' "$5" "$logic" "$card"
else
    echo 'Chip Name: Ascend910C'
fi
""",
        )
        self.executable("msprof", '#!/bin/sh\necho profiler >> "$TRACE"\nexit 77\n')
        # Stop at the selected application for Event timing; its measurements need hardware.
        self.executable(
            "python3",
            f"#!{sys.executable}\n"
            + """import json, os, sys
args = sys.argv[1:]
if "-m" in args and args[args.index("-m") + 1].endswith((".check_eagle_split_norm_rope_concat_prod_accuracy",
                                                         ".profile_eagle_split_norm_rope_concat_layer")):
    record = json.loads(os.environ["EAGLE_TEST_DEVICE_RECORD"])
    case = args[args.index("--case") + 1]
    device = args[args.index("--device-id") + 1]
    with open(os.environ["TRACE"], "a") as trace:
        print(device, case, os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "unset"), record["runtime_ordinal"], file=trace)
    if "--iterations" in args:
        sys.exit(42)
    sys.exit(9 if os.environ.get("FAIL_CASE") == device + "/" + case else 0)
os.execv(sys.executable, [sys.executable, *args])
""",
        )

    def executable(self, name, contents):
        path = self.root / name
        path.write_text(contents, encoding="utf-8")
        path.chmod(0o700)

    def launch(self, kind, *args):
        script = SCRIPTS / f"run_eagle_split_norm_rope_concat_{kind}.sh"
        return subprocess.run(
            ["bash", str(script), "--output", str(self.output), *args],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )

    def test_mask_rejected_before_fused_and_composed_profiling(self):
        for path in ("fused", "composed"):
            with self.subTest(path=path):
                result = self.launch("msprof_op", "--path", path, "--device-id", "12", "--runs", "1")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("ASCEND_RT_VISIBLE_DEVICES", result.stdout)
                self.assertFalse(self.trace.exists(), result.stdout)

    def test_event_only_retains_mask_and_allocated_die(self):
        result = self.launch("msprof_op", "--path", "composed", "--device-id", "12", "--runs", "1", "--no-msprof")
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertEqual(self.trace.read_text().splitlines(), ["12 half-bsnd 8,12 1"])
        record = json.loads((self.output / "device.json").read_text())
        self.assertEqual((record["chip_logic_id"], record["runtime_ordinal"]), (12, 1))

    def test_unmasked_sparse_driver_id_reaches_profiler(self):
        del self.env["ASCEND_RT_VISIBLE_DEVICES"]
        result = self.launch("msprof_op", "--path", "fused", "--device-id", "12", "--runs", "1")
        self.assertEqual(result.returncode, 77, result.stdout)
        self.assertEqual(self.trace.read_text().splitlines(), ["profiler"])
        record = json.loads((self.output / "device.json").read_text())
        self.assertEqual((record["chip_logic_id"], record["runtime_ordinal"], record["npu_id"]), (12, 0, 6))

    def test_stale_build_target_does_not_auto_select_on_a3(self):
        self.env["ASCEND_COMPUTE_UNIT"] = "ascend910b"
        self.env.pop("DEVICE_ID", None)
        del self.env["ASCEND_RT_VISIBLE_DEVICES"]
        result = self.launch("msprof_op", "--path", "fused", "--runs", "1")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("A3 requires --device-id", result.stdout)
        self.assertFalse((self.output / "npu_select.log").exists())

    def test_unmasked_sparse_accuracy_uses_allocated_die(self):
        del self.env["ASCEND_RT_VISIBLE_DEVICES"]
        result = self.launch("prod_accuracy", "--device-id", "8", "--mode", "accuracy")
        self.assertEqual(result.returncode, 0, result.stdout)
        rows = [line.split() for line in self.trace.read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual({(device, mask, ordinal) for device, _, mask, ordinal in rows}, {("8", "unset", "1")})

    def test_failed_shard_reports_failed_and_unrun_cases(self):
        self.env["FAIL_CASE"] = "8/half-bsnd"
        result = self.launch("prod_accuracy", "--device-id", "8", "12", "--mode", "accuracy")
        self.assertNotEqual(result.returncode, 0)
        log = (self.output / "accuracy.log").read_text()
        self.assertNotIn("Selected tests passed", log)
        self.assertIn("Selected tests failed", log)
        self.assertIn("FAIL device=8 mode=accuracy case=half-bsnd distribution=uniform", log)
        self.assertIn("NOT RUN device=8 mode=accuracy case=interleave-bsnd-r128", log)
        self.assertIn("PASS device=12 mode=accuracy case=half-bnsd-r128", log)
        self.assertEqual(log.count("NOT RUN device=8 "), 3)
        self.assertEqual(log.count("PASS device=12 "), 4)
        self.assertEqual(len(self.trace.read_text().splitlines()), 5)

    def test_successful_shards_share_one_verdict(self):
        result = self.launch("prod_accuracy", "--device-id", "8", "12", "--mode", "all", "--stress-runs", "1")
        self.assertEqual(result.returncode, 0, result.stdout)
        log = (self.output / "accuracy.log").read_text()
        self.assertEqual(log.count("Selected tests passed"), 1)
        self.assertEqual(len(self.trace.read_text().splitlines()), 16)
