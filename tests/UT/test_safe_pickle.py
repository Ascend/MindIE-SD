#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for the restricted SafeUnpickler used by share_memory.

NPU-free: only exercises pickle (de)serialization logic, so these run on CPU
and serve as the regression guard for the security hardening.
"""

import io
import os
import pickle
import unittest

from mindiesd.utils.safe_pickle import (
    SafeUnpickler,
    safe_dumps,
    safe_load,
    safe_loads,
)


def _rce_payload(reduce_target, arg):
    """Return a pickle stream whose REDUCE calls ``reduce_target(arg)``.

    This is the exact shape of a malicious payload an attacker would send over
    the PUB socket; loading it with plain ``pickle.loads`` would execute it.
    """

    class _Bomb:
        def __reduce__(self):
            return (reduce_target, (arg,))

    return pickle.dumps(_Bomb())


# Pure-stdlib (de)serialization logic: CPU-compatible, so skip only in NPU-only mode.
@unittest.skipIf(
    os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU."
)
class TestSafeUnpickler(unittest.TestCase):
    def test_roundtrip_plain_builtin_tuple(self):
        # An NPU share handle looks like a tuple of plain ints/bytes.
        handle = (0, 8192, b"\x01\x02\x03\x04", 16, 0)
        self.assertEqual(safe_loads(safe_dumps(handle)), handle)

    def test_safe_load_from_file_object(self):
        blob = safe_dumps({"a": 1, "b": [2, 3]})
        self.assertEqual(safe_load(io.BytesIO(blob)), {"a": 1, "b": [2, 3]})

    def test_safe_loads_accepts_bytearray_and_memoryview(self):
        blob = safe_dumps(42)
        self.assertEqual(safe_loads(bytearray(blob)), 42)
        self.assertEqual(safe_loads(memoryview(blob)), 42)

    def test_dumps_returns_bytes(self):
        self.assertIsInstance(safe_dumps("hello"), bytes)

    def test_blocks_os_system(self):
        # The classic RCE gadget; must be neutralized at find_class time.
        with self.assertRaises(RuntimeError):
            safe_loads(_rce_payload(os.system, "echo should-not-run"))

    def test_blocks_eval(self):
        with self.assertRaises(RuntimeError):
            safe_loads(_rce_payload(eval, "1+1"))  # noqa: S307 - test fixture

    def test_blocks_subprocess_popen(self):
        with self.assertRaises(RuntimeError):
            import subprocess

            safe_loads(_rce_payload(subprocess.Popen, ["echo", "x"]))

    def test_find_class_allows_builtin(self):
        u = SafeUnpickler(io.BytesIO(b""))
        self.assertIs(u.find_class("builtins", "dict"), dict)
        self.assertIs(u.find_class("builtins", "tuple"), tuple)

    def test_find_class_rejects_unknown_module(self):
        # numpy / arbitrary modules must NOT be reconstructable.
        u = SafeUnpickler(io.BytesIO(b""))
        with self.assertRaises(RuntimeError):
            u.find_class("numpy", "array")

    def test_deny_overrides_allow(self):
        # builtins.* is allowlisted, but builtins.eval is explicitly denied.
        u = SafeUnpickler(io.BytesIO(b""))
        with self.assertRaises(RuntimeError):
            u.find_class("builtins", "eval")
        with self.assertRaises(RuntimeError):
            u.find_class("builtins", "exec")

    def test_fixture_is_genuine_rce_under_raw_pickle(self):
        # Sanity: the same payload that SafeUnpickler blocks would succeed
        # under raw pickle.loads, proving the guard is what neutralizes it.
        # Use eval (harmless arg) so the raw-pickle branch has no side effects.
        payload = _rce_payload(eval, "1+1")  # noqa: S307 - test fixture
        self.assertEqual(pickle.loads(payload), 2)  # raw pickle executes it


if __name__ == "__main__":
    unittest.main(verbosity=2)
