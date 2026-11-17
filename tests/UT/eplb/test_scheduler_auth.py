# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""CPU-only regression tests for scheduler authentication and real manager IPC."""

import argparse
import ast
import logging
import multiprocessing
from multiprocessing.connection import AuthenticationError
from multiprocessing.managers import BaseManager
import os
from pathlib import Path
import secrets
import sys
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]


class AuthTestManager(BaseManager):
    """Isolate the IPC test from the NPU-dependent package imports."""


def load_auth_functions():
    source = ROOT / "mindiesd/eplb/eplb_scheduler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_args", "_encode_auth_key", "start_manager_server", "get_manager_client"}
    ]
    namespace = {
        "argparse": argparse,
        "os": os,
        "secrets": secrets,
        "logger": logging.getLogger("scheduler-auth-test"),
        "multiprocessing": multiprocessing,
        "threading": threading,
        "ScheduleManager": AuthTestManager,
    }
    # Match the existing CPU-only setup.py tests: execute the real functions,
    # excluding unrelated imports that require torch_npu.
    exec(  # pylint: disable=exec-used
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace


class TestSchedulerAuth(unittest.TestCase):
    def setUp(self):
        self.auth = load_auth_functions()
        self.env_patch = mock.patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def parse_args(self, *arguments):
        with mock.patch.object(sys, "argv", ["scheduler", *arguments]):
            return self.auth["get_args"]()

    def test_unconfigured_key_is_random_and_not_logged(self):
        with self.assertLogs("scheduler-auth-test", level="WARNING") as logs:
            first = self.parse_args().auth_key
            second = self.parse_args().auth_key
        self.assertRegex(first, r"^[0-9a-f]{32}$")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, "secret_key")
        self.assertNotIn(first, "\n".join(logs.output))
        self.assertNotIn(second, "\n".join(logs.output))
        self.assertIn("EPLB_AUTH_KEY", "\n".join(logs.output))

    def test_environment_key_is_used(self):
        os.environ["EPLB_AUTH_KEY"] = "environment-credential"
        with mock.patch.object(self.auth["logger"], "warning") as warning:
            self.assertEqual(self.parse_args().auth_key, "environment-credential")
        warning.assert_not_called()

    def test_cli_key_overrides_environment(self):
        os.environ["EPLB_AUTH_KEY"] = "environment-credential"
        self.assertEqual(self.parse_args("--auth_key", "cli-credential").auth_key, "cli-credential")

    def test_cli_key_overrides_empty_environment(self):
        os.environ["EPLB_AUTH_KEY"] = ""
        self.assertEqual(self.parse_args("--auth_key", "cli-credential").auth_key, "cli-credential")

    def test_explicit_empty_keys_fail_before_startup(self):
        for empty in ("", "   ", "\t\n"):
            for source in ("cli", "env"):
                with self.subTest(empty=empty, source=source):
                    os.environ["EPLB_AUTH_KEY"] = empty if source == "env" else "environment-credential"
                    args = ("--auth_key", empty) if source == "cli" else ()
                    with mock.patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                        self.parse_args(*args)
                    self.assertEqual(error.exception.code, 2)

    def test_programmatic_empty_keys_rejected_before_manager_creation(self):
        for key in (None, "", "  "):
            for function in ("start_manager_server", "get_manager_client"):
                with self.subTest(key=key, function=function):
                    with mock.patch.dict(self.auth, {"ScheduleManager": mock.Mock()}) as namespace:
                        with self.assertRaises(ValueError):
                            namespace[function](("127.0.0.1", 0), key)
                        namespace["ScheduleManager"].assert_not_called()

    def test_matching_key_connects_and_wrong_keys_are_rejected(self):
        with mock.patch.object(sys, "argv", ["scheduler"]), mock.patch.object(self.auth["logger"], "warning"):
            key = self.auth["get_args"]().auth_key
        server_manager = self.auth["get_manager_client"](("127.0.0.1", 0), key)
        server_manager.start()
        self.addCleanup(server_manager.shutdown)
        client = self.auth["get_manager_client"](server_manager.address, key)
        client.connect()
        for wrong_key in ("secret_key", secrets.token_hex(16)):
            with self.subTest(wrong_key=wrong_key):
                unauthorized = self.auth["get_manager_client"](server_manager.address, wrong_key)
                with self.assertRaises(AuthenticationError):
                    unauthorized.connect()


if __name__ == "__main__":
    unittest.main()
