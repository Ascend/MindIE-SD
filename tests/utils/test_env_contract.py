#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of the Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import unittest

from mindiesd.utils.env import EnvVar, check_string_valid


class TestEnvContract(unittest.TestCase):
    @staticmethod
    def make_env(**changes):
        values = dict(
            mindie_log_level="info",
            mindie_log_stdout="true",
            mindie_log_to_file="false",
            mindie_log_path="/tmp/mindie",
            mindie_log_verbose="false",
            mindie_log_rotate="-s 30 -fs 20 -r 10",
        )
        values.update(changes)
        return EnvVar(**values)

    def test_component_settings_override_global_values(self):
        env = self.make_env(
            mindie_log_level="debug;sd:warn",
            mindie_log_stdout="true;sd:false",
            mindie_log_to_file="false;sd:true",
            mindie_log_verbose="true;sd:0",
            mindie_log_path="/tmp/global;sd:/tmp/component",
            mindie_log_rotate="-s 30;sd:-s weekly -fs 42 -r 8",
        )
        self.assertEqual(env.component_log_level, "warn")
        self.assertEqual(env.component_log_stdout, "false")
        self.assertEqual(env.component_log_to_file, "true")
        self.assertEqual(env.component_log_verbose, "0")
        self.assertEqual(env.component_log_path, "/tmp/component")
        self.assertEqual((env.rotate_cycle, env.rotate_cycle_num), ("weekly", 1))
        self.assertEqual((env.rotate_max_file_size, env.rotate_max_file_num), (42, 8))

    def test_rotation_accepts_lower_and_upper_limits(self):
        for days, size, count in [(1, 1, 1), (180, 500, 64)]:
            with self.subTest(days=days, size=size, count=count):
                env = self.make_env(mindie_log_rotate=f"-s {days} -fs {size} -r {count}")
                self.assertEqual(
                    (env.rotate_cycle_num, env.rotate_max_file_size, env.rotate_max_file_num), (days, size, count)
                )

    def test_rotation_rejects_out_of_range_values(self):
        for value in ["-s 0", "-s 181", "-fs 0", "-fs 501", "-r 0", "-r 65"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_env(mindie_log_rotate=value)

    def test_named_rotation_cycles(self):
        for cycle in ["daily", "weekly", "monthly", "yearly"]:
            with self.subTest(cycle=cycle):
                env = self.make_env(mindie_log_rotate=f"-s {cycle}")
                self.assertEqual((env.rotate_cycle, env.rotate_cycle_num), (cycle, 1))

    def test_all_log_fields_reject_long_or_control_character_values(self):
        fields = [
            "mindie_log_level",
            "mindie_log_stdout",
            "mindie_log_to_file",
            "mindie_log_path",
            "mindie_log_verbose",
            "mindie_log_rotate",
        ]
        for field in fields:
            for value in ["x" * 257, "info\n"]:
                with self.subTest(field=field, value_length=len(value)), self.assertRaises(ValueError):
                    self.make_env(**{field: value})

    def test_string_validator_accepts_printable_unicode_and_rejects_controls(self):
        self.assertTrue(check_string_valid("/tmp/日志-42"))
        for control in ["\f", "\r", "\b", "\t", "\v", "\n", "\x7f"]:
            with self.subTest(control=repr(control)):
                self.assertFalse(check_string_valid("valid" + control))

    def test_null_log_level_disables_logging(self):
        for value in ["null", "debug;sd:null"]:
            with self.subTest(value=value):
                self.assertTrue(self.make_env(mindie_log_level=value).disable_log)


if __name__ == "__main__":
    unittest.main()
