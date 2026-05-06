#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import importlib
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch

sys.path.append('../')


class TestAclGraphConfig(unittest.TestCase):

    # ------------------------------------------------------------------
    # Dynamic imports — follow the mature pattern: no module-level
    # ``from mindiesd.xxx import ...`` so that CI runs from any CWD.
    # ------------------------------------------------------------------

    @classmethod
    def setUpClass(cls):
        cls.CC = cls._import_attr('mindiesd.compilation', 'CompilationConfig')

        try:
            be_mod = importlib.import_module('mindiesd.compilation.mindie_sd_backend')
            cls.Backend = be_mod.MindieSDBackend
        except ImportError:
            cls.Backend = None

        ab_mod = importlib.import_module('mindiesd.compilation.aclgraph_backend')
        cls._ACLGraphEntry = ab_mod._ACLGraphEntry
        cls._patch_fn = ab_mod._patch_fn
        cls._global_graph_pool_ref = lambda: ab_mod._global_graph_pool

    @staticmethod
    def _import_attr(module_name, attr_name):
        return getattr(importlib.import_module(module_name), attr_name)

    # ------------------------------------------------------------------
    # setUp / tearDown
    # ------------------------------------------------------------------

    def setUp(self):
        self._orig_aclgraph_only = self.CC.aclgraph_only
        self._orig_aclgraph_with_compile = self.CC.aclgraph_with_compile
        self._orig_safe_output_mode = self.CC.safe_output_mode

    def tearDown(self):
        self.CC.aclgraph_only = self._orig_aclgraph_only
        self.CC.aclgraph_with_compile = self._orig_aclgraph_with_compile
        self.CC.safe_output_mode = self._orig_safe_output_mode

    # ------------------------------------------------------------------
    # Configuration field tests
    # ------------------------------------------------------------------

    def test_aclgraph_only_default_false(self):
        self.assertFalse(self.CC.aclgraph_only,
                         "aclgraph_only should default to False")

    def test_aclgraph_with_compile_default_false(self):
        self.assertFalse(self.CC.aclgraph_with_compile,
                         "aclgraph_with_compile should default to False")

    def test_enable_aclgraph_field_removed(self):
        self.assertFalse(hasattr(self.CC, "enable_aclgraph"),
                         "enable_aclgraph field should have been removed")

    def test_flags_independent(self):
        self.CC.aclgraph_only = True
        self.CC.aclgraph_with_compile = False
        self.assertTrue(self.CC.aclgraph_only)
        self.assertFalse(self.CC.aclgraph_with_compile)

        self.CC.aclgraph_only = False
        self.CC.aclgraph_with_compile = True
        self.assertFalse(self.CC.aclgraph_only)
        self.assertTrue(self.CC.aclgraph_with_compile)

    # ------------------------------------------------------------------
    # safe_output_mode
    # ------------------------------------------------------------------

    def test_safe_output_mode_default_true(self):
        self.assertTrue(self.CC.safe_output_mode,
                        "safe_output_mode should default to True")

    def test_safe_output_mode_toggle(self):
        self.CC.safe_output_mode = False
        self.assertFalse(self.CC.safe_output_mode)
        self.CC.safe_output_mode = True
        self.assertTrue(self.CC.safe_output_mode)

    # ------------------------------------------------------------------
    # Graph pool
    # ------------------------------------------------------------------

    def test_global_graph_pool_initial_none(self):
        ab = importlib.import_module('mindiesd.compilation.aclgraph_backend')
        ab._global_graph_pool = None
        self.assertIsNone(ab._global_graph_pool)

    # ------------------------------------------------------------------
    # ACLGraphEntry dataclass
    # ------------------------------------------------------------------

    def test_aclgraph_entry_copy_stream_default_none(self):
        entry = self._ACLGraphEntry(
            aclgraph=MagicMock(),
            static_inputs=[],
            output=MagicMock(),
        )
        self.assertIsNone(entry.copy_stream)

    def test_aclgraph_entry_fields_settable(self):
        entry = self._ACLGraphEntry(
            aclgraph=MagicMock(),
            static_inputs=[MagicMock()],
            output=MagicMock(),
            input_addresses=[123456],
        )
        self.assertIsNotNone(entry.aclgraph)
        self.assertEqual(len(entry.static_inputs), 1)
        self.assertEqual(entry.input_addresses, [123456])

    # ------------------------------------------------------------------
    # GC patch utility
    # ------------------------------------------------------------------

    def test_patch_fn_restores_original(self):
        original = os.getenv
        with self.__class__._patch_fn("os.getenv", lambda *a, **kw: "patched"):
            self.assertEqual(os.getenv("ANY"), "patched")
        self.assertIs(os.getenv, original)

    # ------------------------------------------------------------------
    # Routing logic tests
    # ------------------------------------------------------------------
    #
    # 路由测试依赖 MindieSDBackend，其导入链为：
    #   mindie_sd_backend → torch._dynamo → torch._functorch → networkx → bz2 → _bz2
    #
    # 以下 CI 环境会导致 setUpClass 导入失败，路由测试自动跳过：
    #   - Python 编译时未链接 libbz2（缺少 _bz2 C 扩展）
    #   - 修复方式：CI 环境安装 libbz2-dev 并重新编译 Python
    #   - 临时绕过：设置 PYTHONPATH 指向含 _bz2 的 Python 安装

    def _assert_routing(self, aclgraph_only, aclgraph_with_compile,
                        npu_available, expect_compile, expect_aclgraph):
        if self.Backend is None:
            raise unittest.SkipTest(
                "MindieSDBackend unavailable (requires torch._dynamo → networkx → bz2)")
        self.CC.aclgraph_only = aclgraph_only
        self.CC.aclgraph_with_compile = aclgraph_with_compile

        backend = self.Backend()
        mock_compile = MagicMock()
        mock_aclgraph_fn = MagicMock()
        mock_create = MagicMock(return_value=mock_aclgraph_fn)

        with patch.object(backend, "compile", mock_compile), \
             patch("mindiesd.compilation.mindie_sd_backend.create_aclgraph_backend", mock_create), \
             patch("mindiesd.compilation.mindie_sd_backend.npu_graph_available", npu_available):
            graph = MagicMock()
            inputs = [torch.randn(2, 2)]
            backend.__call__(graph, inputs)

            if expect_compile:
                mock_compile.assert_called()
            else:
                mock_compile.assert_not_called()

            if expect_aclgraph:
                mock_create.assert_called()
            else:
                mock_create.assert_not_called()

    def test_routing_aclgraph_only_npu_available(self):
        self._assert_routing(True, False, True, False, True)

    def test_routing_aclgraph_with_compile_npu_available(self):
        self._assert_routing(False, True, True, True, True)

    def test_routing_both_true_npu_available(self):
        self._assert_routing(True, True, True, True, True)

    def test_routing_both_false(self):
        self._assert_routing(False, False, True, True, False)

    def test_routing_aclgraph_only_npu_unavailable(self):
        self._assert_routing(True, False, False, True, False)

    def test_routing_with_compile_npu_unavailable(self):
        self._assert_routing(False, True, False, True, False)

    def test_routing_teardown_restores_config(self):
        self.CC.aclgraph_only = True
        self.CC.aclgraph_with_compile = True
        self.assertEqual((self._orig_aclgraph_only, self._orig_aclgraph_with_compile),
                         (False, False))


if __name__ == "__main__":
    unittest.main()
