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
import unittest
import torch
from unittest.mock import patch, MagicMock, Mock, ANY
from packaging.version import Version

from mindiesd.layers import register_ops

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mindiesd.compilation import MindieSDBackend


class TestRegisterOps(unittest.TestCase):
    
    def setUp(self):
        self.torch_version = Version(torch.__version__.split("+")[0])
        self.test_op_base = "test_op_mindie_sd_"
        self.register_func_path = self._get_register_func_path()

    def _get_register_func_path(self):
        if self.torch_version >= Version("2.2"):
            return 'mindiesd.layers.register_ops._native_register_fake'
        else:
            return 'mindiesd.layers.register_ops._lib.impl'

    def test_check_mindie_operator_exists_existing(self):
        result = register_ops.check_mindie_operator_exists("rope")
        self.assertTrue(result)
    
    def test_check_mindie_operator_exists_nonexistent(self):
        result = register_ops.check_mindie_operator_exists(f"{self.test_op_base}nonexistent")
        self.assertFalse(result)
    
    @patch('mindiesd.layers.register_ops.check_mindie_operator_exists')
    def test_register_mindie_fake_op_decorator(self, mock_check):
        test_op_name = f"{self.test_op_base}decorator"
        mock_check.return_value = True
        
        @register_ops.register_mindie_fake_op(test_op_name)
        def test_fake_func(x, cos, sin, mode):
            return torch.empty_like(x)
        
        self.assertTrue(callable(test_fake_func))
        mock_check.assert_called_once_with(test_op_name)
    
    def test_register_mindie_fake_op_nonexistent_op(self):
        test_op_name = f"{self.test_op_base}nonexistent_decorator"
        with self.assertRaises(RuntimeError):
            @register_ops.register_mindie_fake_op(test_op_name)
            def test_fake_func(x):
                return torch.empty_like(x)
    
    @patch('torch.__version__', new='2.1.0')
    def test_pytorch_21_compatibility(self):
        with patch('torch.library.Library') as mock_library:
            mock_lib_instance = MagicMock()
            mock_library.return_value = mock_lib_instance
            
            import importlib
            importlib.reload(register_ops)
            
            mock_library.assert_called_once_with("mindiesd", "IMPL")
    
    def test_compatible_register_fake_decorator(self):
        test_op_name = f"{self.test_op_base}compatible_decorator"
        with patch(self.register_func_path) as mock_register:
            mock_register.return_value = lambda f: f
            
            def test_fake(x):
                return torch.empty_like(x)
            
            decorator = register_ops._compatible_register_fake(test_op_name)
            decorated_func = decorator(test_fake)
            
            self.assertTrue(callable(decorated_func))
            self.assertEqual(decorated_func.__name__, "test_fake")
            
            if self.torch_version == Version("2.1"):
                mock_register.assert_called_with(test_op_name, ANY, "Meta")
            else:
                mock_register.assert_called_with(f"{test_op_name}")

    def test_compatible_register_fake_wrapper(self):
        test_op_name = f"{self.test_op_base}compatible_wrapper"
        with patch(self.register_func_path) as mock_register:
            mock_register.return_value = lambda f: f
            
            x = torch.randn(2, 4, device='meta')
            
            def test_fake(x):
                self.assertEqual(x.device.type, "meta")
                return torch.empty_like(x)
            
            decorator = register_ops._compatible_register_fake(test_op_name)
            decorated_func = decorator(test_fake)
            
            self.assertTrue(callable(decorated_func))
            
            result = decorated_func(x)
            self.assertEqual(result.device.type, "meta")
            self.assertEqual(result.shape, x.shape)
            
            if self.torch_version == Version("2.1"):
                mock_register.assert_called_with(test_op_name, ANY, "Meta")
            else:
                mock_register.assert_called_with(f"{test_op_name}")


if __name__ == '__main__':
    unittest.main()