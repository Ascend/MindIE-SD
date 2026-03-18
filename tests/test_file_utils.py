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

import unittest
import sys
from io import StringIO
import logging
import stat
import os
sys.path.append('../')
from mindiesd.utils import file_utils
from mindiesd.utils.logs.logging import logger

TEST_PATH = "./test_file"
TEST_PATH_LINK = './test_file/link'
TEST_PATH_INVALID = './test_file/invalid'
TEST_PATH_FILE_SIZE = "./test_file/large_file.txt"
TEST_PATH_FILENUM_PER_DIR = "./test_file/large_dir"
TEST_PATH_FILE_PERMISSION = "./test_file/permission_file.txt"
TEST_PATH_UNDER_DIR = "./test_file/test_under_dir_file"


def create_file(file_path, size=file_utils.MAX_FILE_SIZE + 1):
    if os.path.exists(file_path):
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    file = os.fdopen(os.open(file_path, flags, mode), 'w')
    file.seek(size - 1)
    file.write('\x00')
    file.close()


def create_dir(dir_path, size=file_utils.MAX_FILENUM_PER_DIR + 1):
    if os.path.exists(dir_path):
        return
    os.makedirs(dir_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = stat.S_IWUSR | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for i in range(size):
        with os.fdopen(os.open('{}/{}.txt'.format(dir_path, i), flags, mode), 'w') as f:
            f.write("\n")


@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "NPU", "Skip CPU-compatible tests when MINDIE_TEST_MODE is NPU.")
class TestFileUtils(unittest.TestCase):

    def setUp(self):
        if not os.path.exists(TEST_PATH):
            os.makedirs(TEST_PATH)
        os.chmod(TEST_PATH, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        self.path_test_none = None

        self.path_test_path_length = TEST_PATH
        for _ in range(file_utils.MAX_PATH_LENGTH):
            self.path_test_path_length += "/a"
        
        self.path_test_link = TEST_PATH_LINK
        if not os.path.exists(self.path_test_link):
            os.symlink(os.path.join(os.getcwd(), "test_file"), self.path_test_link)

        self.path_test_not_exist = TEST_PATH_INVALID

        self.path_test_file_size = TEST_PATH_FILE_SIZE
        create_file(self.path_test_file_size)
        
        self.path_test_filenum_per_dir = TEST_PATH_FILENUM_PER_DIR
        create_dir(self.path_test_filenum_per_dir)

        self.path_test_file_permission = TEST_PATH_FILE_PERMISSION
        create_file(self.path_test_file_permission, size=1)

        self.path_test_under_dir = TEST_PATH_UNDER_DIR
        create_dir(self.path_test_under_dir, size=0)
        self.path_test_under_dir_file = os.path.join(self.path_test_under_dir, "test.txt")
        create_file(self.path_test_under_dir_file, size=1)
        self.string_io = StringIO()
 
    def enable_log_capture(self):
        """修改logger中的stream为StringIO,进行日志捕获"""
        
        for handler in logger.handlers:
            if handler.__class__ is logging.StreamHandler:
                handler.stream = self.string_io

    def disable_log_capture(self):
        """回退StringIO为正常日志功能"""
        for handler in logger.handlers:
            if handler.__class__ is logging.StreamHandler:
                handler.stream = sys.stdout

    def test_standardize_path(self):
        error_nums = 0
        test_path = [
            self.path_test_none,
            self.path_test_path_length,
            self.path_test_link
        ]
        for path in test_path:
            try:
                file_utils.standardize_path(path)
            except Exception as e:
                logger.error(e)
                error_nums += 1
        self.assertEqual(error_nums, len(test_path))

        flag = True
        try:
            file_utils.standardize_path(TEST_PATH)
        except Exception as e:
            logger.error(e)
            flag = False
        self.assertTrue(flag)
    
    def test_check_file_safety(self):
        error_nums = 0
        test_path = [
            self.path_test_not_exist,
            self.path_test_filenum_per_dir,
            self.path_test_file_size,
        ]
        for path in test_path:
            try:
                file_utils.check_file_safety(path)
            except Exception as e:
                logger.error(e)
                error_nums += 1
        self.assertEqual(error_nums, len(test_path))

    def test_check_dir_safety(self):
        error_nums = 0
        test_path = [
            self.path_test_not_exist,
            self.path_test_file_size,
            self.path_test_filenum_per_dir
        ]
        for path in test_path:
            try:
                file_utils.check_dir_safety(path)
            except Exception as e:
                logger.error(e)
                error_nums += 1
        self.assertEqual(error_nums, len(test_path))

        flag = True
        try:
            file_utils.check_dir_safety(TEST_PATH)
        except Exception as e:
            logger.error(e)
            flag = False
        self.assertTrue(flag)

    """对于权限/属主等不进行强校验，出现错误进行告警，采用日志捕获的方式进行测试"""
    def test_check_max_permission(self):
        self.enable_log_capture()
        # when file permission is 0o777, larger than MAX_PERMISSION
        os.chmod(self.path_test_file_permission, 0o777)

        # then check the file permission
        with self.assertRaises(PermissionError):
            file_utils.check_max_permission(self.path_test_file_permission, file_utils.CONFIG_FILE_PERMISSION)
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)

        # when file permission is 0o440, smaller than MAX_PERMISSION
        os.chmod(self.path_test_file_permission, 0o440)
        
        file_utils.check_max_permission(self.path_test_file_permission, file_utils.CONFIG_FILE_PERMISSION)
        self.assertNotIn("WARNING", self.string_io.getvalue().rstrip('\n'))
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)
        self.disable_log_capture()

    def test_check_file_under_dir(self):
        self.enable_log_capture()
        # when the files under dir do not meeet the check requirements
        os.chmod(self.path_test_under_dir, 0o750)
        os.chmod(self.path_test_under_dir_file, 0o777)

        with self.assertRaises(PermissionError):
            file_utils.check_file_under_dir(self.path_test_under_dir)
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)

        # when the files under dir meeet the check requirements
        os.chmod(self.path_test_under_dir, 0o750)
        os.chmod(self.path_test_under_dir_file, 0o440)

        file_utils.check_file_under_dir(self.path_test_under_dir)
        self.assertNotIn("WARNING", self.string_io.getvalue().rstrip('\n'))
        # 清空StringIO
        self.string_io.truncate(0)
        self.string_io.seek(0)
        self.disable_log_capture()

if __name__ == '__main__':
    unittest.main()
