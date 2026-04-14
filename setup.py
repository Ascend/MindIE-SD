#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.

# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:

#     http://license.coscl.org.cn/MulanPSL2

# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, 
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import logging
import runpy
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

os.environ["SOURCE_DATE_EPOCH"] = "315532800"
VERSION_ENV = "MINDIE_SD_VERSION_OVERRIDE"
VERSION_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mindiesd", "_version.py")


def get_mindiesd_version():
    version = os.environ.get(VERSION_ENV, None)
    if version:
        logging.info(f"MINDIE_SD_VERSION_OVERRIDE is: {version}")
    else:
        version_ns = runpy.run_path(VERSION_FILE)
        version = version_ns.get("__version__")
        if not version:
            raise RuntimeError(f"Failed to get version from {VERSION_FILE}")
        logging.info(f"Repository version is: {version}")

    mindiesd_version = version.replace("T", "post")
    return mindiesd_version


def copy_so_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    so_files = [f for f in os.listdir(src_dir) if f.endswith('.so')]
    if not so_files:
        logging.warning(f"No .so files found in {src_dir}")
        return
    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        shutil.copy2(src_file, dest_file)
        logging.info(f"Copied {src_file} to {dest_file}")


def ensure_plugin_init():
    plugin_dir = os.path.join(os.getcwd(), 'mindiesd/plugin')
    init_file = os.path.join(plugin_dir, '__init__.py')
    
    os.makedirs(plugin_dir, exist_ok=True)   
    if not os.path.isfile(init_file):
        with open(init_file, 'w') as f:
            pass
    else:
        os.remove(init_file)
        with open(init_file, 'w') as f:
            pass


class CustomBuildPy(_build_py):
    def run(self):
        # 1. 进入 build 目录执行构建脚本
        logging.info(">>> Running build.sh to compile shared libraries...")
        build_script_path = os.path.join(os.path.abspath(os.getcwd()), 'build')
        try:
            subprocess.check_call(
                ['bash', './build.sh'], 
                cwd=build_script_path, 
                stderr=subprocess.STDOUT
                )
        except subprocess.CalledProcessError as e:
            logging.error(f"Build script failed with return code {e.returncode}")
            logging.error(f"Build script output: {e.output if hasattr(e, 'output') else 'No output captured'}")
            raise RuntimeError(f"Build script execution failed with return code {e.returncode}") from e  

        # 2. 把生成的 .so 文件移动到 Python 包目录 mindiesd/plugin
        source_dir = os.path.join(build_script_path, 'build')
        destination_dir = os.path.join(os.path.abspath(os.getcwd()), 'mindiesd', 'plugin')
        copy_so_files(source_dir, destination_dir)
        
        # 3. 继续默认 build_py 流程
        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # 标记为二进制 wheel，否则会生成 py3-none-any
        self.root_is_pure = False


if __name__ == "__main__":
    requirements = ["torch", "torch_npu"]
    mindie_sd_version = get_mindiesd_version()
    ensure_plugin_init()

    setup(
        name="mindiesd",
        version=mindie_sd_version,
        author="ascend",
        description="build wheel for mindie sd",
        setup_requires=[],
        install_requires=requirements,
        zip_safe=False,
        python_requires=">=3.10",
        include_package_data=True,
        packages=find_packages(),
        package_data={
            "": [
                "*.so",  
                "ops/**/*"
            ]
        },
        cmdclass={
            "build_py": CustomBuildPy,
            "bdist_wheel": BDistWheel
        }
    )
