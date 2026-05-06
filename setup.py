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
import sys
import logging
import runpy
import subprocess
import shutil
from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

os.environ["SOURCE_DATE_EPOCH"] = "315532800"
VERSION_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "version.py")


def get_mindiesd_version():
    version_ns = runpy.run_path(VERSION_FILE)
    version = version_ns.get("__version__")
    if not version:
        raise RuntimeError(f"Failed to get version from {VERSION_FILE}")

    logging.info(f"Build version is: {version}")
    return version


def get_python_version():
    """获取 Python 版本字符串，如 py310"""
    try:
        major = sys.version_info.major
        minor = sys.version_info.minor

        if major is None or minor is None:
            raise RuntimeError("Cannot get Python version: version info is None")

        python_version = f"py{major}{minor}"
        logging.info(f"Python version is: {python_version}")
        return python_version
    except Exception as e:
        logging.error(f"Failed to get Python version: {e}")
        raise RuntimeError("Cannot get Python version. Please ensure Python is properly installed.") from e


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
        open(init_file, 'a').close()
    else:
        os.remove(init_file)
        open(init_file, 'a').close()


def run_script(script_path, args=None, cwd=None):
    """执行 shell 脚本"""
    cmd = ['bash', script_path]
    if args:
        cmd.extend(args)

    logging.info(f">>> Running script: {' '.join(cmd)}")
    try:
        subprocess.check_call(
            cmd,
            cwd=cwd,
            stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        logging.error(f"Script failed with return code {e.returncode}")
        raise RuntimeError(f"Script execution failed: {script_path}") from e


def clean_build_dirs(build_dir):
    """清理构建目录"""
    dirs_to_remove = [
        os.path.join(build_dir, "bdist.linux-aarch64"),
        os.path.join(build_dir, "bdist.linux-x86_64"),
        os.path.join(build_dir, "custom_project_tik"),
        os.path.join(build_dir, "lib"),
        os.path.join(build_dir, "output"),
    ]

    logging.info("About to delete the following build-related directories:")
    for dir_path in dirs_to_remove:
        logging.info(f"  - {dir_path}")

    for dir_path in dirs_to_remove:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
        else:
            logging.info(f"Directory does not exist, skipping: {dir_path}")


class CustomBuildPy(_build_py):
    def run(self):
        proj_root = os.path.abspath(os.getcwd())
        build_dir = os.path.join(proj_root, 'build')

        logging.info("=" * 60)
        logging.info("Starting MindIE-SD Build Process")
        logging.info(f"Project root: {proj_root}")
        logging.info(f"Build directory: {build_dir}")
        logging.info("=" * 60)

        get_python_version()

        for script in os.listdir(build_dir):
            script_path = os.path.join(build_dir, script)
            if os.path.isfile(script_path):
                os.chmod(script_path, 0o444)

        try:
            ops_dir = os.path.join(proj_root, 'csrc', 'ops')
            if os.path.isdir(ops_dir):
                logging.info("=" * 60)
                logging.info("Building Ascend operators...")
                logging.info("=" * 60)
                build_ops_script = os.path.join(build_dir, 'build_ops.sh')
                run_script(build_ops_script, args=[build_dir], cwd=build_dir)
            else:
                logging.warning(f"The path of custom op operators {ops_dir} does not exist.")

            plugin_dir = os.path.join(proj_root, 'csrc', 'plugin')
            if os.path.isdir(plugin_dir):
                logging.info("=" * 60)
                logging.info("Building PyTorch plugins...")
                logging.info("=" * 60)
                build_plugin_script = os.path.join(build_dir, 'build_plugin.sh')
                run_script(build_plugin_script, args=[build_dir], cwd=build_dir)
            else:
                logging.warning(f"The path of op plugins {plugin_dir} does not exist.")

            clean_build_dirs(build_dir)

            source_dir = os.path.join(build_dir, 'build')
            destination_dir = os.path.join(proj_root, 'mindiesd', 'plugin')
            copy_so_files(source_dir, destination_dir)

            logging.info("=" * 60)
            logging.info("Build completed successfully!")
            logging.info("=" * 60)

        except Exception as e:
            logging.error(f"Build failed: {e}")
            raise

        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
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
