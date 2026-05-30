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
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # pylint: disable=no-name-in-module

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

os.environ["SOURCE_DATE_EPOCH"] = "315532800"
VERSION_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "version.py")


def get_mindiesd_version():
    version_ns = runpy.run_path(VERSION_FILE)
    version = version_ns.get("__version__")
    if not version:
        raise RuntimeError("Failed to get version from %s" % VERSION_FILE)

    logging.info("Build version is: %s", version)
    return version


def get_python_version():
    """获取 Python 版本字符串，如 py310"""
    try:
        major = sys.version_info.major
        minor = sys.version_info.minor

        if major is None or minor is None:
            raise RuntimeError("Cannot get Python version: version info is None")

        python_version = f"py{major}{minor}"
        logging.info("Python version is: %s", python_version)
        return python_version
    except Exception as e:
        logging.error("Failed to get Python version: %s", e)
        raise RuntimeError("Cannot get Python version. Please ensure Python is properly installed.") from e


def copy_so_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    so_files = [f for f in os.listdir(src_dir) if f.endswith('.so')]
    if not so_files:
        logging.warning("No .so files found in %s", src_dir)
        return
    for so_file in so_files:
        src_file = os.path.join(src_dir, so_file)
        dest_file = os.path.join(dest_dir, so_file)
        shutil.copy2(src_file, dest_file)
        logging.info("Copied %s to %s", src_file, dest_file)


def ensure_plugin_init():
    plugin_dir = os.path.join(os.getcwd(), 'mindiesd/plugin')
    init_file = os.path.join(plugin_dir, '__init__.py')

    os.makedirs(plugin_dir, exist_ok=True)
    with open(init_file, "a", encoding="utf-8"):
        pass


def run_script(script_path, args=None, cwd=None):
    """执行 shell 脚本"""
    cmd = ['bash', script_path]
    if args:
        cmd.extend(args)

    logging.info(">>> Running script: %s", ' '.join(cmd))
    try:
        subprocess.check_call(cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logging.error("Script failed with return code %s", e.returncode)
        raise RuntimeError("Script execution failed: %s" % script_path) from e


def merge_compile_commands(proj_root, build_dir):
    """Merge all compile_commands.json from different build stages into one."""
    import json

    sources = [
        ("AscendC ops", os.path.join(build_dir, "compile_commands_ascendc.json")),
        ("PyTorch plugin", os.path.join(build_dir, "plugin_build", "compile_commands.json")),
        ("TIK ops", os.path.join(build_dir, "compile_commands_tik.json")),
    ]

    merged = []
    seen = set()

    for stage_name, path in sources:
        if not os.path.isfile(path):
            logging.info("compile_commands.json not found for %s: %s", stage_name, path)
            continue

        try:
            with open(path, 'r', encoding="utf-8") as f:
                entries = json.load(f)
        except json.JSONDecodeError as e:
            logging.warning("Failed to parse %s: %s", path, e)
            continue

        if not isinstance(entries, list):
            logging.warning("Unexpected format in %s, expected list", path)
            continue

        added = 0
        for entry in entries:
            key = (
                entry.get("directory", ""),
                entry.get("file", ""),
                entry.get("command", ""),
            )
            if key not in seen:
                seen.add(key)
                merged.append(entry)
                added += 1

        logging.info("Merged %s entries from %s (%s total)", added, stage_name, len(entries))

    if merged:
        output_path = os.path.join(proj_root, "compile_commands.json")
        with open(output_path, 'w', encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        logging.info("Merged compile_commands.json written to %s (%s total entries)", output_path, len(merged))
    else:
        logging.info("No compile_commands.json entries found to merge")


class CustomBuildPy(_build_py):
    def run(self):
        proj_root = os.path.abspath(os.getcwd())
        build_dir = os.path.join(proj_root, 'build')

        logging.info("%s", "=" * 60)
        logging.info("Starting MindIE-SD Build Process")
        logging.info("Project root: %s", proj_root)
        logging.info("Build directory: %s", build_dir)
        logging.info("%s", "=" * 60)

        get_python_version()

        for script in os.listdir(build_dir):
            script_path = os.path.join(build_dir, script)
            if os.path.isfile(script_path):
                os.chmod(script_path, 0o444)

        try:
            ops_dir = os.path.join(proj_root, 'csrc', 'ops')
            if os.path.isdir(ops_dir):
                logging.info("%s", "=" * 60)
                logging.info("Building Ascend operators...")
                logging.info("%s", "=" * 60)
                build_ops_script = os.path.join(build_dir, 'build_ops.sh')
                run_script(build_ops_script, args=[build_dir], cwd=build_dir)
            else:
                logging.warning("The path of custom op operators %s does not exist.", ops_dir)

            plugin_dir = os.path.join(proj_root, 'csrc', 'plugin')
            if os.path.isdir(plugin_dir):
                logging.info("%s", "=" * 60)
                logging.info("Building PyTorch plugins...")
                logging.info("%s", "=" * 60)
                build_plugin_script = os.path.join(build_dir, 'build_plugin.sh')
                run_script(build_plugin_script, args=[build_dir], cwd=build_dir)
            else:
                logging.warning("The path of op plugins %s does not exist.", plugin_dir)

            merge_compile_commands(proj_root, build_dir)

            source_dir = os.path.join(build_dir, 'plugin_build')
            destination_dir = os.path.join(proj_root, 'mindiesd', 'plugin')
            copy_so_files(source_dir, destination_dir)

            logging.info("%s", "=" * 60)
            logging.info("Build completed successfully!")
            logging.info("%s", "=" * 60)

        except Exception as e:
            logging.error("Build failed: %s", e)
            raise

        super().run()


class BDistWheel(_bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        # pylint: disable=attribute-defined-outside-init
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
        package_data={"": ["*.so", "ops/**/*"]},
        cmdclass={"build_py": CustomBuildPy, "bdist_wheel": BDistWheel},
    )
