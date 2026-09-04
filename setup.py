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
from setuptools.dist import Distribution
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel  # pylint: disable=no-name-in-module

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

os.environ["SOURCE_DATE_EPOCH"] = "315532800"
VERSION_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), "version.py")
WHEEL_MODE_ENV = "MINDIESD_WHEEL_MODE"
MULTI_TORCH_PLUGIN_DIR_ENV = "MINDIESD_MULTI_TORCH_PLUGIN_DIR"
SKIP_OPS_BUILD_ENV = "MINDIESD_SKIP_OPS_BUILD"
SKIP_ALL_OPS_PLUGIN_BUILD_ENV = "SKIP_ALL_OPS_PLUGIN_BUILD"
FIXED_WHEEL_MODE = "fixed"
MULTI_TORCH_WHEEL_MODE = "multi_torch"
SUPPORTED_TORCH_PLUGIN_VARIANTS = ("torch26", "torch27", "torch28", "torch29", "torch210")


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


def get_wheel_mode():
    mode = os.environ.get(WHEEL_MODE_ENV, FIXED_WHEEL_MODE).strip().lower()
    if mode not in (FIXED_WHEEL_MODE, MULTI_TORCH_WHEEL_MODE):
        raise RuntimeError(
            f"Unsupported {WHEEL_MODE_ENV}={mode}. Expected one of: {FIXED_WHEEL_MODE}, {MULTI_TORCH_WHEEL_MODE}."
        )

    logging.info("Wheel build mode is: %s", mode)
    return mode


def is_env_enabled(env_name):
    return os.environ.get(env_name, "").strip().lower() in ("1", "true", "yes", "on")


def find_usable_plugin_artifact(build_dir, proj_root):
    """Return the first cached plugin that can be loaded by the current PyTorch."""
    candidates = [
        os.path.join(build_dir, "plugin_build", "libPTAExtensionOPS.so"),
        os.path.join(proj_root, "mindiesd", "plugin", "libPTAExtensionOPS.so"),
    ]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, torch; torch.ops.load_library(sys.argv[1])",
                candidate,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
        logging.warning(
            "Cached plugin artifact %s cannot be loaded by the current PyTorch; "
            "falling back to plugin rebuild. Loader error: %s",
            candidate,
            (result.stderr or result.stdout).strip(),
        )
    return None


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


def copy_multi_torch_plugin_files(proj_root):
    src_root = os.environ.get(
        MULTI_TORCH_PLUGIN_DIR_ENV,
        os.path.join(proj_root, "build", "torch_plugin_variants"),
    )
    dest_root = os.path.join(proj_root, "mindiesd", "plugin")

    logging.info("Using multi torch plugin source directory: %s", src_root)
    missing_variants = []
    for variant in SUPPORTED_TORCH_PLUGIN_VARIANTS:
        variant_src_dir = os.path.join(src_root, variant)
        variant_dest_dir = os.path.join(dest_root, variant)
        so_file = os.path.join(variant_src_dir, "libPTAExtensionOPS.so")
        if not os.path.isfile(so_file):
            missing_variants.append(variant)
            continue

        copy_so_files(variant_src_dir, variant_dest_dir)

    if missing_variants:
        raise RuntimeError(
            "Missing multi torch plugin .so files for variants: %s. "
            "Expected files under %s/<variant>/libPTAExtensionOPS.so." % (", ".join(missing_variants), src_root)
        )


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


def strip_clang_tidy_unsupported_flags(proj_root):
    """删除 compile_commands.json 中 clang-tidy 无法识别的编译选项。

    clang-tidy 用 compile_commands.json 里记录的编译选项驱动 clang 前端。
    CMakeLists 中通过 ``CMAKE_CXX_FLAGS`` 注入的 GCC 专属选项（如
    ``-fabi-version=11``）clang 无法识别，会导致 clang-tidy 报
    "unknown argument" 而失败。这里在合并生成 compile_commands.json 之后把
    这类选项剥离掉，使其可直接用于 clang-tidy。
    """
    import json
    import re

    compile_commands_path = os.path.join(proj_root, "compile_commands.json")
    if not os.path.isfile(compile_commands_path):
        logging.info("Skipping clang-tidy flag stripping: %s not found", compile_commands_path)
        return

    try:
        with open(compile_commands_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning("Failed to read %s: %s", compile_commands_path, e)
        return

    if not isinstance(entries, list):
        logging.warning("Unexpected format in %s, expected list", compile_commands_path)
        return

    # clang 前端无法识别的 GCC 选项。正则同时吞掉前面的空白，保证清理后命令仍规整。
    unsupported_flags = [
        re.compile(r"\s*-fabi-version(?:=\d+|\s+\d+)"),
    ]

    modified = 0
    for entry in entries:
        command = entry.get("command")
        if not isinstance(command, str):
            continue
        cleaned = command
        for pattern in unsupported_flags:
            cleaned = pattern.sub("", cleaned)
        cleaned = cleaned.strip()
        if cleaned != command:
            entry["command"] = cleaned
            modified += 1

    with open(compile_commands_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    logging.info(
        "Stripped unsupported clang-tidy flags from %s (%s/%s entries modified)",
        compile_commands_path,
        modified,
        len(entries),
    )


class CustomBuildPy(_build_py):
    def run(self):
        proj_root = os.path.abspath(os.getcwd())
        build_dir = os.path.join(proj_root, 'build')
        wheel_mode = get_wheel_mode()

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
            if is_env_enabled(SKIP_OPS_BUILD_ENV):
                logging.info("Skipping Ascend operators build because %s is enabled.", SKIP_OPS_BUILD_ENV)
            elif os.path.isdir(ops_dir):
                logging.info("%s", "=" * 60)
                logging.info("Building Ascend operators...")
                logging.info("%s", "=" * 60)
                build_ops_script = os.path.join(build_dir, 'build_ops.sh')
                run_script(build_ops_script, args=[build_dir], cwd=build_dir)
            else:
                logging.warning("The path of custom op operators %s does not exist.", ops_dir)

            if wheel_mode == FIXED_WHEEL_MODE:
                plugin_dir = os.path.join(proj_root, 'csrc', 'plugin')
                if os.path.isdir(plugin_dir):
                    cached_plugin = None
                    if is_env_enabled(SKIP_ALL_OPS_PLUGIN_BUILD_ENV):
                        cached_plugin = find_usable_plugin_artifact(build_dir, proj_root)
                    if cached_plugin:
                        logging.info(
                            "%s=1: skip plugin build, reuse loadable plugin artifact %s",
                            SKIP_ALL_OPS_PLUGIN_BUILD_ENV,
                            cached_plugin,
                        )
                    else:
                        logging.info("%s", "=" * 60)
                        logging.info("Building PyTorch plugins...")
                        logging.info("%s", "=" * 60)
                        build_plugin_script = os.path.join(build_dir, 'build_plugin.sh')
                        run_script(build_plugin_script, args=[build_dir], cwd=build_dir)
                else:
                    logging.warning("The path of op plugins %s does not exist.", plugin_dir)
            else:
                logging.info("%s", "=" * 60)
                logging.info("Packaging prebuilt PyTorch plugin variants...")
                logging.info("%s", "=" * 60)
                copy_multi_torch_plugin_files(proj_root)

            merge_compile_commands(proj_root, build_dir)

            strip_clang_tidy_unsupported_flags(proj_root)

            if wheel_mode == FIXED_WHEEL_MODE:
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


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


if __name__ == "__main__":
    requirements = ["torch", "torch_npu"]
    mindie_sd_version = get_mindiesd_version()
    build_wheel_mode = get_wheel_mode()
    ensure_plugin_init()
    package_data = {"mindiesd": ["ops/**/*"]}
    if build_wheel_mode == MULTI_TORCH_WHEEL_MODE:
        package_data["mindiesd"].append("plugin/**/*.so")
    else:
        package_data["mindiesd"].append("plugin/*.so")

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
        package_data=package_data,
        cmdclass={"build_py": CustomBuildPy, "bdist_wheel": BDistWheel},
        distclass=BinaryDistribution,
    )
