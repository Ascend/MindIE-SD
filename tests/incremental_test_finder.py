#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import os
import re
import sys
import subprocess
import argparse
import glob
import ast
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass
class FileChange:
    """文件变更信息"""
    path: str
    change_type: str  # 'A'=新增, 'M'=修改, 'D'=删除, 'R'=重命名
    old_path: Optional[str] = None  # 重命名时的原路径


@dataclass
class ChangeInfo:
    """变更操作信息"""
    # 按变更类型分类的源码文件
    added_source_files: Set[str] = field(default_factory=set)    # 新增的源码文件
    modified_source_files: Set[str] = field(default_factory=set) # 修改的源码文件
    deleted_source_files: Set[str] = field(default_factory=set)  # 删除的源码文件

    # 测试文件
    deleted_test_files: Set[str] = field(default_factory=set)    # 删除的测试文件

    # 依赖分析结果
    dependency_affected_tests: Set[str] = field(default_factory=set)  # 依赖变更源码的测试

    def get_all_changed_sources(self) -> Set[str]:
        """获取所有变更的源码文件"""
        return self.added_source_files | self.modified_source_files | self.deleted_source_files


class GitChangeDetector:
    """Git变更检测器"""

    def __init__(self, repo_root: str = None, base_branch: str = "master"):
        self.repo_root = repo_root or self._find_repo_root()
        self.base_branch = base_branch

    def _find_repo_root(self) -> str:
        """查找Git仓库根目录"""
        current = os.getcwd()
        while current != '/':
            if os.path.isdir(os.path.join(current, '.git')):
                return current
            parent = os.path.dirname(current)
            if parent == current:  # 到达根目录
                break
            current = parent
        raise RuntimeError("Not in a git repository")

    def _run_git_command(self, cmd: List[str]) -> str:
        """运行Git命令"""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {' '.join(cmd)}")
            print(f"Error: {e.stderr}")
            return ""

    def _ref_exists(self, ref: str) -> bool:
        """检查Git引用是否存在"""
        cmd = ["git", "rev-parse", "--verify", ref]
        try:
            subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_changed_files(self, since_ref: str = None) -> List[FileChange]:
        """
        获取变更的文件列表

        Args:
            since_ref: 对比的Git引用（分支、tag、commit），默认为base_branch

        Returns:
            变更文件列表
        """
        ref = since_ref or self.base_branch
        changes = []

        # 检查基分支是否存在
        if not self._ref_exists(ref):
            print(f"Warning: Base ref '{ref}' not found, skipping branch diff.")
            return changes

        # 获取当前分支与基分支的差异
        diff_cmd = ["git", "diff", "--name-status", f"{ref}...HEAD"]
        output = self._run_git_command(diff_cmd)

        for line in output.split('\n'):
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) < 2:
                continue

            change_type = parts[0][0]  # A/M/D/R

            if change_type == 'R':
                # 重命名: R100\told_path\tnew_path
                if len(parts) >= 3:
                    old_path = parts[1]
                    new_path = parts[2]
                    changes.append(FileChange(new_path, change_type, old_path))
            else:
                # 其他: A/M/D\tpath
                file_path = parts[1]
                changes.append(FileChange(file_path, change_type))

        return changes

    def get_staged_files(self) -> List[FileChange]:
        """获取暂存区的变更文件"""
        changes = []
        cmd = ["git", "diff", "--cached", "--name-status"]
        output = self._run_git_command(cmd)

        for line in output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            change_type = parts[0][0]
            file_path = parts[1]
            changes.append(FileChange(file_path, change_type))

        return changes

    def get_unstaged_files(self) -> List[FileChange]:
        """获取未暂存的变更文件"""
        changes = []
        cmd = ["git", "diff", "--name-status"]
        output = self._run_git_command(cmd)

        for line in output.split('\n'):
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            change_type = parts[0][0]
            file_path = parts[1]
            changes.append(FileChange(file_path, change_type))

        return changes


class SourceDependencyAnalyzer:
    """源码依赖分析器"""

    def __init__(self, repo_root: str):
        self.repo_root = repo_root
        # 缓存模块名到文件路径的映射
        self._module_cache: Dict[str, str] = {}
        # 缓存文件依赖关系
        self._dependency_cache: Dict[str, Set[str]] = {}

    def _get_module_name(self, file_path: str) -> Optional[str]:
        """从文件路径获取模块名"""
        if not file_path.endswith('.py'):
            return None

        # 转换为模块名 (mindiesd/layers/norm.py -> mindiesd.layers.norm)
        rel_path = file_path.replace('/', '.').replace('\\', '.')
        if rel_path.endswith('.py'):
            rel_path = rel_path[:-3]
        return rel_path

    def _extract_imports(self, file_path: str) -> Set[str]:
        """从Python文件中提取导入的模块"""
        imports = set()
        full_path = os.path.join(self.repo_root, file_path)

        if not os.path.isfile(full_path):
            return imports

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    if module:
                        imports.add(module)
                        # 也添加子模块
                        parts = module.split('.')
                        for i in range(1, len(parts)):
                            imports.add('.'.join(parts[:i]))
        except (SyntaxError, UnicodeDecodeError, IOError):
            # 解析失败时返回空集合
            pass

        return imports

    def find_dependent_sources(self, changed_sources: Set[str], other_source_files: List[str]) -> Set[str]:
        """
        查找依赖变更源码的其他源码文件

        Args:
            changed_sources: 变更的源码文件集合
            other_source_files: 其他源码文件列表（排除变更的源码）

        Returns:
            依赖变更源码的文件集合
        """
        dependent_sources = set()

        # 获取变更文件的精确模块名（不包含父模块）
        changed_modules = set()
        for src in changed_sources:
            module = self._get_module_name(src)
            if module:
                changed_modules.add(module)

        if not changed_modules:
            return dependent_sources

        # 检查每个源文件是否依赖变更的模块
        for source_file in other_source_files:
            # 使用缓存或重新解析
            if source_file in self._dependency_cache:
                imports = self._dependency_cache[source_file]
            else:
                imports = self._extract_imports(source_file)
                self._dependency_cache[source_file] = imports

            # 检查是否精确导入变更的模块
            for changed_module in changed_modules:
                if changed_module in imports:
                    dependent_sources.add(source_file)
                    break

        return dependent_sources

    def get_all_source_files(self) -> List[str]:
        """获取项目中所有的Python源文件"""
        source_files = []

        # 扫描 mindiesd 目录
        mindiesd_dir = os.path.join(self.repo_root, 'mindiesd')
        if os.path.isdir(mindiesd_dir):
            for root, _, files in os.walk(mindiesd_dir):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, self.repo_root)
                        source_files.append(rel_path)

        return source_files


class TestMapper:
    """源码与测试用例映射器"""

    # 源码路径到测试路径的映射规则
    MAPPING_RULES = {
        # 精确匹配
        "exact": {
            "mindiesd/layers/flash_attn/attention_forward.py": [
                "tests/layers/flash_attn/test_attn_forward.py",
                "tests/layers/flash_attn/test_attention_func.py",
            ],
            "mindiesd/layers/flash_attn/attention_forward_varlen.py": [
                "tests/layers/flash_attn/test_attention_forward_varlen.py",
            ],
            "mindiesd/layers/flash_attn/sparse_flash_attn.py": [
                "tests/layers/flash_attn/test_sparse_attention.py",
            ],
            "mindiesd/cache_agent/cache_agent.py": [
                "tests/cache/test_cache_agent.py",
            ],
            "mindiesd/cache_agent/attention_cache.py": [
                "tests/cache/test_attention_cache.py",
            ],
            "mindiesd/cache_agent/dit_block_cache.py": [
                "tests/cache/test_dit_block_cache.py",
            ],
            "mindiesd/cache_agent/cache.py": [
                "tests/cache/test_cache_agent.py",
            ],
            "mindiesd/quantization/quantize.py": [
                "tests/quantization/test_quantize.py",
            ],
            "mindiesd/quantization/config.py": [
                "tests/quantization/test_config.py",
            ],
            "mindiesd/quantization/layer.py": [
                "tests/quantization/test_layer.py",
            ],
            "mindiesd/quantization/mode.py": [
                "tests/quantization/test_mode.py",
            ],
            "mindiesd/quantization/utils.py": [
                "tests/quantization/test_quant_utils.py",
            ],
            "mindiesd/compilation/mindie_sd_backend.py": [
                "tests/compilation/test_backend.py",
                "tests/compilation/test_aclgraph_backend.py",
                "tests/UT/compilation/test_aclgraph_config.py",
            ],
            "mindiesd/compilation/aclgraph_backend.py": [
                "tests/compilation/test_aclgraph_backend.py",
                "tests/UT/compilation/test_aclgraph_config.py",
            ],
            "mindiesd/compilation/compiliation_config.py": [
                "tests/compilation/test_aclgraph_backend.py",
                "tests/UT/compilation/test_aclgraph_config.py",
            ],
            "mindiesd/layers/norm.py": [
                "tests/layers/test_norm.py",
                "tests/layers/test_rmsnorm.py",
                "tests/layers/test_layernorm.py",
            ],
            "mindiesd/layers/adalayernorm.py": [
                "tests/layers/test_adalayernorm.py",
            ],
            "mindiesd/layers/rope.py": [
                "tests/layers/test_rope.py",
            ],
            "mindiesd/layers/activation.py": [
                "tests/layers/test_activation.py",
            ],
            "mindiesd/layers/_custom_ops.py": [
                "tests/layers/test_custom_ops.py",
            ],
            "mindiesd/layers/register_ops.py": [
                "tests/layers/test_register_ops.py",
            ],
            # CSRC 算子实现源码映射到 Layers 测试（因为 layers 调用这些算子）
            "csrc/ops/ascendc/op_kernel/ascend_laser_attention.cpp": [
                "tests/plugin/test_la.py",
                "tests/plugin/test_la_preprocess.py",
                "tests/layers/flash_attn/test_attn_forward.py",
                "tests/layers/flash_attn/test_attention_func.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/ops/ascendc/op_kernel/la_preprocess.cpp": [
                "tests/plugin/test_la.py",
                "tests/plugin/test_la_preprocess.py",
                "tests/layers/flash_attn/test_attn_forward.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/ops/ascendc/op_kernel/ada_block_sparse_attention.cpp": [
                "tests/plugin/test_adablocksparseattention.py",
                "tests/layers/flash_attn/test_sparse_attention.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/ops/ascendc/op_kernel/sparse_block_estimate.cpp": [
                "tests/plugin/test_sparseblockestimate.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/plugin/la.cpp": [
                "tests/plugin/test_la.py",
                "tests/plugin/test_la_preprocess.py",
                "tests/layers/flash_attn/test_attn_forward.py",
                "tests/layers/flash_attn/test_attention_func.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/plugin/ada_block_sparse_attention.cpp": [
                "tests/plugin/test_adablocksparseattention.py",
                "tests/layers/flash_attn/test_sparse_attention.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/plugin/rope.cpp": [
                "tests/plugin/test_rope.py",
                "tests/layers/test_rope.py",
                "tests/layers/test_custom_ops.py",
            ],
            "csrc/plugin/adalayernorm.cpp": [
                "tests/plugin/test_adalayernorm.py",
                "tests/layers/test_adalayernorm.py",
            ],
            "csrc/plugin/layernorm.cpp": [
                "tests/plugin/test_layernorm.py",
                "tests/layers/test_norm.py",
                "tests/layers/test_layernorm.py",
            ],
            "csrc/plugin/rainfusionattention.cpp": [
                "tests/plugin/test_rainfusionattention.py",
                "tests/layers/flash_attn/test_attn_forward.py",
                "tests/layers/test_custom_ops.py",
            ],
        },

        # 模式匹配（通配符）
        "pattern": {
            "mindiesd/layers/flash_attn/*.py": "tests/layers/flash_attn/test_*.py",
            "mindiesd/compilation/patterns/*.py": "tests/compilation/patterns/test_*.py",
            "mindiesd/eplb/*.py": "tests/eplb/test_*.py",
            "mindiesd/utils/*.py": "tests/utils/test_*.py",
            "mindiesd/utils/logs/*.py": "tests/utils/logs/test_*.py",
        },

        # 目录匹配
        "directory": {
            "mindiesd/layers/": "tests/layers/",
            "mindiesd/cache_agent/": "tests/cache/",
            "mindiesd/quantization/": "tests/quantization/",
            "mindiesd/compilation/": "tests/compilation/",
            "mindiesd/eplb/": "tests/eplb/",
            "mindiesd/utils/": "tests/utils/",
        },

        # C++插件映射
        "plugin": {
            "csrc/plugin/": "tests/plugin/",
            "csrc/ops/ascendc/op_kernel/": "tests/plugin/",
            "csrc/ops/ascendc/op_host/": "tests/plugin/",
        }
    }

    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def find_related_tests(self, changed_files: List[FileChange]) -> Tuple[Set[str], ChangeInfo]:
        """
        根据变更文件查找相关的测试用例

        Args:
            changed_files: 变更文件列表

        Returns:
            (相关测试文件路径集合, 变更信息)
        """
        related_tests = set()
        change_info = ChangeInfo()

        # 第一步：分类处理变更文件
        for change in changed_files:
            file_path = change.path

            # 如果变更的是测试文件
            if self._is_test_file(file_path):
                if change.change_type == 'D':
                    # 删除的测试直接跳过，但记录下来
                    change_info.deleted_test_files.add(file_path)
                else:
                    # 新增或修改，包含测试
                    related_tests.add(file_path)
                continue

            # 只处理Python文件和C++源文件
            if not self._is_source_file(file_path):
                continue

            # 处理源码文件
            if change.change_type == 'D':
                # 删除的源码，记录并跳过直接映射
                change_info.deleted_source_files.add(file_path)
            elif change.change_type == 'A':
                # 新增的源码
                change_info.added_source_files.add(file_path)
                tests = self._map_to_tests(file_path)
                related_tests.update(tests)
            else:  # 'M' 或其他
                # 修改的源码
                change_info.modified_source_files.add(file_path)
                tests = self._map_to_tests(file_path)
                related_tests.update(tests)

        # 第二步：检查所有变更的源码（新增、修改、删除）是否被其他源码依赖
        all_changed_sources = change_info.get_all_changed_sources()
        if all_changed_sources:
            analyzer = SourceDependencyAnalyzer(self.repo_root)
            all_sources = analyzer.get_all_source_files()

            # 从所有源码中排除变更的源码本身
            other_sources = [s for s in all_sources if s not in all_changed_sources]

            dependent_sources = analyzer.find_dependent_sources(
                all_changed_sources,
                other_sources
            )

            # 为依赖的源码查找测试
            for dep_source in dependent_sources:
                tests = self._map_to_tests(dep_source)
                change_info.dependency_affected_tests.update(tests)

            related_tests.update(change_info.dependency_affected_tests)

        return related_tests, change_info

    def _is_test_file(self, file_path: str) -> bool:
        """判断是否为测试文件"""
        return file_path.startswith("tests/") and \
               (file_path.startswith("test_") or "/test_" in file_path) and \
               file_path.endswith(".py")

    def _is_source_file(self, file_path: str) -> bool:
        """判断是否为源文件（需要测试的代码文件）"""
        source_extensions = ['.py', '.cpp', '.h', '.hpp', '.c', '.cc']
        return any(file_path.endswith(ext) for ext in source_extensions)

    def _map_to_tests(self, source_path: str) -> List[str]:
        """将源码路径映射到测试路径"""
        tests = []

        # 1. 精确匹配
        if source_path in self.MAPPING_RULES["exact"]:
            tests.extend(self.MAPPING_RULES["exact"][source_path])

        # 2. 模式匹配
        for pattern, test_pattern in self.MAPPING_RULES["pattern"].items():
            if fnmatch(source_path, pattern):
                # 将通配符转换为具体文件
                matched_tests = self._expand_test_pattern(test_pattern)
                tests.extend(matched_tests)

        # 3. 目录匹配
        for src_dir, test_dir in self.MAPPING_RULES["directory"].items():
            if source_path.startswith(src_dir):
                # 根据源文件推断测试文件名
                relative_path = source_path[len(src_dir):]
                test_file = self._infer_test_file(relative_path, test_dir)
                if test_file:
                    tests.append(test_file)

        # 4. 插件匹配
        for plugin_dir, test_dir in self.MAPPING_RULES["plugin"].items():
            if source_path.startswith(plugin_dir):
                tests.extend(self._find_plugin_tests(source_path, test_dir))

        # 去重并过滤存在的测试文件
        unique_tests = list(set(tests))
        return [t for t in unique_tests if self._test_file_exists(t)]

    def _expand_test_pattern(self, pattern: str) -> List[str]:
        """展开测试文件通配符模式"""
        full_pattern = os.path.join(self.repo_root, pattern)
        files = glob.glob(full_pattern)
        return [os.path.relpath(f, self.repo_root) for f in files]

    def _infer_test_file(self, relative_path: str, test_dir: str) -> Optional[str]:
        """根据源文件推断测试文件名"""
        # 去掉.py后缀，加上test_前缀
        base_name = os.path.basename(relative_path)
        name_without_ext = os.path.splitext(base_name)[0]
        test_name = f"test_{name_without_ext}.py"
        return os.path.join(test_dir, test_name)

    def _find_plugin_tests(self, source_path: str, test_dir: str) -> List[str]:
        """查找插件相关的测试"""
        tests = []
        # 从源文件名推断测试名
        base_name = os.path.basename(source_path)
        name_without_ext = os.path.splitext(base_name)[0]

        # 常见的插件测试映射
        plugin_test_mapping = {
            "la": ["test_la.py", "test_la_preprocess.py"],
            "ada_block_sparse_attention": ["test_adablocksparseattention.py"],
            "sparse_block_estimate": ["test_sparseblockestimate.py"],
            "adalayernorm": ["test_adalayernorm.py"],
            "layernorm": ["test_layernorm.py"],
            "rope": ["test_rope.py"],
            "rainfusionattention": ["test_rainfusionattention.py"],
        }

        name_lower = name_without_ext.lower()
        for key, test_files in plugin_test_mapping.items():
            if key in name_lower:
                for tf in test_files:
                    test_path = os.path.join(test_dir, tf)
                    if test_path not in tests:
                        tests.append(test_path)

        return tests

    def _test_file_exists(self, test_path: str) -> bool:
        """检查测试文件是否存在"""
        full_path = os.path.join(self.repo_root, test_path)
        return os.path.isfile(full_path)


class IncrementalTestFinder:
    """增量测试运行器"""

    def __init__(self, repo_root: str = None, base_branch: str = "main"):
        self.repo_root = repo_root or GitChangeDetector().repo_root
        self.detector = GitChangeDetector(self.repo_root, base_branch)
        self.mapper = TestMapper(self.repo_root)

    def get_incremental_tests(self, since_ref: str = None,
                              include_staged: bool = True,
                              include_unstaged: bool = True) -> Tuple[Set[str], ChangeInfo, List[FileChange]]:
        """
        获取增量测试列表

        Args:
            since_ref: 对比的Git引用
            include_staged: 是否包含暂存区变更
            include_unstaged: 是否包含未暂存变更

        Returns:
            (测试文件集合, 变更信息, 变更文件列表)
        """
        all_changes = []

        # 获取分支间差异
        branch_changes = self.detector.get_changed_files(since_ref)
        all_changes.extend(branch_changes)

        # 获取暂存区变更
        if include_staged:
            staged_changes = self.detector.get_staged_files()
            all_changes.extend(staged_changes)

        # 获取未暂存变更
        if include_unstaged:
            unstaged_changes = self.detector.get_unstaged_files()
            all_changes.extend(unstaged_changes)

        # 去重（基于文件路径）
        unique_changes = {}
        for change in all_changes:
            if change.path not in unique_changes:
                unique_changes[change.path] = change
        all_changes = list(unique_changes.values())

        # 查找相关测试
        related_tests, change_info = self.mapper.find_related_tests(all_changes)

        return related_tests, change_info, all_changes

    def print_test_plan(self, tests: Set[str], change_info: ChangeInfo, changes: List[FileChange]):
        """打印测试计划"""
        print("=" * 80)
        print("增量测试计划")
        print("=" * 80)

        print("\n📁 变更文件:")
        for change in changes:
            type_icon = {"A": "➕", "M": "✏️", "D": "🗑️", "R": "📝"}.get(change.change_type, "❓")
            print(f"  {type_icon} {change.path}")

        # 显示变更统计
        if change_info.added_source_files:
            print(f"\n📥 新增源码 ({len(change_info.added_source_files)}个):")
            for src in sorted(change_info.added_source_files):
                print(f"  ➕ {src}")

        if change_info.modified_source_files:
            print(f"\n✏️  修改源码 ({len(change_info.modified_source_files)}个):")
            for src in sorted(change_info.modified_source_files):
                print(f"  ✏️  {src}")

        if change_info.deleted_source_files:
            print(f"\n🗑️  删除源码 ({len(change_info.deleted_source_files)}个):")
            for src in sorted(change_info.deleted_source_files):
                print(f"  🗑️  {src}")

        # 显示跳过的删除测试
        if change_info.deleted_test_files:
            print(f"\n⏭️  跳过的已删除测试 ({len(change_info.deleted_test_files)}个):")
            for test in sorted(change_info.deleted_test_files):
                print(f"  ⏭️  {test}")

        # 显示依赖分析结果
        if change_info.dependency_affected_tests:
            print(f"\n🔗 依赖变更源码的测试 ({len(change_info.dependency_affected_tests)}个):")
            for test in sorted(change_info.dependency_affected_tests):
                print(f"  ⚠️  {test}")

        # 显示需要运行的测试
        if tests:
            print(f"\n🧪 需要运行的测试用例 ({len(tests)}个):")
            for test in sorted(tests):
                print(f"  ✅ {test}")
        else:
            print(f"\n🧪 没有需要运行的测试用例")

        print("\n" + "=" * 80)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="增量测试工具")
    parser.add_argument("--base", "-b", default="master",
                       help="基分支或commit (默认: master)")
    parser.add_argument("--repo", "-r", default=None,
                       help="仓库根目录 (默认: 自动检测)")
    parser.add_argument("--no-staged", action="store_true",
                       help="不包含暂存区变更")
    parser.add_argument("--no-unstaged", action="store_true",
                       help="不包含未暂存变更")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="仅显示测试计划，不执行测试")

    args = parser.parse_args()

    # 创建运行器
    runner = IncrementalTestFinder(
        repo_root=args.repo,
        base_branch=args.base
    )

    # 获取增量测试
    tests, change_info, changes = runner.get_incremental_tests(
        since_ref=args.base,
        include_staged=not args.no_staged,
        include_unstaged=not args.no_unstaged
    )

    # 打印测试计划
    runner.print_test_plan(tests, change_info, changes)

    if args.dry_run:
        return

    if not tests:
        print("\n没有需要运行的测试。")
        return

    # 运行测试
    print("\n🚀 开始运行测试...\n")
    # 返回测试文件列表供 run.py 使用
    return tests


if __name__ == "__main__":
    main()
