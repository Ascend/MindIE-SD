#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Pure-logic unit tests for the new compilation-backend routing primitives.

Covers:
  * ``BackendMode.coerce``              (mindiesd/compilation/compiliation_config.py)
  * ``CompilationConfig.resolved_backend_mode``  (same file, legacy-bool compat mapping)
  * ``PatternBase.backend``             (mindiesd/compilation/passes/register_pattern_to_pass.py)

These modules are loaded *without* importing torch. ``compiliation_config.py`` only
needs the stdlib (dataclasses + enum), and ``register_pattern_to_pass.py`` only pulls
torch in through its sibling ``pattern_match_pass``; we stub that sibling so the file
loads torch-free. Either way we never trigger ``mindiesd/__init__.py`` (which requires
torch/torch_npu, unavailable on this Windows/Python-3.13 dev machine).
"""

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-root resolution (works regardless of CWD because __file__ is absolute).
# test file: <root>/tests/UT/compilation/test_backend_mode.py  -> parents[3]
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_compiliation_config():
    """Load the ``compiliation_config`` module.

    Prefers a normal package import. If that fails — typically because
    ``mindiesd/__init__.py`` pulls in torch which is absent on this host — fall
    back to loading the file directly via importlib so the package ``__init__``
    is bypassed entirely.
    """
    # Strategy 1: normal import (works wherever torch/torch_npu is installed).
    try:
        return importlib.import_module("mindiesd.compilation.compiliation_config")
    except Exception:
        pass

    # Strategy 2: load by absolute file path, skipping the package __init__.
    target = _REPO_ROOT / "mindiesd" / "compilation" / "compiliation_config.py"
    if not target.exists():
        raise FileNotFoundError(f"compiliation_config.py not found at {target}")
    spec = importlib.util.spec_from_file_location("mindiesd.compilation.compiliation_config", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pattern_base():
    """Load ``PatternBase`` from ``register_pattern_to_pass.py`` torch-free.

    That file's only torch-bearing dependency is the relative import
    ``from .pattern_match_pass import PatternMatchPass``. We install a tiny stub
    for that sibling (plus namespace-package placeholders for its parents) so the
    relative import resolves without ever importing torch, letting us exercise
    the pure default ``PatternBase.backend()``.
    """
    # Strategy 1: normal import.
    try:
        mod = importlib.import_module("mindiesd.compilation.passes.register_pattern_to_pass")
        return mod.PatternBase
    except Exception:
        pass

    # Strategy 2: file load with a stubbed pattern_match_pass to avoid torch.
    pkg = "mindiesd.compilation.passes"
    pm_name = pkg + ".pattern_match_pass"

    if pm_name not in sys.modules:
        stub = types.ModuleType(pm_name)

        class _DummyPatternMatchPass:  # minimal stand-in; only needs to exist.
            def __init__(self, *args, **kwargs):
                pass

            def register_pattern(self, *args, **kwargs):
                pass

        stub.PatternMatchPass = _DummyPatternMatchPass
        sys.modules[pm_name] = stub

    # Ensure each parent package exists as a namespace package so the relative
    # import ``.pattern_match_pass`` resolves against ``__package__``.
    parent = pkg
    while parent:
        if parent not in sys.modules:
            ns = types.ModuleType(parent)
            ns.__path__ = []  # mark as package
            sys.modules[parent] = ns
        parent = parent.rsplit(".", 1)[0] if "." in parent else ""

    target = _REPO_ROOT / "mindiesd" / "compilation" / "passes" / "register_pattern_to_pass.py"
    if not target.exists():
        raise FileNotFoundError(f"register_pattern_to_pass.py not found at {target}")
    spec = importlib.util.spec_from_file_location(pkg + ".register_pattern_to_pass", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg + ".register_pattern_to_pass"] = mod
    spec.loader.exec_module(mod)
    return mod.PatternBase


# ---------------------------------------------------------------------------
# Module-level load of compiliation_config (stdlib-only; should always succeed).
# Capture any error so test classes can SkipTest with a clear reason rather than
# erroring at collection time.
# ---------------------------------------------------------------------------
_CONFIG_LOAD_ERROR = None
try:
    _cfg = _load_compiliation_config()
    BackendMode = _cfg.BackendMode
    CompilationConfig = _cfg.CompilationConfig
except Exception as exc:  # pragma: no cover - defensive; path is verified.
    _cfg = None
    BackendMode = None
    CompilationConfig = None
    _CONFIG_LOAD_ERROR = exc


# ===========================================================================
# BackendMode.coerce
# ===========================================================================
class TestBackendModeCoerce(unittest.TestCase):
    """``BackendMode.coerce`` accepts enum name, enum value, instances, and
    whitespace-padded strings; returns None for None / unknown strings / other
    types.
    """

    @classmethod
    def setUpClass(cls):
        if BackendMode is None:
            raise unittest.SkipTest(f"BackendMode unavailable: {_CONFIG_LOAD_ERROR!r}")

    def test_enum_name(self):
        # Task: 枚举名 "NPUGRAPH_EX"
        self.assertIs(BackendMode.coerce("NPUGRAPH_EX"), BackendMode.NPUGRAPH_EX)

    def test_enum_value(self):
        # Task: 枚举值 "npugraph_ex"
        self.assertIs(BackendMode.coerce("npugraph_ex"), BackendMode.NPUGRAPH_EX)

    def test_none_returns_none(self):
        # Task: None -> None (used by resolved_backend_mode to detect "not set")
        self.assertIsNone(BackendMode.coerce(None))

    def test_invalid_string_returns_none(self):
        # Task: 非法字符串 -> None. Note "aclgraph" is *not* a valid enum value
        # (the real value is "aclgraph_only"), so it is treated as unknown.
        for bad in ("invalid", "aclgraph", "ACLGRAPH", "", "npugraph"):
            with self.subTest(value=bad):
                self.assertIsNone(BackendMode.coerce(bad))

    def test_existing_instance_returned_as_is(self):
        # Task: 传入已是 BackendMode 实例 -> 原样返回 (identity preserved)
        for member in BackendMode:
            with self.subTest(member=member):
                self.assertIs(BackendMode.coerce(member), member)

    def test_whitespace_padded_string(self):
        # Task: 带空格的字符串 -> strip 后命中枚举值
        self.assertIs(BackendMode.coerce("  npugraph_ex  "), BackendMode.NPUGRAPH_EX)
        # 带空格的枚举名同样成立
        self.assertIs(BackendMode.coerce("\t ACLGRAPH_ONLY \n"), BackendMode.ACLGRAPH_ONLY)

    def test_all_members_round_trip_via_name_and_value(self):
        # 覆盖所有成员：name 与 value 双向都能命中对应成员
        for member in BackendMode:
            with self.subTest(member=member):
                self.assertIs(BackendMode.coerce(member.name), member)
                self.assertIs(BackendMode.coerce(member.value), member)

    def test_non_string_non_enum_returns_none(self):
        # 非字符串/非枚举/非 None 的输入落入兜底分支 -> None
        self.assertIsNone(BackendMode.coerce(123))
        self.assertIsNone(BackendMode.coerce(["NPUGRAPH_EX"]))


# ===========================================================================
# CompilationConfig.resolved_backend_mode
# ===========================================================================
class TestResolvedBackendMode(unittest.TestCase):
    """Priority rules for ``resolved_backend_mode``:

    explicit backend_mode > aclgraph_with_compile > aclgraph_only > DEFAULT.

    Each test mutates ``CompilationConfig`` *class* attributes, so setUp backs
    them up and tearDown restores them to avoid leaking state across tests
    (mirrors the save/restore pattern in test_aclgraph_config.py).
    """

    @classmethod
    def setUpClass(cls):
        if CompilationConfig is None:
            raise unittest.SkipTest(f"CompilationConfig unavailable: {_CONFIG_LOAD_ERROR!r}")
        cls.CC = CompilationConfig

    def setUp(self):
        # Save the three attributes resolved_backend_mode consults.
        self._orig_backend_mode = self.CC.backend_mode
        self._orig_aclgraph_only = self.CC.aclgraph_only
        self._orig_aclgraph_with_compile = self.CC.aclgraph_with_compile

    def tearDown(self):
        # Restore exactly — never pollute the class for sibling tests.
        self.CC.backend_mode = self._orig_backend_mode
        self.CC.aclgraph_only = self._orig_aclgraph_only
        self.CC.aclgraph_with_compile = self._orig_aclgraph_with_compile

    def _assert_resolved(self, backend_mode, with_compile, only, expected):
        """Set the three class attrs and assert resolved_backend_mode()."""
        self.CC.backend_mode = backend_mode
        self.CC.aclgraph_with_compile = with_compile
        self.CC.aclgraph_only = only
        self.assertEqual(self.CC.resolved_backend_mode(), expected)

    # ---- truth table ------------------------------------------------------

    def test_a_explicit_backend_mode_wins_over_legacy_bools(self):
        # (a) 显式 backend_mode=NPUGRAPH_EX 即使两个旧布尔都 True 仍返回 NPUGRAPH_EX
        self._assert_resolved(BackendMode.NPUGRAPH_EX, True, True, BackendMode.NPUGRAPH_EX)

    def test_b_with_compile_when_backend_mode_none(self):
        # (b) backend_mode=None, with_compile=True, only=False -> ACLGRAPH_WITH_COMPILE
        self._assert_resolved(None, True, False, BackendMode.ACLGRAPH_WITH_COMPILE)

    def test_c_aclgraph_only_when_backend_mode_none(self):
        # (c) backend_mode=None, only=True, with_compile=False -> ACLGRAPH_ONLY
        self._assert_resolved(None, False, True, BackendMode.ACLGRAPH_ONLY)

    def test_d_default_when_all_unset(self):
        # (d) backend_mode=None, 两个布尔都 False -> DEFAULT
        self._assert_resolved(None, False, False, BackendMode.DEFAULT)

    def test_e_with_compile_priority_when_both_bools_true(self):
        # (e) backend_mode=None, 两个布尔都 True -> ACLGRAPH_WITH_COMPILE
        #     (with_compile 优先于 only，与原 mindie_sd_backend 判断顺序一致)
        self._assert_resolved(None, True, True, BackendMode.ACLGRAPH_WITH_COMPILE)

    def test_f_string_enum_value_aclgraph_only(self):
        # (f) 识别得到的枚举*值*字符串 "aclgraph_only" -> ACLGRAPH_ONLY。
        # 注：任务描述里写作 "aclgraph"，但枚举真实 value 是 "aclgraph_only"；
        # "aclgraph" 既非 name 也非 value，coerce 视为非法（见下条用例）。
        self._assert_resolved("aclgraph_only", False, False, BackendMode.ACLGRAPH_ONLY)

    def test_f_bare_aclgraph_unrecognized_falls_through(self):
        # 裸字符串 "aclgraph" 不是合法 name/value -> coerce 返回 None ->
        # 旧布尔也都 False -> 回落到 DEFAULT。如实覆盖真实行为。
        self._assert_resolved("aclgraph", False, False, BackendMode.DEFAULT)

    def test_explicit_string_value_beats_legacy_bools(self):
        # 显式 backend_mode 即便用字符串形式，也优先于旧布尔：with_compile=True
        # 也不会覆盖显式的 DEFAULT。
        self._assert_resolved("default", True, True, BackendMode.DEFAULT)

    # ---- fixture safety ---------------------------------------------------

    def test_teardown_restores_class_attributes(self):
        # 仿照 test_aclgraph_config.test_routing_teardown_restores_config：
        # 用例里改写属性，断言 setUp 捕获的原值仍是默认；tearDown 会复原。
        self.CC.backend_mode = BackendMode.NPUGRAPH_EX
        self.CC.aclgraph_only = True
        self.CC.aclgraph_with_compile = True
        self.assertEqual(self._orig_backend_mode, None)
        self.assertFalse(self._orig_aclgraph_only)
        self.assertFalse(self._orig_aclgraph_with_compile)


# ===========================================================================
# PatternBase.backend
# ===========================================================================
class TestPatternBaseBackend(unittest.TestCase):
    """The default ``PatternBase.backend()`` returns ``"inductor"``; subclasses
    that do not override it inherit the same default.
    """

    @classmethod
    def setUpClass(cls):
        try:
            cls.PatternBase = _load_pattern_base()
        except Exception as exc:  # pragma: no cover - skip if torch-free load fails
            raise unittest.SkipTest(f"PatternBase unavailable (torch-free load failed): {exc!r}")

    def test_default_backend_is_inductor(self):
        # PatternBase.backend() 默认返回 "inductor"
        self.assertEqual(self.PatternBase.backend(), "inductor")

    def test_subclass_inherits_default(self):
        # 不覆写 backend() 的具体子类仍得到 "inductor"
        class _Concrete(self.PatternBase):
            @staticmethod
            def name():
                return "concrete"

            @staticmethod
            def inputs():
                return []

            @staticmethod
            def pattern(*args, **kwargs):
                pass

            @staticmethod
            def replacement(*args, **kwargs):
                pass

        self.assertEqual(_Concrete.backend(), "inductor")


if __name__ == "__main__":
    unittest.main(verbosity=2)
