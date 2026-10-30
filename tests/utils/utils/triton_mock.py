# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2

# pylint: disable=attribute-defined-outside-init

"""Import-only Triton doubles; CPU tests must explicitly mock kernel launches."""

import sys
from importlib.util import spec_from_loader
from dataclasses import dataclass
from types import ModuleType


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = fn.__name__
        self.__doc__ = fn.__doc__
        self.__module__ = fn.__module__
        self.__wrapped__ = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __getitem__(self, grid):
        raise AssertionError("CPU test must mock the Triton launch boundary or use the CPU kernel interpreter")


def _jit(fn=None, **kwargs):
    return JITFunction if fn is None else JITFunction(fn)


class _LanguageModule(ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(f"CPU Triton language operation requires a test interpreter: {name}")


class _UnavailableDriver:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def unavailable(*args, **kwargs):
            raise AssertionError(f"CPU tests must mock the Triton driver interface: {name}")

        setattr(self, name, unavailable)
        return unavailable


def mock_triton():
    """Prevent imports of the Ascend compiler and hardware discovery in CPU mode."""
    names = (
        "triton",
        "triton.language",
        "triton.language.extra",
        "triton.language.extra.cann",
        "triton.language.extra.cann.extension",
        "triton.extension",
        "triton.extension.buffer",
        "triton.extension.buffer.language",
        "triton.runtime",
        "triton.runtime.driver",
        "triton.runtime.jit",
        "triton.backends",
        "triton.backends.compiler",
        "triton.compiler",
        "triton.compiler.compiler",
    )
    for name in names:
        module = _LanguageModule(name)
        module.__file__ = __file__
        module.__path__ = []
        module.__spec__ = spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    triton = sys.modules["triton"]
    triton.__version__ = "0.0.0"
    triton.jit = _jit
    triton.cdiv = lambda numerator, denominator: (numerator + denominator - 1) // denominator
    triton.next_power_of_2 = lambda value: 1 << (max(1, value) - 1).bit_length()
    sys.modules["triton.runtime.jit"].JITFunction = JITFunction

    @dataclass(frozen=True)
    class GPUTarget:
        backend: str
        arch: object
        warp_size: int

    sys.modules["triton.backends.compiler"].GPUTarget = GPUTarget
    language = sys.modules["triton.language"]
    language.constexpr = type("constexpr", (), {})
    language.tensor = type("tensor", (), {})
    language.dtype = type("dtype", (), {})
    for name in ("float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint32"):
        setattr(language, name, name)
    driver = sys.modules["triton.runtime.driver"]
    driver.active = _UnavailableDriver()
    driver.active.utils = _UnavailableDriver()
