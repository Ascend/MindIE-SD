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

"""Restricted (de)serialization helpers that prevent pickle-driven RCE.

``pickle`` is unsafe to apply to untrusted bytes: a crafted pickle stream can
import and call arbitrary callables (e.g. ``os.system`` / ``subprocess.Popen``),
which yields remote code execution on whoever calls ``pickle.loads``.

Share-memory / TP pipelines exchange objects (NPU shared-storage handles,
tensors, ...) over ZMQ sockets, so the receiving side must never feed raw
socket bytes into ``pickle.loads``. The helpers here provide drop-in
replacements that only allow an explicit, safe set of modules/classes to be
reconstructed, and reject everything else by default.

Usage::

    from mindiesd.utils.safe_pickle import safe_dumps, safe_loads

    sock.send(safe_dumps(handle))          # instead of sock.send_pyobj(handle)
    handle = safe_loads(sock.recv())        # instead of sock.recv_pyobj()
"""

import io
import pickle
from typing import Any


class SafeUnpickler(pickle.Unpickler):
    """A ``pickle.Unpickler`` subclass with an allowlist + denylist.

    ``find_class`` is the hook pickle calls whenever it needs to resolve a
    module/class name embedded in the stream. We override it to:

    1. Reject a small set of well-known dangerous callables outright.
    2. Only reconstruct classes whose module is covered by an explicit prefix
       allowlist.
    3. Reject anything else (deny-by-default).
    """

    # Modules that are considered safe to reconstruct objects from.
    # Kept tight on purpose: NPU shared-memory handles are plain builtins
    # (int/bytes tuples), torch/torch_npu cover tensor/storage types.
    ALLOWED_MODULE_PREFIXES = {
        # --- Python builtins / common infrastructure ---
        "builtins.",
        "collections.",
        "copyreg.",
        "functools.",
        "itertools.",
        "operator.",
        "types.",
        "weakref.",
        # --- PyTorch types (tensors, storages, dtypes, devices) ---
        "torch.",
        "torch._tensor.",
        "torch.storage.",
        "torch._C.",
        # --- Huawei Ascend NPU types ---
        "torch_npu.",
    }

    # Callables that are blocked even if their module were otherwise allowed.
    # These are the classic gadgets used to turn pickle into RCE.
    DENY_CLASSES = {
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("builtins", "compile"),
        ("builtins", "__import__"),
        ("os", "system"),
        ("os", "popen"),
        ("subprocess", "Popen"),
        ("subprocess", "run"),
        ("subprocess", "call"),
        ("subprocess", "check_output"),
        ("codecs", "decode"),
        ("types", "CodeType"),
        ("types", "FunctionType"),
        ("pickle", "loads"),
        ("pickle", "load"),
    }

    def find_class(self, module, name):
        # Gate 1: block known-dangerous gadgets explicitly.
        if (module, name) in self.DENY_CLASSES:
            raise RuntimeError(
                f"Blocked unsafe class loading ({module}.{name}): "
                f"refusing to deserialize a callable that enables arbitrary code execution."
            )

        # Gate 2: only reconstruct allowlisted modules.
        if any((module + ".").startswith(prefix) for prefix in self.ALLOWED_MODULE_PREFIXES):
            return super().find_class(module, name)

        # Gate 3: deny-by-default for everything else.
        raise RuntimeError(
            f"Blocked unsafe class loading ({module}.{name}): "
            f"module is not in SafeUnpickler's allowlist. If this type is "
            f"legitimately required, add '{module}.' to ALLOWED_MODULE_PREFIXES."
        )


def safe_dump(obj: Any, fp, protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """Drop-in replacement for ``pickle.dump`` (serialization is safe as-is)."""
    pickle.dump(obj, fp, protocol=protocol)


def safe_dumps(obj: Any, protocol: int = pickle.HIGHEST_PROTOCOL) -> bytes:
    """Drop-in replacement for ``pickle.dumps`` (serialization is safe as-is)."""
    return pickle.dumps(obj, protocol=protocol)


def safe_load(fp) -> Any:
    """Drop-in replacement for ``pickle.load`` that blocks unsafe class loading."""
    return SafeUnpickler(fp).load()


def safe_loads(data: Any) -> Any:
    """Drop-in replacement for ``pickle.loads`` that blocks unsafe class loading.

    Accepts ``bytes`` / ``bytearray`` / ``memoryview`` as well as any object
    exposing the buffer protocol (e.g. ``zmq.Frame``).
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    else:
        # zmq.Frame and other buffer-protocol objects
        buf = bytes(memoryview(data))
    return SafeUnpickler(io.BytesIO(buf)).load()
