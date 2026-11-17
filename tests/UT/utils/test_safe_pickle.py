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


import array
import collections
import io
import pickle

import pytest

from mindiesd.utils.safe_pickle import SafeUnpickler, safe_dump, safe_dumps, safe_load, safe_loads


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview, lambda value: array.array("B", value)])
def test_buffer_roundtrip(wrap):
    value = {"handle": (7, b"storage", None), "shape": [1, 2, 3]}
    assert safe_loads(wrap(safe_dumps(value))) == value


def test_stream_and_allowlisted_collection():
    value = collections.OrderedDict([("first", 1), ("second", 2)])
    stream = io.BytesIO()
    safe_dump(value, stream)
    stream.seek(0)
    assert safe_load(stream) == value
    unpickler = SafeUnpickler(io.BytesIO())
    assert unpickler.find_class("builtins", "set") is set


@pytest.mark.parametrize("module,name", sorted(SafeUnpickler.DENY_CLASSES))
def test_dangerous_globals_rejected_without_execution(module, name):
    # Protocol-0 GLOBAL invokes find_class; no REDUCE gadget is executed.
    payload = f"c{module}\n{name}\n.".encode()
    with pytest.raises(RuntimeError, match="Blocked unsafe class"):
        safe_loads(payload)


@pytest.mark.parametrize("module", ["pathlib", "torch_evil", "collections_evil", "builtins_evil"])
def test_unknown_and_prefix_lookalike_modules_rejected(module):
    with pytest.raises(RuntimeError, match="allowlist"):
        SafeUnpickler(io.BytesIO()).find_class(module, "Payload")


def test_invalid_buffer_and_corrupt_payload():
    with pytest.raises(TypeError):
        safe_loads(object())
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        safe_loads(b"invalid")
