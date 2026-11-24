# pylint: disable=redefined-outer-name

"""Shared fixtures for the MoE layer tests."""

import pytest

from mindiesd.layers.moe.moe_context import reset_moe_context
from mindiesd.layers.moe.token_dispatcher import DynamicDispatcher


@pytest.fixture(autouse=True)
def clean_split_cache():
    """Reset the process-wide split caches around every test.

    The dispatcher caches pinned host buffers and NPU events across calls; without
    this reset a test that changes the rank count sees the previous test's buffers.
    """
    DynamicDispatcher.clear_split_cache()
    yield
    DynamicDispatcher.clear_split_cache()


@pytest.fixture(autouse=True)
def clean_moe_context():
    """Start every test from an empty communicator context.

    `moe()` installs the context per call; a test that leaves an EP group or a
    quantization mode behind would silently change what the next one measures.
    """
    reset_moe_context()
    yield
    reset_moe_context()
