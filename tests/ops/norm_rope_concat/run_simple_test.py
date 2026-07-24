#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""
Quick smoke test for mindiesd.norm_rope_concat.

Usage:
    python run_simple_test.py              # auto-detect NPU
    python run_simple_test.py --cpu         # run reference-only on CPU
    python run_simple_test.py --npu 0       # specify NPU device id
"""

import argparse
import sys
import traceback

import torch


# Import reference implementation from the test module
from test_norm_rope_concat import torch_norm_rope_concat_ref


def _has_npu():
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def _has_mindiesd():
    try:
        return hasattr(torch.ops, 'mindiesd') and hasattr(torch.ops.mindiesd, 'norm_rope_concat')
    except Exception:
        return False


def run_reference_test():
    """Run pure-PyTorch reference tests (always works, no NPU needed)."""
    print("=" * 70)
    print("Phase 1: Reference Implementation Tests (CPU)")
    print("=" * 70)

    passed = 0
    failed = 0

    # Test 1: Pure transpose
    try:
        B, S, H, D = 1, 4, 2, 8
        q = torch.randn(B, S, H, D)
        result = torch_norm_rope_concat_ref(q, q, q)
        assert result[0].shape == (B, H, S, D), f"Shape mismatch: {result[0].shape}"
        print("   [PASS] Test 1: Pure transpose (no norm/rope/encoder)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 1: {e}")
        failed += 1

    # Test 2: LayerNorm affine
    try:
        B, S, H, D = 2, 3, 4, 16
        q = torch.randn(B, S, H, D)
        w = torch.ones(D)
        b = torch.zeros(D)
        result = torch_norm_rope_concat_ref(q, q, q,
            norm_query_weight=w, norm_query_bias=b,
            norm_key_weight=w, norm_key_bias=b,
            norm_type=2, is_training=True)
        assert result[0].shape == (B, H, S, D)
        assert result[3].shape == (B, S, H, 1)  # mean
        print("   [PASS] Test 2: LayerNorm affine (training mode)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 2: {e}")
        failed += 1

    # Test 3: RoPE INTERLEAVE
    try:
        B, H, D = 1, 1, 4
        q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        rope_sin = torch.ones(1, D)
        rope_cos = torch.ones(1, D)
        result = torch_norm_rope_concat_ref(q, q, q,
            rope_sin=rope_sin, rope_cos=rope_cos, rope_type=1)
        # cos=1, sin=1 → (x1-x2, x2+x1, x3-x4, x4+x3)
        out = result[0]
        assert abs(out[0, 0, 0, 0].item() - (-1.0)) < 1e-5, f"Expected -1, got {out[0,0,0,0]}"
        assert abs(out[0, 0, 0, 1].item() - 3.0) < 1e-5, f"Expected 3, got {out[0,0,0,1]}"
        print("   [PASS] Test 3: RoPE INTERLEAVE correctness")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 3: {e}")
        failed += 1

    # Test 4: Concat order
    try:
        B, H, D = 1, 2, 4
        Sq, Seq = 3, 2
        q_self = torch.ones(B, Sq, H, D)
        q_enc = torch.ones(B, Seq, H, D) * 2
        result = torch_norm_rope_concat_ref(q_self, q_self, q_self,
            encoder_query=q_enc, encoder_key=q_enc, encoder_value=q_enc,
            concat_order=0)
        q_bhsd = result[0].permute(0, 2, 1, 3)
        assert (q_bhsd[:, :Sq] == 1.0).all(), "self part should be 1"
        assert (q_bhsd[:, Sq:] == 2.0).all(), "encoder part should be 2"
        print("   [PASS] Test 4: Concat order (BEFORE_ENCODER)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 4: {e}")
        failed += 1

    # Test 5: All norm types
    try:
        B, S, H, D = 1, 3, 2, 8
        q = torch.randn(B, S, H, D)
        w_ = torch.ones(D)
        b_ = torch.zeros(D)
        for nt in [0, 1, 2, 3, 4]:
            result = torch_norm_rope_concat_ref(q, q, q,
                norm_query_weight=w_, norm_query_bias=b_,
                norm_key_weight=w_, norm_key_bias=b_,
                norm_type=nt, is_training=(nt in (1, 2)))
            assert result[0].shape == (B, H, S, D)
        print("   [PASS] Test 5: All 5 norm_type values")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 5: {e}")
        failed += 1

    print(f"\n   Reference tests: {passed} passed, {failed} failed")
    return failed == 0


def run_npu_test(device_id=0):
    """Run actual NPU operator tests."""
    print("\n" + "=" * 70)
    print(f"Phase 2: NPU Operator Tests (device: npu:{device_id})")
    print("=" * 70)

    if not _has_npu():
        print("   [SKIP] NPU not available")
        return True

    if not _has_mindiesd():
        print("   [SKIP] mindiesd.norm_rope_concat not registered")
        return True

    device = torch.device(f"npu:{device_id}")
    passed = 0
    failed = 0

    # Test 1: Basic forward
    try:
        B, S, H, D = 1, 4, 2, 8
        query = torch.randn(B, S, H, D, device=device)
        key = torch.randn(B, S, H, D, device=device)
        value = torch.randn(B, S, H, D, device=device)

        result = torch.ops.mindiesd.norm_rope_concat(query, key, value)
        q_out, k_out, v_out = result[0], result[1], result[2]
        assert q_out.shape == (B, H, S, D), f"Shape: {q_out.shape}"
        assert k_out.shape == (B, H, S, D)
        assert v_out.shape == (B, H, S, D)
        print("   [PASS] Test 1: Basic forward (no norm/rope/encoder)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 1: {e}")
        traceback.print_exc()
        failed += 1

    # Test 2: LayerNorm affine + training
    try:
        B, S, H, D = 1, 5, 4, 16
        query = torch.randn(B, S, H, D, device=device)
        key = torch.randn(B, S, H, D, device=device)
        value = torch.randn(B, S, H, D, device=device)
        nqw = torch.ones(D, device=device)
        nqb = torch.zeros(D, device=device)
        nkw = torch.ones(D, device=device)
        nkb = torch.zeros(D, device=device)

        result = torch.ops.mindiesd.norm_rope_concat(
            query, key, value,
            norm_query_weight=nqw, norm_query_bias=nqb,
            norm_key_weight=nkw, norm_key_bias=nkb,
            norm_type=2, is_training=True)
        assert result[0].shape == (B, H, S, D)
        assert result[3].shape == (B, S, H, 1)  # norm_query_mean
        assert result[4].shape == (B, S, H, 1)  # norm_query_rstd
        print("   [PASS] Test 2: LayerNorm affine (training mode)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 2: {e}")
        traceback.print_exc()
        failed += 1

    # Test 3: With encoder (cross-attention)
    try:
        B, H, D = 1, 4, 16
        Sq, Sk, Sv = 5, 5, 5
        Seq, Sek, Sev = 3, 3, 3

        query = torch.randn(B, Sq, H, D, device=device)
        key = torch.randn(B, Sk, H, D, device=device)
        value = torch.randn(B, Sv, H, D, device=device)
        eq_ = torch.randn(B, Seq, H, D, device=device)
        ek_ = torch.randn(B, Sek, H, D, device=device)
        ev_ = torch.randn(B, Sev, H, D, device=device)

        nqw = torch.ones(D, device=device)
        nqb = torch.zeros(D, device=device)
        nkw = torch.ones(D, device=device)
        nkb = torch.zeros(D, device=device)
        naqw = torch.ones(D, device=device)
        naqb = torch.zeros(D, device=device)
        nakw = torch.ones(D, device=device)
        nakb = torch.zeros(D, device=device)

        rope_seq = Sq + Seq
        rope_sin = torch.randn(rope_seq, D, device=device)
        rope_cos = torch.randn(rope_seq, D, device=device)

        result = torch.ops.mindiesd.norm_rope_concat(
            query, key, value,
            encoder_query=eq_, encoder_key=ek_, encoder_value=ev_,
            norm_query_weight=nqw, norm_query_bias=nqb,
            norm_key_weight=nkw, norm_key_bias=nkb,
            norm_added_query_weight=naqw, norm_added_query_bias=naqb,
            norm_added_key_weight=nakw, norm_added_key_bias=nakb,
            rope_sin=rope_sin, rope_cos=rope_cos,
            norm_type=2, norm_added_type=2, rope_type=1,
            concat_order=0, eps=1e-5, is_training=True)

        assert result[0].shape == (B, H, Sq + Seq, D), f"query output: {result[0].shape}"
        assert result[1].shape == (B, H, Sk + Sek, D), f"key output: {result[1].shape}"
        assert result[2].shape == (B, H, Sv + Sev, D), f"value output: {result[2].shape}"
        print("   [PASS] Test 3: With encoder (cross-attention + norm + rope)")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 3: {e}")
        traceback.print_exc()
        failed += 1

    # Test 4: RMSNorm affine
    try:
        B, S, H, D = 1, 4, 4, 16
        query = torch.randn(B, S, H, D, device=device)
        key = torch.randn(B, S, H, D, device=device)
        value = torch.randn(B, S, H, D, device=device)
        nqw = torch.ones(D, device=device)
        nkw = torch.ones(D, device=device)

        result = torch.ops.mindiesd.norm_rope_concat(
            query, key, value,
            norm_query_weight=nqw, norm_key_weight=nkw,
            norm_type=4, is_training=False)
        assert result[0].shape == (B, H, S, D)
        print("   [PASS] Test 4: RMSNorm affine")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 4: {e}")
        traceback.print_exc()
        failed += 1

    # Test 5: FP16
    try:
        B, S, H, D = 1, 4, 2, 16
        query = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
        key = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
        value = torch.randn(B, S, H, D, dtype=torch.float16, device=device)
        nqw = torch.ones(D, dtype=torch.float16, device=device)
        nqb = torch.zeros(D, dtype=torch.float16, device=device)
        nkw = torch.ones(D, dtype=torch.float16, device=device)
        nkb = torch.zeros(D, dtype=torch.float16, device=device)
        rope_sin = torch.randn(S, D, dtype=torch.float16, device=device)
        rope_cos = torch.randn(S, D, dtype=torch.float16, device=device)

        result = torch.ops.mindiesd.norm_rope_concat(
            query, key, value,
            norm_query_weight=nqw, norm_query_bias=nqb,
            norm_key_weight=nkw, norm_key_bias=nkb,
            rope_sin=rope_sin, rope_cos=rope_cos,
            norm_type=2, rope_type=1, is_training=False)
        assert result[0].dtype == torch.float16
        assert result[0].shape == (B, H, S, D)
        print("   [PASS] Test 5: FP16 input")
        passed += 1
    except Exception as e:
        print(f"   [FAIL] Test 5: {e}")
        traceback.print_exc()
        failed += 1

    print(f"\n   NPU tests: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Smoke test for norm_rope_concat")
    parser.add_argument("--cpu", action="store_true", help="Run reference-only tests on CPU")
    parser.add_argument("--npu", type=int, default=0, help="NPU device ID (default: 0)")
    args = parser.parse_args()

    print("norm_rope_concat — Smoke Test")
    print(f"PyTorch: {torch.__version__}")
    print(f"NPU available: {_has_npu()}")
    print(f"mindiesd op registered: {_has_mindiesd()}")

    all_ok = True

    # Always run reference tests
    all_ok &= run_reference_test()

    # Run NPU tests if not explicitly CPU-only
    if not args.cpu:
        all_ok &= run_npu_test(args.npu)
    else:
        print("\n   [INFO] Skipping NPU tests (--cpu flag)")

    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

