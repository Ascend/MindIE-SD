#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Eagle quantized block sparse attention (EQBSA, Q/K INT8 + V FP8)."""

import math

from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry

from ._common import (
    MfuMbuSummaryMixin,
    sparse_block_indices,
    sparse_kv_tokens,
    tensor_bytes,
)

EQBSA_BLOCK_SHAPE = (128, 128)
EQBSA_Q_QUANT_BLOCK = 64
EQBSA_SUPPORTED_DTYPES = ("fp8",)


@ProviderRegistry.register_base_impl("eqbsa", "ComputeEngine")
class EagleQuantBlockSparseAttentionOp(MfuMbuSummaryMixin, BasicOp):
    """Benchmark the pre-quantized EQBSA operator in BNSD layout.

    Q/K are sequence-block INT8 (64-token quant blocks), V is per-channel FP8
    E4M3, and the scales are read by the fused kernel. Quantization is outside
    the timed region so latency represents EQBSA itself. FLOPs are discounted
    by the block mask's realized retention (kept blocks, tail block at its real
    length), not by the requested `sparsity`.
    """

    # BasicOp initializes these through prepare_args/vendor_impl lifecycle hooks.
    # pylint: disable=attribute-defined-outside-init

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type != "default":
            raise ValueError(f"eqbsa only supports arg_type=default, got {self.arg_type}")

        # Reproducibility follows BSA: the seed fixes backend-created Q/K/V
        # and scale data. The sparse mask needs no seed: it is built from a
        # deterministic block rule (see _sparse_mask_info).
        seed = int(self.args_dict.get("seed", 42))
        import torch

        torch.manual_seed(seed)
        npu = getattr(torch, "npu", None)
        if npu is not None and hasattr(npu, "manual_seed"):
            npu.manual_seed(seed)

        self.batch_size = int(self.args_dict["batch_size"])
        self.num_heads = int(self.args_dict["num_heads"])
        self.num_kv_heads = int(self.args_dict.get("num_kv_heads", 32))
        self.head_dim = int(self.args_dict["head_dim"])
        self.q_len = int(self.args_dict["q_len"])
        self.kv_len = int(self.args_dict.get("kv_len", self.q_len))
        self.causal = bool(self.args_dict.get("causal", False))
        self.sparsity = float(self.args_dict.get("sparsity", 0.8))
        self.dtype = self.args_dict.get("dtype", "fp8")
        self.softmax_scale = self.head_dim ** (-0.5)

        self._validate_args()
        self.flops_calc()

    def _validate_args(self):
        if self.dtype not in EQBSA_SUPPORTED_DTYPES:
            raise ValueError(f"eqbsa dtype {self.dtype} not in {EQBSA_SUPPORTED_DTYPES}")
        if self.num_heads <= 0 or self.num_kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("num_heads/num_kv_heads/head_dim must be positive")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads {self.num_heads} must be divisible by num_kv_heads {self.num_kv_heads}")
        if self.q_len <= 0 or self.kv_len <= 0:
            raise ValueError("q_len/kv_len must be positive")
        if self.causal:
            raise ValueError("eqbsa benchmark only supports the operator's non-causal path")
        if self.sparsity < 0 or self.sparsity >= 1:
            raise ValueError(f"sparsity must be in [0, 1), got {self.sparsity}")

    def flops_calc(self):
        # Same useful attention accounting as BSA: QK^T and PV, discounted by
        # the retained block fraction. Fused dequant is part of the kernel but
        # is not added as attention FLOPs, keeping EQBSA comparable to BSA.
        self.kv_blocks = math.ceil(self.kv_len / EQBSA_BLOCK_SHAPE[1])
        self.kept_kv_blocks = len(sparse_block_indices(self.kv_blocks, self.sparsity))
        self.retained_kv_tokens = sparse_kv_tokens(self.kv_len, EQBSA_BLOCK_SHAPE[1], self.sparsity)
        self.actual_sparsity = 1.0 - self.kept_kv_blocks / self.kv_blocks
        valid_parts = self.q_len * self.retained_kv_tokens
        self.calc_flops = 2 * (self.batch_size * self.num_heads * self.head_dim * valid_parts * 2)

    def summary(self, latency_us, kernel_mapping=None):
        """Add the realized sparsity so reports show (requested vs actual)."""
        target_dict = super().summary(latency_us, kernel_mapping or {})
        actual_sparsity = getattr(self, "actual_sparsity", None)
        if target_dict and actual_sparsity is not None:
            target_dict["actual_sparsity"] = actual_sparsity
        return target_dict

    def vendor_parser(self):
        pass

    def _tensor_info(self, shape, torch_dtype):
        from xpu_perf.micro_perf.core.utils import OpTensorInfo

        return OpTensorInfo(shape=shape, dtype=torch_dtype, device=self.backend.get_torch_device_name())

    def _quantized_input_info(self, key, shape, torch_dtype, quant_shapes, quant_cache):
        """Create matching quantized data/scale via the operator UT quant path.

        The first creator invocation quantizes one bf16 Q/K/V set once; later
        creators reuse its outputs so a quant tensor and its scale are paired.
        """

        def ensure(device):
            if not quant_cache:
                import torch

                q = torch.randn(size=quant_shapes["q"], dtype=torch.float32, device=device).to(torch.bfloat16)
                k = torch.randn(size=quant_shapes["k"], dtype=torch.float32, device=device).to(torch.bfloat16)
                v = torch.randn(size=quant_shapes["v"], dtype=torch.float32, device=device).to(torch.bfloat16)
                (
                    query_quant,
                    key_quant,
                    value_quant,
                    query_scale,
                    key_scale,
                    value_scale,
                ) = self._quantize(q, k, v)
                print(
                    "[eqbsa] prequant "
                    f"q={tuple(q.shape)}:{q.dtype}, "
                    f"k={tuple(k.shape)}:{k.dtype}, "
                    f"v={tuple(v.shape)}:{v.dtype}"
                )
                print(
                    "[eqbsa] quantized "
                    f"q={tuple(query_quant.shape)}:{query_quant.dtype}, "
                    f"k={tuple(key_quant.shape)}:{key_quant.dtype}, "
                    f"v={tuple(value_quant.shape)}:{value_quant.dtype}, "
                    f"q_scale={tuple(query_scale.shape)}:{query_scale.dtype}, "
                    f"k_scale={tuple(key_scale.shape)}:{key_scale.dtype}, "
                    f"v_scale={tuple(value_scale.shape)}:{value_scale.dtype}"
                )
                quant_cache.update(
                    {
                        "q": query_quant,
                        "k": key_quant,
                        # The plugin receives INT8 storage while value_dtype
                        # declares the logical FP8 E4M3 content (op UT bitcast).
                        "v": value_quant.view(torch.int8),
                        "query_scale": query_scale,
                        "key_scale": key_scale,
                        "value_scale": value_scale,
                    }
                )
            return quant_cache[key]

        def creator(size, dtype, device):
            return ensure(device)

        info = self._tensor_info(shape, torch_dtype)
        info.creator = creator
        return info

    def _sparse_mask_info(self, shape):
        """Per-row block mask built from the same rule as `flops_calc`.

        Every query row keeps the identical `sparse_block_indices` block set,
        so no row is left without keys and the FLOPs discount equals the
        mask's realized retention exactly.
        """
        import torch

        def creator(size, dtype, device):
            size = tuple(size)
            # Shape guard: flops_calc accounts self.kv_blocks, so a diverging
            # mask shape would silently skew MFU.
            if size[-1] != self.kv_blocks:
                raise ValueError(f"mask kv blocks {size[-1]} differ from accounted {self.kv_blocks}")
            mask = torch.zeros(size=size, dtype=torch.int8, device=device)
            mask[..., sparse_block_indices(size[-1], self.sparsity)] = 1
            return mask.to(dtype)

        info = self._tensor_info(shape, torch.int8)
        info.creator = creator
        return info

    def vendor_impl(self):
        import torch

        b, n, nk, d = self.batch_size, self.num_heads, self.num_kv_heads, self.head_dim
        sq, skv = self.q_len, self.kv_len
        q_blocks = math.ceil(sq / EQBSA_Q_QUANT_BLOCK)
        kv_blocks = math.ceil(skv / EQBSA_Q_QUANT_BLOCK)
        quant_shapes = {
            "q": [b, n, sq, d],
            "k": [b, nk, skv, d],
            "v": [b, nk, skv, d],
        }
        quant_cache = {}

        self.input_tensor_info = {
            "q": self._quantized_input_info("q", [b, n, sq, d], torch.int8, quant_shapes, quant_cache),
            "k": self._quantized_input_info("k", [b, nk, skv, d], torch.int8, quant_shapes, quant_cache),
            "v": self._quantized_input_info("v", [b, nk, skv, d], torch.int8, quant_shapes, quant_cache),
            "query_scale": self._quantized_input_info(
                "query_scale", [b, n, q_blocks], torch.float32, quant_shapes, quant_cache
            ),
            "key_scale": self._quantized_input_info(
                "key_scale", [b, nk, kv_blocks], torch.float32, quant_shapes, quant_cache
            ),
            "value_scale": self._quantized_input_info(
                "value_scale", [b, nk, d], torch.float32, quant_shapes, quant_cache
            ),
            "block_sparse_mask": self._sparse_mask_info(
                [b, n, math.ceil(sq / EQBSA_BLOCK_SHAPE[0]), math.ceil(skv / EQBSA_BLOCK_SHAPE[1])]
            ),
        }
        self.output_tensor_info = {
            "out": self._tensor_info([b, n, sq, d], torch.bfloat16),
        }

        q_numel = b * n * sq * d
        kv_numel = b * nk * skv * d
        scale_bytes = 4.0 * (b * n * q_blocks + b * nk * kv_blocks + b * nk * d)
        mask_numel = b * n * math.ceil(sq / EQBSA_BLOCK_SHAPE[0]) * math.ceil(skv / EQBSA_BLOCK_SHAPE[1])
        # EQBSA: Q/K INT8 and V FP8 both occupy one byte per element.
        self.input_tensor_size = (
            tensor_bytes(q_numel, self.dtype)
            + tensor_bytes(kv_numel, self.dtype)
            + tensor_bytes(kv_numel, "fp8")
            + scale_bytes
            + tensor_bytes(mask_numel, "fp8")
        )
        self.output_tensor_size = tensor_bytes(q_numel, "bf16")
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.executed_path = "qk_int8_pv_fp8"
        self.actual_seq_lengths = [sq] * b
        self.actual_seq_lengths_kv = [skv] * b

        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        raise NotImplementedError
