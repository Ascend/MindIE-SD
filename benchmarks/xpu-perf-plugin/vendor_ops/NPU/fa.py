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

"""NPU flash attention vendor implementation for Ascend core ops."""

import logging
import math

from _quant import is_quant_unsupported
from xpu_perf.micro_perf.core.op import ProviderRegistry

logger = logging.getLogger(__name__)


@ProviderRegistry.register_vendor_impl("fa", "NPU")
class NPUFlashAttentionOp:
    """NPU FA: dispatch by quant_format following mindiesd/quantization/layer.py.

    bf16   -> torch_npu.npu_fusion_attention
    fp8/mxfp8 -> npu_dynamic_mx_quant + fused_infer_attention_score_v2
    mxfp4  -> torch.ops.mindiesd.quant_flash_attn

    Quantized paths fall back to bf16 FA when the platform lacks
    DynamicMxQuant (e.g. Ascend910_93), keeping the benchmark runnable; the
    schema byte accounting still reflects the requested dtype.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k = tensor_mapping["k"]
        v = tensor_mapping["v"]
        n = self.num_heads
        d = self.head_dim
        scale = 1.0 / math.sqrt(d)
        # Record what actually executed so reports can tell real quantized
        # runs from bf16 fallbacks (see MfuMbuSummaryMixin.executed_path).
        self.executed_path = self.dtype

        if self.dtype in ("fp8", "mxfp8"):
            try:
                return self._quant_fa_fp8(q, k, v, n, d, scale)
            except RuntimeError as exc:
                if not is_quant_unsupported(exc):
                    raise
                logger.warning(
                    "DynamicMxQuant unsupported on this platform; %s FA falls back to bf16",
                    self.dtype,
                )
                self.executed_path = "bf16_fallback"
                return self._bf16_fa(q, k, v, n, scale)

        if self.dtype == "mxfp4":
            try:
                return self._quant_fa_mxfp4(q, k, v, n, d, scale)
            except RuntimeError as exc:
                if not is_quant_unsupported(exc):
                    raise
                logger.warning("mxfp4 FA falls back to bf16 (DynamicMxQuant unsupported)")
                self.executed_path = "bf16_fallback"
                return self._bf16_fa(q, k, v, n, scale)

        return self._bf16_fa(q, k, v, n, scale)

    def _bf16_fa(self, q, k, v, n, scale):
        import torch_npu

        return torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            input_layout="BNSD",
            scale=scale,
            pre_tockens=2147483647,
            next_tockens=2147483647,
            head_num=n,
        )[0]

    def _quant_fa_fp8(self, q, k, v, n, d, scale):
        import torch
        import torch_npu

        from mindiesd.layers.flash_attn.fused_infer_attention_score import (
            fused_infer_attention_score_v2,
        )

        q_scale_dtype = torch.float8_e4m3fn
        qq, q_scale = torch_npu.npu_dynamic_mx_quant(q, dst_type=q_scale_dtype, axis=-1)
        kk, k_scale = torch_npu.npu_dynamic_mx_quant(k, dst_type=q_scale_dtype, axis=-1)
        vv, v_scale = torch_npu.npu_dynamic_mx_quant(v, dst_type=q_scale_dtype, axis=-1)
        out = fused_infer_attention_score_v2(
            qq,
            kk,
            vv,
            input_layout="BNSD",
            num_query_heads=n,
            softmax_scale=scale,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            query_quant_mode=6,
            key_quant_mode=6,
            value_quant_mode=8,
            dequant_scale_query=q_scale,
            dequant_scale_key=k_scale,
            dequant_scale_value=v_scale,
            out_dtype=q.dtype,
        )[0]
        return out

    def _quant_fa_mxfp4(self, q, k, v, n, d, scale):
        import torch
        import torch_npu

        from mindiesd.layers.flash_attn.common import AttentionParam
        from mindiesd.quantization.layer import (
            MXFP4_FA_SEQ_PAD_BASE,
            MXFP4_K_QUANT_MODE,
            MXFP4_Q_QUANT_MODE,
            MXFP4_V_QUANT_MODE,
            _dynamic_mx_quant_fa,
            _get_qfa_seqused,
            _pad_fa_seq_before_quant,
            _reshape_mxfp4_v_scale_for_fa,
        )

        query, s, padded_s = _pad_fa_seq_before_quant(q, MXFP4_FA_SEQ_PAD_BASE, "BNSD")
        key, kv_s, padded_kv_s = _pad_fa_seq_before_quant(k, MXFP4_FA_SEQ_PAD_BASE, "BNSD")
        value, _, _ = _pad_fa_seq_before_quant(v, MXFP4_FA_SEQ_PAD_BASE, "BNSD")
        batch_size = query.shape[0]
        seq_param = AttentionParam(
            batch_size, n, d, padded_s, padded_kv_s, torch.int32, str(query.device)
        )
        seqused_q, seqused_kv = _get_qfa_seqused(seq_param)

        quant_q, q_scale = _dynamic_mx_quant_fa(query, axis=-1)
        quant_k, k_scale = _dynamic_mx_quant_fa(key, axis=-1)
        quant_v, v_scale = _dynamic_mx_quant_fa(value, axis=2)
        v_scale = _reshape_mxfp4_v_scale_for_fa(v_scale, "BNSD")

        qfa_metadata = torch.ops.mindiesd.quant_flash_attn_metadata(
            num_heads_q=n,
            num_heads_kv=n,
            head_dim=d,
            q_quant_mode=MXFP4_Q_QUANT_MODE,
            k_quant_mode=MXFP4_K_QUANT_MODE,
            v_quant_mode=MXFP4_V_QUANT_MODE,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            batch_size=batch_size,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            q_dtype=torch_npu.float4_e2m1fn_x2,
            k_dtype=torch_npu.float4_e2m1fn_x2,
            v_dtype=torch_npu.float4_e2m1fn_x2,
            mask_mode=0,
            win_left=2147483647,
            win_right=2147483647,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_out="BNSD",
        )
        out, _ = torch.ops.mindiesd.quant_flash_attn(
            quant_q,
            quant_k,
            quant_v,
            q_scale,
            k_scale,
            v_scale,
            q_quant_mode=MXFP4_Q_QUANT_MODE,
            k_quant_mode=MXFP4_K_QUANT_MODE,
            v_quant_mode=MXFP4_V_QUANT_MODE,
            block_table=None,
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
            seqused_q=seqused_q,
            seqused_kv=seqused_kv,
            sinks=None,
            attn_mask=None,
            metadata=qfa_metadata,
            q_dtype=torch_npu.float4_e2m1fn_x2,
            k_dtype=torch_npu.float4_e2m1fn_x2,
            v_dtype=torch_npu.float4_e2m1fn_x2,
            q_descale_dtype=torch_npu.float8_e8m0fnu,
            k_descale_dtype=torch_npu.float8_e8m0fnu,
            v_descale_dtype=torch_npu.float8_e8m0fnu,
            softmax_scale=scale,
            mask_mode=0,
            win_left=2147483647,
            win_right=2147483647,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="BNSD",
            layout_kv="BNSD",
            layout_out="BNSD",
            return_softmax_lse=0,
        )
        if out.shape[2] != s:
            out = out[:, :, :s, :]
        return out
