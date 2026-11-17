# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.

"""Complete current/legacy LA layers, starting from the same business inputs."""

import torch

from mindiesd.layers.flash_attn import ascend_laser_attention as la
from mindiesd.layers.flash_attn.common import AttentionParam


class LegacyLaserAttention:
    """Dtype path from 07b8d842; keep casts inside the measured layer call.

    Padding is shared with production because its implementation is unchanged.
    Pre/postprocessing are frozen here; production helpers must not replace them.
    """

    @staticmethod
    def la_preprocess_input(query, key, value):
        if query.dtype != torch.float16:
            query = query.to(torch.float16)
            key = key.to(torch.float16)
            value = value.to(torch.float16)
        query = la.AscendLaserAttention.pad(query, 256, -2)
        query = la.AscendLaserAttention.pad(query, 128, -1)
        key = la.AscendLaserAttention.pad(key, 256, -2)
        key = la.AscendLaserAttention.pad(key, 128, -1)
        value = la.AscendLaserAttention.pad(value, 256, -2)
        value = la.AscendLaserAttention.pad(value, 128, -1)
        return query.contiguous(), key.contiguous(), value.contiguous()

    @staticmethod
    def la_postprocess_output(attention_out, dtype, qseqlen, head_dim):
        if dtype != attention_out.dtype:
            attention_out = attention_out.to(torch.float16).to(dtype)
        return attention_out[:, :, :qseqlen, :head_dim]

    @classmethod
    def forward_attn_bnsd(cls, attn_param, query, key, value, mask=None, scale=None):
        if la.is_a5_device():
            raise la.ParametersInvalid(la._A5_LA_UNSUPPORTED_MSG)
        if not attn_param.head_first:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        if mask is not None:
            mask = ~mask.to(torch.bool)
        new_query, new_key, new_value = LegacyLaserAttention.la_preprocess_input(query, key, value)
        pre_tokens = la.MAX_TOKEN
        if attn_param.kv_seqlen % 256 != 0:
            pre_tokens = (attn_param.kv_seqlen // 256 + 1) * 256 - attn_param.kv_seqlen
        _, output = la.ops.laser_attention(
            new_query,
            new_key,
            new_value,
            None,
            None,
            None,
            scale,
            attn_param.head_num,
            "BNSD",
            1.0,
            pre_tokens,
            1,
            True,
        )
        output = LegacyLaserAttention.la_postprocess_output(
            output,
            query.dtype,
            attn_param.q_seqlen,
            attn_param.head_dim,
        )
        if not attn_param.head_first:
            output = output.transpose(1, 2)
        return output


LAYERS = {
    "current": la.AscendLaserAttention.forward_attn_bnsd,
    "legacy": LegacyLaserAttention.forward_attn_bnsd,
}


def make_inputs(device, seed=20260921):
    """Generate the agreed raw BSND BF16 inputs, before any layer conversion."""
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def generate(shape):
        uniform = torch.empty(shape).uniform_(-5, 5, generator=generator)
        mu = torch.empty(()).uniform_(-5, 5, generator=generator).item()
        sigma = torch.empty(()).uniform_(0.1, 2, generator=generator).item()
        normal = torch.empty(shape).normal_(mu, sigma, generator=generator)
        choose_uniform = torch.rand(shape, generator=generator) < 0.5
        return torch.where(choose_uniform, uniform, normal).to(torch.bfloat16).to(device)

    query = generate((1, 5280, 24, 128))
    key = generate((1, 8800, 24, 128))
    value = generate((1, 8800, 24, 128))
    param = AttentionParam(1, 24, 128, 5280, 8800, torch.bfloat16, False)
    return param, query, key, value
