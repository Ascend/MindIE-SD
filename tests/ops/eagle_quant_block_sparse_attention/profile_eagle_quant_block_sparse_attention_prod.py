# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
"""Profile one EQBSA or CANN FP8 BSA forward with pre-quantized inputs."""

import json
from tests.ops.eagle_quant_block_sparse_attention.eqbsa_cases import (
    CANN_BLOCK_KV,
    CANN_BLOCK_Q,
    parse,
    print_preview,
    scene_name,
    selections,
)
from tests.ops.eagle_quant_block_sparse_attention.eagle_quant_block_sparse_attention_accuracy_common import (
    contract,
    dut,
    initialize,
    make_inputs,
    make_source_inputs,
    metadata,
    to_device,
)


def profile_cann(s, device, torch):
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import (
        _adapt_mask_for_block_sizes,
        _fp8_quant_qkv,
        _get_rot_matrices,
        rain_fusion_attention_v3,
    )

    src, mask, parameters = make_source_inputs(s)
    query, key, value = (src[k].to(f"npu:{device}") for k in ("query", "key", "value"))
    del src
    q_rot, k_rot = _get_rot_matrices(query.device, query.dtype, s["head_dim"])
    q, k, v, qs, ks, vs = _fp8_quant_qkv(
        query,
        key,
        value,
        q_rot,
        k_rot,
        block_size_q=CANN_BLOCK_Q,
        block_size_kv=CANN_BLOCK_KV,
        layout=s["layout"],
    )
    mask = _adapt_mask_for_block_sizes(mask, CANN_BLOCK_Q, CANN_BLOCK_KV, pool_size=128).to(query.device)
    torch.npu.synchronize()
    data = dict(
        query=q,
        key=k,
        value=v,
        query_scale=qs,
        key_scale=ks,
        value_scale=vs,
        block_sparse_mask=mask,
    )
    info = metadata(data, s, parameters)
    info["scenario"] = scene_name(s)
    info["attributes"]["block_shape"] = [CANN_BLOCK_Q, CANN_BLOCK_KV]
    info["generation"]["encoding"] = "Q/K Hadamard; Q/K/V block FP8 E4M3; adjacent KV mask blocks any-merged"
    print(json.dumps(info), flush=True)
    print("START single CANN FP8 BSA forward", flush=True)
    with torch.inference_mode():
        output = rain_fusion_attention_v3(
            q,
            k,
            v,
            block_sparse_mask=mask,
            scale=s["head_dim"] ** -0.5,
            head_num=s["num_query_heads"],
            num_key_value_heads=s["num_kv_heads"],
            input_layout=s["layout"],
            actual_seq_lengths=[s["query_seq_len"]],
            actual_seq_lengths_kv=[s["kv_seq_len"]],
            block_size_q=CANN_BLOCK_Q,
            block_size_kv=CANN_BLOCK_KV,
            inner_precise=4,
            q_dequant_scale=qs,
            k_dequant_scale=ks,
            v_dequant_scale=vs,
        )
        torch.npu.synchronize()
    if output.shape != q.shape or output.dtype != torch.bfloat16:
        raise RuntimeError(f"unexpected CANN FP8 BSA output shape={output.shape} dtype={output.dtype}")


def main():
    args, plan = parse("profile")
    if args.dry_run:
        print_preview(args, plan, perf=True)
        return
    s = selections(plan, perf=True)[0]
    torch, _ = initialize(args.device_id)
    print(f"START {scene_name(s)} input generation/quantization", flush=True)
    if s["impl"] == "cann_fp8_bsa":
        profile_cann(s, args.device_id, torch)
        print("DONE one CANN FP8 BSA forward; use msprof op Task Duration(us)", flush=True)
        return
    cpu, parameters = make_inputs(s)
    print("START copying encoded inputs to physical NPU", flush=True)
    inputs = to_device(cpu, args.device_id)
    del cpu
    torch.npu.synchronize()
    print(json.dumps(metadata(inputs, s, parameters)), flush=True)
    print("START single target forward", flush=True)
    with torch.inference_mode():
        outputs = dut(inputs, s)
        torch.npu.synchronize()
    contract(outputs, s, finite=False)  # Metadata only; no post-DUT reduction or comparison.
    print("DONE one EQBSA forward; use msprof op Task Duration(us)", flush=True)


if __name__ == "__main__":
    main()
