# Version Compatibility Notes

## Product Version Information

| Item | Content |
| -------- | ------ |
| Product Name | MindIE SD |
| Product Version | 3.2.0 |
| Version Type | Release |
| Maintenance Cycle | Three Months |

## Related Product Version Compatibility

| Product Name | Version |
| -------- | ------ |
| CANN | 9.2.0 |
| TorchNPU | 26.2.0 |
| MindCluster | 26.2 |
| CCAE | iMaster CCAE V100R026C10SPC100 |
| Ascend HDK | See [CANN Version Compatibility Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/9.0.1release-notes.md) for version compatibility |

# Version Compatibility

MindIE SD components must be used with matching versions. Do not mix components across different versions.

**Table 1** Software Version Compatibility

`Y` indicates a supported combination, `/` indicates no compatibility entry, and `To be confirmed` indicates that compatibility must be confirmed and documented before release.

| MindIE SD Version | CANN 9.2.0 | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1 | CANN 8.5.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y | / | / |
| 3.1.0 | / | Y | Y | Y | / | / |
| 3.0.0 | / | / | / | Y | Y | Y |
| 2.3.0 | / | / | / | / | / | Y |

| MindIE SD Version | MindCluster 26.2 | MindCluster 26.1.0 | MindCluster 26.0.0 | MindCluster 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y |
| 3.1.0 | / | Y | Y | Y |
| 3.0.0 | / | / | Y | Y |
| 2.3.0 | / | / | / | Y |

| MindIE SD Version | TorchNPU 26.2.0 | TorchNPU 26.1.0 | TorchNPU 26.0.0 | TorchNPU 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y |
| 3.1.0 | / | Y | Y | Y |
| 3.0.0 | / | / | Y | Y |
| 2.3.0 | / | / | / | Y |

| MindIE SD Version | CCAE iMaster CCAE V100R026C10SPC100 | CCAE iMaster CCAE V100R026C00SPC010 | CCAE iMaster CCAE V100R025C30SPC100 |
| ----------------- | ---------- | ---------- | ---------- |
| 3.2.0 | To be confirmed | To be confirmed | To be confirmed |
| 3.1.0 | Y | Y | Y |
| 3.0.0 | / | Y | Y |
| 2.3.0 | / | / | Y |

# Version Usage Notes

- The following capabilities are only available on 950PR&950DT products: the `quant_four_over_six_a5` quantization operator, the MXFP4 mode of rf_v3 block sparse attention (`precision="mxfp4"`), the EQBSA qmode1 mix path, and the fused path of the FFN fused operator (other platforms automatically fall back to the native operator chain).
- The vLLM-Omni joint images only support the linux/amd64 platform.

# 3.2.0 Release Notes

## New Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Functional quantized attention API. Introduces the public API `mindiesd.quant_attention`, which consolidates quantized attention execution into a reusable functional interface that accepts floating-point Q/K/V directly. `precision` supports FP8, MXFP8, and MXFP4; BNSD/BSND layouts, optional Q/K rotation matrices, and attention window configuration are supported. `FP8RotateQuantFA` shares the same execution logic as the public entry, and MXFP4/MXFP8 integration retains the legacy timestep fallback. |
| 2    | FP8 attention mode enum and FIA full-quantization path extension. Adds the `FP8FAMode` enum (`HIGH_PRECISION` (default) and `C8V16_TILING512`); `QuantConfig`/`OnlineQuantConfig` gain the `fp8_fa_mode` option, which binds the quantization block, `value_quant_mode`, and `inner_precise` in one setting. The FIA operator adds an FP8 per-block V512 quantization path and a C8V16 (inner_precise=4) full-quantization path. |
| 3    | Eagle block sparse attention operator family (EQBSA/EBSA). Adds `eagle_quant_block_sparse_attention`, a quantized block sparse attention operator (Q/K INT8 + V FP8 mixed precision, combining quantization and block sparsity to reduce long-sequence overhead), with the qmode1 mix path completed for 950PR&950DT products; adds `eagle_block_sparse_attention`, a non-quantized block sparse attention operator with BlockK support relaxed to 64 (16-aligned); the EagleQBSA of MiniMax-H3 is routed through rf_v3 sparse attention. |
| 4    | BF16-to-FP4 adaptive quantization operator. Adds the `quant_four_over_six_a5` operator, supporting 4/6 adaptive dynamic MX quantization from BF16 input to FP4 (FLOAT4_E2M1) with E8M0 scaling and blocksize=32, adapted to 950PR&950DT products. |
| 5    | FFN fused operator. Adds an FFN forward fused operator that merges the linear→act→linear sequence (three kernel launches) into a single kernel; activations include gelu/silu/swiglu; adapted to 950PR&950DT products, with automatic fallback to the native operator chain on other platforms. Provides the Python API `mindiesd.layers.eagle_ffn_linear`. |
| 6    | USP sequence-parallel attention. Adds the `mindiesd.layers.usp.usp_attention` interface, combining Ulysses sequence parallelism, KV AllGather, communication quantization (FP8 E4M3/MXFP8, with quantized payload and scale communicated separately), and attention computation in a single call, with head-chunked dual-stream overlap pipelined execution. |
| 7    | Sparse attention enhancements. `sparse_attention` gains the `video_spans` parameter, completing spatial rearrangement of multiple video clips, global block selection, and original token order restoration in a single RainFusion call, while non-video tokens remain dense; the BLKK alignment constraint of SparseLinearAttention is relaxed to 16 on 950PR&950DT products. |
| 8    | MXFP4 block sparse attention. rf_v3 sparse attention adds the `precision="mxfp4"` mode: Q/K are quantized to FP4 E2M1 + E8M0 scaling after Hadamard rotation and executed via `aclnnBlockSparseAttentionV3`; only supported on 950PR&950DT products. |
| 9    | Compilation fusion pattern extension. Wan2.2 adds 5 categories of compilation fusion patterns (qk_norm, adaLN modulation, RoPE, FastGelu, residual gate) with bf16/fp32 dual-precision variants and model-level bf16 precision unification; MiniMax-H3 adds RMSNorm/RoPE fusion and AdaLN/SwiGLU/residual-gate Triton single-kernel fusion; Qwen-Image adds RoPE and norm_out AdaLN fusion patterns; 13 new `FusionPatterns` switches are added. |
| 10   | GMMFR fusion for MXFP8 MoE. Supports GMMFR fusion in MXFP8-quantized MoE scenarios, reducing kernel launches and data movement in MoE forward computation. |
| 11   | New fused operators `mul_add` and `norm_rope_concat`. `mul_add` implements the gated-broadcast fusion y = a + b*c; `norm_rope_concat` fuses LayerNorm/RMSNorm + RoPE + KV Concat into a single operator, with support for outputting mean/rstd in training mode. |
| 12   | mindie_bench operator performance benchmarking tool. Adds the `benchmarks/scripts/mindie_bench.py` CLI (run/report/compare subcommands) with a built-in NPU backend and FA/BSA/GMM/MM operator definitions, supporting sequence-length sweeps, baseline snapshots, HTML/CSV reports, and CI threshold gating. |
| 13   | Build acceleration. Adds the `SKIP_ALL_OPS_PLUGIN_BUILD` environment variable: operator and plugin compilation is skipped when cached artifacts are complete and compatible, with automatic fallback to a full build otherwise; the default behavior is unchanged. |
| 14   | DummyRun toolchain enhancement. Quantization options are unified into `--quant {fp32,bf16,w8a8}`, with device-aware w8a8 quantization format selection (W8A8-MXFP8 on 950PR&950DT products, W8A8-DYNAMIC INT8 on Atlas 800I A2 inference server / Atlas 800I A3 SuperPoD server); Wan2.2 dummy run adds frame count and resolution configuration. |
| 15   | Deployment and ecosystem extension. The vLLM-Omni joint images are aligned to base image v0.28.0 with a new variant for 950PR&950DT products; new inference deployment guides for MiniMax-H3, Qwen-Image-2.1, and DiffSynth-Engine, plus a FLUX.1-dev Cache-DiT example, are added. |

## Modified Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | MoE Dispatcher parameter renaming and semantic clarification. `tokens_full` is renamed to `inputs_sharded` (with the default value inverted accordingly; semantics unchanged), and `reduce_results` is renamed to `reduce_routed_out`; `top_k` becomes a required parameter of `resolve_dispatcher_class`; `fused_moe`/`moe` gain `return_dispatcher_type` to return the actually selected dispatcher type. See "Upgrade Impact" for migration notes. |
| 2    | TIK operator build fault tolerance. When the CANN toolkit no longer ships the TBE/TIK project templates, the build now detects the absence and skips TIK operator compilation with a WARNING instead of failing the entire build; the AscendC operator build flow is unaffected. |
| 3    | Frequency regulator plugin execution adjustment. Manual stream synchronization and D2H/H2D state copies are replaced by unified fully asynchronous execution through the `EXEC_NPU_CMD` framework, which now manages the workspace/executor/ACL Tensor lifecycles. |
| 4    | DummyRun example dependency adjustment. `examples/dummy_run` upgrades its diffusers dependency from 0.38.0 to 0.40.0 and transformers to 5.2.0. |

## Removed Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Removed the redundant arch35 kernel copies of the FIA and Eagle operators (249 files in total); header includes are unified to the canonical path, and a missing canonical path now fails at compile time instead of silently falling back. This change only affects internal C++ source organization: no public API is removed, and runtime algorithms and precision paths are unchanged. |

## API Change Notes

API changes include additions, modifications, deprecations, and removals. API changes only reflect code-level modifications and do not include documentation improvements in language, formatting, or links.

- New: APIs introduced in this version.
- Modified: APIs changed compared to the previous version.
- Deprecated: APIs that cease evolution from the declared version and may be removed one year after declaration.
- Removed: APIs removed in this version.

| Class Name / API Signature | Change Type | Description |
| :----------- | :------- | :------- |
| def mindiesd.quant_attention | New | Functional public API for quantized attention; `precision` supports fp8/mxfp8/mxfp4 |
| class mindiesd.FP8FAMode | New | FP8 attention mode enum (HIGH_PRECISION / C8V16_TILING512) |
| def mindiesd.layers.eagle_ffn_linear | New | FFN fused operator API; takes the fused path on 950PR&950DT products |
| def mindiesd.layers.residual_gate_add | New | Residual gate fused operator API |
| def mindiesd.layers.usp.usp_attention | New | USP sequence-parallel attention API (module-level export, not added to the top-level mindiesd namespace) |
| New torch.ops.mindiesd operators | New | eagle_block_sparse_attention, eagle_quant_block_sparse_attention, quant_four_over_six_a5, mul_add, norm_rope_concat |
| def mindiesd.fused_moe | Modified | Parameter `tokens_full` renamed to `inputs_sharded` (default value inverted accordingly), `reduce_results` renamed to `reduce_routed_out`; new `return_dispatcher_type` parameter |
| def mindiesd.sparse_attention | Modified | New `video_spans` parameter for multi-video sparse attention; rf_v3 adds the `precision="mxfp4"` mode |
| class mindiesd.QuantConfig | Modified | New `fp8_fa_mode` option |
| class mindiesd.OnlineQuantConfig | Modified | New `fp8_fa_mode` option |

## Resolved Issues

| No. | Category | Issue Description |
| :--- | :--- | :--- |
| 1 | Operator & Compilation | Some CANN toolkits no longer ship the TBE/TIK project templates, causing msopgen to fail and the entire operator build to fail. |
| 2 | Operator & Compilation | The EQBSA operator failed to compile under CANN 9.2 due to a non-static `__simd_vf__`. |
| 3 | Operator & Compilation | Redefined `Ops::Base::ToString` in quant_four_over_six_a5 and eagle_ffn caused full-build link failures. |
| 4 | Operator & Compilation | The eagle_ffn operator's `math_util.h` include path was incompatible across CANN versions, causing build failures. |
| 5 | Operator & Compilation | Since torch_npu 2.9, the dtype cast decomposition change caused all RMSNorm/RoPE compilation patterns to mismatch. |
| 6 | Operator & Compilation | The Wan residual-gate compilation pattern incorrectly matched FLUX's reversed operand order (x + gate * y), causing incorrect broadcast semantics. |
| 7 | Stability & Precision | RainFusion rf_v2 produced a block mask inconsistent with the NPU kernel tiling on irregularly aligned video grids (e.g., MiniMax H3 1344×768), causing corrupted video output. |
| 8 | Stability & Precision | The `attn_cache` of `lru_cache_by_attn_param` was a module-level global dictionary shared by all decorated functions, causing cross-function cache pollution. |
| 9 | Stability & Precision | `MXFP8RotateQuantFA` passed sequence lengths as device tensors, triggering `aten._local_scalar_dense` and causing a graph break at every attention layer during DiT aclgraph capture. |
| 10 | Stability & Precision | EPLB `UpdateTaskTransfer.update_emit_task` used a blocking `queue.put`, deadlocking the worker thread when the instruction queue was full. |
| 11 | Stability & Precision | The frequency regulator plugin's manual stream synchronization and D2H/H2D state copies caused resource management issues, leading to coredump/507035 errors. |
| 12 | Stability & Precision | Quantized layers modified `self.bias` dtype in place inside forward, repeatedly invalidating Dynamo guards and triggering recompilation. |
| 13 | Compatibility | A new Atlas 800I A3 SuperPoD server model reports soc_version=256, which fell outside the recognition range, causing device recognition and operator loading failures. |

## Known Issues

| No. | Category | Issue Description |
| :--- | :--- | :--- |
| 1 | Operators | Matrix multiplication operators based on CATLASS and Triton are being continuously added. |
| 2 | Usability | End-to-end integration and runnable examples for third-party frameworks such as Cache DiT and DiffSynth-Engine are being continuously improved. |
| 3 | Performance | Parallel communication-computation overlap and fusion schemes are being continuously extended, and joint optimization of EPLB and MoE fusion is in progress. |

# Upgrade Impact

## Impact on Running Systems During Upgrade

- Business impact

  Software version upgrade will cause service interruption.

- Network communication impact

  No impact on network communication.

## Impact on Running Systems After Upgrade

- MoE API parameters have been renamed: existing code calling `fused_moe(..., tokens_full=..., reduce_results=...)` with keyword arguments must migrate to `inputs_sharded=` / `reduce_routed_out=`; code calling `resolve_dispatcher_class` directly must pass `top_k` explicitly. The default-value semantics are unchanged (the former `tokens_full=True` is equivalent to the current `inputs_sharded=False`).
- Under newer CANN toolkits (which no longer ship the TBE/TIK project templates), TIK operators are automatically skipped during build; build pipelines that depend on TIK operator artifacts should take note.
- The dummy run example CLI has changed: the `--compute-precision` and boolean `--quant` arguments have been removed in favor of the unified `--quant {fp32,bf16,w8a8}`; the diffusers dependency of `examples/dummy_run` has been upgraded to 0.40.0.
- The vLLM-Omni joint images have been upgraded to base image v0.28.0 (Atlas 800I A2 inference server / Atlas 800I A3 SuperPoD server) with a new variant for 950PR&950DT products, and image tag naming has been updated accordingly.

# Virus Scan Results

No virus scan results are available yet. They will be supplemented before the release.

# Vulnerability Patch List

No new vulnerability patches in this version.

# Revision History

| Document Version | Release Date | Description |
| -------- | -------- | -------- |
| 01 | 2026-10-15 | First official release. |
