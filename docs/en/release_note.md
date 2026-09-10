# Version Compatibility Description

## Product Version Information

| Item | Content |
| -------- | ------ |
| Product Name | MindIE SD |
| Product Version | 3.1.0 |
| Version Type | Official|
| Maintenance Period | Three months |

## Version Compatibility of Related Products

| Product Name | Version |
| -------- | ------ |
| CANN | 9.1.0 |
| TorchNPU | 26.1.0 |
| CCAE | iMaster CCAE V100R026C10SPC100 |
| Ascend HDK | For version compatibility, see [CANN Version Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/9.0.1release-notes.md) |

## Version Compatibility Description

MindIE components must be used together as a matched set. Do not mix components across versions.

**Table 1** Software version compatibility description

| MindIE SD Version | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1 | CANN 8.5.0 |
| -------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 3.1.0          | Y          | Y          | Y          | /          | /          |
| 3.0.0          | /          | /          | Y          | Y          | Y          |
| 2.3.0          | /          | /          | /          | /          | Y          |

| MindIE SD Version | TorchNPU 26.1.0 | TorchNPU 26.0.0 | TorchNPU 7.3.0 |
| -------------- | --------------- | --------------- | -------------- |
| 3.1.0          | Y               | Y               | Y              |
| 3.0.0          | /               | Y               | Y              |
| 2.3.0          | /               | /               | Y              |

| MindIE SD Version | CCAE iMaster CCAE V100R026C10SPC100 | CCAE iMaster CCAE V100R026C00SPC010 | CCAE iMaster CCAE V100R025C30SPC100 |
| -------------- | -------------- | -------------- | -------------- |
| 3.1.0          | Y              | Y              | Y              |
| 3.0.0          | /              | Y              | Y              |
| 2.3.0          | /              | /              | Y              |

## Version Usage Notes

None

## 3.1.0 Change Description

### New Features

| Number | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Ported the Fused Infer Attention Score (FIA) operator to the MindIE SD managed domain. The operator now supports a dedicated FP8 E4M3FN per‑block path, with host checker and tiling key validation to reject unsupported quant modes (noquant/INT8/HIFLOAT8/MXFP8), providing an extensible foundation for low‑precision and sparse attention acceleration. |
| 2    | Added full MoE inference support, including a fused_moe kernel interface for NPU‑accelerated forward computation. Integrated `torch_npu.npu_moe_gating_top_k` and `npu_moe_gating_top_k_softmax` to offload expert selection to the NPU, reducing kernel launches and data movement. Extended MoE quantization with both W8A8 MXFP8 and INT8 paths, validated with HunyuanImage-3.0 in vLLM-Omni. |
| 3    | Enhanced sparse attention (SLA/BSA) with an AscendC backend for Block Sparse Attention. Upgraded `aclnnBlockSparseAttention` to V2 with FP8 quantization compatibility and expanded block size support (q=128, kv=256/512). Automatic fallback to V1 is provided for older CANN versions that only expose the V1 symbol. |
| 4    | Extended quantization capabilities with online quantization (OnlineQuantConfig) supporting FA and MM algorithms with fallback options. Added MXFP8 dynamic quantization (MXFP8_DYNAMIC) that applies rotation and MXFP8 quantization to Q/K before FA, and introduced MXFP4 quantized Flash Attention (quant_flash_attn) with deployment‑side MXFP4 FA logic. |
| 5    | Introduced a new Wheel packaging mode (MINDIESD_WHEEL_MODE=multi_torch). A single wheel now supports torch 2.6, 2.7, 2.8, 2.9, and 2.10, dynamically selecting the appropriate `libPTAExtensionOPS.so` at runtime based on `torch.__version__`. The default fixed‑torch‑version build remains unchanged. |
| 6    | Added unified device detection (`is_a5_device()`) for Ascend 950PR and Ascend 950DT. Public APIs such as `attention_forward` automatically route to `fused_attn_score` on these devices, with clear error messages and migration guidance for direct calls to deprecated operators. |
| 7    | Added support for a Frequency Regulator operator plugin, including a C++ wrapper, aclnn two‑stage flow, BackendSelect registration, and Python API export. |
| 8    | Streamlined deployment and ecosystem integration: merged vLLM-Omni and MindIE-SD into a unified Docker image with consistent naming, version tagging, and OCI metadata; switched the pip installation source to the public PyPI; and added a developer skill set under .agents/skills/ (ascend‑deploy, auto‑optimization, code‑standards, compilation‑dev). |

### Modified Features

| Number | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Adjusted operator compilation scope. Removed the original CANN version detection and operator filtering logic, and unified compilation to build all operators for all platforms (`ascend910`/`ascend910b`/`ascend910_93`/`ascend950`) in a single pass. This fixes the issue where generic operators such as `laser_attention` and `ada_block_sparse_attention` failed to generate kernels for 910B/910 platforms under CANN ≥ 9.0. Compilation behavior can now be overridden via the `ASCEND_OP_NAME` and `ASCEND_COMPUTE_UNIT` environment variables.|
| 2    | Adjusted MoE Dispatcher default strategy. Previously, the strategy defaulted to `static` for the Atlas 800I A2 Inference Server and `dynamic` for the Atlas 800I A3 SuperPoD Server/Ascend 950PR/Ascend 950DT. It is now selected based on the relationship between `top_k` and `ep_size`, with `dynamic` preferred in MoE EP scenarios to mitigate the performance degradation caused by cross-node all-to-all communication.|
| 3    | Refactored logging system. Unified log output for the MindIE SD Python module, with component identifiers now included in both default and verbose modes. Reduced verbosity in default INFO logs (normal workflow messages downgraded to DEBUG) and enhanced WARNING/ERROR logs with clearer problem descriptions, potential root causes, and remediation suggestions. |
| 4    | Added security compilation options. Incorporated security compilation and linking flags into the operator build following security hardening guidelines, improving the security of the generated artifacts.  |

### Deleted Features

| Number | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Deleted the `csrc/ops/ascendc/` directory and all its operator source files, and consolidated them under per-op directory management. Synchronously updated the source file mappings in test cases that point to the deleted directory. |

### Interface Changes

The interface changes described in this section include additions, modifications, deprecations, and removals. Interface changes reflect only code-level modifications and do not include improvements to the documentation itself in terms of language, formatting, links, and so on.

- New: indicates an interface newly added in this version.

- Modified: indicates that this interface has been modified compared with the previous version.

- Deprecated: indicates that this interface stops evolving as of the version in which the deprecation statement is made, and may be removed one year after the statement.

- Deleted: indicates that the interface is removed in this version.

| Class Name/API Prototype | Category | Change Description |
| :----------- | :------- | :------- |
| def mindiesd.frequency_regulator | New | Frequency regulator operator interface |
| def mindiesd.fused_moe | New | Fused MoE operator interface, supporting open-source frameworks to perform MoE forward computation on NPUs |
| class mindiesd.OnlineQuantConfig | New | Online quantization configuration class |
| class mindiesd.TimestepManager | New | Timestep manager class |
| class mindiesd.TimestepPolicyConfig | New | Timestep policy configuration class |
| class mindiesd.QuantConfig | New | Quantization configuration class (export restored; not exported in mindiesd.__init__ in 3.0.0) |
| def mindiesd.sparse_attention | Modification | Upgraded to aclnnBlockSparseAttentionV2, compatible with FP8 quantization; automatically falls back to V1 on older CANN versions that contain only V1 aclnn symbols |

### Resolved Issues

| Serial Number | Category | Problem Description |
| :--- | :--- | :--- |
| 1 | Security | ZMQ shared memory handle broadcast uses pickle to directly deserialize the socket byte stream, posing an arbitrary code execution (RCE) risk. |
| 2 | Security | The LayerNorm operator has a size_t computation underflow that leads to out-of-bounds iteration and potential memory corruption. |
| 3 | Operator and Compilation | In CANN≥9.0 environments, general-purpose operators such as laser_attention and ada_block_sparse_attention do not generate kernels for the 910B/910 platforms, resulting in missing operators on the corresponding platforms. |
| 4 | Operator and Compilation | TIK operator build linking fails when CMake≥4.1.0. |
| 5 | Operator and Compilation | When ABSA and FIA operators are compiled simultaneously, a tiling registration key conflict causes the process to crash and exit (core dump). |
| 6 | Operator and Compilation | Missing header files or incorrect include paths during operator compilation cause build failures. |
| 7 | Stability and Precision | Repeated calls to `enable_offload` register the forward hook multiple times, and the forward event recording order is incorrect. |
| 8 | Stability and Precision | When triton is not installed, `import mindiesd` triggers `std::bad_alloc`. |
| 9 | Stability and Precision | A defect in the input_layout validation of the `sparse_block_estimate` fake operator leads to uninitialized variables and out-of-bounds access. |
| 10 | Stability and Precision | MoE W8A8 dynamic quantization has abnormal precision. |
| 11 | Stability and Precision | During ACLGraph capture, the static input data_ptr contains stale data, affecting execution precision. |
| 12 | Stability and Precision | For some Hunyuan models using the FA512 operator, the V-side per-block dequant scale is blocked by a hardcoded block_size=256 validation. |
| 13 | Stability and Precision | On older CANN versions that contain only V1 aclnn symbols, block_sparse_attention V2 fails at dlsym, making the path unavailable. |
| 14 | Logging | The log output points, formats, and levels of each module are inconsistent, with redundant information in default scenarios and insufficient information in abnormal scenarios. |
| 15 | Testing | The block_sparse_attention UT supports only Ascend 950PR / Ascend 950DT devices and cannot be executed in other environments. |

### Known Issues

| Serial Number | Category | Problem Description |
| :--- | :--- | :--- |
| 1 | Operator | Continuously supplement matrix multiplication operators implemented based on CATLASS and Triton. |
| 2 | Usability Improvement | Continuously improve end-to-end integration and runnable examples for third-party frameworks such as Cache DiT and DiffSynth-Engine. |
| 3 | Performance Improvement | Continuously expand parallel compute-communication overlap and fusion solutions, and advance the joint optimization of EPLB and MoE fusion. |

## Upgrade Impact

### Impact on the Current System During the Upgrade

- Impact on services

  The software version upgrade causes service interruption.

- Impact on network communication

  There is no impact on network communication.

### Impact on the Current System After the Upgrade

- If the original code directly calls legacy paths of FIA such as noquant/INT8/HIFLOAT8/MXFP8/PA/mask/rope/prefix/sparse, it must be migrated to the FP8 E4M3FN per-block path; otherwise, it will be blocked by the dual verification of host checker and tiling key.

- Log levels and content have been adjusted: default INFO logs are streamlined, and services that rely on logs for monitoring and alerting need to confirm whether the log content is sufficient.

## Vulnerability Patch List

| Serial Number | Category | Vulnerability Description |
| :--- | :--- | :--- |
| 1 | Security | The ZMQ shared memory handle broadcast uses pickle to deserialize the socket byte stream, posing an arbitrary code execution (RCE) risk. A restricted deserializer SafeUnpickler has been introduced, which only allows loading safe types from the whitelist. |
| 2 | Security | The size_t computation underflow in the LayerNorm operator leads to out-of-bounds iteration and potential memory corruption. A `TORCH_CHECK` validation has been added before dimension computation. |
