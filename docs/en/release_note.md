# Version Mapping

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-06-05T08:13:32.356Z pushedAt=2026-06-08T10:50:46.023Z -->

## Product Version Information

| Item | Content |
| -------- | ------ |
| Product Name | MindIE SD |
| Product Version | 3.0.0 |
| Version Type | Official release |
| Maintenance Period | Three months |

## Related Product Version Mapping

| Product Name | Version |
| -------- | ------ |
| CANN | 8.5.1 |
| Ascend Extension for PyTorch | 7.3.0 |
| Ascend HDK | For version compatibility details, see [CANN Version Compatibility](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/releasenote/releasenote_0000.html) (Note: The HDK versions compatible with CANN 8.5.1 and CANN 8.5.0 are the same) |

# Version Compatibility

MindIE SD components must be used together as a matched set. Do not mix components across different versions.

**Table 1** Software version compatibility notes

| CANN | Ascend Extension for PyTorch |
| ---- | ---------------------------- |
| 8.5.1 | 7.3.0                        |

# Important Notes

N/A

# 3.0.0 Release Notes

## Added Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Enhanced quantization capabilities: Added support for FA dynamic FP8 and a new W4A4_DYNAMIC quantization format. Extended the common logic for W4A4 quantization algorithms and introduced `W4A4MXFP4DualQuantLinear`, improving adaptability and deployment flexibility across various quantization scenarios. |
| 2    | Enhanced operator and plugin capabilities: Added `aclnn LayerNorm` plugin and public interface. Introduced `adaLayerNormV2` plugin and its layer implementation. Extended `aclnn` capabilities with `sparse_block_estimate`, `block_sparse_attention`, `laser_attention`, and `la_preprocess`, improving operator coverage.. |
| 3    | Enhanced runtime capability: Added multi-instance shared memory support, allowing multiple instances to share weight memory and reduce redundant usage. Introduced block-level CPU offload, enabling fine-grained dynamic swapping of modules between CPU and NPU to alleviate memory pressure. |
| 4    | Enhanced Attention and layout adaptation: `attention_forward` and `rf_v2` support BNSD layout input, and support specifying the use of FA through environment variables, reducing the adaptation cost for upper-layer model integration. |
| 5    | Enhanced scheduling and serving capabilities: Added DyEPLB scheduling and new service example support, and completed service-side synchronization adaptation and accuracy fixes for wan2.2, improving usability in deployment and inference scenarios. |
| 6    | Optimized low-level implementation of quantization operators: Migrated all torch-atb based quantization operators to native aclnn operators, improving compatibility and stability. Added support for torch.compile and other compilation optimizations, enhancing framework adaptability. |

## Modified Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Removed the `_mindie_sd` suffix from custom plugin operator naming, unified the namespace to `mindiesd`, and further standardized naming conventions. |
| 2    | In FA quantization scenarios, adjusted the FIA operator output format to be consistent with the input query format, reducing compatibility issues caused by format inconsistency. |
| 3    | Completed adaptation for newly added constraints of the npu_quant_matmul operator, reducing integration risks caused by new constraints. |
| 4    | Completed systematic adaptation of the aclnn compilation project, enhancing the build chain, directory management, and error handling capabilities, thereby improving the efficiency and stability of operator project builds. |

## Deleted Features

N/A

## API Change Description

The following documents API changes, including additions, modifications, deprecations, and deletions. Changes reflect only code-level updates, not documentation improvements such as language, formatting, or link adjustments.

- Addition: New APIs introduced in this release.

- Modification: APIs changed from the previous version.

- Deprecation: APIs that are no longer evolved as of this release and may be removed after one year.

- Deletion: APIs deleted in this release.

| Class Name/API Prototype | Change Category | Change Description |
| :----------- | :------- | :------- |
| • def mindiesd.layernorm_scale_shift<br>• def mindiesd.fast_layernorm<br>• def mindiesd.sparse_attention | Added | Newly added API |
| • class mindiesd.Linear<br>• class mindiesd.QuantFA | Deleted | Deleted API |

## Resolved Issues

| Serial No. | Category | Problem Description |
| :--- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 1 | Installation and Compatibility | Missing `libopapi.so` during post-installation testing of MindIE-SD caused failures in basic functionality and validation. |
| 2 | Installation and Compatibility | Incompatibility with newer torch versions, which impacted usage alongside updated inference images. |
| 3 | Installation and Compatibility | The build package was missing a plugin, which affected installation integrity and plugin loading. |
| 4 | Operator and Compilation Scenarios | After enabling compile in a new environment, Flux.1-dev failed to call aclnnAdaLayerNorm, rendering the compilation acceleration path unavailable. |
| 5 | Operator and Compilation Scenarios | Incorrect API usage in `test_rainfusionattention.py` caused related test execution failures. |
| 6 | Cache and Test Quality | The block_end validation in the DiT Cache Agent was inconsistent with the left-closed, right-open rule, affecting cache scenario usage. |
| 7 | Cache and Test Quality | The test accuracy comparison method was limited, using only cosine similarity, resulting in an incomplete accuracy evaluation dimension. |

## Known Issues

| Serial No. | Category | Problem Description |
| :--- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 1 | Operators | Lack of matrix multiplication operators implemented based on CUTLASS and Triton. |
| 2 | Usability Improvements | Need to support more extensions, such as cache-dit. |
| 3 | Performance Improvements | Need to support more parallel compute-communication overlapping and fusion solutions. |

# Upgrade Impact

## Impact on the Current System During the Upgrade

- Impact on Services

  Service interruption will occur during the software version upgrade.

- Impact on Network Communication

  None

## Impact on the Current System After Upgrade

N/A

# Vulnerability Patch List

N/A
