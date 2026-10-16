# Version Compatibility Notes

## Product Version Information

| Item | Content |
| -------- | ------ |
| Product Name | MindIE SD |
| Product Version | 3.0.0 |
| Version Type | Official Release |
| Maintenance Cycle | Three Months |

## Related Product Version Compatibility

| Product Name | Version |
| -------- | ------ |
| CANN | 9.0.0 |
| Ascend Extension for PyTorch | 7.3.0 |
| Ascend HDK | See [CANN Version Compatibility Notes](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/releasenote_0000.html) for version compatibility |

# Version Compatibility

MindIE SD components must be used with matching versions. Do not mix components across different versions.

**Table 1** Software Version Compatibility

| CANN | Ascend Extension for PyTorch |
| ---- | ---------------------------- |
| 9.0.0 | 7.3.0                        |

# Version Usage Notes

None

# 3.0.0 Release Notes

## New Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Enhanced quantization capabilities. Support for FA dynamic FP8, new W4A4_DYNAMIC quantization format, completion of W4A4 quantization algorithm common logic, and new W4A4MXFP4DualQuantLinear capability, improving adaptability and deployment flexibility across quantization scenarios. |
| 2    | Enhanced operator and plugin capabilities. New aclnn LayerNorm plugin and external API, new adaLayerNormV2 plugin and layer implementation, along with additional aclnn capabilities including sparse_block_estimate, block_sparse_attention, laser_attention, and la_preprocess, increasing operator coverage. |
| 3    | Enhanced runtime capabilities. New multi-instance shared memory for sharing weight memory across instances to reduce duplication; new block-level CPU offload for fine-grained dynamic transfer of modules between CPU and NPU, alleviating memory pressure. |
| 4    | Enhanced attention and layout adaptation. attention_forward and rf_v2 now support BNSD layout input; FA can be enabled via environment variable, reducing integration cost for upper-layer model integration. |
| 5    | Enhanced scheduling and serving capabilities. New Dynamic EPLB scheduling, new service-based examples, and wan2.2 service-side synchronization and accuracy fixes, improving availability for serving and inference scenarios. |
| 6    | Optimized quantization operator backend. Migrated existing torch-atb-based quantization operators to aclnn native operators, improving compatibility and stability, with full support for compilation optimization features such as torch.compile, enhancing framework adaptability. |

## Modified Features

| No. | Details |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | Removed `_mindie_sd` suffix from custom plugin operator names and unified namespace to `mindiesd` for consistent naming. |
| 2    | In FA quantization scenarios, FIA operator output format now matches input query format to reduce format incompatibility issues. |
| 3    | Adapted npu_quant_matmul operator with added constraints to mitigate integration risks from constraint changes. |
| 4    | Systematic adaptation of aclnn compilation project with enhanced build chains, directory management, and error handling to improve build efficiency and stability. |

## Removed Features

None

## API Change Notes

API changes include additions, modifications, deprecations, and removals. API changes only reflect code-level modifications and do not include documentation improvements in language, formatting, or links.

- New: APIs introduced in this version.
- Modified: APIs changed compared to the previous version.
- Deprecated: APIs that cease evolution from the declared version and may be removed one year after declaration.
- Removed: APIs removed in this version.

| Class Name / API Signature | Change Type | Description |
| :----------- | :------- | :------- |
| • def mindiesd.layernorm_scale_shift<br>• def mindiesd.fast_layernorm<br>• def mindiesd.sparse_attention | New | New APIs |
| • class mindiesd.Linear<br>• class mindiesd.QuantFA | Removed | Removed APIs |

## Resolved Issues

| No. | Category | Issue Description |
| :--- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 1 | Installation & Compatibility | libopapi.so missing when running tests after installing compiled MindIE-SD, affecting post-installation testing and basic functionality verification. |
| 2 | Installation & Compatibility | Insufficient compatibility with newer torch versions, affecting integration with newer inference images. |
| 3 | Installation & Compatibility | Build package missing plugin, affecting package completeness and plugin capability loading. |
| 4 | Operator & Compilation | Flux.1-dev failing aclnnAdaLayerNorm call with compile enabled in new environments, blocking compilation acceleration path. |
| 5 | Operator & Compilation | Incorrect API usage in test_rainfusionattention.py causing test execution failures. |
| 6 | Cache & Test Quality | Block_end validation in DiT Cache Agent inconsistent with left-closed right-open interval semantics, affecting cache usage. |
| 7 | Cache & Test Quality | Single-metric accuracy comparison (only cosine similarity), incomplete accuracy evaluation dimensions. |

## Known Issues

| No. | Category | Issue Description |
| :--- | :----------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------- |
| 1 | Operators | Lack of matrix multiplication operators based on CUTLASS and Triton |
| 2 | Usability | Need to support more extensions, such as cache-dit |
| 3 | Performance | Need to support more parallel computation masking and fusion schemes |

# Upgrade Impact

## Impact on Running Systems During Upgrade

- Business impact

  Software version upgrade will cause service interruption.

- Network communication impact

  No impact on network communication.

## Impact on Running Systems After Upgrade

None

# Vulnerability Patch List

None
