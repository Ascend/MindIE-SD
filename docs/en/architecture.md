# Architecture Design

## Architecture Goals

MindIE SD aims to build an Ascend-native multimodal acceleration series suite, working alongside industry model suites (e.g., diffusers) to achieve efficient multimodal inference on Ascend. It primarily focuses on providing key operators and fused operators for multimodal generation, combined with Ascend-native quantization/sparsity algorithms, compute-to-memory tradeoff strategies, multi-card parallelism, and other approaches to enable rapid migration and Ascend acceleration of diffusers models. In the future, it will further expand to multimodal understanding, omnimodal, and other acceleration scenarios.

By design, modules are independently decoupled and can be used individually or in combination. The industry already has acceleration methods such as Cache-DiT and xDiT, whose effects are similar to the cache and parallelism modules, presenting a choice of approach. However, other components in MindIE SD can still be used independently alongside them.

**Key Features:**

- Ascend-native acceleration operators: Provides Ascend-native multimodal FA, MM, MoE, quant operators, and fused operators, accessible externally through the layer module. For details, see [Core Acceleration API](./features/core_layers.md).
- Quantization and sparsity capabilities: Provides Ascend-tailored algorithm combinations for Ascend data types and compute distribution, imported through the quantization module. For details, see [Sparsity](./features/sparse.md) and [Quantization](./features/quantization.md).
- Compute-to-memory tradeoff: Provides cache algorithms at DiT module, DiT block, attention, and other granularities to support acceleration in different view scenarios. For details, see [Compute-to-Memory Tradeoff](./features/cache.md), [CPU Offload](./features/cpu_offload.md), and [VRAM Sharing](./features/share_memory.md).
- Multi-card parallelism: Provides parallelism capabilities such as CFG and USP, as well as dynamic expert load balancing (EPLB) for MoE scenarios, integrated into the acceleration operator APIs for automatic enablement after interface replacement. For details, see [Multi-Card Parallelism](./features/parallelism.md) and [Dynamic Expert Load Balancing](./features/DyEPLB.md).
- FA_Power_Cap technology: Splits FA execution and reorders FA with communication for long-sequence video generation, reducing average power consumption and improving end-to-end performance. For details, see [FA_Power_Cap Technology](./features/fa_power_cap.md).
- Automatic affinity acceleration: Based on the torch.compile inductor mechanism, custom fusion passes are implemented to achieve Ascend-native operator substitution.

>[!NOTE] Note
>
>- diffusers models accelerated on Ascend using MindIE SD are published on [Modelers](https://modelers.cn/models?name=MindIE&page=1&size=16) and [ModelZoo](https://www.hiascend.com/software/modelzoo).
>For features related but not core to this repository, samples are provided in the examples directory for reference. For example, see [Service Deployment](../../examples/service) for servitization deployment samples, and [Cache](../../examples/cache) for multimodal inference acceleration samples.

## Architecture Overview

As shown below, MindIE SD provides Ascend acceleration capabilities externally based on the PyTorch framework. Each acceleration capability can be used independently, primarily including the cache, parallelism, quantization, layer, and kernel modules.

MindIE SD's relevant interfaces follow the diffusers interface definitions. Some diffusers models accelerated on Ascend using MindIE SD are published on [Modelers](https://modelers.cn/models?name=MindIE&page=1&size=16)/[ModelZoo](https://www.hiascend.com/software/modelzoo), and simple plugin-based adaptation directly on diffusers is also supported.

![MindIE SD Architecture Overview](../figures/architecture_overview.png)

**Basic Features:**

- layer module: Provides basic external acceleration interfaces (including attn, moe, quant, and other feature layers). It is the foundation for advanced features and can be used independently.
- kernel module: Provides high-performance Ascend kernels for multimodal generation, supporting operator integration via programming languages such as AscendC and Triton.
- compilation module: Based on FX graph capabilities, enables fusion passes after compilation is turned on to achieve automatic Ascend affinity acceleration. For details, see [Compilation Features](./features/compilation.md).

**Advanced Features:**

- quantization module: Supports automatic enablement of quantization capabilities.
- cache module: Provides compute-to-memory tradeoff acceleration capability implementation.
- parallelism module: Provides multi-card parallel distributed acceleration capabilities, requiring collaboration with the layer module and PyTorch.

## Directory Structure

```text
|- benchmarks         // Core kernel performance monitoring and compilation acceleration effect monitoring
|- build              // Build scripts
|- csrc               // Ascend kernel source code location
|- docs               // Project documentation
|- examples
  |- cache            // Cache feature sample: enable cache for model acceleration
  |- service          // Servitization sample: convert command-line mode to servitization
  |- wan              // Model inference sample: model inference commands and parameter configuration
|- mindiesd
  |- cache_agent      // Advanced feature: provides cache capabilities
  |- compilation      // Provides compilation capabilities, implementing automatic graph modification based on FX graph (while still maintaining single-operator dispatch)
  |- eplb             // Advanced feature: provides expert parallel load balancing capabilities
  |- layers           // Provides basic PyTorch layer interfaces
  |- quantization     // Advanced feature: provides quantization capabilities
  |- utils            // Core utility module, providing shared infrastructure services and common functionality
|- tests              // Test cases
```
