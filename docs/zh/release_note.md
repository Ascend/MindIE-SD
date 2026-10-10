# 版本配套说明

## 产品版本信息

| 项目 | 内容 |
| -------- | ------ |
| 产品名称 | MindIE SD |
| 产品版本 | 3.2.0 |
| 版本类型 | Release版本 |
| 维护周期 | 三个月 |

## 相关产品版本配套说明

| 产品名称 | 版本 |
| -------- | ------ |
| CANN | 9.2.0 |
| TorchNPU | 26.2.0 |
| MindCluster | 26.2 |
| CCAE | iMaster CCAE V100R026C10SPC100 |
| Ascend HDK | 版本配套关系参见 [CANN版本配套说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/9.0.1release-notes.md) |

# 版本兼容性说明

MindIE各组件需要配套使用，请勿跨版本混用各组件。

**表 1** 软件版本兼容性说明

`Y` 表示支持配套，`/` 表示无配套条目，`待确认` 表示配套关系尚未确认，需在版本发布前确认并补充。

| MindIE SD 版本 | CANN 9.2.0 | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1 | CANN 8.5.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y | / | / |
| 3.1.0 | / | Y | Y | Y | / | / |
| 3.0.0 | / | / | / | Y | Y | Y |
| 2.3.0 | / | / | / | / | / | Y |

| MindIE SD 版本 | MindCluster 26.2 | MindCluster 26.1.0 | MindCluster 26.0.0 | MindCluster 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y |
| 3.1.0 | / | Y | Y | Y |
| 3.0.0 | / | / | Y | Y |
| 2.3.0 | / | / | / | Y |

| MindIE SD 版本 | TorchNPU 26.2.0 | TorchNPU 26.1.0 | TorchNPU 26.0.0 | TorchNPU 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- | ---------- |
| 3.2.0 | Y | Y | Y | Y |
| 3.1.0 | / | Y | Y | Y |
| 3.0.0 | / | / | Y | Y |
| 2.3.0 | / | / | / | Y |

| MindIE SD 版本 | CCAE iMaster CCAE V100R026C10SPC100 | CCAE iMaster CCAE V100R026C00SPC010 | CCAE iMaster CCAE V100R025C30SPC100 |
| ----------------- | ---------- | ---------- | ---------- |
| 3.2.0 | 待确认 | 待确认 | 待确认 |
| 3.1.0 | Y | Y | Y |
| 3.0.0 | / | Y | Y |
| 2.3.0 | / | / | Y |

# 版本使用注意事项

- 以下能力仅支持 Ascend 950PR&950DT系列产品：`quant_four_over_six_a5` 量化算子、rf_v3 块稀疏注意力 MXFP4 档位（`precision="mxfp4"`）、EQBSA qmode1 mix 路径、FFN 融合算子的融合路径（其他平台自动回退原生算子链）。
- vLLM-Omni 联合镜像仅支持 linux/amd64 平台。

# 3.2.0更新说明

## 新增特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | 量化注意力函数式接口。新增公共 API `mindiesd.quant_attention`，将量化注意力执行逻辑沉淀为可复用函数式接口，调用方直接传入浮点 Q/K/V；`precision` 支持 FP8/MXFP8/MXFP4 三档，支持 BNSD/BSND 布局、可选 Q/K 旋转矩阵与注意力窗口配置；`FP8RotateQuantFA` 与公共入口统一执行逻辑，MXFP4/MXFP8 接入保留 legacy timestep 回退。 |
| 2    | FP8 注意力模式枚举与 FIA 全量化路径扩展。新增 `FP8FAMode` 枚举（`HIGH_PRECISION`（默认）与 `C8V16_TILING512`），`QuantConfig`/`OnlineQuantConfig` 新增 `fp8_fa_mode` 配置项，一次性绑定量化块、`value_quant_mode` 与 `inner_precise`；FIA 算子新增 FP8 per-block V512 量化路径与 C8V16（inner_precise=4）全量化路径。 |
| 3    | Eagle 块稀疏注意力算子族（EQBSA/EBSA）。新增 `eagle_quant_block_sparse_attention` 量化块稀疏注意力算子（Q/K INT8 + V FP8 混合精度，叠加量化与块稀疏降低长序列开销），并补齐 Ascend 950PR&950DT系列产品 qmode1 mix 路径；新增 `eagle_block_sparse_attention` 非量化块稀疏注意力算子，BlockK 支持放宽至 64（16 对齐）；MiniMax-H3 模型的 EagleQBSA 经 rf_v3 稀疏注意力路由接入。 |
| 4    | BF16→FP4 自适应量化算子。新增 `quant_four_over_six_a5` 算子，支持 BF16 输入到 FP4（FLOAT4_E2M1）的 4/6 自适应动态 MX 量化（E8M0 缩放、blocksize=32），适配 Ascend 950PR&950DT系列产品。 |
| 5    | FFN 融合算子。新增 FFN 前向融合算子，将 linear→act→linear 三次 kernel launch 融合为单 kernel，激活支持 gelu/silu/swiglu，适配 Ascend 950PR&950DT系列产品，其他平台自动回退原生算子链；提供 Python 接口 `mindiesd.layers.eagle_ffn_linear`。 |
| 6    | USP 序列并行注意力。新增 `mindiesd.layers.usp.usp_attention` 接口，单次调用组合 Ulysses 序列并行、KV AllGather、通信量化（FP8 E4M3/MXFP8，量化 payload 与 scale 分离通信）与注意力计算，支持按 head 切分的双流 overlap 流水执行。 |
| 7    | 稀疏注意力能力增强。`sparse_attention` 新增 `video_spans` 参数，单次 RainFusion 调用完成多视频 clip 的空间重排、全局 block 选择与原始 token 顺序恢复，非视频 token 保持 dense 计算；Ascend 950PR&950DT系列产品上 SparseLinearAttention 的 BLKK 对齐约束放宽至 16。 |
| 8    | MXFP4 块稀疏注意力。rf_v3 稀疏注意力新增 `precision="mxfp4"` 档位，Q/K 经 Hadamard 旋转后量化为 FP4 E2M1 + E8M0 缩放，经 `aclnnBlockSparseAttentionV3` 执行，仅支持 Ascend 950PR&950DT系列产品。 |
| 9    | 编译融合 pattern 扩展。Wan2.2 新增 5 类编译融合 pattern（qk_norm、adaLN 调制、RoPE、FastGelu、残差门控）并支持 bf16/fp32 双精度与模型级 bf16 精度统一；MiniMax-H3 新增 RMSNorm/RoPE 融合与 AdaLN/SwiGLU/残差门控 Triton 单 kernel 融合；Qwen-Image 新增 RoPE 与 norm_out AdaLN 融合 pattern；配套新增 `FusionPatterns` 开关 13 个。 |
| 10   | MoE MXFP8 的 GMMFR 融合支持。MXFP8 量化 MoE 场景下支持 GMMFR 融合，减少 MoE 前向计算的 kernel launch 与数据搬移。 |
| 11   | 新融合算子 `mul_add` 与 `norm_rope_concat`。`mul_add` 实现 y = a + b*c 门控广播融合；`norm_rope_concat` 实现 LayerNorm/RMSNorm + RoPE + KV Concat 三合一融合，支持训练模式输出 mean/rstd。 |
| 12   | mindie_bench 算子性能基准工具。新增 `benchmarks/scripts/mindie_bench.py` 命令行工具（run/report/compare 子命令），内置 NPU 后端与 FA/BSA/GMM/MM 算子定义，支持序列长度扫描、基线快照、HTML/CSV 报告与 CI 阈值门禁。 |
| 13   | 构建加速。新增 `SKIP_ALL_OPS_PLUGIN_BUILD` 环境变量，算子与 plugin 缓存产物完整且兼容时跳过编译，缺失或不兼容时自动回退全量编译，默认行为不变。 |
| 14   | DummyRun 工具链增强。量化参数统一为 `--quant {fp32,bf16,w8a8}`，w8a8 按设备感知选择量化格式（Ascend 950PR&950DT系列产品走 W8A8-MXFP8，Atlas 800I A2推理服务器/Atlas 800I A3超节点服务器走 W8A8-DYNAMIC INT8）；Wan2.2 dummy run 新增帧数与分辨率配置。 |
| 15   | 部署与生态扩展。vLLM-Omni 联合镜像基础镜像对齐 v0.28.0 并新增 Ascend 950PR&950DT系列产品版本；新增 MiniMax-H3、Qwen-Image-2.1、DiffSynth-Engine 推理部署指南与 FLUX.1-dev Cache-DiT 示例。 |

## 修改特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | MoE Dispatcher 接口参数重命名与语义澄清。`tokens_full` 更名为 `inputs_sharded`（默认值相应取反，语义保持不变），`reduce_results` 更名为 `reduce_routed_out`；`resolve_dispatcher_class` 的 `top_k` 调整为必填参数；`fused_moe`/`moe` 新增 `return_dispatcher_type`，可返回实际选用的 dispatcher 类型。迁移说明见“升级影响”。 |
| 2    | TIK 算子构建容错。当 CANN toolkit 不再交付 TBE/TIK 工程模板时，构建流程由整次失败调整为探测缺失后跳过 TIK 算子构建并输出 WARNING，AscendC 算子构建流程不受影响。 |
| 3    | 频率调节 plugin 执行方式调整。由手工流同步与 D2H/H2D 状态拷贝，统一调整为 `EXEC_NPU_CMD` 框架全异步执行，workspace/executor/ACL Tensor 生命周期由框架统一管理。 |
| 4    | DummyRun 示例依赖版本调整。`examples/dummy_run` 依赖 diffusers 由 0.38.0 升级至 0.40.0、transformers 升级至 5.2.0。 |

## 删除特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | 删除 FIA 与 Eagle 算子的 arch35 冗余内核副本（共 249 个文件），头文件引用统一收口至正式路径，正式路径缺失时编译期直接报错，不再静默回退。本次删除仅涉及内部 C++ 源码组织，不删除任何对外接口，运行时算法与精度路径不变。 |

## 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

- 新增：表示此次版本新增的接口。
- 修改：表示本接口相比于上个版本有修改。
- 废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
- 删除：表示该接口在此次版本被移除。

| 类名/API原型 | 变更类别 | 变更说明 |
| :----------- | :------- | :------- |
| def mindiesd.quant_attention | 新增 | 量化注意力函数式公共接口，`precision` 支持 fp8/mxfp8/mxfp4 |
| class mindiesd.FP8FAMode | 新增 | FP8 注意力模式枚举（HIGH_PRECISION / C8V16_TILING512） |
| def mindiesd.layers.eagle_ffn_linear | 新增 | FFN 融合算子接口，Ascend 950PR&950DT系列产品走融合路径 |
| def mindiesd.layers.residual_gate_add | 新增 | 残差门控融合算子接口 |
| def mindiesd.layers.usp.usp_attention | 新增 | USP 序列并行注意力接口（模块级导出，未加入 mindiesd 顶层命名空间） |
| torch.ops.mindiesd 新增算子 | 新增 | eagle_block_sparse_attention、eagle_quant_block_sparse_attention、quant_four_over_six_a5、mul_add、norm_rope_concat |
| def mindiesd.fused_moe | 修改 | 参数 `tokens_full` 更名为 `inputs_sharded`（默认值相应取反）、`reduce_results` 更名为 `reduce_routed_out`，新增 `return_dispatcher_type` |
| def mindiesd.sparse_attention | 修改 | 新增 `video_spans` 参数支持多视频稀疏注意力；rf_v3 新增 `precision="mxfp4"` 档位 |
| class mindiesd.QuantConfig | 修改 | 新增 `fp8_fa_mode` 配置项 |
| class mindiesd.OnlineQuantConfig | 修改 | 新增 `fp8_fa_mode` 配置项 |

## 已解决的问题

| 序号 | 类别 | 问题描述 |
| :--- | :--- | :--- |
| 1 | 算子与编译 | 部分 CANN toolkit 不再交付 TBE/TIK 工程模板导致 msopgen 失败、整次算子构建失败。 |
| 2 | 算子与编译 | EQBSA 算子在 CANN 9.2 下因 `__simd_vf__` 非静态导致编译失败。 |
| 3 | 算子与编译 | quant_four_over_six_a5 与 eagle_ffn 算子 `Ops::Base::ToString` 重定义导致全量构建链接失败。 |
| 4 | 算子与编译 | eagle_ffn 算子 `math_util.h` 在不同 CANN 版本间包含路径不兼容导致构建失败。 |
| 5 | 算子与编译 | torch_npu 2.9 起 dtype cast 分解方式变化导致 RMSNorm/RoPE 编译 pattern 全部失配。 |
| 6 | 算子与编译 | Wan 残差门控编译 pattern 误匹配 FLUX 的反序操作数（x + gate * y），导致 broadcast 语义错误。 |
| 7 | 稳定性与精度 | RainFusion rf_v2 在非规则对齐视频网格（如 MiniMax H3 1344×768）上 block mask 与 NPU kernel 分块不一致，导致视频花图。 |
| 8 | 稳定性与精度 | `lru_cache_by_attn_param` 的 `attn_cache` 为模块级全局字典，所有被装饰函数共享导致跨函数缓存污染。 |
| 9 | 稳定性与精度 | `MXFP8RotateQuantFA` 以 device 端张量传入序列长度触发 `aten._local_scalar_dense`，导致 DiT aclgraph capture 在每个注意力层 graph break。 |
| 10 | 稳定性与精度 | EPLB `UpdateTaskTransfer.update_emit_task` 使用阻塞式 `queue.put`，指令队列满时 worker 线程死锁。 |
| 11 | 稳定性与精度 | 频率调节 plugin 手工流同步与 D2H/H2D 状态拷贝存在资源管理问题，导致 coredump/507035 异常。 |
| 12 | 稳定性与精度 | 量化层 forward 内就地修改 `self.bias` dtype 导致 Dynamo guard 反复失效、重复编译。 |
| 13 | 兼容性 | Atlas 800I A3超节点服务器新机型 soc_version=256 超出识别区间，导致设备无法识别、算子加载失败。 |

## 遗留问题

| 序号 | 类别 | 问题描述 |
| :--- | :--- | :--- |
| 1 | 算子 | 持续补齐基于 CATLASS 和 Triton 实现的矩阵乘法算子。 |
| 2 | 易用性提升 | 持续完善 Cache DiT、DiffSynth-Engine 等第三方框架的端到端对接与可运行样例。 |
| 3 | 性能提升 | 持续扩展并行通算掩盖与融合方案，推进 EPLB 与 MoE 融合的联合优化。 |

# 升级影响

## 升级过程中对现行系统的影响

- 对业务的影响

  软件版本升级过程中会导致业务中断。

- 对网络通信的影响

  对网络通信无影响。

## 升级后对现行系统的影响

- MoE 接口参数已重命名：以关键字方式调用 `fused_moe(..., tokens_full=..., reduce_results=...)` 的既有代码需迁移为 `inputs_sharded=` / `reduce_routed_out=`；直接调用 `resolve_dispatcher_class` 的代码需显式传入 `top_k`。参数默认值语义保持不变（原 `tokens_full=True` 等价于现 `inputs_sharded=False`）。
- 新版 CANN toolkit（不再交付 TBE/TIK 工程模板）环境下 TIK 算子自动跳过构建，依赖 TIK 算子产物的构建流程需关注。
- dummy run 示例命令行参数调整：`--compute-precision` 与布尔 `--quant` 参数已移除，统一改用 `--quant {fp32,bf16,w8a8}`；`examples/dummy_run` 依赖 diffusers 升级至 0.40.0。
- vLLM-Omni 联合镜像基础镜像升级至 v0.28.0（Atlas 800I A2推理服务器/Atlas 800I A3超节点服务器）并新增 Ascend 950PR&950DT系列产品版本，镜像 tag 命名同步更新。

# 病毒扫描结果

当前暂无病毒扫描结果，将在版本发布前补充。

# 漏洞修补列表

本版本无新增漏洞修补。

# 修订记录

| 文档版本 | 发布日期 | 修改说明 |
| -------- | -------- | -------- |
| 01 | 2026-10-15 | 第一次正式发布。 |
