# 版本配套说明

## 产品版本信息

| 项目 | 内容 |
| -------- | ------ |
| 产品名称 | MindIE SD |
| 产品版本 | 3.1.0 |
| 版本类型 | 正式版本 |
| 维护周期 | 三个月 |

## 相关产品版本配套说明

| 产品名称 | 版本 |
| -------- | ------ |
| CANN | 9.0.1 |
| Ascend Extension for PyTorch | 26.0.0 |
| MindCluster | 7.3.0 |
| CCAE | iMaster CCAE V100R026C10SPC100 |
| Ascend HDK | 版本配套关系参见 [CANN版本配套说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/releasenote/9.0.1release-notes.md) |

# 版本兼容性说明

MindIE各组件需要配套使用，请勿跨版本混用各组件。

**表 1**  软件版本兼容性说明

| MindIE SD 版本 | CANN 9.1.0 | CANN 9.0.1 | CANN 9.0.0 | CANN 8.5.1 | CANN 8.5.0 |
| ----------------- | ---------- | ---------- | ---------- |---------- |---------- |
| 3.1.0             |Y            | Y          | Y          |            |            |
| 3.0.0             |              |             | Y          |Y          |Y          |
| 2.3.0             |              |             |             |            |Y          |

| MindIE SD 版本 | MindCluster 26.1.0 | MindCluster 26.0.0 | MindCluster 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- |
| 3.1.0             |Y            |Y          |Y            |
| 3.0.0             |              |Y          |Y            |
| 2.3.0             |              |            |Y            |

| MindIE SD 版本 | Ascend Extension for PyTorch 26.1.0 | Ascend Extension for PyTorch 26.0.0 | Ascend Extension for PyTorch 7.3.0 |
| ----------------- | ---------- | ---------- | ---------- |
| 3.1.0             |Y            |Y          |Y            |
| 3.0.0             |              |Y          |Y            |
| 2.3.0             |              |            |Y            |

| MindIE SD 版本 | CCAE iMaster CCAE V100R026C10SPC100 | CCAE iMaster CCAE V100R026C00SPC010 | CCAE iMaster CCAE V100R025C30SPC100 |
| ----------------- | ---------- | ---------- | ---------- |
| 3.1.0             |Y            |Y          |Y            |
| 3.0.0             |              |Y          |Y            |
| 2.3.0             |              |            |Y            |

# 版本使用注意事项

暂无

# 3.1.0更新说明

## 新增特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | 注意力算子（Fused Infer Attention Score，FIA）完整迁移。将 FIA 算子迁移至 MindIE-SD 自管域；能力收口为 FP8 E4M3FN per-block 专用路径，通过 host checker 与 tiling key 双重校验拦截 noquant/INT8/HIFLOAT8/MXFP8 等历史路径，为低精度与稀疏注意力加速提供可演进的算子底座。 |
| 2    | MoE 全栈推理支持。新增 `fused_moe` 融合算子接口，支持开源框架在 NPU 上执行 MoE 前向计算；接入 `torch_npu.npu_moe_gating_top_k` 与 `npu_moe_gating_top_k_softmax`，将专家选择阶段下沉到 NPU，减少 kernel launch 次数与数据搬移；扩展 MoE 量化支持 W8A8 MXFP8 与 INT8 两条推理路径。基于 vLLM-Omni 中 HunyuanImage-3.0 完成适配验证。 |
| 3    | 稀疏注意力（SLA / BSA）能力增强。为 Sparse Linear Attention 补充 AscendC backend 的 Block Sparse Attention 算子后端；`aclnnBlockSparseAttention` 升级到 V2，兼容 FP8 量化路径，block_size 支持 q=128、kv=256/512；在仅含 V1 aclnn 符号的老版本 CANN 上自动回退到 V1。 |
| 4    | 量化能力增强。新增在线量化（`OnlineQuantConfig`）支持，提供 FA 与 MM 算法配置及回退手段；新增 FA MXFP8 动态量化（`MXFP8_DYNAMIC`），在传入 FA 前对 Q、K 完成旋转与 MXFP8 量化；新增 MXFP4 量化 Flash Attention（`quant_flash_attn`）及部署端 mxfp4 FA 逻辑，扩展低精度注意力场景覆盖。 |
| 5    | 多 torch 版本 Wheel 打包。新增 `MINDIESD_WHEEL_MODE=multi_torch` 打包模式，单一 wheel 可同时适配 torch 2.6/2.7/2.8/2.9/2.10，运行时根据 `torch.__version__` 自动选择对应 `libPTAExtensionOPS.so`；默认仍保持原有固定 torch 版本构建方式不变。 |
| 6    | A5 设备适配与自动路由。新增 `is_a5_device()` 统一识别 A5 设备，`attention_forward` 等公共 API 在 A5 上自动路由到 `fused_attn_score`，并对直接调用下线算子的场景给出清晰报错与迁移指引。 |
| 7    | 频率调节（Frequency Regulator）算子。新增频率优化算子的 MindIE-SD plugin 支持，包括 C++ wrapper、aclnn 两阶段流、BackendSelect 注册与 Python API 导出。 |
| 8    | 部署与生态扩展。新增 vLLM-Omni + MindIE-SD 合并镜像，并将 omni 镜像重构为独立 MindIE-SD 镜像，统一镜像名、版本 Tag 与 OCI 元数据，pip 安装源切换为公共 PyPI；新增 `.agents/skills/` 开发者技能集（ascend-deploy、auto-optimization、code-standards、compilation-dev）。 |

## 修改特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | 算子编译范围调整。移除原 CANN 版本检测与算子过滤逻辑，统一一次编译全部算子、全部平台（ascend910/ascend910b/ascend910_93/ascend950），修复 CANN≥9.0 环境下 laser_attention、ada_block_sparse_attention 等通用算子未生成 910B/910 平台 kernel 的问题；支持通过 `ASCEND_OP_NAME` / `ASCEND_COMPUTE_UNIT` 环境变量覆盖。 |
| 2    | MoE Dispatcher 默认策略调整。由“A2 默认 static、A3/A5 默认 dynamic”调整为依据 `top_k` 与 `ep_size` 关系精细选择，大 EP 场景优先 dynamic，降低跨机 all-to-all 通信劣化影响。 |
| 3    | 日志系统重构。统一 MindIE SD Python 模块日志出口，默认与 verbose 模式均包含组件标识；精简默认 INFO 场景日志（正常流程降级为 DEBUG），并增强 WARNING/ERROR 日志的问题描述、可能根因与修复建议。 |
| 4    | 安全编译选项。按安全编译指南为算子工程新增安全编译与链接选项，提升产物安全性。 |

## 删除特性

| 编号 | 详细 |
| :--- | :----------------------------------------------------------------------------------------------------------- |
| 1    | 删除 `csrc/ops/ascendc/` 目录及其全部算子源文件，统一收敛至 per-op 目录管理；同步更新测试用例中指向被删目录的源文件映射。 |

## 接口变更说明

本章节的接口变更说明包括新增、修改、废弃和删除。接口变更只体现代码层面的修改，不包含文档本身在语言、格式、链接等方面的优化改进。

- 新增：表示此次版本新增的接口。
- 修改：表示本接口相比于上个版本有修改。
- 废弃：表示该接口自作出废弃声明的版本起停止演进，且在声明一年后可能被移除。
- 删除：表示该接口在此次版本被移除。

| 类名/API原型 | 变更类别 | 变更说明 |
| :----------- | :------- | :------- |
| def mindiesd.frequency_regulator | 新增 | 频率调节算子接口 |
| def mindiesd.fused_moe | 新增 | 融合 MoE 算子接口，支持开源框架在 NPU 上执行 MoE 前向计算 |
| class mindiesd.OnlineQuantConfig | 新增 | 在线量化配置类 |
| class mindiesd.TimestepManager | 新增 | 时间步管理类 |
| class mindiesd.TimestepPolicyConfig | 新增 | 时间步策略配置类 |
| class mindiesd.QuantConfig | 新增 | 量化配置类（恢复导出，3.0.0 未在 mindiesd.__init__ 导出） |
| def mindiesd.sparse_attention | 修改 | 升级至 aclnnBlockSparseAttentionV2，兼容 FP8 量化；在仅含 V1 aclnn 符号的老版本 CANN 上自动回退 V1 |

## 已解决的问题

| 序号 | 类别 | 问题描述 |
| :--- | :--- | :--- |
| 1 | 安全 | ZMQ 共享内存句柄广播使用 pickle 直接反序列化 socket 字节流，存在任意代码执行（RCE）风险。 |
| 2 | 安全 | LayerNorm 算子 size_t 计算下溢导致越界迭代与潜在内存破坏。 |
| 3 | 算子与编译 | CANN≥9.0 环境下 laser_attention、ada_block_sparse_attention 等通用算子未生成 910B/910 平台 kernel，导致对应平台算子缺失。 |
| 4 | 算子与编译 | CMake≥4.1.0 时 TIK 算子构建链接失败。 |
| 5 | 算子与编译 | 同时编译 ABSA 与 FIA 算子时因 tiling 注册键冲突导致进程崩溃退出（core dump）。 |
| 6 | 算子与编译 | 算子编译时缺少头文件或包含路径错误，导致构建失败。 |
| 7 | 稳定性与精度 | `enable_offload` 多次调用重复注册 forward hook，且 forward 事件录制顺序错误。 |
| 8 | 稳定性与精度 | 未安装 triton 时 `import mindiesd` 触发 `std::bad_alloc`。 |
| 9 | 稳定性与精度 | `sparse_block_estimate` 假算子 input_layout 校验缺陷导致未初始化变量与越界。 |
| 10 | 稳定性与精度 | MoE W8A8 动态量化精度异常。 |
| 11 | 稳定性与精度 | ACLGraph capture 时静态输入 data_ptr 存在陈旧数据，影响执行精度。 |
| 12 | 稳定性与精度 | 部分 Hunyuan 模型使用 FA512 算子时，V 侧 per-block dequant scale 被硬编码 block_size=256 校验拦截。 |
| 13 | 稳定性与精度 | block_sparse_attention V2 在仅含 V1 aclnn 符号的老版本 CANN 上 dlsym 失败，导致路径不可用。 |
| 14 | 日志 | 各模块日志出口、格式与级别不一致，默认场景信息冗余、异常场景信息不足。 |
| 15 | 测试 | block_sparse_attention UT 仅支持 A5 设备，其余环境无法执行。 |

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

- 若原有代码直接调用 FIA 的 noquant/INT8/HIFLOAT8/MXFP8/PA/mask/rope/prefix/sparse 等历史路径，需迁移至 FP8 E4M3FN per-block 路径，否则将被 host checker 与 tiling key 校验拦截。
- 日志级别与内容已调整：默认 INFO 日志精简，依赖日志进行监控告警的业务需确认日志内容是否充分。

# 漏洞修补列表

| 序号 | 类别 | 漏洞说明 |
| :--- | :--- | :--- |
| 1 | 安全 | ZMQ 共享内存句柄广播使用 pickle 反序列化 socket 字节流，存在任意代码执行（RCE）风险。已引入受限反序列化器 SafeUnpickler，仅允许加载白名单内的安全类型。 |
| 2 | 安全 | LayerNorm 算子 size_t 计算下溢导致越界迭代与潜在内存破坏。已在维度计算前增加 `TORCH_CHECK` 校验。 |
