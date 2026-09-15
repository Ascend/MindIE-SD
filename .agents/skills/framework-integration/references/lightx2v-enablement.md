# LightX2V：特性开启方式与合入版机制（框架差异记录）

> 定位：本文件只放 **LightX2V 侧特有**的内容——开启方式（配置键 / 注册表 / 命令）、框架侧前置与坑、
> 特性开关面板、合入版（上游 PR #1471）机制与代码地图。**通用方法与判定纪律**见
> `model-auto-optimization/references/lossless-methodology-notes.md`、
> `dit-perf-opt/references/combination-search.md`（有损组合协议）、
> `accuracy-gate/references/quality-gate.md`（质量门禁）；
> **能力支持面**（✅/🟡/❌/❓ 与证据码 **L1**）见 `framework-support-matrix.md`。
> **本文件不写绝对耗时与绝对质量分值**（二者只在本次环境成立，不可迁移）；**要写**大致加速比
> （量级/约数）与质量变化度（降幅/差值）；
> 实测记录与产物坐标见 §6。
> 与 `vllm-omni-enablement.md` / `cache-dit-enablement.md` 为**同级并列**：各自管本框架的开启方式，
> 互不抄写；框架侧**结构性开发**（从无到有）不在本文件，见 `../SKILL.md` §2（分支 B）。
> 路径占位符：`{model_weight_dir}`（权重根）、`{repo}`（代码仓根）、`{run_results_dir}`（运行产物根）、
> `{env_host}` / `{env_container}`（运行环境）。
> 本文件的方向性结论一律是该框架 × 该模型 × 该规模的**观测**，不是执行序（换框架 / 模型 / 规模须重判）。

## 1. 画像与版本边界

### 1.1 画像速答（迁移清单）

- 形态：三方推理框架 **LightX2V**（`lightx2v` pip 包 + `lightx2v_platform` 平台算子层），
  在昇腾容器上把 mindiesd 的算子融合接入其 MiniMax-H3 推理路径。
- **算子选择机制 = 注册表**：`RMS_WEIGHT_REGISTER` / `ROPE_REGISTER` / `ATTN_WEIGHT_REGISTER`，
  外加**平台级** `PLATFORM_*_REGISTER`（平台注册表 merge 进主注册表，import `lightx2v_platform` 时触发）。
- 版本口径：**上游合入版 PR #1471**（`feat(ascend): add MiniMax-H3 fused RoPE and MindIE SD compile
  backend`）——本文件**以合入版（平台注册表架构）为准**；合入前「全局改共享类」的版本仅在历史提交中，
  **勿再按旧姿势操作**。
- 接入面速答（迁移清单）：

  | 算子 | 抽象接口 | 接入方式（合入版） | 方向（本组合观测） |
  |---|---|---|---|
  | RMSNorm | 有（`rms_type`） | 运行时注册表：`"rms_type": "npu_rms_norm"` | 单点收益为正（与 rope 合计计） |
  | RoPE | 有（`rope_type`） | 运行时注册表：`"rope_type": "minimax_h3_npu_rope"`（走 mindiesd `rotary_position_embedding`） | 单点收益为正（与 rms 合计计） |
  | AdaLN / SwiGLU / residual gate | 无 | compile：`"use_compile": true` + `"compile_backend": "mindie"` | 正收益但幅度小，且**随序列变长被通信占比稀释** |
  | Ulysses a2a | 有（`seq_p_a2a_backend`） | 框架**原生可选后端**：`"seq_p_a2a_backend": "hccl_eager"`（不编进图） | 通信段耗时明显下降（额外红利） |

- **策略判定（可移植的决策序）**：有抽象接口的算子走**运行时注册表替换**（配置字段切换）；
  无接口的算子走 **compile**（须先解决 §3.2 的三条坑）；多卡 collective 用框架**原生可选后端留在
  eager**——三者都**不改框架核心共享代码**。
- **设计原则（durable）**：框架接入优先「**平台注册 + 配置可选**」，不要全局改共享类。早期版本曾把
  a2a 以 `@torch._dynamo.disable` 全局钉在 `TorchUlyssesA2A.exchange`（影响所有平台所有 Ulysses
  用户），评审后收敛为平台注册的可选后端 `hccl_eager`，common 侧 `TorchUlyssesA2A` 恢复纯净。

### 1.2 版本边界与依赖前置

- **代码 / 配置必须配套**：合入版配置含 `seq_p_a2a_backend: "hccl_eager"`；若代码是合入前版本会报
  unknown a2a backend；若代码已合入而配置缺该键，a2a 会回编译图并丢掉通信红利。**远端代码升级后
  务必同步配置**。
- **同步回退风险（硬约束）**：mindiesd 从 dev-skills 分支整仓回填远端会**覆盖会话内未合入的修复**
  ——本案例实际复现两处并已修复回填（详见 §5）：① `minimax_h3_swiglu_pattern.py` 需
  **split_twice 变体**，丢失后 swiglu 融合**静默消失**（墙钟回归，**日志无痕、只有 profile 可见**）；
  ② **注册顺序**：`enable_minimax_h3_gate` 必须先于 `enable_wan_residual_gate`。
  核验姿势：对比 `kernel_details.csv` 中 swiglu / gather_residual_gate 的 kernel 计数。
- 实测口径说明：本案例数据来自「**合入前代码 + 本地镜像 #1471 等价机制**」的远端（机制等价已验证）；
  迁移时按 §1.1 的合入版姿势重新核对配置键。

## 2. 启动与并行前置

### 2.1 环境坐标（占位符）

| 项 | 值 |
|---|---|
| 远端设备 | `{env_host}`（root，容器 `{env_container}`） |
| 模型 | `{model_weight_dir}/MiniMax-H3`（transformer + text encoder 两份大权重） |
| 代码 | `{repo}/LightX2V`（合入版 = 上游 main）+ `{repo}/MindIE-SD`（mindiesd） |
| 并行 | USP4（`tensor_p=1`、`seq_p=4`、ulysses a2a），`torchrun` 4 卡 |
| 序列 | 5s：local 9467 / global 37751；15s：local ~27276 / global ~109103（token 数，随分辨率 / 时长换算） |

### 2.2 启动前置与坑

- **HCCL 端口**：默认端口段被既有进程占用时报错 → `HCCL_NPU_SOCKET_PORT_RANGE=20000-21000`
  （RoCE 端口用 `ss` 查不到，"看似空闲"仍可能冲突）。
- **卡选择**：先 `npu-smi info` 看 Health / 占用，**避开 Alarm 卡**；对比必须**同卡组**；
  `UB LINK ERROR` 等瞬时错误重跑即可；**组可用性必须用真实多卡 run 验证**（`npu-smi` Health=OK
  ≠ 可用；驱动损伤残留态表现为整组每步时长均匀变慢且无热降频斜率）——见 dit-parallel-opt
  `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4（该文件的 §1–§5 是**多卡运行期劣化**的唯一真源）。
- **CRLF**：Windows 上编写的 `.sh` 上传前必须转 LF。
- **SIGKILL 副作用**：强杀进行中的多卡任务可伤 NPU 驱动（整组变慢、端口 bind 泄漏）→ 换卡组恢复；
  脚本里的卡组须**硬编码本地**（远程执行器会重传覆盖 `sed` 修改）；多轮实验后端口段易耗尽，
  换新段或驱动复位。

## 3. 特性开关面板（开关 / 日志与计数契约 / 坑）

### 3.1 运行时算子接入（有抽象接口）

#### npu_rms_norm

- 文件：`lightx2v_platform/ops/norm/ascend_npu/npu_rms_norm.py`
- 注册：`@PLATFORM_RMS_WEIGHT_REGISTER("npu_rms_norm")`；配置：`"rms_type": "npu_rms_norm"`
  （H3 weights 构建读 `config.get("rms_type", "torch_native")`）。

#### minimax_h3_npu_rope（合入版，走 mindiesd 算子）

- 文件：`lightx2v_platform/ops/rope/ascend_npu/minimax_h3_npu_rope.py`
- 注册：`@PLATFORM_ROPE_REGISTER("minimax_h3_npu_rope")`；配置：`"rope_type": "minimax_h3_npu_rope"`。
- **懒加载**：模块级 `_load_mindiesd_rope()`（`lru_cache`）——`import mindiesd` 成功返回
  `rotary_position_embedding`；`ImportError` 告警返回 None（走 `TorchRealRope` fallback）；
  其他异常 `raise RuntimeError`。实例缓存 `_mindiesd_rope` / `_fallback_rope`。
- **算子姿势**：H3 部分旋转（`rotary_dim=96` / `head_dim=128`），调用
  `mindiesd.layers.rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half",
  head_first=False, fused=True)`：

  - x 传 4D SBND：`x_rot.unsqueeze(1)` = `[L,1,H,D]`
  - cos/sin **显式传 4D S11D** `[L,1,1,D]`（先 `cos.to(x.dtype)`）
  - ⚠️ **坑**：mindiesd 该算子的 2D `[S,D]` cos 路径假设 x 是 `[B,S,N,D]`（`head_first=False`
    保留 i=1 维）；对 SBND 会 reshape 成 `[1,1,1,D]` 报形状错 ⇒ **必须显式升 4D S11D 传入**
    （与 mindiesd compile pattern 传 4D cos 一致）。

- **校验（类内 `_validate_inputs`）**：x 3D；freqs 为 `(cos, sin)` tuple；cos/sin 同形同设备；
  cos 2D 且 seq 对齐；`rotary_dim` 为正偶数、≤ head 且等于 cos 宽度。
- fallback（无 mindiesd）：`TorchRealRope(layout="split_half")`，数值 bit-exact；
  mindiesd 算子路径与 rotate-half 语义一致（bf16 计算，距 fp32 参考 ≤1 ULP）。

### 3.2 compile 接入（无抽象接口 / 图形态不匹配）

- 姿势：把 adaln / swiglu / gate 经 `torch.compile(block_runner, backend={MindIE})` 接入编译图；
  配置面 `"use_compile": true` + `"compile_backend": "mindie"`（经平台注册表解析，见 §4）。
- **必须修的三条坑**：

  1. **a2a 不能进编译图**：Dynamo 追踪 `dist.all_to_all_single` → `split_sizes=[1,1,...]` →
     HCCL 退化为变长 `hcom_alltoallv`（单次耗时比固定 alltoall **放大数十倍**）→ 墙钟大幅劣化。
     处置：collective 留 eager（`@torch.compiler.disable`），**优先用框架原生可选后端**
     `seq_p_a2a_backend="hccl_eager"`，不要全局改共享 collective 类。
  2. **backend 实例复用**：每次新建 `MindieSDBackend()` → Dynamo `BACKEND_MATCH failure` 反复重编译，
     `recompile_limit` 后**静默退回 eager**。必须缓存**单实例**。
  3. **swiglu pattern 双 split 变体**：LightX2V 的 `chunk(2)` 被 Dynamo 展开成**两个独立 split 节点**
     （diffusers 是单 split 双 getitem）→ mindiesd pattern 需 `split_twice` 变体。
- **负收益起点与修复语义**：初版曾出现**显著负收益**（AOTAutograd 把 custom op 当黑盒 → 边界插入
  大量连续化拷贝 kernel），修复后转为**正收益**——「compile 在某框架为负」属**架构 / 热路径差异**，
  不可跨框架照搬结论。
- ⚠️ **compile × head-parallel 不兼容**（见 dit-parallel-opt `ascend-topology-bandwidth-diag.md` §2）：
  逐头 python 循环触发 Dynamo `recompile_limit` → 部分帧静默回退 eager；组合不要用于生产。
- ⚠️ **compile × rf3 待解**：稀疏后端与 compile 同开时仍有 Dynamo trace 期错误（首步取证未完）⇒
  生产叠加需先解决。

### 3.3 时间步优化（原生步数裁剪）

- 姿势：纯配置项 `infer_steps` 裁剪（框架自带能力，**无需改代码**）。
- 方向（**本组合观测**）：收益来自**步数减少本身**（纯配置、框架自带能力，无需改代码）；
  **质量变化度**：24 步 vs 30 步帧 SSIM **降约 0.06（近无感档）**，20 步降约 0.11（激进档）；
  性能：Run DiT 降约三成（24 步）/近五成（20 步）。当时「优于为 H3 移植 Taylor
  cache」的结论**前提是框架 cache 不可用**（`feature_caching` 对 H3 显式 NotImplemented）——cache
  接线后应重做「裁剪步数 × cache」叠加评估，**勿沿用旧表述**。

### 3.4 量化（线性层）

- **框架内建路线的阻塞（准确表述）**：框架侧仅 `dit_quantized`（per-output-channel 预量化 ckpt）一条
  路，其 scheme 全集 = fp8/int8 × `{q8f, sgl, torchao, triton, vllm}`（`H3_CHANNEL_QUANT_SCHEMES`：
  `{name}.weight` 量化 + `{name}.weight_scale` per-out-channel；模板 `configs/minimax_h3/fp8/*.json`）
  ——**该 scheme 族为第三方非昇腾实现，Ascend 无 kernel** ⇒ NPU 上框架内建路线跑不通（**非「框架不
  支持量化」**）。
- **可解路线（已落地）**：H3 权重树为自研 `WeightModule` / `MMWeight`（**0 个 `nn.Linear`**，
  `quantize()` 直接接线无对象）⇒ 改为**注册原生 scheme** `dit_quant_scheme="npu-w8a8-mxfp8"`
  （`mm_weight.py` 的 `MMWeightNpuW8a8Mxfp8`，内部直连 mindiesd
  `W8A8MXFP8OnlineQuantLinear` / `npu_quant_matmul`；`model.py` 门对 online scheme 免预量化 ckpt）。
- 方向（**本组合观测**）：该原生 scheme 在真实模型上**量化生效 + compile 生效**，步时下降
  **约一成多**（较 bf16 clean-window，2× 复现）、**质量变化度**：帧 SSIM **降约 0.03（近无损档）**；
  monkeypatch 版同效。**非 docs `quantize()` 接口，W8A8_DYNAMIC 未接**。

### 3.5 稀疏 FA（`rf_v3` / `video_spans` 平台扩展）

- 接入姿势（**零框架源码改动**）：平台侧新注册 attn `npu_flash_attn_rf3`（`npu_flash_attn` 超类 +
  模块级配置 `rf_v3` / `video_spans` 分支），launcher 在 infer 首调 `configure_sparse`
  （`span = text + audio` 取 indices、`grid=[37,24,42]`）。
- 方向（**本组合观测**）：精度上 **eager 形态质量可用**，**采纳档 `sp≤0.5`**（质量优先选 bf16 稀疏、
  性能优先选 `mix`——`mix` = Q/K INT8 + V FP8 块量化）；稀疏度更高为**激进备选**（质量显著下降）。
  性能随稀疏度提高而提升；**质量变化度**（同 seed / 同配置 / 同窗 21 帧门禁 vs dense）：
  sp0.3 SSIM **降约 0.03**、sp0.5 降约 0.04、sp0.8 降约 0.19（显著档）；性能 bf16 sp0.5 降约三成、
  `mix` sp0.6 降约四成半（读数见归档）。
- **质量结论更正（重要留痕）**：同 seed / 同配置 / 同窗帧门禁下，质量梯度**平滑单调**；
  历史「某稀疏区间质量平台化」的结论**证伪**（早期轮次与 dense 基线不同配置 / seed / 口径混淆所致，
  证据作废）。**质量判定必须同 seed / 同配置 / 同窗**。
- 前置与待解：**compile × rf3** 有 Dynamo trace 期错误（首步取证未完）→ 生产叠加需先解决；
  vLLM-Omni 的 RAINFUSION 几何契约（video 为 packed tail + 不规则尾 promote 进 prefix + prefix 全保留）
  在本轮证明**非质量必需**（两框架同为 mindiesd `rf_v3` 路径）。
- **fp8 a2a comm 在本环境不可用**：`seq_p_fp8_comm` 走 vllm 的
  `dynamic_per_token_scaled_fp8_quant`（Ascend 未注册）；naive per-token fp8 回退可跑通，但
  **量化开销 > 通信节省** ⇒ **否决**。

### 3.6 计数契约与收益判定

- **统计口径（重要）**：多卡日志中 Run DiT 各 rank 不同 ⇒ 对比**必须固定 rank0**（或取 max）；
  用 `head -1` 混 rank 会**夸大收益**（本案例曾把小幅收益误报成数倍于真值的收益）；per-step **p50 比
  avg 稳**（avg 被首步编译拉高）。
- **kernel 级改善 ≠ 墙钟收益**：某 2D 融合 kernel 在 kernel 级为正、但墙钟无收益（`.contiguous()`
  拷贝 + 启动开销抵消）⇒ **必须以 rank0 墙钟为准，kernel 级只作解释**。
- **收益判定用 kernel diff**（图命中 ≠ 运行期生效），再配墙钟步时；只 rank0 采集。
- **质量档位口径（阈值口径）**：以帧 PSNR/SSIM 分「近无感 / 可见 / 显著下降」三档，**阈值数值以运行时
  profile 为准**（`accuracy-gate/references/quality-gate.md` + 仓库 `evals/`；图像与视频的数值域不同、**不可跨域迁移**）。
  无 VLM 时视觉门判 `inconclusive`，须并排帧存证。
- **热降频口径**：满载长跑后段会热降频（步长抬高、与代码/租户无关）⇒ 对比一律用
  **rank0 clean-window（steps 2-14）avg/p50**。**口径单点在 perf-gate**
  （`../../perf-gate/references/measurement-discipline.md` §7：clean-window 定义与热降频判定），
  **本框架的读数**在 `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §5。

## 4. 合入版机制与代码地图（#1471，配置全部可选、common 零侵入）

- **compile backend 注册表**：`lightx2v/utils/registry_factory.py` 新增 `COMPILE_BACKEND_REGISTER`
  （merge `PLATFORM_COMPILE_BACKEND_REGISTER`）；平台注册
  `lightx2v_platform/compilation/ascend_npu/mindie.py` 的
  `@PLATFORM_COMPILE_BACKEND_REGISTER("mindie")` **懒工厂**（`import` mindiesd 失败 raise
  RuntimeError，明确提示需 MindIE-SD）。core 侧 `BaseTransformerInfer._create_compile_backend`
  **只查注册表，未知名 raise（不静默回退）**；不再硬编码 mindiesd 包名。
- **a2a 可选后端**：`lightx2v_platform/ops/a2a/ascend_npu/ulysses_a2a.py` 的
  `@PLATFORM_A2A_BACKEND_REGISTER("hccl_eager")`（`@torch.compiler.disable` 的
  `HcclEagerUlyssesA2A`）；common `create_ulysses_a2a_backend` 保留 torch / round_robin 内建后查
  `A2A_BACKEND_REGISTER`，未知名 raise。
- **平台导入顺序**：`lightx2v_platform/ops/__init__.py` 的 ascend 分支**先注册 `.a2a.ascend_npu`
  再注册 attn**（attn 可能 import 公共 a2a 工厂）。
- `compile_dynamic` 配置键**已移除**（实测无收益：Sym 动态 shape 抵消固定红利）。
- **配置示例**（合入版，`configs/platforms/ascend_npu/minimax_h3_t2av_sp_compile_5s.json`）：

  ```json
  {
    "infer_steps": 30,
    "target_video_length": 120,
    "target_height": 768,
    "target_width": 1344,
    "attn_type": "npu_flash_attn",
    "rms_type": "npu_rms_norm",
    "rope_type": "minimax_h3_npu_rope",
    "use_compile": true,
    "compile_backend": "mindie",
    "parallel": {
      "tensor_p_size": 1,
      "seq_p_size": 4,
      "seq_p_attn_type": "ulysses",
      "seq_p_a2a_backend": "hccl_eager"
    }
  }
  ```

## 5. 回修与坑（`[探针]` 标注）

> 以下为**未合入上游 / 本地镜像的平台侧改动** ⇒ 按本仓「经验 vs 探针」判定为**探针**，不作为推荐
> 姿势与报表宣称；由探针发现的 **durable 约束**（注册顺序、变体必要性、配置配套）按经验处理。

1. `[探针]` **平台侧原生量化 scheme** `dit_quant_scheme="npu-w8a8-mxfp8"`
   （`mm_weight.py` / `model.py` 门；远端带 `.bak` 备份）——见 §3.4。
2. `[探针]` **平台侧稀疏 attn 注册** `npu_flash_attn_rf3` + launcher 的 `configure_sparse` 首调
   （零框架源码改动，但为平台层新增）——见 §3.5。
3. `[探针]` **swiglu pattern 的 `split_twice` 变体**（mindiesd 侧 pattern 变体）——丢失即融合静默
   消失，只有 profile 可见 ⇒ **同步回退后必须复采 profile 核验**（§1.2）。
4. `[探针]` **注册顺序约束**：`enable_minimax_h3_gate` 必须先于 `enable_wan_residual_gate`，
   否则 wan 泛型 `x+y*gate` 会抢占 H3 `index_select` 位点并在 2D 下运行期 fallback（日志可见
   `[residual_gate_add] fallback (ndim)`，kernel 级 `gather_residual_gate` 消失）。

## 6. 产物坐标指针（实测记录不入 skills）

- 运行产物根 `{run_results_dir}/`：serve 日志（per-step cost / 墙钟）、`ASCEND_PROFILER_OUTPUT/`
  （`kernel_details.csv` + `trace_view.json` + `step_trace_time.csv`）、质量对产物；采集方法见
  `profiling-collect`（标准 CANN profiler 姿势 + 少步快速采集经验）。
- **本次迁移归档**：`{run_results_dir}/archive/lightx2v-mindiesd-case.md` —— 本框架链的**绝对耗时、
  绝对加速比与质量数值**原文；只在原环境成立，**不可跨模型 / 框架 / 规模 / 窗口引用**。
- 会话过程文档（**框架仓内，非本仓、不入库**）：稀疏/量化收口与
  无损矩阵文档、跨框架 GAP·COMPARE 文档。
- 支持矩阵：**L1** 证据码的**能力面**留在 `framework-support-matrix.md`（含最后一列 last-checked），
  **开启方式**指回本文件。

## 7. 维护与更新

- 触发（版本 / 合入态势）：LightX2V 偏离上游合入版 PR #1471 的平台注册表架构（退回「全局改共享类」姿势）、或 `lightx2v_platform` 的 `PLATFORM_*_REGISTER` 机制调整时，§1.1 策略判定、§1.2「代码 / 配置必须配套」（`seq_p_a2a_backend`、`rms_type`、`rope_type`、`use_compile` / `compile_backend`）与 §4 代码地图须重核。
- 触发（使能集合与探针）：mindiesd 侧 swiglu 的 `split_twice` 变体或注册顺序约束（`enable_minimax_h3_gate` 先于 `enable_wan_residual_gate`）失效，或 §5 的两条平台侧探针（`dit_quant_scheme="npu-w8a8-mxfp8"`、`npu_flash_attn_rf3` 与 launcher 的 `configure_sparse`）被回填覆盖时，§3.2 的 compile 三坑、§3.4 量化与 §3.5 稀疏的档位结论须重测。
- 触发（支持面与口径）：`framework-support-matrix.md` 的 **L1** 证据码行刷新，或 clean-window / 质量档位口径（`accuracy-gate` 与仓库 `evals/`）变更时，§3.6 的 rank0 clean-window 与质量档位结论须同步复核。
- 复核：最小核对 = 同配置跑一次并只在 rank0 采 kernel diff，看 `kernel_details.csv` 中 `swiglu` / `gather_scale_shift` / `gather_residual_gate` 融合 kernel 是否仍出现、`hcom_alltoallv` 是否仍完全消失（a2a 走 `hccl_eager` 未退化）；mindiesd 同步回退后必须复采 profile（swiglu 融合丢失**日志无痕、只有 profile 可见**）。
