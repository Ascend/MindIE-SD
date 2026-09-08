# 实例：LightX2V MiniMax-H3 接入 MindIE-SD（本框架实测案例）

> ⚠️ **本文件是 LightX2V 特定实例**，结论（接入方式、compile 修复、收益数据）
> 仅在该框架 + USP4 + MiniMax-H3 配置下成立。移植到其他三方框架时，
> 用 SKILL.md §使能与验证回路重新验证，勿直接照搬。
>
> ✅ **已合入上游**：PR `feat(ascend): add MiniMax-H3 fused RoPE and MindIE SD compile backend`
> （#1471，commit `5c225825`）。本文档以**合入版（平台注册表架构）**为准；
> 合入前"全局改共享类"的版本仅在历史提交中，勿再按旧姿势操作。
>
> 场景：三方推理框架 LightX2V（`lightx2v` pip 包 + `lightx2v_platform` 平台算子层），
> 在远端昇腾容器（Ascend 950PR ×8）上，把 mindiesd 的算子融合接入其 MiniMax-H3
> 推理路径，并用标准 CANN profiler 做算子级验证。

## 目录

- [1. 环境与登录](#1-环境与登录)
- [2. 接入策略（核心结论，合入后最终形态）](#2-接入策略核心结论合入后最终形态)
- [3. 运行时接入实现（保留）](#3-运行时接入实现保留)
  - [3.1 npu_rms_norm](#31-npu_rms_norm)
  - [3.2 minimax_h3_npu_rope（合入版，走 mindiesd 算子）](#32-minimax_h3_npu_rope合入版走-mindiesd-算子)
- [4. 性能数据（R18 rank0 校正口径）](#4-性能数据r18-rank0-校正口径)
  - [4.1 墙钟（5s / 15s，4 卡 USP4，30 步，rank0）](#41-墙钟5s--15s4-卡-usp430-步rank0)
  - [4.2 kernel 级 diff（compile vs runtime，CANN profiler，1 profiled step，rank0）](#42-kernel-级-diffcompile-vs-runtimecann-profiler1-profiled-steprank0)
- [5. compile 接入（成功路径 + 合入版机制）](#5-compile-接入成功路径--合入版机制)
  - [5.1 必须修的坑（原理；合入版机制见 5.2）](#51-必须修的坑原理合入版机制见-52)
  - [5.2 合入版机制与代码地图（#1471，配置全部可选、common 零侵入）](#52-合入版机制与代码地图1471配置全部可选common-零侵入)
  - [5.3 最终结论（R18 rank0 口径）](#53-最终结论r18-rank0-口径)
- [6. 采集方法（标准 CANN profiler）](#6-采集方法标准-cann-profiler)
- [7. 经验教训](#7-经验教训)
- [8. R19 增补（2026-09-04 复核：mindiesd 同步回退 + 热降频口径）](#8-r19-增补2026-09-04-复核mindiesd-同步回退--热降频口径)
- [10. R22-R25 增补（S4 有损 + 无损矩阵要点）](#10-r22-r25-增补s4-有损--无损矩阵要点)
- [11. 跨框架差距：LightX2V-H3 待补充能力清单](#11-跨框架差距lightx2v-h3-待补充能力清单2026-09-对比-skills-其他框架后提炼)

## 1. 环境与登录

| 项 | 值 |
|---|---|
| 远端设备 | <远端 IP>（root，容器 `{容器名}`） |
| 模型 | `{model_dir}/MiniMax-H3`（transformer 62GB + text encoder 63GB） |
| 代码 | `{repo}/LightX2V`（合入版 = 上游 main）+ `{repo}/MindIE-SD`（mindiesd） |
| 并行 | USP4（tensor_p=1, seq_p=4, ulysses a2a），torchrun 4 卡 |
| 序列 | 5s：local 9467 / global 37751；15s：local ~27276 / global ~109103 |

## 2. 接入策略（核心结论，合入后最终形态）

LightX2V 的算子通过注册表选择实现（`RMS_WEIGHT_REGISTER` / `ROPE_REGISTER` /
`ATTN_WEIGHT_REGISTER` + 平台级 `PLATFORM_*_REGISTER`）。合入后 MiniMax-H3 的
DiTBlock 采用**三路组合**，全部由配置驱动、零侵入：

| 算子 | 抽象接口 | 接入方式（合入版） | 实测结果（rank0） |
|---|---|---|---|
| RMSNorm | 有（`rms_type`） | 运行时：`"rms_type": "npu_rms_norm"` | 累计 -10%（含 rope） |
| RoPE | 有（`rope_type`） | 运行时：`"rope_type": "minimax_h3_npu_rope"`（mindiesd `rotary_position_embedding`） | 累计 -10% |
| AdaLN / SwiGLU / residual gate | 无 | compile：`"use_compile": true` + `"compile_backend": "mindie"` | 5s Run DiT **-1.8%~-7.5%**；15s **-1.2%** |
| Ulysses a2a | 有（`seq_p_a2a_backend`） | 可选后端：`"seq_p_a2a_backend": "hccl_eager"`（不编进图） | 通信 **-35%**（与 compile 一起） |

**策略判定**：有抽象接口的算子走运行时注册表替换（配置字段切换）；
无接口的算子走 compile（须先解决 §5.1 的 3 个坑）；多卡 collective 用框架
**原生可选后端**留在 eager——三者都不改框架核心共享代码。

> ⚠️ 早期版本曾把 a2a 以 `@torch._dynamo.disable` 全局钉在 `TorchUlyssesA2A.exchange`
> （影响所有平台所有 Ulysses 用户）；评审后收敛为平台注册的可选后端 `hccl_eager`，
> common 侧 `TorchUlyssesA2A` 恢复纯净。**这是可移植的设计原则**：
> 框架接入应优先"平台注册 + 配置可选"，不要全局改共享类。

## 3. 运行时接入实现（保留）

### 3.1 npu_rms_norm

- 文件：`lightx2v_platform/ops/norm/ascend_npu/npu_rms_norm.py`
- 注册：`@PLATFORM_RMS_WEIGHT_REGISTER("npu_rms_norm")`
- 配置：`"rms_type": "npu_rms_norm"`（H3 weights 构建读 `config.get("rms_type", "torch_native")`）
- 平台注册表 merge 进主 `RMS_WEIGHT_REGISTER`（import `lightx2v_platform` 时触发）

### 3.2 minimax_h3_npu_rope（合入版，走 mindiesd 算子）

- 文件：`lightx2v_platform/ops/rope/ascend_npu/minimax_h3_npu_rope.py`
- 注册：`@PLATFORM_ROPE_REGISTER("minimax_h3_npu_rope")`；配置 `"rope_type": "minimax_h3_npu_rope"`
- **懒加载**：模块级 `_load_mindiesd_rope()`（`lru_cache`）——`import mindiesd` 成功返回
  `rotary_position_embedding`；`ImportError` 告警返回 None（走 TorchRealRope fallback）；
  其他异常 raise `RuntimeError`。实例缓存 `_mindiesd_rope` / `_fallback_rope`。
- **算子姿势**：H3 部分旋转（rotary_dim=96 / head_dim=128），调用
  `mindiesd.layers.rotary_position_embedding(x, cos, sin, rotated_mode="rotated_half",
  head_first=False, fused=True)`：
  - x 传 4D SBND `x_rot.unsqueeze(1)` = `[L,1,H,D]`
  - cos/sin 显式传 4D S11D `[L,1,1,D]`（先 `cos.to(x.dtype)`）
  - ⚠️ **坑**：mindiesd 该算子的 2D `[S,D]` cos 路径假设 x 是 `[B,S,N,D]`
    （head_first=False 保留 i=1 维）；对 SBND 会 reshape 成 `[1,1,1,D]` 报形状错。
    必须显式升 4D S11D 传入（与 mindiesd compile pattern 传 4D cos 一致）。
- **校验（类内 `_validate_inputs`）**：x 3D、freqs 为 (cos,sin) tuple、cos/sin 同形同设备、
  cos 2D 且 seq 对齐、rotary_dim 正偶数 ≤ head 且等于 cos 宽度。
- fallback（无 mindiesd）：`TorchRealRope(layout="split_half")`，数值 bit-exact。
  mindiesd 算子路径与 rotate-half 语义一致（bf16 计算，距 fp32 参考 ≤1 ULP）。

## 4. 性能数据（R18 rank0 校正口径）

### 4.1 墙钟（5s / 15s，4 卡 USP4，30 步，rank0）

| 运行（5s） | rank0 per-step sum | rank0 Run DiT | p50 |
|---|---|---|---|
| runtime（融合，无 compile） | 144.64s | 155.24s | 3.954s |
| compile（干净环境重跑） | **136.77s** | **143.65s** | 3.831s |

- 5s：Run DiT **-1.8%~-7.5%**（两次 full 跑 -1.8%~-2.1%，干净重跑 -7.5%）；
  per-step sum **-2.0%~-5.4%**；一次 +3.5% 离群确认是运行波动
- 15s：Run DiT **1376.4s vs 1392.6s = -1.2%**（per-step 1349.6 vs 1366.1，一致）
- 合计 vs torch-native 基线（rms/rope 未融合）：约 **-12%~-15%**（运行时融合 -10% + compile 再降）

> ⚠️ 墙钟口径铁律：Run DiT 必须固定 **rank0**（4 卡同步收尾 rank）；`head -1` 混 rank
> 曾把 -2% 误报成 -6.5%（R18 修正）。per-step p50 比 avg 稳（avg 被首步编译拉高）。

### 4.2 kernel 级 diff（compile vs runtime，CANN profiler，1 profiled step，rank0）

| 指标 | runtime | compile | Δ |
|---|---|---|---|
| kernel 总数 | 3600 | 3050 | **-550（-15.3%）** |
| kernel 总耗时 | 3968ms | 3576ms | **-9.9%** |
| 通信 | 962ms | 621ms | **-35%** |

- 新增融合 kernel：`swiglu` 50×32ms、`gather_scale_shift` 200×19ms、`gather_residual_gate` 100×11ms
- 消除算子链：Mul **-90%**、Add **-77%**、Silu **-99%**、IndexSelect **-87%**
- `hcom_alltoallv` **完全消失**（a2a 用 `hccl_eager` 留 eager 后不再退化；此前 Dynamo 把
  `split_sizes` 推断成 `[1,1,...]`，单次 200-270ms）
- 通信 -35% 是额外红利：collective 出图后与编译计算重叠改善

15s kernel 级：`swiglu` 93.7ms + `gather_scale_shift` 55.5ms + `gather_residual_gate`
32.3ms（共 181.5ms）替代 99.4ms 旧链；kernel 数 -15.3%、kernel 耗时 -333ms（-0.7%）。
⚠️ 15s 通信占 kernel 总耗时 **57.6%**（28002/48584ms）——长序列 a2a 数据线性增长，
是 Ulysses 层面瓶颈，compile 不可及，稀释了融合收益（解释 15s 墙钟仅 -1.2%）。

## 5. compile 接入（成功路径 + 合入版机制）

把 adaln/swiglu/gate 经 `torch.compile(block_runner, backend={MindIE})` 接入编译图。
初版曾 +69% 负收益（AOTAutograd 把 custom op 当黑盒，边界插大量连续化拷贝 kernel），
修复后转为正收益。

### 5.1 必须修的坑（原理；合入版机制见 5.2）

1. **a2a 不能进编译图**：Dynamo 追踪 `dist.all_to_all_single` → `split_sizes=[1,1,...]`
   → HCCL 退化 `hcom_alltoallv`（200-270ms/次 vs 4ms），墙钟 +50%。
2. **backend 实例复用**：每次新建 `MindieSDBackend()` → Dynamo `BACKEND_MATCH failure`
   反复重编译，`recompile_limit(8)` 后**静默退回 eager**。必须缓存单实例。
3. **swiglu pattern 双 split 变体**：LightX2V 的 `chunk(2)` 被 Dynamo 展开成两个独立
   split 节点（diffusers 是单 split 双 getitem）→ mindiesd pattern 需 `split_twice` 变体。

### 5.2 合入版机制与代码地图（#1471，配置全部可选、common 零侵入）

- **compile backend 注册表**：`lightx2v/utils/registry_factory.py` 新增
  `COMPILE_BACKEND_REGISTER`（merge `PLATFORM_COMPILE_BACKEND_REGISTER`）。
  平台注册：`lightx2v_platform/compilation/ascend_npu/mindie.py`
  `@PLATFORM_COMPILE_BACKEND_REGISTER("mindie")` 懒工厂（`import` mindiesd 失败 raise
  RuntimeError，明确提示需 MindIE-SD）。
  core 侧 `BaseTransformerInfer._create_compile_backend` 只查注册表，未知名 **raise**
  （不静默回退）；不再硬编码 mindiesd 包名。
- **a2a 可选后端**：`lightx2v_platform/ops/a2a/ascend_npu/ulysses_a2a.py`
  `@PLATFORM_A2A_BACKEND_REGISTER("hccl_eager")`（`@torch.compiler.disable` 的
  `HcclEagerUlyssesA2A`）。common `create_ulysses_a2a_backend` 保留 torch/round_robin
  内建后查 `A2A_BACKEND_REGISTER`，未知名 raise。
- 平台导入顺序：`lightx2v_platform/ops/__init__.py` ascend 分支**先注册 `.a2a.ascend_npu`**
  再注册 attn（attn 可能 import 公共 a2a 工厂）。
- `compile_dynamic` 配置键：合入时已移除（R16 实测无收益，Sym 动态 shape 抵消固定红利）。
- **配置示例**（`configs/platforms/ascend_npu/minimax_h3_t2av_sp_compile_5s.json`，合入版）：

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

> ⚠️ 代码/配置必须配套：合入版配置含 `seq_p_a2a_backend: "hccl_eager"`，若代码是合入前
> 版本会报 unknown a2a backend；若代码合入而配置缺该键，a2a 回编译图丢 -35% 通信红利。
> 远端代码升级后务必同步配置。

### 5.3 最终结论（R18 rank0 口径）

| 指标 | 5s (4卡) | 15s (4卡) |
|---|---|---|
| kernel 总耗时 | -9.9%（3968→3576ms） | -0.7%（-333ms） |
| kernel 数 | -550（-15.3%） | -550（-15.3%） |
| 通信 | -35%（962→621ms） | — |
| Run DiT 墙钟 | **-1.8%~-7.5%**（干净环境 -7.5%） | **-1.2%** |
| per-step sum | **-2.0%~-5.4%** | -1.2% |

子图审计（DUMP_N=8）：block 被 a2a/FA graph-break 拆成 ~8 个子图，调制/adaln/swiglu/gate
段全部融合；唯一未融合是 gate-msa 残差（`residual + gate*attn_out` 跨子图边界），属拆图
固有代价。**DiTBlock 中所有可融合算子已全部融合**（gate-msa 2D kernel 原型已试回退：
kernel 级 -0.7% 但墙钟无收益）。

## 6. 采集方法（标准 CANN profiler）

对任意三方框架：

```python
handler = torch_npu.profiler.tensorboard_trace_handler(PROF_OUT)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    record_shapes=True,
    on_trace_ready=handler,
) as prof:
    <框架推理 1 步>
    torch.npu.synchronize()
```

warmup 5 步在 profiler 外；只 rank0 采集；产出 `ASCEND_PROFILER_OUTPUT/`
（kernel_details.csv + trace_view.json + step_trace_time.csv），打包回传后
喂 `profiling-analyze/scripts/analyze_trace.py` + `compare_traces.py`。

> 少步快速采集（经验，见 profiling-collect SKILL「少步快速采集经验」）：eager / 图已编译稳定时
> profiler 外 warmup 可压到 **1 步**（`H3_WARMUP_STEPS=1`），采第 2 步 1 步即可——单步即代表
> 算子形态与 kernel 序，kernel diff 用 1 步数据足够；compile/首次 JIT 仍须预热覆盖编译（≥10 步）。

## 7. 经验教训

- **统计口径（重要）**：多卡日志 Run DiT 各 rank 不同，对比必须固定 rank0（或 max）；
  `head -1` 混 rank 会夸大收益（曾把 -2% 误报成 -6.5%）；per-step p50 比 avg 稳
- **kernel 级改善 ≠ 墙钟收益**：gate-msa 2D 融合 kernel -0.7% 但墙钟无收益（拷贝开销抵消），
  必须以 rank0 墙钟为准
- **框架接入优先"平台注册 + 配置可选"**：a2a 留 eager、compile backend 合入后都收敛为
  `PLATFORM_*_REGISTER` + 配置键（`hccl_eager` / `mindie`），common 零改动——全局改共享类
  会被评审要求收敛（本案例真实发生过）
- **卡选择**：npu-smi 先看 Health/占用；避开 Alarm 卡；对比必须同卡组；UB LINK ERROR 重跑
- **HCCL 端口**：16666 冲突 → `HCCL_NPU_SOCKET_PORT_RANGE=20000-21000`（RoCE 端口 ss 查不到）
- **CRLF**：Windows .sh 上传前转 LF

## 8. R19 增补（2026-09-04 复核：mindiesd 同步回退 + 热降频口径）

- **同步回退风险**：mindiesd 从 dev-skills 分支整仓回填远端会**覆盖 R18 会话内未合入的修复**。
  本案例实际复现两处并已修复回填（详见 `LightX2V/session_work/R19_CLOSURE_LIGHTX2V_H3.md`）：
  1. `minimax_h3_swiglu_pattern.py` 需 **split_twice 变体**（LightX2V chunk(2) 展开为两个独立
     split 节点）——丢失后 swiglu 融合静默消失（墙钟 +2% 回归），且 profile 才可见（日志无痕）
  2. **注册顺序**：`enable_minimax_h3_gate` 必须先于 `enable_wan_residual_gate`，否则 wan 泛型
     `x+y*gate` 抢占 H3 index_select 位点并在 2D 下运行期 fallback（日志可见
     `[residual_gate_add] fallback (ndim)`，kernel 级 gather_residual_gate 消失）
  - 核验姿势：对比 kernel_details.csv 中的 swiglu_kernel / gather_residual_gate_kernel 计数
- **热降频 → clean-window 口径**：950PR 满载 ~600W/卡，机箱热时 30 步长跑在 ~14-17 步后
  降频 4→7-8s/步（83-86°C，与 compile/租户无关）→ 与历史绝对数字不可比；对比一律用
  **rank0 steps 2-14 clean-window avg/p50**（该窗口与无降频时代数字吻合）
- **fp8 a2a comm 在此环境不可用**：`seq_p_fp8_comm` 走 vllm `dynamic_per_token_scaled_fp8_quant`
  （Ascend 未注册）；naive per-token fp8 回退跑通但量化开销 > 通信节省（+1.9%），否决
- **收益判定**：kernel diff 为准（图命中 ≠ 运行期生效），配墙钟 step 时

## 10. R22-R25 增补（S4 有损 + 无损矩阵要点）

完整文档：`LightX2V/session_work/R22_S4_CLOSURE_LIGHTX2V_H3.md`、`R23_LOSSLESS_MATRIX_LIGHTX2V_H3.md`

- **S4 采纳项**：compile + `infer_steps=24`（原生步数裁剪，Run DiT -28~-37%，帧 SSIM vs 30 步
  0.937，近无感；20 步 -47%/0.894 为激进档）——纯配置、机器自带能力；当时「优于为 H3 移植
  Taylor cache」成立的前提是框架 cache 不可用（`feature_caching` 对 H3 显式 NotImplemented）；
  bench 侧 mindiesd CacheAgent 接入（§11 P0）后可做「24 步 × cache」叠加评估，勿沿用旧表述
- **S4 采纳项（2026-09 增）**：+ 原生线性量化 scheme `dit_quant_scheme="npu-w8a8-mxfp8"`
  （§11 P1 落地；最终部署配置 `r28_final_24step_npu_mxfp8.json`）：24 步 Run DiT ≈100-118s、
  **clean-window ≈3.55-3.62s/步（较 bf16 -13%，2× 复现），帧 SSIM 0.972（近无损）**
- **mindiesd rf_v3 稀疏 FA 可配置接入 H3**（零框架源码改动）：平台新注册 attn
  `npu_flash_attn_rf3`（npu_flash_attn 超类 + 模块级配置 rf_v3/video_spans 分支）+ launcher 在
  infer 首调 `configure_sparse`（span=text+audio 取 indices、grid=[37,24,42]）。性能 bf16
  sp0.5 -28%、mix sp0.6 -45%。
  **2026-09 质量结论更正（同 seed/同配置/同窗 21 帧门禁 vs dense bf16）**：质量梯度**平滑单调**——
  sp0.3 **SSIM 0.975**（近无损）、sp0.5 **0.960**（良好）、sp0.8 **0.81**（激进档）；
  历史「sp0.3-0.6 平台 0.82-0.84」**证伪**（R19 时代与 dense 基线不同配置/seed/口径混淆所致，
  证据作废）。eager 形态质量可用（采纳档 sp≤0.5）；**compile×rf3 仍有 Dynamo trace 期错误**
  （首步取证未完）→ 生产叠加需先解 compile×rf3；vllm-omni RAINFUSION 几何契约（video 为
  packed tail + 不规则尾 promote 进 prefix + prefix 全保留；A5 上与 LightX2V 同为 mindiesd rf_v3
  路径）本轮证明非质量必需（未加契约梯度已正常）
- **S4 量化阻塞**：H3 `dit_quantized` 需 Ascend int8/fp8 GEMM kernel——本机 triton int8 GEMM
  MLIR 编译失败、fp8-triton Assertion、vllm/sgl/q8f 等第三方方案在昇腾无可用 kernel（格式已查明：
  `{name}.weight` 量化 + `{name}.weight_scale` per-out-channel）
- **质量门禁首案例校准**（MindIE-SD `evals/` quality_compare，帧 PSNR/SSIM）：近无感 ≈SSIM 0.94+/
  PSNR 29+；可见 0.80-0.90；显著 <0.80（无 VLM → 定量 + 并排帧）
- **无损矩阵要点**：融合阶梯 kernel 5700/4426ms→3600/3968(-10.4% rms/rope 运行时)→3200/3629
  (-18% +compile adaln/swiglu/gate)；并行选型依赖拓扑（UB 岛 bulk、SYS 跨岛 head-parallel，
  见 parallelism-strategy `ascend-topology-bandwidth-diag.md`）；HCCL 微基准须先 `set_device`
- **并行形态补充（矩阵列证据锚）**：TP2×SP2（tensor_p=2）实测可用（UB 岛 3.97-4.04s/步）但
  差于 USP4 bulk → 回退（R23 §B）；TP 单用未单独核；H3 SP 仅 Ulysses（model.py 门：非
  Ulysses parallel_type 直接 NotImplemented）→ **RSP 无 Ring 实现**（非 attn_mask 问题）
- **显存补充（矩阵列证据锚）**：已启用 `vae_cpu_offload=true + vae_decode_parallel=true`；
  H3 model 支持 model/block CPU offload（model.py:62；`*_block_offload` 配置存在但未跑墙钟）；
  mindiesd `enable_offload` 异步档（层粒度）未核
- **量化阻塞的准确表述（§10 上一格修正依据）**：框架侧仅 `dit_quantized`（per-output-channel
  预量化 ckpt）一条路，且其 kernel 尝试（triton int8 GEMM / fp8-triton）在本机编译失败；
  **不等于「本 NPU 无任何量化 kernel」**——mindiesd `quantization/quantize.py` 的
  W8A8_DYNAMIC/W8A8_MXFP8（含 online）在本机存在、同族 harness 于同 NPU 实证跑通
  （28 Linear 量化 + compile PASS）→ 阻塞实为「框架未接线 mindiesd quantize 路线」（§11 P1）。
  补充（框架内建面）：LightX2V `dit_quantized` 支持 scheme 全集 = fp8/int8 ×
  {q8f,sgl,torchao,triton,vllm}（model.py `H3_CHANNEL_QUANT_SCHEMES`，per-output-channel +
  `{name}.weight_scale`），模板 `configs/minimax_h3/fp8/*.json`——**该 scheme 族为第三方非昇腾实现（q8f/sgl/torchao/vllm，Ascend 无 kernel）** → NPU 上框架内建路线跑不通（非框架不支持）
- 环境：SIGKILL 进行中多卡任务可伤 NPU 驱动（全组 ~10× 慢、端口 bind 泄漏）→ 换卡组恢复；
  脚本卡组须硬编码本地（远程执行器会重传覆盖 sed 修改）；多轮实验后端口段易耗尽，用新段
  或驱动复位

## 11. 跨框架差距：LightX2V-H3 待补充能力清单（2026-09 对比 skills 其他框架后提炼）

> 方法：以 vllm-omni×H3（V1）/ DiffSynth×Qwen-Image（V2）/ mindiesd 库能力为基准，对照本 case
> 全部实测/阻塞记录，识别「框架需要补充的能力」+ 落点 + 前置。完整矩阵见
> `framework-support-matrix.md`（LightX2V #1471 列）；此处只记**结论与触发**。

- **P0 DiTCache/AttentionCache**（最大显性缺口）：框架侧 `feature_caching` 对 H3 显式
  NotImplemented（model.py:447）；但 mindiesd `cache_agent/`（CacheAgent + DiTBlockCache +
  AttentionCache）已具备、V1 组合内 3.65×、V2 实证**bench 侧零框架改动接入** CacheAgent
  （DiTBlockCache -17~-27%/step、AttentionCache -17.6%、双缓存互斥同开=fail-closed 假加速、
  计数契约 reuse/compute、CFG-on shape 约束）→ LightX2V 补缓存=接线工作，先照 DSE 先例
  bench 侧接入验证（注意 compile 分段下 per-block 拦截点需先核验），可行再 runner 级集成；
  与已采纳的 `infer_steps=24` 时间步做叠加评估（S4-2 协议）
- **P1 mindiesd `quantize()` online 路线 = 线性层量化 seam**（W8A8_DYNAMIC / W8A8_MXFP8）：
  本 case §10 的量化阻塞仅针对 `dit_quantized`（预量化 ckpt，第三方非昇腾 scheme）与 triton int8/fp8
  GEMM 两条路线；同 NPU 上 mindiesd `quantization/quantize.py` 的 W8A8_MXFP8/DYNAMIC（含
  online）已有同族 harness 实证（28 Linear 量化 + compile PASS、vllm-omni INT8 -22%）→
  缺口 = LightX2V 编译前接入 `quantize(model, online_config=…)`，需 smoke 验证本机 kernel +
  compile 兼容 + 质量门禁。
  **2026-09 已按原生方案落地**：LightX2V H3 权重树为自研 `WeightModule`/`MMWeight`（0 个
  nn.Linear，`quantize()` 直接接线无对象）→ 注册原生 scheme `dit_quant_scheme="npu-w8a8-mxfp8"`
  （mm_weight.py `MMWeightNpuW8a8Mxfp8`，内部直连 mindiesd `W8A8MXFP8OnlineQuantLinear`
  /`npu_quant_matmul`；model.py 门对 online scheme 免预量化 ckpt；远端带 .bak 备份）→ 真实
  模型 24 步验证：**clean-window 3.567/3.550 vs bf16 4.058/4.043（-12.1~-12.2%），帧 SSIM
  0.972（近无损）**（350 DiT 模块量化 + compile 生效；monkeypatch 版同效 -13%/0.966）
- **P2 稀疏/注意力量化 seam（2026-09 状态更新：质量门禁通过、eager 可采纳）**：rf3 mix
  （Q/K INT8 + V FP8 块量化）性能已实证（sp0.6 -45%；全长 probe sp0.7→2.01×、sp0.8→2.06×）。
  历史「质量双因受限（mix sp0 0.902 / sp>0 平台 0.82-0.84）」**证伪**——同 seed/同配置/同窗
  21 帧门禁（vs dense bf16）：sp0.3 bf16 SSIM **0.975**（近无损）、sp0.5 bf16 **0.960**/mix
  **0.957**（良好）、sp0.8 bf16 **0.81**（激进档）；性能（eager fused 24 步 clean-window）：
  dense 4.074 → sp0.3 -10.6% → sp0.5 bf16 -17.7%/mix -23.1% → sp0.8 -27.5% →
  **采纳档 sp≤0.5（eager；质量优先 bf16、性能优先 mix）**；sp0.8 为激进备选（质量 0.81）。
  **前置**：compile×rf3 有 Dynamo trace 期错误（首步取证未完）→ 生产叠加需先解 compile×rf3；
  vllm-omni RAINFUSION 几何契约（video tail + 不规则尾 promote 进 prefix）本轮证明非质量必需
  （两框架同 mindiesd rf_v3 路径）
  核验 950PR 实现/scale 语义；质量优先选 bf16 稀疏、性能优先修 mix
- **P1 ACLGraph 生效核验**：docs 称 MindieSDBackend 内自动启用，LightX2V compile 流未核验——
  若未生效可能为免费红利；需 mindiesd 侧图捕获计数手段
- **P2**：rf_v2（docs 主表，V1 视频档 +1.30× 实证；rf3 质量受限时的另路，需核验 npu FA rf_v2
  在 950PR 可用性）；TP×USP 组合复测（V1 最优无损为 TP2×USP2）；offload 档位对齐 DLO（15s/
  长序列启动时）；comm masking 在 15s/长序列（comm 57.6%）重估
- **P3（硬前置/负证据）**：FA 量化（mindiesd FP8/MXFP8 FA——**A5 已微测可用**（`npu_fused_infer_attention_score_v2` fp8，cos≈0.998、rel≈5%，2026-09），端到端质量/收益未验证；LightX2V 未接线）、RSP（Ring 不支持 attn_mask，上游）、CFG 并行
- **移植纪律**：vllm-omni 的 compile 为负收益而 LightX2V 为正（架构/热路径差异）——上表任何
  补充都必须按「使能与验证回路」在 LightX2V 本体重验（kernel diff + rank0 墙钟 + 质量门禁），
  不直接照搬其他框架数字
