# MiniMax-H3 Dummy Run 适配记录

> **本文件的定位（先读）**：本文件是**模型底座**（H3 的几何 / 组件 / 仓库格式 / 构造与命令），
> 按 `.agents/README.md` §7 属「换框架仍成立、**换模型不成立**」一类。
> **可迁移的方法已下沉到各自能力的真源**，本文件只保留 H3 侧的实测细节；**与真源冲突时以真源为准**：
>
> | 本文件章节 | 方法真源（可迁移部分去这里） |
> |---|---|
> | §7 / §8 / §11 / §12 融合 pattern 与图形态 | `../../pattern-dev/references/fusion-graph-forms-and-semantics.md` + `pattern-dev` SKILL 的 Phase 2/5 |
> | §9 4 卡 CP + 通信掩盖 | `../../dit-parallel-opt/SKILL.md` §CP/掩盖 + `../../dit-parallel-opt/references/comm-masking-method.md` |
> | §13 w8a8(MXFP8) 编译图与数据格式 | `../../quantization-dev/references/online-quant-contract.md`（契约） |
> | §6 计算精度 / §A6 量化档位语义 | `../../dit-perf-opt/references/quant-tier-device-mapping.md`（选档语义） |
> | §B' profile 目录隔离 | `../../profiling-collect/references/profile-dir-isolation.md` |
> | §C compile vs eager 双报表口径 | `../../profiling-analyze/references/eager-vs-compile-report.md` |
>
> **目录** · [1. 仓库双格式](#1-仓库双格式最易踩坑) · [2. 配置获取（gated → modelscope）](#2-配置获取gated--modelscope) · [3. 依赖版本](#3-依赖版本) · [4. 组件清单](#4-组件清单) · [5. 关键适配点](#5-关键适配点) · [6. BF16 计算精度](#6-bf16-计算精度--compute-precision默认-bf16) · [7. RMSNorm 融合（H3 侧坐标）](#7-rmsnorm-融合h3-侧坐标) · [8. 其余融合（H3 侧坐标）](#8-其余融合h3-侧坐标) · [9. CP + 通信掩盖（H3 侧配置事实）](#9-4-卡-context-parallel--通信掩盖h3-侧并行配置事实) · [10. 验证结果](#10-验证结果远端-910b-npu2-layersdiffusers-0400-隔离安装256384124) · [11. SwiGLU + AdaLN + gate（H3 侧坐标）](#11-swiglu--adaln--gate-融合h3-侧实现坐标与实测细节) · [12. qk_norm + RoPE 大融合（负面结论归档）](#12-qk_norm--rope-大融合负面结论归档防重复实验) · [13. w8a8(MXFP8) 编译图（H3 侧图节点清单）](#13-w8a8mxfp8-编译图h3-侧图节点清单)
>
> MiniMax-H3（33B 全模态生成模型，T2VA / FL2VA / Ref2VA 工作流）在 `examples/dummy_run/` 的
> dummy run 适配要点。依据：`examples/dummy_run/minimax_h3_infer.py` 与
> `examples/dummy_run/model/minimax_h3_model.py`（2026-08 实测通过）。
>
> ⚠️ **数字纪律**：本文件的**绝对耗时与绝对加速比已按 `.agents/README.md` §7 移出**，归档于
> 会话产物目录 `{run_results_dir}/archive/minimax-h3-notes-numbers.md`；正文只保留
> **比例关系 / 占比 / 判定阈值 / 结构契约 / 计数契约**，方向性结论标注「本组合观测」。

## 1. 仓库双格式（最易踩坑）

`MiniMaxAI/MiniMax-H3`（modelscope 镜像 `MiniMax/MiniMax-H3` 同构）**同一仓库混两种布局**：

| 位置 | 格式 | 特征 | 用途 |
|---|---|---|---|
| 仓库**根目录** | diffusers 格式 | `model_index.json` / `modular_model_index.json`，`_class_name: MiniMaxH3ModularPipeline`；`transformer/`、`vae/`、`audio_vae/`、`text_encoder/`、`tokenizer/`、`processor/`、`scheduler/`、`audio_scheduler/` | **dummy run 只能用它** |
| `FL2VA/`、`Ref2VA/` 子目录 | vLLM-Omni 格式 | `model_index.json` 的类名（`MiniMaxH3DiTModel`、`MiniMaxH3Qwen3VLHFEncoder` 等）与 transformer 配置键（`ffn_hidden_size`、`latents_dim` 等）在 diffusers 0.40 中**均不存在** | vLLM-Omni 部署 |

- vLLM-Omni 部署目录（如 `{model_weight_dir}/MiniMax-H3/FL2VA`）**不能**当 `--config_cache`
- 判定方法：看 `model_index.json` 的 `_class_name` 是否为 `MiniMaxH3ModularPipeline`，
  或 `transformer/config.json` 是否有 `num_refiner_layers` / `ffn_dim` / `in_channels` 键

## 2. 配置获取（gated → modelscope）

- HF 上为 **gated 模型**，需审批；modelscope 镜像无需鉴权
- 脚本内 modelscope 优先 + HF 回退；下载模式只拉配置，不含权重：

```python
snapshot_download(
    "MiniMax/MiniMax-H3",
    allow_patterns=["*.json", "*.txt", "*.model", "*.py", "tokenizer*"],
    ignore_patterns=["*.safetensors", "*.bin", "*.msgpack", "*.ckpt", "*.pth", "*.index.json"],
    max_workers=1,
)
```

> 陷阱：`*.safetensors` 不匹配 `*.safetensors.index.json`（后缀是 .json），需追加 `*.index.json`，
> 否则会拉下 KB 级索引文件（无害但多余）。

## 3. 依赖版本

- **diffusers >= 0.40.0**（`MiniMaxH3ModularPipeline` 于 0.40 引入；dummy_run requirements 由 0.38.0 升到 0.40.0）
- transformers >= 4.56.0（Qwen3-VL）
- 远端验证隔离：`pip install --target /tmp/dif040_site --no-deps diffusers==0.40.0` + `PYTHONPATH=/tmp/dif040_site`，
  不污染已安装的 vllm-omni / mindiesd 环境（见 env-install 故障排查）

## 4. 组件清单

| 组件 | 类 | subfolder | 说明 |
|---|---|---|---|
| Transformer | `MiniMaxH3Transformer3DModel` | `transformer` | 50 层 → 2 层；`num_refiner_layers` 2 → 1；meta → to_empty bf16 |
| Text Encoder | `Qwen3VLForConditionalGeneration` | `text_encoder` | 64 层 → 2 层，vision `depth` 27 → 1 |
| Video VAE | `AutoencoderKLMiniMaxH3` | `vae` | 仅统计参数量，t2va 前向不调用 |
| Audio VAE | `AutoencoderKLMiniMaxH3Audio` | `audio_vae` | 同上（自定义 remote code：DAC/BigVGAN） |
| Scheduler | `MiniMaxH3Scheduler` | `scheduler` | shift=12.0 |
| Audio Scheduler | `MiniMaxH3Scheduler` | `audio_scheduler` | shift=3.0 |
| Tokenizer / Processor | `Qwen2TokenizerFast` / `Qwen3VLProcessor` | `tokenizer` / `processor` | 真实词表（KB 级） |

## 5. 关键适配点

1. **scheduler 步数**：`MiniMaxH3Scheduler.set_timesteps` 要求 `num_inference_steps >= 2`，
   2 步 = 1 次 transformer 前向。**不要沿用其他模型的 1 步**
2. **几何约束**：`num_frames` 必须为 `17n+5`（最小 124 = 17×7+5，时长 5–15s @ 24fps）；
   `height`/`width` 必须是 32 的倍数（`vae_spatial_compression_ratio 16 × patch_w 2`）
3. **单卡 O(seq²)**：全自注意力 packed 序列，768×1344×124 帧 QK^T ≈ 160GB（bf16）单卡不可行；
   默认 256×384 小画布（seq ≈ 4K），可 `--height/--width` 调整
4. **`text_encoder_layer`**：完整模型在 Qwen3-VL 第 50 层 hidden state 做条件，且该属性是
   **read-only property**（`get_qwen3vl_prompt_embeds` 校验层数 > 该值）；截断为 2 层后
   必须**子类覆盖**该 property 为 1
5. **decode 裁剪**：`MiniMaxH3VideoDecodeStep` 只接受 `output_type` ∈ {pil, np, pt}，无 latent 选项；
   自定义 remote code 的 VAE decode 在 NPU 未验证 → 子类 `MiniMaxH3Blocks` 去掉
   `MiniMaxH3DecodeStep`，t2va 去噪后直接返回 latents
   - **import 位置**：`MiniMaxH3DecodeStep` 在 `diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3`
     （不在 `decoders.py`，那里只有 `MiniMaxH3VideoDecodeStep` / `MiniMaxH3AudioDecodeStep` / `MiniMaxH3AfterDenoiseStep`）
6. **计时 hook**：文本编码块直接驱动 `text_encoder.model` **子模块**，顶层 forward hook 不触发；
   `_PhaseTimer` 需对该子模块单独挂 hook
7. **`expandable_segments`**：meta → to_empty 构造 VAE 时分配显示约 +9.7GB（参数仅 5.2GB），
   是分配粒度现象，非错误

## 6. BF16 计算精度（--compute-precision，默认 bf16）

- **机制（模型级，编译侧零隐式精度转换）**：MiniMax-H3 DiT **没有 fp32 强制岛**——所有投影/norm
  按 `get_parameter_dtype(...)` 对齐输入，`_apply_rotary_emb` 把 rope cos/sin cast 到 hidden dtype
  → **只需把权重 cast 到 bf16，整个 DiT block stack 即原生 bf16 计算**（无需 wan 的源码级
  `.float()` 改写）。`--compute-precision` 取值 `bf16`（默认）/ `fp32`；bf16 时对
  transformer/text_encoder/vae/audio_vae 执行 `.to(torch.bfloat16)` + eager 部分 `Tensor.float`
  patch 兜底。编译图验证（`_verify_compute_precision_graph`）确认无 fp32/int32 计算输入。
- **实测（eager，256×384×124，2 layers）**：切 bf16 后 **transformer 前向与总推理耗时均下降约一个
  数量级、峰值显存同步明显下降**（本组合观测；绝对数字归档于
  `{run_results_dir}/archive/minimax-h3-notes-numbers.md`）。
- **⚠️ compile 陷阱（H3 侧两条）**：① `torch.compile` 把 forward 包成 `(*args, **kwargs)`，而 H3 的
  denoise 块用 `inspect.signature(transformer.forward)` 过滤 `denoiser_input_fields` → 5 个行索引参数
  （token_tags/position_ids/video_indices/audio_indices/text_indices）被丢弃，报
  `missing 5 required positional arguments`；修法 = 用 `_CompiledDiT` wrapper（显式声明完整 forward 签名）
  再 `register_components(transformer=...)`。② 之后 pipeline 的 `patch_size` / `canvas_multiple` property
  访问 `transformer.config`，wrapper 必须暴露 `.config`，否则报
  `'MiniMaxH3DummyPipeline' object has no attribute 'canvas_multiple'`。

## 7. RMSNorm 融合（H3 侧坐标）

> **方法真源 → `../../pattern-dev/references/fusion-graph-forms-and-semantics.md`**（RMSNorm 的
> torch 2.11 前置分解时机与 before/after-freezing 窗口判据）与 `../../pattern-dev/references/mismatch-catalog.md`
> （手写链 vs `torch.rms_norm` 作 pattern 的 target 差异：`add_.Scalar` vs `add.Scalar`、输入 cast）。

- **实现**：`patterns/minimax_h3_rmsnorm_pattern.py`（`register_replacement`，bf16/fp32 × 3D/4D 四变体）。
  **无需修改 `mindie_sd_backend.py`**（曾临时改过，torch 2.11 实测可还原，还原后性能保持、单测 3/3 通过）。
- **计数口径（本组合实测）**：RmsNorm ×14 新增、InplaceCopy_Cast 61→26（-96%）、Pow -99% / Mean -95% /
  Rsqrt -75%；模型 RMSNorm 总数 **14**（2 layers×4 + token_refiner 4 + final 1 + norm_out 1）
  → **14/14 全部命中**（eager 的 23 个 Pow 中 9 个为非 RMSNorm 平方运算）。
- **AB 判据**：`enable_minimax_h3_rmsnorm=False` 回到基线耗时、`True` 为下降后耗时（方向确定，非噪声）；
  transformer 与 kernel 总耗时均明显下降（绝对数字归档于 `{run_results_dir}/archive/minimax-h3-notes-numbers.md`）。

## 8. 其余融合（H3 侧坐标）

> **方法真源 → `../../pattern-dev/references/fusion-graph-forms-and-semantics.md`**（RoPE `rotate_half`
> 部分旋转链形态、dtype 提升 R1 的识别与修复）+ `../../pattern-dev/SKILL.md` Phase 2/5（注册顺序防误匹配）。

- **RoPE（已实现）**：`patterns/minimax_h3_rope_pattern.py`（`register_replacement`，bf16/fp32 双变体）。
  H3 侧形态：匹配 rotate_half 部分旋转链（slice 96 / split / neg / cat / mul×2 / add，外圈 slice/cat 保留，
  `npu_rotary_mul` **只旋转 96 通道部分**）；**必须注册在 `wan_residual_gate` 之前**——wan 的
  residual+gate pattern 会误匹配 H3 的 rope 子图（`x_rot*cos+rotated*sin` 被当 `x+y*gate`，4D 走 fallback
  造成轻微负收益，AB 证实）。剩余 `RotaryV2_Slice` ×4（x_rot 切片物化）未消。
- **AdaLN 调制 / SwiGLU 的算子探针结论（H3 侧事实，2026-08 probe）**：
  - AdaLN 链 `x*(1+scale_idx)+shift_idx`：`ops.adaln/adaln_v2`（weight=None 纯调制）实测
    **CheckShape failed**（`aclnnAdaLayerNorm` 要求 weight/bias 非 None 或特定 shape），不可复用；
    现有 `muls_add` 仅标量 scale。
  - SwiGLU：`npu_swiglu` 存在（CANN 25.7）但语义是 **`gate*silu(hidden)`**，与 diffusers SwiGLU 的
    **`silu(gate)*hidden` 顺序相反**，不可直接替换；`npu_ffn(act="swiglu")` 权重方向要求 w1 的 k 维 = x 的 k 维
    （与 H3 图不符）。
- **`enable_wan_residual_gate`** 对 H3 图的 **3D 残差**子图也会匹配但 fallback（y 为 2D），存在轻微负收益；
  RoPE 注册顺序已消除 rope 部分，3D 残差部分保留。

## 9. 4 卡 Context Parallel + 通信掩盖（H3 侧并行配置事实）

> **方法真源 → `../../dit-parallel-opt/references/comm-masking-method.md` 与
> `../../dit-parallel-opt/SKILL.md` 的 CP·掩盖节**（CP 机制、掩盖注入点与 comm stream 姿势、正确性判据、
> 掩盖率上限 `1-1/n` 与 c/f、「改了并行但不报错也没生效」的判定、host-bound 剩余瓶颈与下一步方向）；
> 取数口径归 `../../perf-gate/`。本节只留 **H3 侧并行配置事实**，绝对耗时/加速比归档于
> `{run_results_dir}/archive/minimax-h3-notes-numbers.md`。

- **档位形态（本组合实测过的两种，换形态 = 重判）**：
  - **非 USP**：仅 seq 分片、attention **未 wire** `_parallel_config` → profile 只有 allGather、无 allToAll；
  - **USP4**：`ulysses=4`、FA 切头（all_to_all）参与掩盖。
- **world size / rank 布局**：`torchrun` **4 进程**；`device_start + local_rank` 映射到 **NPU 4-7**
  （单 UB 岛 4 卡组）；`dit_world = 4`。
- **seq 可整除性**：seq 不可被 4 整除时必须开 **`ulysses_anything=True`**（`PartitionAnythingSharder`；
  否则 `EquipartitionSharder` 断言 `size % mesh == 0` 失败）。
- **H3 侧的 wire 前置（易静默降级）**：`apply_context_parallel` 只挂分片/聚合 hook，**不设置 attention
  processor 的 `_parallel_config`**（类属性默认 None）→ 必须给每个 `attn.processor._parallel_config` 设
  `ParallelConfig(context_parallel_config=cp_cfg)`（是 `ParallelConfig` **包装**，不是
  `ContextParallelConfig`），才会触发 Ulysses 的 all_to_all FA 切头路径。
- **脚本（H3 侧坐标）**：`examples/dummy_run/minimax_h3_parallel.py`（runner）+
  `examples/dummy_run/masking.py`（掩盖注入，含 pad+等分 all_to_all）+
  `mindiesd/parallel/`（comm stream 基础设施，自 framework 仓库移植）。
- **方向（本组合观测）**：掩盖把 comm stream 上的未掩盖通信从**主导项**压到**几乎可忽略**；
  USP 下 wall 由 **rank 间不均衡转为均衡**。该 4 卡档仍为 **host-bound**（Free 与设备等待与墙钟同量级，
  即设备等 host）——下一步方向见方法真源。

## 10. 验证结果（远端 910B NPU，2 layers，diffusers 0.40.0 隔离安装，256×384×124）

> 参数量与 latents 形状属**结构契约**，保留；**绝对耗时 / 峰值显存 / 加速比已归档**
> （`{run_results_dir}/archive/minimax-h3-notes-numbers.md`），下表只留**档位关系与方向**。

```text
transformer params: 1.75 B | text_encoder: 2.73 B | vae: 2.60 B | audio_vae: 0.15 B | Total: 7.24 B

| 配置 | 相对其它档的关系 | 编译图验证 |
|---|---|---|
| eager fp32 | 基线（最慢档） | — |
| eager bf16（默认） | 前向与总推理较 fp32 大幅下降（约一个数量级），峰值显存明显下降 | — |
| compile bf16 | 与 eager bf16 同量级（差异在噪声内） | PASSED（无 fp32/int32 计算节点） |
| eager w8a8（FFN 融合默认开启） | 低于 eager bf16（量化单点为正） | — |
| compile w8a8（同） | **本表最优档**（量化 × compile 叠加仍为正，明显优于 eager） | PASSED |

Video latents: (1, 24, 37, 16, 24) | Audio latents: (2, 32, 207) | Verification: PASSED
```

> w8a8 行为 2026-09-06 于 A310-50（A5 → MXFP8）复测：FFN hidden 站点融合（`mindiesd::mm_swiglu_mxquant`）
> 已**默认开启**（无 MMX_FFN_FUSION 开关）；数值**位级一致**（同 seed latents mean_rel=0.0）。

- **质量变化度**（本表口径，同 seed latents 对拍）：`compile` 相对 `eager`（同精度档）**位级一致
  （mean_rel=0.0）** ⇒ compile 档不引入质量变化；`w8a8` 档 vs `bf16` 的变化度**未在本 dummy 口径下测**。

## 11. SwiGLU + AdaLN + gate 融合（H3 侧实现坐标与实测细节）

> **方法真源 → `../../pattern-dev/references/fusion-graph-forms-and-semantics.md` +
> `../../pattern-dev/SKILL.md` Phase 2/5**（三算子图形态与语义、融合边界（免 cat / 表 L2 驻留 / gather 行内核）、
> i32 索引与多行并行）；**收益归因 → `../../pattern-dev/references/benefit-rootcause-guide.md` §3 R5 与案例 4**。

### H3 侧实现坐标

- `mindiesd/layers/scale_shift.py`：triton 算子族——`gather_scale_shift`(AdaLN，表+索引)、
  `gather_residual_gate`(gate 融合)、`swiglu`(免 cat)、`scale_shift`(plain 兜底)；均 i32 索引 + 3 行/program
- `mindiesd/compilation/patterns/minimax_h3_swiglu_pattern.py`（bf16/fp32 双变体，`split_size=14336` 精确匹配）
- `mindiesd/compilation/patterns/minimax_h3_adaln_pattern.py`（匹配 index_select×2 + add/mul/add 链）
- `mindiesd/compilation/patterns/minimax_h3_gate_pattern.py`（匹配 index_select(gate) + mul + add；
  **注册在 `wan_residual_gate` 之前**防误匹配）
- 四段注册 + 开关 `enable_minimax_h3_swiglu/adaln/gate`（默认 True）；单测三份（均 1 passed）

### H3 真实图形态（verified by dump, before-freezing）

```text
SwiGLU: matmul_4 [1,1,28672] -> split.Tensor(matmul_4,14336,-1)
        -> getitem_10(hidden前半) / getitem_11(gate后半)
        -> silu(getitem_11) -> mul(getitem_10, silu)
AdaLN:  index_select(scale_table) -> add(·,1.0) -> mul(x,·)
        -> index_select(shift_table) -> add(mul,·)
```

### H3 侧实测事实（只留计数与形态，绝对数归归档）

- **算子验收值（H3 接口事实）**：SwiGLU 需把 chunk 顺序对调为 `[gate,hidden]`（`npu_swiglu` 语义是
  `first_half*silu(second_half)`），swapped-order err=1e-4(bf16)；AdaLN 三种 shape err=0.0039(bf16)；
  调制表仅 `[3,D]`=64KB（**L2 驻留**，故行内核能一次吸收 2 个 index_select）。
- **最终 kernel 构成（all on，315 kernels vs baseline 339）**：`gather_scale_shift_kernel` ×10、
  `gather_residual_gate_kernel` ×8、`swiglu_kernel` ×3；IndexSelect_GatherV2 **28→2**、Silu 仅剩 4 个小实例。
- **多时长验证（5s/10s/15s = 124/243/345 帧）**：5 个 pattern（rmsnorm/rope/adaln/swiglu/gate）
  all_on vs all_off 全部 PASSED + compute-precision PASSED；**收益比例在各时长稳定在同一量级
  （不随时长漂移）**，而绝对耗时随规模**超线性**增长（FA O(seq²)）。
- **帧数 snap 到 `17n+5`**：传 120/240/360 会被 diffusers 向上取整；**15s 不能传 360**
  （取整 362 超上限 360），**合法上限 345**。
- ⚠️ `patterns/__init__.py` 含他人工作区改动（QwenRope/DropoutZero 等），推送时勿覆盖。

## 12. qk_norm + RoPE 大融合：**负面结论归档（防重复实验）**

> **方法真源 → `../../pattern-dev/references/fusion-graph-forms-and-semantics.md` +**
> **`../../pattern-dev/SKILL.md` Phase 2/5**（融合边界判据、大段一次匹配的 pattern 写法、短行 triton 形态天花板）。

- **结论（一行，勿重复实验）**：qk_norm + RoPE 结构上**可行**（norm 输出 cast 后直接进
  `_apply_rotary_emb`、无 GEMM 间隔、单站点 norm 输出无第三消费者），但 **triton 版模型级净负**：
  v1 标量行循环模型级最慢、v2 2D tile + partner gather 仍慢（gather 被标量化）、v3 对齐半块重载
  在 BR=128 最快但 **fused 4 站点合计仍约为旧链的 1.7 倍量级** → **默认关**，2026-09 **整体移除代码**。
  Ascend C 侧复核后**收益上限也仅为个位数百分比**（读流量减半只换来个位数百分比改善 ⇒ 瓶颈是短行
  D=128 的**行内归约指令/延迟**，不是读带宽，Ascend C 的流量经验无法进一步传导）。
- **H3 侧站点事实**：站点 = transformer block×2 ×(q,k) = **4/step**；refiner 的 qk_norm 无 rope，**不纳入**。
  单站点现链 = RmsNorm + RotarySlice + RotaryV2 + ConcatD + cast/tensorMove（各几十微秒量级、合计亚毫秒量级）。
- **已移除清单（勿按清单复原）**：`mindiesd/layers/norm_rope.py`、`patterns/minimax_h3_norm_rope_pattern.py`、
  `FusionPatterns.enable_minimax_h3_norm_rope`、注册条目与对应单测。
- **该模型测出的三条留档判据**：① pattern 输出 cast 的 `_to_copy` **只写 dtype+layout（省略 device）**
  ——inductor matcher 对 device kwarg 宽容，带 device 会在 pattern trace 时 `_to_copy` 解包递归爆栈；
  ② 注册必须**最先**（`passes/__init__.py` 字典首项，先于 rmsnorm/rope 单点 pattern）；
  ③ 单测必须**隔离**（仅启用本 pattern），全开会让单点 pattern 抢跑成"假通过"。
- **收益池定位（本组合观测）**：GEMM 约四分之三 / FA 约一成半，norm+rope 区仅几个百分点。

## 13. w8a8(MXFP8) 编译图：**H3 侧图节点清单**

> **契约真源 → `../../quantization-dev/references/online-quant-contract.md`**（编码公式、舍入、scale 粒度、退化块、布局）；
> **选档语义 → `../../dit-perf-opt/references/quant-tier-device-mapping.md`**；
> **graph-entry 改图机制 → `../../pattern-dev/references/graph-pattern-rewrite-guide.md`**。本节只留 H3 侧图节点清单。

- **量化档位**：`--quant w8a8` 按设备选算法——A5 → **W8A8-MXFP8**、A2/A3 → W8A8-DYNAMIC(INT8)
  （`model/common/quantization.py`）；dummy 下 transformer 的 `nn.Linear` 全量化（**28/28、0 残留**）。
- **每个量化 Linear 展开为**：`npu_dynamic_mx_quant`（激活按 k 分块 32 出 fp8e4m3 + e8m0 scale）
  - `npu_quant_matmul`（V5 / QuantBatchMatmulV3）出 bf16。
- **FFN 形态（diffusers 0.40 `SwiGLU`）**：单个 `Linear(D→2F)` → `chunk` → `hidden*silu(gate)`；
  真图 `Qmm([S,2F]) → view[1,S,2F] → split → silu → mul → view[S,-1] → DxQ → Qmm`，view 尺寸带**动态 S**
  （同图 S=1/3967）⇒ trace 式 pattern 无法命中。
- **FFN 融合算子（H3 侧接线事实）**：`mindiesd::mm_swiglu_mxquant`（`mindiesd/layers/mm_swiglu_mxquant.py`，
  catlass 65 语义、单 problem、model-order W、kernel 内缓存列序调换）；**不接入 dummy eager 代码**，
  compile 侧默认开启（`enable_minimax_h3_ffn_fusion=True`），真图命中走 **graph-entry 手写 CallFunction 树
  - handler 手动改图**（参考 `torch/_inductor/fx_passes/mkldnn_fusion.py` 的 `_recover_linear`）：
  全 Arg 叶子、view 尺寸 `Ignored`、共享子节点 `_users=MULTIPLE`，由 `register_ffn_fusion_graph_entries()`
  注册进 pattern_pass；**fusion on 时同步禁 triton swiglu pattern**（fusion 拥有 FFN 站点）。
  实测 compile 图 3 个 fused op、eager/compile 同 seed latents **位级一致（mean_rel=0.0）**。
- **适用边界（为何 wan/flux 无 mm_swiglu_mxquant）**：该算子只匹配 **H3 特有 FFN hidden 形态**
  （单个 `Linear(D→2F)` 输出直接 `chunk→hidden*silu(gate)` + 输出 MX 量化直供 out-proj）。Wan2.2 与
  FLUX 的 FFN 均为 **GELU 单分支 MLP**（无 `[hidden|gate]` 两半、无 SwiGLU、无 out==2F 配对）→
  结构判定（hidden out==2F + out in==F 容器内唯一）天然不命中、融合为 **0 站点**，属**预期而非缺陷**；
  它们的 compile 收益走 rms/rope/gate/adaln pattern + 小 kernel 消减。若未来要对 GELU-FFN 模型做同类
  融合，需另起 GELU 语义 kernel（本项目无此诉求）。
- **跨栈数据格式一致性（实测于 m=128/k=512 档，其它 shape 未验证）**：`npu_dynamic_mx_quant` 的
  scale 布局（A: `[m,ceil(k/32)/2,2]`；W 按 k 量化: `[n,ceil(k/32)/2,2]`，e8m0 字节）与 catlass
  例程（53/65）输入布局**字节一致**，可直接喂入；解码对拍相对误差 ~2%（e4m3 量化级）。
  **语义核验建议**：以 fp32 真实输出例程（catlass 53）或量化输出解码对拍作基准，**勿只依赖量化输出的自比对**。

## 维护与更新

当MiniMax-H3 dummy run 适配点变化时，按 dev-workflow 的复盘流程更新本文件。
