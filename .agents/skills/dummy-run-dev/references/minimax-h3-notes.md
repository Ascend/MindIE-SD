# MiniMax-H3 Dummy Run 适配记录

> **目录** · [1. 仓库双格式](#1-仓库双格式最易踩坑) · [2. 配置获取（gated → modelscope）](#2-配置获取gated--modelscope) · [3. 依赖版本](#3-依赖版本) · [4. 组件清单](#4-组件清单) · [5. 关键适配点](#5-关键适配点) · [6. BF16 计算精度](#6-bf16-计算精度--compute-precision默认-bf16) · [7. RMSNorm 融合](#7-rmsnorm-融合pattern-matcher-机制2026-08-实测) · [8. 其余融合](#8-其余融合2026-08-实测) · [9. CP + 通信掩盖](#9-4-卡-context-parallel--通信掩盖2026-08-实测) · [10. 验证结果](#10-验证结果远端-910b-npu2-layersdiffusers-0400-隔离安装256384124) · [11. SwiGLU + AdaLN + gate 融合](#11-swiglu--adaln--gate-融合2026-08-实现全模型验证完成总收益--186ms)
>
> MiniMax-H3（33B 全模态生成模型，T2VA / FL2VA / Ref2VA 工作流）在 `examples/dummy_run/` 的
> dummy run 适配要点。依据：`examples/dummy_run/minimax_h3_infer.py` 与
> `examples/dummy_run/model/minimax_h3_model.py`（2026-08 实测通过）。

## 1. 仓库双格式（最易踩坑）

`MiniMaxAI/MiniMax-H3`（modelscope 镜像 `MiniMax/MiniMax-H3` 同构）**同一仓库混两种布局**：

| 位置 | 格式 | 特征 | 用途 |
|---|---|---|---|
| 仓库**根目录** | diffusers 格式 | `model_index.json` / `modular_model_index.json`，`_class_name: MiniMaxH3ModularPipeline`；`transformer/`、`vae/`、`audio_vae/`、`text_encoder/`、`tokenizer/`、`processor/`、`scheduler/`、`audio_scheduler/` | **dummy run 只能用它** |
| `FL2VA/`、`Ref2VA/` 子目录 | vLLM-Omni 格式 | `model_index.json` 的类名（`MiniMaxH3DiTModel`、`MiniMaxH3Qwen3VLHFEncoder` 等）与 transformer 配置键（`ffn_hidden_size`、`latents_dim` 等）在 diffusers 0.40 中**均不存在** | vLLM-Omni 部署 |

- vLLM-Omni 部署目录（如 `<model_weight_dir>/MiniMax-H3/FL2VA`）**不能**当 `--config_cache`
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
  不污染已部署的 vllm-omni / mindiesd 环境（见 ascend-deploy §2 Step 8）

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
- **实测（eager，256×384×124，2 layers）**：transformer **319.8ms(fp32) → 30.9ms(bf16) ≈ 10.4×**；
  总推理 338.3ms → 50.7ms（≈6.7×）；峰值显存 21.90GB → **13.50GB**。
- **⚠️ compile 陷阱 1（forward 签名）**：`torch.compile` 把 forward 包装为 `(*args, **kwargs)`，
  而 MiniMax-H3 的 denoise 块用 `inspect.signature(transformer.forward)` 过滤
  `denoiser_input_fields` → 5 个行索引参数（token_tags/position_ids/video_indices/audio_indices/
  text_indices）被丢弃，forward 报 `missing 5 required positional arguments`。修复：用
  `_CompiledDiT` wrapper（显式声明完整 forward 签名，内部转发 compiled 模块）再
  `register_components(transformer=...)`。
- **⚠️ compile 陷阱 2（config 属性）**：`register_components` 后 pipeline 的 `patch_size` /
  `canvas_multiple` property 访问 `transformer.config`，wrapper 必须暴露 `.config`；
  否则 property 内部 AttributeError 被 Python 视为属性缺失，最终报
  `'MiniMaxH3DummyPipeline' object has no attribute 'canvas_multiple'`。

## 7. RMSNorm 融合（pattern matcher 机制，2026-08 实测）

- **方案**：`patterns/minimax_h3_rmsnorm_pattern.py`（register_replacement，bf16/fp32 × 3D/4D 四变体）。
  **无需修改 `mindie_sd_backend.py`**（曾临时改过，torch 2.11 实测可还原——见下）。
- **分解时机（torch 2.11 实测，910B）**：Dynamo/aot_autograd **在 freeze 前**就把
  torch.rms_norm 分解成链（before-freezing 图直接是 `_to_copy(f32)→pow→mean→add.Scalar→
  rsqrt→mul→mul`）→ before_freezing 的 pattern matcher 一次运行即命中。
  **旧结论"必须 after_freezing 二次运行"基于 torch 2.9**（当时 aot 保留单节点、freeze 才分解）；
  torch 2.11 已前置分解，该改动还原（`git checkout mindie_sd_backend.py`）后性能保持
  25.87ms（vs 修改时 25.76ms，噪声级），单元测试 3/3 通过。
- **为什么手写链而非 `torch.rms_norm` 作 pattern**：make_fx 对 torch.rms_norm 的自动分解产生
  `add_.Scalar`（inplace，composite op + python dispatcher 展开），而真实图产生 `add.Scalar`
  （非 inplace）——target 不同 0 命中。手写链精确固定每个 target（add.Scalar、mean dim
  [x.dim()-1]、_to_copy(f32) 输入 cast、不含输出 cast 以避免 `_to_copy(bf16, layout, device)`
  的 kwargs 差异）。
  **注意**：make_fx 空 decomp 表下 torch.rms_norm 仍展开成链（composite 机制，与 decomp 表
  无关）；`pre_dispatch=True` 的 make_fx 可保留 `aten.rms_norm` 单节点（备选方案，若未来
  torch 恢复 freeze 后分解可改用单节点 pattern + pre_dispatch trace fn）。
- **实测（compile bf16 vs eager bf16）**：transformer **30.9ms → 26.65ms（-4.4ms）**；
  kernel 总耗时 32.4 → 28.0ms（-13.6%）；RmsNorm ×14 新增（0.45ms）、InplaceCopy_Cast
  61→26（-96%）、Pow -99% / Mean -95% / Rsqrt -75%。模型 RMSNorm 总数 14（2 layers×4 +
  token_refiner 4 + final 1 + norm_out 1）→ **14/14 全部命中**（eager 的 23 个 Pow 中 9 个为
  非 RMSNorm 平方运算）。AB 验证：enable_minimax_h3_rmsnorm=False → 30.93ms vs True → 26.65ms。

## 8. 其余融合（2026-08 实测）

- **RoPE 融合（已实现）**：`patterns/minimax_h3_rope_pattern.py`（register_replacement，
  bf16/fp32 双变体）。匹配 rotate_half 部分旋转链（slice 96/split/neg/cat/mul×2/add，
  外圈 slice/cat 保留，npu_rotary_mul 只旋转 96 通道部分）。**注册在 wan_residual_gate 之前**
  ——wan 的 residual+gate pattern 会误匹配 MiniMax 的 rope 子图（`x_rot*cos+rotated*sin` 被当
  `x+y*gate`，4D 走 fallback 造成 ~0.26ms 负收益，AB 证实）。
  **dtype 提升 bug（2026-08-22 修复）**：replacement 收到 pattern 匹配的 fp32 cos/sin
  （pattern 内 `_to_copy(bf16)` 节点被消费），rope.py `x.to(cos.dtype)` 把 bf16 x 提升到
  fp32 再 `type_as(x)` 降回 → 每处 rope 两个大张量 Cast（50+67us，4 处 ≈0.47ms 纯浪费），
  导致 RoPE 收益 ≈0。修复：replacement 显式 `_to_copy(cos, dtype=x.dtype)` → Cast 4+4us，
  RotaryV2 本体 113→35.5us，**RoPE 收益 ≈0 → -0.68ms**（both 25.76ms）。
  剩余：`RotaryV2_Slice` 81us×4 ≈0.33ms（x_rot 切片物化）——优化方向见
  `refs/minimax_profiles/bf16_compile_ab_report.md` §7（npu_rotary_mul slice 变体
  支持 full-head 输入的可行性调研）。
- **AdaLN 调制 / SwiGLU（未实现，2026-08 probe 结论）**：
  - AdaLN 调制链 `x*(1+scale_idx)+shift_idx`：`ops.adaln/adaln_v2`(weight=None 纯调制)实测
    **CheckShape failed**(aclnnAdaLayerNorm 要求 weight/bias 非 None 或特定 shape),不可复用;
    现有 `muls_add` 仅标量 scale。需 tensor-scale 融合算子(收益 ~0.3-0.5ms)。
  - SwiGLU：`npu_swiglu` 存在(CANN 25.7)但语义是 **`gate*silu(hidden)`**,与 diffusers
    SwiGLU 的 `silu(gate)*hidden` **gate/hidden 顺序相反**,不可直接替换;
    `npu_ffn(act="swiglu")` 权重方向要求 w1 的 k 维 = x 的 k 维(与图不符)。
  - 两者均无现成算子,列为后续项。
- **注意**：`enable_wan_residual_gate` 对 MiniMax 图的 3D 残差子图也会匹配但 fallback
  （y 为 2D），存在轻微负收益；可通过收紧 pattern 锚定或注册顺序消除（当前 RoPE 已消除
  rope 部分，3D 残差部分保留）。

## 9. 4 卡 Context Parallel + 通信掩盖（2026-08 实测）

### 方案

- **CP 机制**：diffusers 0.40 自带 `_cp_plan`(Ulysses-anything,seq 分片)+
  `apply_context_parallel`(hooks)。runner 用 torchrun 4 进程,`device_start + local_rank`
  映射到 NPU 4-7。seq 不可被 4 整除时启用 `ulysses_anything=True`(PartitionAnythingSharder,
  否则 EquipartitionSharder 断言 size%mesh==0 失败)。
- **⚠️ 关键：attention 需手动 wire `_parallel_config`**：`apply_context_parallel` 只挂
  分片/聚合 hook,**不设置 attention processor 的 `_parallel_config`**(类属性默认 None) →
  `dispatch_attention_fn(parallel_config=None)` 走非 CP 路径,只有 seq 分片+输出 gather
  (profile 只有 allGather 无 allToAll)。必须给每个 `attn.processor._parallel_config`
  设 `ParallelConfig(context_parallel_config=cp_cfg)`(注意是 ParallelConfig 包装,不是
  ContextParallelConfig),才会触发 Ulysses 的 all_to_all FA 切头路径。
- **掩盖**：`mindiesd/parallel/`（自 framework 仓库移植）的 HCCL ctypes + 独立 comm stream
  (compute 记录 ready 事件 → comm stream 等 → HCCL 集合 → 记录 done → compute 等) 实现
  通信与计算重叠。monkey-patch `funcol.all_to_all_single` / `funcol.all_gather_tensor`
  为 masked 版(见 `examples/dummy_run/masking.py`),零 diffusers 源码改动。
- **正确性**：masked all_to_all(等分)/all_gather 与 torch.distributed **逐字节一致(err=0.0)**。

### 效果(4 卡, 256×384×124, 2 layers, bf16, 910B NPU 4-7)

#### A. 非 USP(仅 seq 分片,attention 未 wire)

| 指标 | unmasked CP | masked CP | 改善 |
|---|---|---|---|
| transformer wall | 27.7ms | 28.1ms | ~持平(host-bound) |
| kernel 总耗时 | 37.5ms | **12.1ms** | **-67.7%** |
| Communication(未掩盖) | **31.4ms** | **1.5ms** | **-95.2%** |
| Stage(profiler 设备时间线) | **71.5ms** | **35.5ms** | **-50.3%** |

#### B. USP4(ulysses=4,FA 切头,wire 后)

| 指标 | USP unmasked | USP full-mask (ag+a2a) | 改善 |
|---|---|---|---|
| transformer wall | 42.9/83.5ms(rank 不均) | **36.8ms(均衡)** | 显著 |
| kernel 总耗时 | 22.19ms | **19.45ms** | -12.3% |
| Communication(未掩盖) | 8.34ms | **5.43ms** | **-35%** |
| Free(设备空闲) | 24.3ms | 28.7ms | ~持平(host-bound) |

### 结论与剩余瓶颈

1. **通信掩盖有效**：非 USP 的 KV all-gather 从 25.7ms → 0.66ms(comm stream);
   USP 下 all_gather + all_to_all 掩盖使 wall 42.9→36.8ms(均衡)、通信 -35%。
2. **FA 切头(all_to_all)掩盖完成(2026-08)**:`HcclAlltoAllV`(split 路径)在本 CANN 9.1.0
   环境 **SIGSEGV**(等分 HcclAlltoAll 正常)。**解决方案(pad+等分)**:Ulysses-anything 的
   all_to_all 里 input 按 in_sizes 等分块、output 按 out_sizes 组装(s_local 各 rank 可
   不同);把每个 input 块 pad 到 S_PAD(128 倍数,由全局 max(out_sizes) 推导,全 rank 一致),
   用等分 HcclAlltoAll(count=S_PAD×row_elems)交换,再 slice 各块前 out_sizes[j] 行。
   实测 err=0.0(最小复现),kernel 名从 hcom_alltoallv 变为 hcom_alltoall(等分)。
3. **剩余瓶颈是 host-bound**：Free 24-29ms(设备等 host)+ DAVID_EVENT_WAIT(跨流事件同步)
   595 kernels 的 Python enqueue + Ulysses-anything 每次 attention 的
   `gather_size_by_comm` 冗余 broadcast ×8。
4. **下一步方向**：① host 预取/软件流水(提前 enqueue 下一层通信,与当前层计算重叠);
   ② 缓存 `gather_size_by_comm`(静态 seq);③ 减少事件同步(批量发起 → 一次等待);
   ④ compile+CP 兼容性。
5. **脚本**：`examples/dummy_run/minimax_h3_parallel.py`(runner) +
  `examples/dummy_run/masking.py`(掩盖注入,含 pad+等分 all_to_all) +
  `mindiesd/parallel/`(comm stream 基础设施,从 framework 仓库移植;AlltoAllV 缺陷已绕过)。

## 10. 验证结果（远端 910B NPU，2 layers，diffusers 0.40.0 隔离安装，256×384×124）

```text
transformer params: 1.75 B | text_encoder: 2.73 B | vae: 2.60 B | audio_vae: 0.15 B | Total: 7.24 B

| 配置 | transformer (timed) | 总推理 | 峰值显存 | 编译图验证 |
|---|---|---|---|---|
| eager fp32 | 319.8 ms | 338.3 ms | 21.90 GB | — |
| eager bf16（默认） | 30.9 ms | 50.7 ms | 13.50 GB | — |
| compile bf16 | 31.0 ms | 51.5 ms | 13.50 GB | PASSED（无 fp32/int32 计算节点） |

Video latents: (1, 24, 37, 16, 24) | Audio latents: (2, 32, 207) | Verification: PASSED
```

## 11. SwiGLU + AdaLN + gate 融合（2026-08 实现,全模型验证完成,总收益 -1.86ms）

### 目标与算子验证

- **SwiGLU**：MiniMax FFN 的 `split->silu(gate)->mul(hidden,silu)` 链 →
  `npu_swiglu`(需把 chunk 顺序对调为 [gate,hidden],因为 npu_swiglu 语义是
  `first_half*silu(second_half)`)；**swapped-order err=1e-4(bf16)** ✓
- **AdaLN**：`x*(1+scale)+shift`(scale/shift 是 index_select 表行 [S,D]) →
  新增 triton 算子 `mindiesd/layers/scale_shift.py`
  (`mindiesd::gather_scale_shift`, 吸收 2 个 index_select; 表 [3,D] L2 驻留),
  **三种 shape err=0.0039(bf16)** ✓
- 演进: 初版 plain `scale_shift`(1D flatten) 负收益 +0.64ms → BS8192 +0.26ms
  → **gather 融合(行内核)转正 -0.24ms**, 详见下方分析

### 实现文件

- `mindiesd/layers/scale_shift.py`：triton 算子族——`gather_scale_shift`(AdaLN,
  表+索引)、`gather_residual_gate`(gate 融合)、`swiglu`(免 cat)、`scale_shift`
  (plain 兜底)；均 i32 索引 + 3 行/program
- `mindiesd/compilation/patterns/minimax_h3_swiglu_pattern.py`(register_replacement,
  bf16/fp32 双变体,split_size=14336 精确匹配; replacement=triton swiglu 免 cat)
- `mindiesd/compilation/patterns/minimax_h3_adaln_pattern.py`(register_replacement,
  匹配 index_select×2 + add/mul/add 链)
- `mindiesd/compilation/patterns/minimax_h3_gate_pattern.py`(register_replacement,
  匹配 index_select(gate) + mul + add; 注册在 wan_residual_gate 之前)
- 四段注册 + `enable_minimax_h3_swiglu/adaln/gate` 开关
- 单测:`test_minimax_h3_swiglu_pattern.py` / `test_minimax_h3_adaln_pattern.py` /
  `test_minimax_h3_gate_pattern.py`(**均 1 passed**)

### 真实图形态(verified by dump, before-freezing)

```text
SwiGLU: matmul_4 [1,1,28672] -> split.Tensor(matmul_4,14336,-1)
        -> getitem_10(hidden前半) / getitem_11(gate后半)
        -> silu(getitem_11) -> mul(getitem_10, silu)
AdaLN:  index_select(scale_table) -> add(·,1.0) -> mul(x,·)
        -> index_select(shift_table) -> add(mul,·)
```

### ✅ 全模型验证 + AdaLN 负收益根因与转正（2026-08-23 完成）

环境解封后（0-3 卡可用）完成逐 pass AB 与 kernel diff（compile bf16, device 0,
transformer timed）：

| 配置 | transformer ms |
| --- | --- |
| baseline（both off） | 25.83~25.96 |
| SwiGLU on（cat+npu_swiglu 旧方案） | 25.42~25.48（-0.35ms）✓ |
| AdaLN on（plain BS1024 初版） | 26.47（+0.64ms）✗ |
| AdaLN on（plain BS8192 调优） | 26.22（+0.26ms）✗ |
| AdaLN on（gather 1 行） | 25.72（-0.24ms）✓ |
| AdaLN on（gather 3 行 + i32） | 25.51（-0.45ms）✓ |
| both on（gather + npu_swiglu） | 24.96（-1.00ms）✓ |
| **both on（+ triton swiglu 免 cat + gate 融合）** | **24.10（-1.86ms）✓** |

**最终 kernel 构成（all on, 315 kernels vs baseline 339）**:

- `gather_scale_shift_kernel` ×10（AdaLN, 326us）
- `gather_residual_gate_kernel` ×8（**gate 融合**, 349us, 每 site 43.6us vs 原链 ~124us）
- `swiglu_kernel` ×3（**triton swiglu 免 cat**, 518us vs cat+npu_swiglu 1068us）
- IndexSelect_GatherV2 28→2；Silu 仅剩 4 个小实例

**AdaLN 负收益根因与转正（kernel diff 实证，详见 compilation-dev skill
`benefit-rootcause.md` R5 + 案例 4）**:

1. 初版 plain scale_shift 负收益根因：triton 1D kernel（BS1024, 230us/site,
   0.74TB/s）比 3 个 aclnn 逐元素 kernel（117us/site）慢 ~2x；且 2 个
   index_select gather（33us/site）在 pattern 外，[S,D] scale/shift 物化+重读
   170MB 冗余流量；
2. **转正关键①（gather 融合）**：调制表只有 [3,D]=64KB（3 模态，L2 驻留）→
   pattern 扩展匹配 2 个 index_select 节点，triton 行内核（grid=(S,)）单 kernel
   吸收全部 → 150us/site → 94.7us/site，-0.24ms；
3. **转正关键②（cannbot-skills 指导）**：i64→i32 索引（Avoid 标量降级）+
   每 program 3 行（`tl.static_range(3)`, 3×5376<UB 16384, 尾部 ROWS=1 微
   kernel）→ 94.7→66us/site，-0.24→-0.45ms；
4. **SwiGLU 免 cat**：原 cat([gate,hidden])+npu_swiglu 的 cat 拼 [1,S,2F]
   大张量 ~190us/site；triton 行 kernel 直接读 proj（hidden*silu(gate)）
   → 单 kernel 172.5us/site（bench 276 vs 550us），省 ~0.5ms；
5. **gate 融合**：`hidden + gate_table[idx]*attn/ff` 每 block 3 处，gate 表
   [3,D] L2 驻留 → triton `gather_residual_gate`（i32+3行, 43.6us/site vs
   原链 ~124us），省 ~0.4ms；注册在 wan_residual_gate 之前防误匹配；
6. 结论修正：**"triton 打不过 aclnn" 是伪结论**，真实瓶颈是 kernel 形态
   （流量冗余 + 融合边界 + i64 降级 + 并行度）；外部技能库
   `cannbot-skills/ops/triton-latency-optimizer`（discrete_memory_access /
   avoid_scalar_lowering / vector_core_partition）直接指导了 i32 与多行并行；
7. `enable_minimax_h3_adaln/swiglu/gate` 默认 True，总收益 -1.86ms
   （25.96→24.10ms, -7.2%）；warmup 仍有 triton JIT（~1-2s）。

### 多时长视频验证（5s/10s/15s，2026-08-23）

5 个 pattern（rmsnorm/rope/adaln/swiglu/gate）在 3 个时长全验证通过
（device 0，compile bf16，256×384，all_on vs all_off）：

| 时长(帧数) | all_on | all_off | 收益 |
| --- | --- | --- | --- |
| 5s (124=17×7+5) | 24.18ms | 30.95ms | **-6.77ms（-21.9%）** |
| 10s (243=17×14+5) | 51.32ms | 67.94ms | **-16.62ms（-24.5%）** |
| 15s (345=17×20+5) | 77.83ms | 101.59ms | **-23.76ms（-23.4%）** |

- 全部 PASSED + compute-precision PASSED（无 fp32/tf32/int32 计算节点）
- 帧数 snap 到 17n+5：传 120/240/360，diffusers 向上取整；**15s 不能传 360**
  （取整 362 超上限 360），合法上限 345
- 收益百分比稳定（~22-24%），时长增长超线性（FA O(seq²)，24→51→78ms）

- 注意:`patterns/__init__.py` 含他人工作区改动(QwenRope/DropoutZero 等),
  推送时勿覆盖;远端部署需确认不引入他人 pattern 冲突

## 维护与更新

当MiniMax-H3 dummy run 适配点变化时，按 dev-workflow 的复盘流程更新本文件。
