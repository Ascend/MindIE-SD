# 框架 × MindIE-SD 特性支持矩阵

> 定位：MindIE-SD 特性**命名与能力**的唯一真相源是 `docs/zh/features/*`
> （自动同步版：`performance-optimization/references/mindiesd-features.md`）。
> 本表 = 各三方推理框架（vLLM-Omni / DiffSynth-Engine / LightX2V）对上述特性的**支持面**——
> 支持状态 + 框架侧使能姿势 + 已知坑，供：使能前查支持面、使能失败排查、S4 组合前裁剪候选、
> 框架版本升级后感知特性增长。
>
> 口径：状态仅对「框架版本 + 模型」成立（案例结论单点不迁移）；`❓` 表示未核验，禁止当作支持。
>
> 状态图例：`✅` 案例实证支持 · `🟡` 可配置但有已知限制/负面 · `❌` 不支持或阻塞 · `❓` 未核验。
> 证据码：V1 = `references/vllm-omni-minimax-h3-case.md`（950PR / MiniMax-H3-FL2VA / vLLM-Omni 0.28）、
> V2 = `references/diffsynth-engine-case.md` / `diffsynth-engine-notes.md`（Qwen-Image / DiffSynth-Engine）、
> V3 = `references/vllm-omni-qwen-image-case.md`（950PR / **Qwen-Image-2512 图像 1024² 20 步** / vLLM-Omni 0.28）、
> L1 = `references/lightx2v-mindiesd-case.md`（MiniMax-H3 / LightX2V #1471）、G = 本技能正文（通用回路）。
> L1 列 last-checked：2026-09-05（对照 case §10 新增锚点 + 远端源码复核修订：量化阻塞表述、
> TP/RSP/offload/缓存/时间步格与 §二 姿势段；修订明细见 case §10「并行形态/显存/量化阻塞表述」三条）。
> vLLM-Omni 列 last-checked：2026-09-06（0.28 disable-vllm-ascend；V1=H3-FL2VA 576p 视频 950PR×4；
> **V3=Qwen-Image-2512 1024²/20 步 950PR×1-2（2026-09-05/06）：量化✅/Cache✅/2 卡 TP1×USP2 采纳（无损最优
> 4.35s）、TP2=S4 有损基底、rf_v2 图像不可用（0.4/0.8 档位无关）、4-rank（TP4/USP2）短任务通信病态回退、
> compile 输出非无损**；图像质量域（同 seed 像素对比）显著高于
> 视频，有损阈值不可跨域迁移）。
> env A（2026-09-07 闭环，950PR×2 TP2，**显式 TORCH_SDPA 基线 436.7s@60**，editable mindiesd 双树并存）：
> FA -17.4%、稀疏 rf_v2 0.8 **-38%**、Cache 单点 **-77%（83.4s）**、组合 mxfp8+FFN-MX+Cache **4.70×（92.9s）**；
> 数值口径与双树选树规则见 §二 env A 行。

## 〇、framework × 特性/能力 支持档位（图例与记录规则）

状态图例（✅/🟡/❌/❓）表达"验证到什么程度"；本图例表达"**该能力在 framework×模型组合下处于
什么实现就绪状态**"，供特性覆盖清单排序与执行序（先做什么、哪些要投入开发）使用：

| 档位 | 定义 | 判例（该能力的判定粒度） | 流程含义 |
|------|------|--------------------------|----------|
| **已支持** | 能力在该框架组合下可直接使用（默认或开关/配置即生效） | 如 LightX2V `npu-w8a8-mxfp8` 原生 scheme、vLLM-Omni cache_dit/量化开关 | 零开发：直接做（覆盖清单判「做」） |
| **待配置** | 特性面存在但**部分能力不全/未接线**，需配置/接线补齐特定能力 | 如"量化有支持但不支持 mxfp8 型 w8a8"——能力实体（kernel/loader/机制）已具备，缺框架接线/开关/量化描述符；mindiesd `quantize` 本机已具备而框架未接线（case §11 P0/P1；Cache 按新规不做 mindiesd 额外适配） | framework-feature-enablement 接线（记 P 级计划）；成本=配置接线 |
| **待开发** | 特性在该框架组合下**没有支持，需完整实现** | 能力实体（kernel/消费者/机制）不存在：需 operator-dev 新算子 / framework-extension-dev 结构性实现 / 框架侧新机制 | 先估成本 + 用户确认（SKILL §0 补齐策略②③）再投入；默认不静默做 |
| **上界（预留）** | 理想档/天花板探针定义——**空间预留，待人工填写** | 如"某档假设全收益/无副作用时的收益上限"（后续按需求补判例与口径） | 只诊断不宣称；不进入 delivery（同 purpose 的 unsafe_probe 纪律） |

判定粒度（能力级，不按特性大面一刀切）：

- **量化**：按能力组合判——`w8a8`（类型 mxFP8 e4m3）/ `w8a8`（int8 dynamic）/ `w4a4`（类型 mxFP4）/
  `f8`（FA 侧，路径 FP8RotateQuantFA 或框架案例实现）；组合（w8a8f8）逐能力判；与报表修饰符
  白名单同源（overview-report §2.1）。
- **稀疏**：**与 mindiesd 稀疏算子对齐**逐算子判——rf_v2 / ada_bsa /（平台扩展 rf_v3、video_spans）
  等（mindiesd-features.md 为算子面真源），逐算子 × 框架给出档位。
- **cache**：DiTCache / AttentionCache 分判；**编译/融合**：按 pattern 族（RMSNorm/RoPE/…）与
  后端（compile/GraphPatternEntry/API 接入）分判；**并行**：USP/TP/RSP/CFG/CP 分判。
- 记录方式：刷新单元格时建议首词标档位（如 `待配置 L1（…）`、`已支持 V1（…）`），与既有
  状态图例 + 证据码并置；档位归属有争议时以「能力实体是否存在（kernel/loader/消费者）+ 是否
  仅差接线」为准（存在但差接线=待配置；不存在=待开发）。
- 档位进入流程：run-state「特性覆盖清单」列记录 + 排序（已支持/待配置 先做，待开发先估成本并
  §0 确认）；上界档预留见上，定义由人工后续填写。

## 一、支持矩阵（特性行 = docs 规范名）

### 量化（docs: `quantization.md`；接口 `quantize(model, quant_desc_*.json)`，描述符由 msmodelslim 预导出）

> 报表枚举映射（`overview-report.md` §2.1）：docs 档位名 → 报表特性名 = 量化(修饰符)，
> 如 线性层 `W8A8_MXFP8` → `量化(w8a8)`（实现即 MXFP8 e4m3）、FA 8bit（FP8）→ `量化(f8)`
> （统称；docs 侧实现名 = `FP8RotateQuantFA`，见 quantization.md；EagleQBSA（Q/K INT8 + V
> per-channel FP8）为框架案例（vLLM mix）实现——报表说明列必须写明实际路径）；本表行名保留
> docs 名。

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| W8A8_DYNAMIC（INT8 online） | ✅ V1（60 步 96.4s = **-18.3% vs lossless 118.0s**（1.22×）；vs 344.9s SDPA ≈3.6×；576p 档）；✅ V3（Qwen-Image 1024²/20 步 **-12.7%**（4.33s vs TP2 4.96s）；质量 37.5 PSNR/0.986 SSIM） | ❓ | ❌ L1（框架未接线 mindiesd quantize；仅 `dit_quantized` 预量化 ckpt 路 + triton int8 GEMM 本机编译失败——非「本 NPU 无 kernel」，见 case §10/§11 P1） |
| W8A8_MXFP8 | 🟡 V1（2026-09-06 修复闭环：mindiesd dev `mm_swiglu_mxquant`（catlass 集成）与 vllm-omni 0.28 **mxfp8 档接线打通**——compile 侧 FFN-MX 融合对 vLLM-Omni H3 真图**命中 52/52**（vLLM 变体 GraphPatternEntry + C++ 布局自适应 vLLM GEMM-ready (K,N)/scale (c,N,2)）；60 步 TP2 e2e **-5.8%**（Bon 299.1s vs Boff 317.5s）；质量门=量化级容差（单层 rel 0.36%、输出差异 ≤ mxfp8 量化分散度 bf16-vs-mxfp8 ~19dB 同量级）；仅适用 SwiGLU/MoE（H3），Qwen-Image 无 SwiGLU 不适用；int8 档仍不适用（i8 域 ≠ MX e4m3，须 `method:mxfp8`）。见 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/fix_validation.md） | ❓ | 🟡 L1（**原生 scheme `npu-w8a8-mxfp8` 已实现**（内部 mindiesd MXFP8/npu_quant_matmul）实测 -12%/SSIM 0.97，见 case §11；非 docs `quantize()` 接口，DYNAMIC 未接） |
| W8A16 / W4A16 / W4A4_MXFP4_* 等其余档 | ❓ | ❓ | ❓ |
| FA 量化（FP8，`FP8RotateQuantFA`） | ❓ | ❓ | ❌ L1（框架未接线；kernel 侧 mindiesd FP8/MXFP8 FA（v2 op）**A5 微测可用** 2026-09：cos≈0.998、rel≈5%，见 case §11） |

### 稀疏（docs: `sparse.md`；接口 `sparse_attention(q,k,v, sparse_type=…)`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| rf_v2（RainFusion2.0） | ✅ V1（视频档 sparsity=0.8 实测；另 EagleQBSA mix）；env A 2026-09-07（950PR×2 TP2）：0.8 档 **-38%**（222.7s@60；`end_step=0` 才真实参与，语义=末 N 步保留 dense）；🟡 V3（**Qwen-Image 图像形态不可用**：staying dense——层未声明 qkv_layout，rf_v2 需 BSND video 段，图像 2D 无视频段；**0.8 与 0.4（<60% 小档）两档均 staying dense（档位无关）**；输出=lossless 逐字节；经验：图像若未来有 2D 稀疏路径应从 <60% 起试、视频可大稀疏） | ❓ | ❓（平台 rf 系只试过 rf3/video_spans（下行）；rf_v2 在 950PR npu FA 可用性未核） |
| ada_bsa | ❓ | ❓ | ❓ |
| rf_v3 / video_spans（平台扩展，**非 docs 主表**） | ❓ | ❓ | 🟡 L1（可接入；eager 质量梯度正常：sp0.3/0.5 近无损-良好 SSIM 0.975/0.960、sp0.8 0.81（同 seed 门禁，2026-09）；compile×rf3 trace 待解；历史「0.82 平台」证伪，见 case §10） |

### 编译路径（docs: `compilation.md`；入口 `torch.compile(backend=MindieSDBackend())`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 编译后端 MindieSDBackend | 🟡 V1（可注入 52 block，kernel 级为负→默认关）；🟡 V3（Qwen-Image 60 modules 编译成功但**输出非逐字节一致**（compiled FA 数值/seed 语义，torch_npu 告警）+ 仅 ~-4% → 默认关）；🟡（2026-09-06 H3 FFN MX 融合**修复闭环**：vLLM 变体 GraphPatternEntry + C++ 布局自适应后 `mm_swiglu_mxquant` 命中 52/52、Qmm/DxQ -52，60 步 e2e **-5.8%**——首个 vLLM-Omni 真图上 compile pattern 融合**命中且为正收益**的案例；输出非位级（量化级近似，差异 ≤ mxfp8 分散度）；见 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/fix_validation.md） | ✅ V2（`_compiled_call_impl` 原地写入 + backend 单例） | ✅ L1（`compile_backend=mindie` 平台注册表） |
| Pattern 融合：RMSNorm / RoPE / AdaLayerNorm / fastGELU / Mul+Add | ✅ V1（eager 单算子已覆盖，勿再叠 compile） | ✅ V2（qwen_rope 等命中集合） | ✅ L1（rms_type/rope_type 注册表 + pattern 变体） |
| ACLGraph 静态图捕获 | ❓ | ❓ | ❓ |

### 显存（docs: `cpu_offload.md`；接口 `enable_offload(model, blocks, …)`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 异步 CPU Offload | ✅ V1（框架侧 DLO 解锁 TP1×USP4 BF16；普通 layerwise 950PR 会 OOM）；❓ V3（Qwen-Image 单卡 ~55GB/128GB 显存足 → offload 未测 N/A） | ❓ | ❓（框架自有 vae_cpu_offload/VAE 并行已启用、model/block offload 支持（model.py:62，`*_block_offload` 配置在）；mindiesd `enable_offload` 异步档未核，见 case §10） |

### 并行 / 通信（docs: `parallelism.md` / `usp.md`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| Ulysses 序列并行 (USP) | ✅ V1（TP2×USP2 最优无损；TP1×USP4 需降显存后）；✅ V3（Qwen-Image **2 卡 TP1×USP2 采纳 = 无损最优 4.35s**（-12% vs TP2、-30% vs TP1）；4-rank USP 形态短任务病态回退 27.8s） | ❓ | ✅ L1（seq-parallel + `hccl_eager`） |
| 张量并行 (TP) | ✅ V1（TP2 起点）；✅ V3（Qwen-Image **TP2 = S4 有损基底 4.96s**（次优并行）；TP4 4-rank 通信病态回退 27.8s） | ❓ | 🟡 L1（TP2×SP2 实测可用但 UB 岛差于 bulk → 回退；TP 单用未单独核，见 case §10） |
| CFG 并行 | ❓ | ❓ | ❓ |
| RSP（环状序列并行） | ❌ V1（Ring 不支持 attn_mask，mask_sp_padding 开关无效 → 序列并行选 USP） | ❓ | ❌ L1（H3 SP 仅 Ulysses，model.py 门对非 Ulysses 直接 NotImplemented；无 Ring 实现——非 attn_mask 问题） |
| head-parallel / CP 变体（**框架自有，非 docs 特性**） | ❓（USP 为主） | ❓ | ✅ L1（SYS 跨岛更优 -11.5%；compile × head-parallel 不兼容） |

### 缓存加速（docs: `cache.md`；接口 `CacheConfig(method=…, …) + CacheAgent`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| DiTCache（method=`"dit_block_cache"`） | ✅ V1（组合内 3.65×；S4-2 frontier：int8+mix+Cache **28.2s@60**；**别名**：case 曾记 Cache-DiT/DiTBlockCache）；env A 2026-09-07（950PR×2 TP2，SDPA 基线）：单点 **-77%（83.4s@60）**，组合 量化(mxfp8)+FFN-MX+Cache **4.70×（92.9s）**；✅ V3（Qwen-Image **单独 Cache 单点 -13.7%**（4.28s，默认档 th=0.24）；+量化(w8a8) 组合 **-24.0%**（3.77s = 1.65× vs TP1 基线）；步跳过随图 4.29-4.54s 双峰） | ✅ V2（DiTBlockCache；双缓存互斥、CFG-on shape 约束） | ❌ L1（feature_caching NotImplemented → 按新规 **Cache 与框架对齐、mindiesd 不额外适配**；原 cache_agent 接入 P0 转为探针观察，不沉淀推荐姿势） |
| AttentionCache（method=`"attention_cache"`） | ❓ | ✅ V2（与 DiTCache 互斥） | ❌ L1（同 DiTCache：框架未接线；按新规不做 mindiesd 额外适配，原 bench 侧 CacheAgent 备选转探针观察） |

### 时间步优化（docs: cache 章「时间步优化」）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 时间步优化（减少/跳过步数） | ❓（案例用 60/10 步非裁剪） | ❓ | ✅ L1（`infer_steps=24` 采纳，-28~-37% Run DiT，帧 SSIM 0.937）；「优于移植 cache」为当时结论（cache 未接线）；cache 叠加为 §11 P0 待验 |

## 二、姿势与坑（关键格，细节回 case 文件）

- **vLLM-Omni（V1）**：compile 注入 = `OMNI_MINDIE_COMPILE=1` 门控 + backend 单例，52 block 编译成功、pattern 注册、输出逐字节一致但 kernel 级为负 → 默认关；offload 用 DLO（`--enable-distributed-layerwise-offload`），普通 layerwise 在 950PR 触发 OOM；量化/稀疏/缓存叠加实测组合见 case §4 表；无 comm-stream 掩盖机制（0.28，单步 Overlapped=0；未重叠通信占比随并行策略 **USP2 27.5% > TP4 17% > USP4(int8) 10.5%**，见 case §7/报表 §2.3——掩盖收益上限随策略 10-27% 步时）。
- **vLLM-Omni compile×FFN-MX 融合（2026-09-06 修复闭环，首个正向 compile pattern 案例）**：H3 FFN hidden 站点 `mm_swiglu_mxquant`（Qmm+swiglu+输出量 三合一 catlass kernel）经 **vLLM 变体 GraphPatternEntry + C++ 布局自适应** 在 vLLM-Omni 真图命中 **52/52**（fused 0→52，Qmm/DxQ 各 -52），60 步 TP2 mxfp8 e2e **-5.8%**（299.1 vs 317.5s）。关键：① 必须 `method:mxfp8`（MX e4m3 域）非 int8；② vLLM-Omni 权重 GEMM-ready (K,N) fp8 + scale (c,N,2)、行序 gate-first → C++ `AdaptLayoutCached` 检测 `w.size(0)==k` 缓存 `w^T`（免 row-swap）；③ npu_swiglu 单融合 op → 独立 pattern 变体（无 view/split 形态）；④ kernel launch 前需 `aclrtSynchronizeStream`（裸 aclrtLaunch 与异步转置拷贝跨流竞态 → AI Core 507015）；⑤ 输出为量化级近似非位级（单层 rel 0.36%，视频差异 ≤ mxfp8 相对 bf16 分散度 ~19dB 同量级）。证据 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/。
- **vLLM-Omni H3 完整重跑闭环（2026-09-07，env A 新基线 + 最新 MAO 规范 run-state/stage_gate/双报表）**：基线 = **env A TP2×2 卡 SDPA（TORCH_SDPA 显式）436.7s@60**。⚠️ 环境修正：env A mindiesd editable 常驻（MindIE-SD-028）→ 无显式 attention config 时默认 FLASH_ATTN，**基线须 TORCH_SDPA 显式**；CMP 树（MindIE-SD-CMP，mm_swiglu v5）与 028 树并存——**CMP sparse 缺 `video_spans`（rf_v2 不兼容）→ 稀疏档须 028 树**，融合档须 CMP 树。结果（60 步同窗 r1 稳定对，异常窗剔除）：FA 360.7s(-17.4%)；compile bf16 无正收益回退；mxfp8+FFN-MX 304s(-6~7% vs unfused，fused=52)；稀疏 rf_v2 0.8 222.7s(-38%，⚠️ `end_step` 语义=保留末 N 步 dense，误设 60=全程 dense fail-closed——配置后须输出对比确认参与)；Cache DiT-Cache 83.4s(-77%，最优单点)；**最强组合 量化(mxfp8)+FFN-MX+Cache 92.9s（4.70×）**。质量 vs FA lossless：Cache 21.7/0.754、combo 18.5/0.673（阈值 16/0.51 内，视觉 inconclusive 未宣称通过）。共享宿主热节流/多租户使同档热窗值高 20-100%（r2/s 异常剔除留痕）。产物 runs/20260907_minimax-h3_optimization/（overview/detail/final/evidence）+ D:\framework\agentic/（run-state/stage_gate S0-S4+close error=0）。
- **vLLM-Omni（V3，Qwen-Image-2512 图像）**：稀疏 rf_v2 使能前置判据 = 层须声明 `qkv_layout='BSND'`（视频轴）——图像 2D 无 → **staying dense 兜底（输出=lossless 逐字节，fail-closed 判未生效；0.4/0.8 档位无关）**；短任务（20 步）并行选 **2 卡 TP1×USP2（4.35s）> TP2（4.96s，S4 有损基底）**（4-rank TP4/USP2 通信病态 ~27.8s）；compile 对 qwen-image 输出**非逐字节**（compiled FA 数值/seed 语义）→ 双重否决；TP2 单步 comm 31.2%（Overlapped=0）→ 掩盖空间上限 ~31-37% 步时（框架结构性缺口）；图像同 seed 像素质量域（w8a8 37.5/0.986）远高于视频（19.6/0.679）→ 有损阈值不可跨域迁移；拓扑先查 UB 岛（npu-smi -t topo，0-3/4-7 各为岛，跨岛 SYS）同岛选卡；多进程残留需 `pkill -9 -f 'vLLM-Omni::DiffusionWorker'`；JSON 型 CLI 参数经多层 shell 丢引号 → 走文件传递（细节见 case §5/§6）。
- **DiffSynth-Engine（V2）**：compile = `compile_backend="mindie"` + `_compiled_call_impl` 原地写入 + backend 实例复用（勿重建）；Qwen-Image RoPE 实数域改写命中 `qwen_rope_pattern`；DiTCache 与 AttentionCache 互斥、CFG-on 有 shape 约束；attention 进图中性回退（不启用）。
- **LightX2V（L1）**：版本口径 = 上游合入版 **#1471** 机制（`COMPILE_BACKEND_REGISTER` /
  `hccl_eager` / rms·rope 注册表）；实测数据来自「合入前代码 + 本地镜像 #1471 等价机制」的远端
  （机制等价已验证，见 case）。compile = `use_compile:true` + `compile_backend:"mindie"` +
  `seq_p_a2a_backend:"hccl_eager"`（collective 留 eager 防 `hcom_alltoallv` 退化）；
  **量化 ❌ 的准确含义 = 框架未接线 mindiesd quantize**（仅 `dit_quantized` 预量化 ckpt 路，
  triton int8/fp8 GEMM 本机编译失败）；mindiesd `quantize.py` W8A8_DYNAMIC/MXFP8（online）本机
  已具备、同 NPU 同族 harness 实证 → 阻塞可解（§11 P1，待 smoke）；缓存 ❌ = feature_caching
  对 H3 NotImplemented → 按新规 Cache 与框架对齐、mindiesd 不额外适配（原 cache_agent 接入转探针观察）
  （§11 P0）→ 时间步 24 步为现行采纳项，cache 叠加待验；SYS 跨岛并行选 head-parallel（UB 岛选
  bulk），compile × head-parallel 不兼容（Dynamo 静默回退）；TP2×SP2 实测可用但 UB 岛回退；
  RSP 无 Ring 实现（H3 SP 仅 Ulysses）。
  **跨框架差距与待补充能力（P1 mindiesd quantize online 路线 / ACLGraph 核验等；缓存 bench 侧接入按
  Cache 对齐框架新规转探针观察）→ `lightx2v-mindiesd-case.md` §11**（2026-09 探针确认 mindiesd 侧
  quantize 已具备，缺口=框架接线；Cache 不额外适配）。

## 三、刷新协议（防表过期——本表硬配套）

1. **触发点**：每次模型优化进入 S0/S4 或某特性使能失败时核对一次对应框架列；框架版本升级
   （vLLM-Omni / DiffSynth-Engine / LightX2V / mindiesd）；`docs/zh/features` 变更后
   （`refresh_features.py` 同步 mindiesd-features.md 后对照新特性名）；新增 case 回填时。
2. **动作**：对照框架源码开关/env/接口 + 最小使能冒烟 → 更新单元格
   （`✅/🟡/❌/❓` + 版本 + 日期）；`❓` 升级为确定态须带证据码。
3. **留痕**：表头按框架记录「版本 + last-checked」；状态变更在对应 case 摘要补一句。
4. **docs 联动（特性增长感知）**：mindiesd 新增/改名特性先走
   `performance-optimization` 的 features 刷新规则更新真相源，再同步本表行名与状态。
5. **机器化 seam 联动**：本表的状态/姿势与组合前静态判定用的机器声明
   `model-auto-optimization/scripts/feature_declarations.json` 互为印证——新增特性或改支持状态时，
   同步检查声明是否需要增行/改 seam（seam_check 语义以声明为准，本表供人读与框架列状态）。

## 四、与相邻能力边界

- mindiesd 自身 kernel/pattern 能力清单 → 槽位 S2-1（README §4）+ `mindiesd-features.md`（接口真相源）
- 特性选档与组合 seam → `performance-optimization/references/combination-search.md`（组合前先按本表裁剪候选）
- 端到端质量判定 → `quality-gate.md` + 仓库 `evals/`
- 本表只登记「框架侧支持状态 + 姿势指针」，实测细节与回修回 case 文件。

## 维护与更新

- 按 §三 协议每次刷新；docs 特性命名漂移时同步本表行名。
- 本文件被 framework-feature-enablement（Reference Files）与 model-auto-optimization（S1/S4）引用。
