# 框架 × MindIE-SD 特性支持矩阵

> 定位：MindIE-SD 特性**命名与能力**的唯一真相源是 `docs/zh/features/*`（仓内直读，无镜像副本）。
> 本表 = 各三方推理框架（vLLM-Omni / DiffSynth-Engine / LightX2V）对上述特性的**支持面**——
> 支持状态 + 框架侧使能姿势 + 已知坑，供：使能前查支持面、使能失败排查、S4 组合前裁剪候选、
> 框架版本升级后感知特性增长。
>
> 口径：状态仅对「框架版本 + 模型」成立（案例结论单点不迁移）；`❓` 表示未核验，禁止当作支持。
>
> 状态图例：`✅` 案例实证支持 · `🟡` 可配置但有已知限制/负面 · `❌` 不支持或阻塞 · `❓` 未核验。
> 证据码：V1 = vLLM-Omni 0.28 × MiniMax-H3-FL2VA 实测（576p 视频；机型/卡数坐标见归档；**能力面=本矩阵各格，
> 开启方式见 `references/vllm-omni-enablement.md`，实测数字归档于会话产物目录 `{run_results_dir}/archive/`**）、
> V2 = `references/diffsynth-engine-enablement.md`（Qwen-Image / DiffSynth-Engine；原 `-case.md` +
> `-notes.md` 两件已合并为本件，绝对数字归档于 `{run_results_dir}/archive/`）、
> V3 = vLLM-Omni 0.28 × **Qwen-Image-2512 图像 1024² 20 步**实测（**能力面=本矩阵各格，
> 开启方式见 `references/vllm-omni-enablement.md`，实测数字归档于 `{run_results_dir}/archive/`**）、
> V4 = `references/vllm-omni-train-aware-enablement.md`（MiniMax-H3-FL2VA **视频 15s 768P** /
> vLLM-Omni 0.28 **USP4**；**通用方法见 `references/train-aware-lossy-method.md`**）、
> L1 = `references/lightx2v-enablement.md`（MiniMax-H3 / LightX2V #1471；**能力面=本矩阵各格，
> 开启方式见该文件，实测数字归档于 `{run_results_dir}/archive/lightx2v-mindiesd-case.md`**）、
> G = 本技能正文（通用回路）。
> **V1 列的子批次**（同框架同模型、不同基线/补丁批，仍记 V1 而不再派生新码）：
> **V1a** = 2026-09-06 `mm_swiglu_mxquant`/mxfp8 档 compile 融合修复闭环（证据
> `runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/fix_validation.md`）；
> **V1b** = 2026-09-07 env A 完整重跑闭环（基线改为显式 `TORCH_SDPA`，产物
> `runs/20260907_minimax-h3_optimization/`）。表内两处裸写日期的条目即这两个子批次。
> L1 列 last-checked：2026-09-05（对照 `lightx2v-enablement.md` §3.4/§3.5 新增锚点 + 远端源码复核修订：
> 量化阻塞表述、TP/RSP/offload/缓存/时间步格与 §二 姿势段；修订明细见归档
> `{run_results_dir}/archive/lightx2v-mindiesd-case.md` §10「并行形态/显存/量化阻塞表述」三条）。
> vLLM-Omni 列 last-checked：2026-09-06（0.28 disable-vllm-ascend；V1=H3-FL2VA 576p 视频（机型/卡数坐标见归档）；
> **V3=Qwen-Image-2512 1024²/20 步（机型/卡数坐标见归档；2026-09-05/06）：量化✅/Cache✅/2 卡 TP1×USP2 采纳（无损最优
> 形态）、TP2=S4 有损基底、rf_v2 图像不可用（0.4/0.8 档位无关）、4-rank（TP4/USP2）短任务通信病态回退、
> compile 输出非无损**；图像质量域（同 seed 像素对比）显著高于
> 视频，有损阈值不可跨域迁移；绝对耗时/加速比见归档 `{run_results_dir}/archive/`）。
> vLLM-Omni 列 last-checked 追加 2026-09-12（V4 = **长视频 15s/768P/USP4**）：**单点收益排序随规模反转**——
> 长序列档 **削注意力的维度 > 缓存维度 > 线性层量化维度**，短序列档则是缓存维度主导（注意力 O(S²)
> 成为主项后线性层量化占比被稀释）⇒ **单点排序只在本任务口径下有效，禁止跨时长/分辨率/拓扑引用**；
> **量化维度的边际收益随注意力被压缩单调上升**（单点很小 → 稀疏之上更大 → 稀疏+缓存之上最大，即
> 正协同的逐步墙钟证据），故「单点收益低 ≠ 可裁」，缓存维度可独立连乘；**训练感知档**
> （`少步蒸馏` 可叠 `量化`+`稀疏`；`VAE解码替换` 为框架未提供的待开发项）与
> **少步档缓存结构性失效（字节级等价判据）** 见新增「训练感知」表。
> vLLM-Omni 列 last-checked 追加 2026-09-13（V4 续：**CP 形态与稀疏/掩盖的叠加面**）——
> 复合「AllGather-KV × Ulysses」**可跑通、可叠分块掩盖（逐字节精确）、与稀疏可叠加**；
> **环状形态对稀疏不可用**（`ring_degree>1` 硬拒 + 路径绕过注意力后端 ⇒ 稀疏静默失效）；
> 4 卡 `usp=2, allgather-degree=2` 在本负载下**未跑通**；形态抉择与 CP×稀疏契约分别见
> `dit-parallel-opt/references/parallel-form-selection-method.md` 与 `cp-sparse-combination-method.md`。
> **本表的数字口径**（纪律见 `.agents/README.md` §7「数字纪律（强制）」）：**不写绝对耗时与绝对
> 质量分值**（二者只在本次环境成立）；**要写**大致加速比（量级/约数）与质量变化度（相对基线的
> 差值/降幅）；比例关系（排序/独立性/协同强度/能否完全解释/是否节省的全部来源/线性或反转）与
> 方向选择（标「本组合观测」）保留；策略与方向**以最近一次实测为准**，历史结论只作参照。
> env A（2026-09-07 闭环，TP2（机型/卡数坐标见归档），**显式 TORCH_SDPA 基线**，editable mindiesd 双树并存）：
> FA / 稀疏 rf_v2 / Cache 单点逐个递增、组合最强（**单点排序与协同强度见 §二 env A 段**）；
> 数值口径与双树选树规则同上；**绝对耗时与绝对加速比归档于 `{run_results_dir}/archive/`**。

## 〇、framework × 特性/能力 支持档位（图例与记录规则）

状态图例（✅/🟡/❌/❓）表达"验证到什么程度"；本图例表达"**该能力在 framework×模型组合下处于
什么实现就绪状态**"，供特性覆盖清单排序与执行序（先做什么、哪些要投入开发）使用：

| 档位 | 定义 | 判例（该能力的判定粒度） | 流程含义 |
|------|------|--------------------------|----------|
| **已支持** | 能力在该框架组合下可直接使用（默认或开关/配置即生效） | 如 LightX2V `npu-w8a8-mxfp8` 原生 scheme、vLLM-Omni cache_dit/量化开关 | 零开发：直接做（覆盖清单判「做」） |
| **待配置** | 特性面存在但**部分能力不全/未接线**，需配置/接线补齐特定能力 | 如"量化有支持但不支持 mxfp8 型 w8a8"——能力实体（kernel/loader/机制）已具备，缺框架接线/开关/量化描述符；mindiesd `quantize` 本机已具备而框架未接线（`lightx2v-enablement.md` §3.4/§3.5 + 本表「跨框架待补充能力清单」；Cache 按新规不做 mindiesd 额外适配） | framework-integration 接线（记 P 级计划）；成本=配置接线 |
| **待开发** | 特性在该框架组合下**没有支持，需完整实现** | 能力实体（kernel/消费者/机制）不存在：需 operator-dev 新算子 / 本技能 §2 分支 B 结构性实现 / 框架侧新机制 | 先估成本 + 用户确认（SKILL §0 缺口补齐策略①之分支 B）再投入；默认不静默做 |
| **上界（预留）** | 理想档/天花板探针定义——**空间预留，待人工填写** | 如"某档假设全收益/无副作用时的收益上限"（后续按需求补判例与口径） | 只诊断不宣称；不进入 delivery（同 purpose 的 unsafe_probe 纪律） |

判定粒度（能力级，不按特性大面一刀切）：

- **量化**：按能力组合判——`w8a8`（类型 mxFP8 e4m3）/ `w8a8`（int8 dynamic）/ `w4a4`（类型 mxFP4）/
  `f8`（FA 侧，路径 FP8RotateQuantFA 或框架案例实现）；组合（w8a8f8）逐能力判；与报表修饰符
  白名单同源（overview-report §2.1）。
- **稀疏**：**与 mindiesd 稀疏算子对齐**逐算子判——rf_v2 / ada_bsa /（平台扩展 rf_v3、video_spans）
  等（`docs/zh/features/sparse.md` 为算子面真源），逐算子 × 框架给出档位。
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
| W8A8_DYNAMIC（INT8 online） | ✅ V1（576p 视频档：单点为正（约 1.2 倍量级），vs 显式 SDPA 基线更大；**质量变化度**：vs lossless 的 SSIM **降约三成**（差值约 0.3），视觉门 inconclusive；绝对数字见归档 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`）；✅ V3（Qwen-Image 1024²/20 步：单点为正（约一成多）、**质量变化度**：**质量基本无感**（SSIM 降约 0.01；读数见归档）；🟡 V4（**长视频档单点收益最低**：序列变长后注意力 O(S²) 主导、线性层量化占比被稀释 ⇒ 单点收益远小于削注意力/缓存的维度——**但边际收益随注意力被压缩单调上升**（稀疏之上更大、稀疏+缓存之上最大）⇒ 量化应与稀疏同部署（正协同），**勿按单点收益否定它**；⚠️ 单点收益与协同强度均**不可跨规模引用**） | ❓ | ❌ L1（框架未接线 mindiesd quantize；仅 `dit_quantized` 预量化 ckpt 路 + triton int8 GEMM 本机编译失败——非「本 NPU 无 kernel」，见 `lightx2v-enablement.md` §3.4 与归档 §10/§11 P1） |
| W8A8_MXFP8 | 🟡 V1（2026-09-06 修复闭环：mindiesd dev `mm_swiglu_mxquant`（catlass 集成）与 vllm-omni 0.28 **mxfp8 档接线打通**——compile 侧 FFN-MX 融合对 vLLM-Omni H3 真图**全部站点命中**（vLLM 变体 GraphPatternEntry + C++ 布局自适应 vLLM GEMM-ready (K,N)/scale (c,N,2)）；单点为正（绝对数字见归档）；质量门=**量化级容差**（单层 rel 与输出差异 ≤ mxfp8 相对 bf16 的量化分散度同量级）；仅适用 SwiGLU/MoE（H3），Qwen-Image 无 SwiGLU 不适用；int8 档仍不适用（i8 域 ≠ MX e4m3，须 `method:mxfp8`）。见 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/fix_validation.md） | ❓ | 🟡 L1（**原生 scheme `npu-w8a8-mxfp8` 已实现**（内部 mindiesd MXFP8/npu_quant_matmul），质量近无损档，见 `lightx2v-enablement.md` §3.4；非 docs `quantize()` 接口，DYNAMIC 未接） |
| W8A16 / W4A16 / W4A4_MXFP4_* 等其余档 | ❓ | ❓ | ❓ |
| FA 量化（FP8，`FP8RotateQuantFA`） | ❓ V4（0.28 侧**无全注意力 FA 8bit 接线**——只有稀疏块路径的框架案例实现（见 V1 EagleQBSA）；故 V4 的 15s 档三元 `Cache + 量化(w8a8f8) + 稀疏` 无实测路径，按 ❓ 登记、不作锚点不宣称） | ❓ | ❌ L1（框架未接线；kernel 侧 mindiesd FP8/MXFP8 FA（v2 op）**目标代际 kernel 级微测可用**（量化级精度）2026-09，端到端质量/收益未验证，见 `lightx2v-enablement.md` §3.5） |

### 稀疏（docs: `sparse.md`；接口 `sparse_attention(q,k,v, sparse_type=…)`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| rf_v2（RainFusion2.0） | ✅ V1（视频档 sparsity=0.8 实测；另 EagleQBSA mix）；env A 2026-09-07（TP2；机型/卡数坐标见归档）：单点为正、量级居前（绝对数字见归档）；⚠️ `end_step` 语义=末 N 步保留 dense，误设=全程 dense 则 fail-closed；🟡 V3（**Qwen-Image 图像形态不可用**：staying dense——层未声明 qkv_layout，rf_v2 需 BSND video 段，图像 2D 无视频段；**大档与 <60% 小档均 staying dense（档位无关）**；输出=lossless 逐字节；经验：图像若未来有 2D 稀疏路径应从 <60% 起试、视频可大稀疏） | ❓ | ❓（平台 rf 系只试过 rf3/video_spans（下行）；rf_v2 在目标机型 npu FA 可用性未核） |
| ada_bsa | ❓ | ❓ | ❓ |
| rf_v3 / video_spans（平台扩展，**非 docs 主表**） | ❓ | ❓ | 🟡 L1（可接入；eager 质量梯度**平滑单调**（近无损→良好→激进档，同 seed 门禁 2026-09）；采纳档 sp≤0.5；compile×rf3 trace 待解；历史「某区间质量平台化」**证伪**，见 `lightx2v-enablement.md` §3.5 与归档 §10） |

### 编译路径（docs: `compilation.md`；入口 `torch.compile(backend=MindieSDBackend())`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 编译后端 MindieSDBackend | 🟡 V1（**整 transformer 编译**：可注入全部 block，kernel 级为负→默认关）；🟡 V3（Qwen-Image 全部 modules 编译成功但**输出非逐字节一致**（compiled FA 数值/seed 语义，torch_npu 告警）+ 收益小（几个百分点）→ 默认关）；🟡 **V1a**（**compile 内的 FFN 融合 pattern**（与上行对象不同，勿混读）：2026-09-06 修复闭环后 `mm_swiglu_mxquant` **全部站点命中**（被替代链计数归零）、单点为正——首个 vLLM-Omni 真图上 compile pattern 融合**命中且为正收益**的案例；输出非位级（量化级近似，差异 ≤ mxfp8 分散度）；绝对数字见归档 `{run_results_dir}/archive/`；见 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/fix_validation.md） | ✅ V2（`_compiled_call_impl` 原地写入 + backend 单例） | ✅ L1（`compile_backend=mindie` 平台注册表） |
| Pattern 融合：RMSNorm / RoPE / AdaLayerNorm / fastGELU / Mul+Add | ✅ V1（eager 单算子已覆盖，勿再叠 compile） | ✅ V2（qwen_rope 等命中集合） | ✅ L1（rms_type/rope_type 注册表 + pattern 变体） |
| ACLGraph 静态图捕获 | ❓ | ❓ | ❓ |

### 显存（docs: `cpu_offload.md`；接口 `enable_offload(model, blocks, …)`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 异步 CPU Offload | ✅ V1（框架侧 DLO 解锁原本因单卡容量不可行的高序列并行 BF16 形态；普通 layerwise 在部分代际会 OOM（须逐代际验证））；❓ V3（Qwen-Image 单卡显存足 → offload 未测 N/A） | ❓ | ❓（框架自有 vae_cpu_offload/VAE 并行已启用、model/block offload 支持（model.py:62，`*_block_offload` 配置在）；mindiesd `enable_offload` 异步档未核，见 `lightx2v-enablement.md` §3.6 与归档 §10） |

### 并行 / 通信（docs: `parallelism.md` / `usp.md`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| Ulysses 序列并行 (USP) | ✅ V1（TP2×USP2 最优无损；TP1×USP4 需降显存后）；✅ V3（Qwen-Image **2 卡 TP1×USP2 采纳 = 无损最优形态**（优于 TP2、更优于 TP1）；4-rank USP 形态短任务病态回退；绝对数字见归档） | ❓ | ✅ L1（seq-parallel + `hccl_eager`） |
| 张量并行 (TP) | ✅ V1（TP2 起点）；✅ V3（Qwen-Image **TP2 = S4 有损基底**（次优并行形态）；TP4 4-rank 通信病态回退） | ❓ | 🟡 L1（TP2×SP2 实测可用但 UB 岛差于 bulk → 回退；TP 单用未单独核，见 `lightx2v-enablement.md` §3.6 与归档 §10） |
| CFG 并行 | ❓ | ❓ | ❓ |
| RSP（环状序列并行） | ❌ V1（Ring 不支持 attn_mask，mask_sp_padding 开关无效 → 序列并行选 USP） | ❓ | ❌ L1（H3 SP 仅 Ulysses，model.py 门对非 Ulysses 直接 NotImplemented；无 Ring 实现——非 attn_mask 问题） |
| head-parallel / CP 变体（**框架自有，非 docs 特性**） | ✅ V4（**复合 AllGather-KV × Ulysses**（`--usp k --allgather-degree m`）可跑通、可叠分块掩盖（逐字节精确）、可与稀疏叠加；但**端到端不如纯 8 卡序列并行**（本组合观测：慢约一成，成因是跨岛 K/V 汇聚 + 形态自带计算；抉择按 GQA/带宽条件化，见 `dit-parallel-opt/references/parallel-form-selection-method.md`）；**环状形态对稀疏不可用**（硬拒 + 路径绕过后端 ⇒ 稀疏静默失效）；4 卡 `usp=2, allgather-degree=2` **未跑通**） | ❓ | ✅ L1（SYS 跨岛组更优（本组合观测）；compile × head-parallel 不兼容） |

### 缓存加速（docs: `cache.md`；接口 `CacheConfig(method=…, …) + CacheAgent`）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| DiTCache（method=`"dit_block_cache"`） | ✅ V1（组合内为主导演进项；S4-2 frontier 由它主导；**别名**：曾记 Cache-DiT/DiTBlockCache）；✅ **V1b**（2026-09-07 env A，TP2（机型/卡数坐标见归档），SDPA 基线）：**单点降幅居首**、组合最强；**质量变化度**：单点 vs lossless SSIM **降约三成**，叠加稀疏/量化后进一步降到**降约五成**（绝对数字见归档 `{run_results_dir}/archive/`）；✅ V3（Qwen-Image **单独 Cache 单点为正**（默认档，约一成多）；+量化(w8a8) 组合更大（约两成多）；**质量变化度**：图像档 SSIM 降约 0.04（叠加档同级，视觉门 inconclusive）；步跳过随图双峰） | ✅ V2（DiTBlockCache；双缓存互斥、CFG-on shape 约束） | ❌ L1（feature_caching NotImplemented → 按新规 **Cache 与框架对齐、mindiesd 不额外适配**；原 cache_agent 接入 P0 转为探针观察，不沉淀推荐姿势） |
| AttentionCache（method=`"attention_cache"`） | ❓ | ✅ V2（与 DiTCache 互斥） | ❌ L1（同 DiTCache：框架未接线；按新规不做 mindiesd 额外适配，原 bench 侧 CacheAgent 备选转探针观察） |

### 时间步优化（docs: cache 章「时间步优化」）

| docs 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 时间步优化（减少/跳过步数） | ❓（案例用 60/10 步非裁剪）；V4 说明：训练感知档 50→4 步属**换权重** → 归 `少步蒸馏`（训练感知组），**不记本行**；且 **4 步档 Cache 字节级失效**（`max_warmup_steps` 吃满步数预算 → 开/关两档产物 md5 相同） | ❓ | ✅ L1（`infer_steps` 裁剪为采纳项：收益**与步数比例同量级**，帧 SSIM 属近无感档；「优于移植 cache」为当时结论（cache 未接线）；cache 叠加为待验项，见 `lightx2v-enablement.md` §3.3） |

### 训练感知（S5：换入**外部训练权重**——非 docs 特性，按权重来源归组）

| 特性 | vLLM-Omni 0.28 | DiffSynth-Engine | LightX2V #1471 |
|---|---|---|---|
| 少步蒸馏（换入蒸馏适配器） | ✅ V4（服务侧 `--task-type fl2va --lora-backend peft --lora-path …` + 请求 `num_inference_steps=4` + `lora={…scale}`；**仅 0.28 有此链**；两条前置契约——`model_index.json` 不得 pin `base_schedule`、适配器 metadata 的 `base_schedule` 使步数语义 = **denoiser 评估次数**（4 ≠ 5 个 sigma 点）；装载计数契约 = 计数行与实际适配器模块数一致（rank 递进 + 各 worker 激活），防 no-op 假加速；收益呈**线性于步数**、不随负载规模变） | ❓ | ❓（框架侧 LoRA/蒸馏权重链未核） |
| VAE解码替换（换入外部训练的小型自编码器） | ❌（**待开发**：框架未提供该接口）；V4 已以 **fork 探针** `[探针]` 验证可行性（新增解码器模块 + pipeline env 分派，**默认关 + `.bak` 保留、未合入上游**）。要点：换入前先核**参考实现的判定契约**（末层 `12 = 3×patch²` 通道 + `pixel_shuffle` + 因果记忆块 + 时序 ×4 ⇒ 逐帧 2D 网络是**架构级错配**，扫参不可补）；latent 须**原样喂入**（上游已自行 `*std+mean`，重复归一化会过驱动头部非线性）；末层无激活 ⇒ 输出无界，照抄 `clamp(0,1)` 会把大比例像素钉成纯黑白（观感=块状），后处理须权重相关重验。收益：**端到端节省全部来自 decode 阶段**（diffuse 不变）；质量须与原生解码器**同 latent 对拍**（灰度相关 + 钳位占比 + 输出 std）+ SSIM，并声明画质档位（本例属**预览级**——依 `quality-gate.md` 给保画质档 / 预览档两条并列建议） | ❓ | ❓ |

## 二、姿势与坑（关键格；机制细节见各框架 enablement 文件，实测数字见会话产物归档）

- **vLLM-Omni（V1）**：compile 注入 = `OMNI_MINDIE_COMPILE=1` 门控 + backend 单例，全部 block 编译成功、pattern 注册、输出逐字节一致但 kernel 级为负 → 默认关；offload 用 DLO（`--enable-distributed-layerwise-offload`），普通 layerwise 在部分代际触发 OOM（须逐代际验证）；量化/稀疏/缓存叠加实测组合见归档 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md` §4 表（开启方式见 `references/vllm-omni-enablement.md` §3）；无 comm-stream 掩盖机制（0.28，单步 Overlapped=0；未重叠通信占比随并行策略 **USP2 > TP4 > USP4(int8)**，见 `vllm-omni-enablement.md` §2.4 / 归档 §7 / 报表 §2.3——掩盖收益上限 = 该占比，步时占比随策略变化）。
- **vLLM-Omni compile×FFN-MX 融合（**V1a**，2026-09-06 修复闭环，首个正向 compile pattern 案例）**：H3 FFN hidden 站点 `mm_swiglu_mxquant`（Qmm+swiglu+输出量 三合一 catlass kernel）经 **vLLM 变体 GraphPatternEntry + C++ 布局自适应** 在 vLLM-Omni 真图**全部站点命中**（被替代链计数归零；站点计数见归档 `{run_results_dir}/archive/`），端到端为**正收益**（绝对数字见归档 `{run_results_dir}/archive/`）。关键：① 必须 `method:mxfp8`（MX e4m3 域）非 int8；② vLLM-Omni 权重 GEMM-ready (K,N) fp8 + scale (c,N,2)、行序 gate-first → C++ `AdaptLayoutCached` 检测 `w.size(0)==k` 缓存 `w^T`（免 row-swap）；③ npu_swiglu 单融合 op → 独立 pattern 变体（无 view/split 形态）；④ kernel launch 前需 `aclrtSynchronizeStream`（裸 aclrtLaunch 与异步转置拷贝跨流竞态 → AI Core 507015）；⑤ 输出为量化级近似非位级（单层 rel 与视频差异均 ≤ mxfp8 相对 bf16 的量化分散度同量级）。证据 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/。
- **vLLM-Omni H3 完整重跑闭环（**V1b**，2026-09-07，env A 新基线 + MAO 规范 run-state/stage_gate/双报表）**：基线 = **env A TP2×2 卡 SDPA（TORCH_SDPA 显式）**。⚠️ 环境修正：env A mindiesd editable 常驻（MindIE-SD-028）→ 无显式 attention config 时默认 FLASH_ATTN，**基线须 TORCH_SDPA 显式**；CMP 树（MindIE-SD-CMP，mm_swiglu v5）与 028 树并存——**CMP sparse 缺 `video_spans`（rf_v2 不兼容）→ 稀疏档须 028 树**，融合档须 CMP 树。结果（同窗 r1 稳定对，异常窗剔除；**绝对耗时与绝对质量分值见归档**）：FA 单点为正；compile bf16 无正收益回退；mxfp8+FFN-MX 单点为正（融合站点全部命中）；稀疏 rf_v2 0.8 单点为正（⚠️ `end_step` 语义=保留末 N 步 dense，误设=全程 dense fail-closed——配置后须输出对比确认参与）；**Cache 单点降幅居首**；**最强组合 = 量化(mxfp8)+FFN-MX+Cache（由 Cache 主导）**，**质量变化度**：缓存单点 vs lossless SSIM 降约三成、叠加稀疏/量化后降约五成（**阈值内但视觉 inconclusive，未宣称通过**；阈值见运行 profile、读数见归档）。共享宿主热节流/多租户使同档热窗值高 20-100%（r2/s 异常剔除留痕）。产物 `{run_results_dir}/` 下 runs/20260907_minimax-h3_optimization/（overview/detail/final/evidence）+ 编排产物目录（run-state/stage_gate S0-S4+close error=0）。
- **vLLM-Omni（V3，Qwen-Image-2512 图像）**：稀疏 rf_v2 使能前置判据 = 层须声明 `qkv_layout='BSND'`（视频轴）——图像 2D 无 → **staying dense 兜底（输出=lossless 逐字节，fail-closed 判未生效；0.4/0.8 档位无关）**；短任务（20 步）并行方向 **2 卡 TP1×USP2 优于 TP2（TP2 = S4 有损基底）**，4-rank（TP4/USP2）通信病态回退（**本组合观测，绝对耗时见归档**）；compile 对 qwen-image 输出**非逐字节**（compiled FA 数值/seed 语义）→ 双重否决；TP2 单步未重叠通信占比约三成（Overlapped=0）→ 掩盖空间上限 = 该占比（框架结构性缺口）；图像同 seed 像素质量域**远高于**视频档（无轨迹混沌）→ 有损阈值不可跨域迁移；拓扑先查 `npu-smi -t topo`（同域 = UB/HCCS 全互联、跨域 = SYS/PCIe）并按同域选卡；多进程残留需 `pkill -9 -f 'vLLM-Omni::DiffusionWorker'`；JSON 型 CLI 参数经多层 shell 丢引号 → 走文件传递（细节见 `references/vllm-omni-enablement.md` §1/§2/§3/§5，实测数字见归档 `{run_results_dir}/archive/`）。
- **vLLM-Omni（V4，MiniMax-H3 长视频 15s/768P/USP4，2026-09-12）**：**单点收益排序随时长反转**——
  长序列档 **削注意力的维度 > 缓存维度 > 线性层量化维度**（注意力 O(S²) 成为主项后线性层量化占比
  被稀释），短序列档则是缓存维度主导 ⇒ **单点结论不跨规模迁移**（换规模 = 新对照）。
  长视频档的资源投放顺序为「削注意力项在前、线性层量化作低成本叠加项」——**该排序是本组合观测，
  不是可移植执行序**（换时长 / 分辨率 / 拓扑 / 框架 = 重判）。协同用「与单点连乘的比值」定位（≈1 独立可连乘 /
  明显 >1 正协同），**机理用逐步墙钟直接证**：量化维度的边际收益随注意力被压缩**单调上升**
  ⇒ 单点收益低 ≠ 可裁；三元由「两两 × 剩余单点」**完全解释** ⇒ 归因完成。
  **数字口径**：不记绝对耗时与绝对质量分值（随模型 × 框架能力 × 负载规模变化，纪律见
  `.agents/README.md` §7「数字纪律」）；**大致加速比（量级/约数）与质量变化度照记**；
  **比例关系与方向选择留存**，但方向须标注为**本组合观测**——**同一方法在不同框架下由不同「能力
  实体」承载（融合 op / eager 路径 / 框架自带开关 / 待开发缺口），收益不同则优选方向也不同**，
  故本节的行结论**不是可移植的优化路径**；本表的作用是给出各框架的**能力面**，
  方向须由使用方**在任务内动态判定**（测单点 → 比值法定协同 → 结合能力面 → 定方向）。USP4 前提 =
  `--tensor-parallel-size 1 --usp 4 --ring 1 --enable-distributed-layerwise-offload --text-encoder-tp-size 4
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling`（TP1 使 DiT 全量复制 → **DLO 强制**；
  `--text-encoder-tp-size` **依赖并行形态**（本组合观测，换并行形态 = 重判）：**序列并行（USP）形态**须与 DiT world 对齐否则子组 `None` 触发 assert；**纯 TP 形态（usp=1）本组合下整体不可用**（tetp 对齐到 DiT world → 视觉 seam `pixel_values and image_grid_thw must be provided together`；改用默认 1 → 更晚在 `_build_denoise_inputs` 报 `IndexError: tuple index out of range`；两次失败点不同 ⇒ 根因是「无序列并行」形态本身，非 tetp 取值））；`duration` 框架上限 [4,15] s、fps 固定 24。
  质量口径：预览级替换件须与原生解码器**同 latent 对拍**（灰度相关对尺度不变 ⇒ 判「接对与否」；
  PSNR 对丢高频的预览件不敏感 ⇒ **不可用于判接错**），外观看输出 std/mean/钳位像素占比。
  热漂移自查：同配置冷/热两窗比对（本例步速率与稳态 e2e 漂移均在噪声地板内）——
  有该证据才可裁重复次数，**因裁剪被清理掉的阶段拆分必须标「推算值」不得当实测引用**。
  开启方式见 `references/vllm-omni-train-aware-enablement.md`；通用方法见 `references/train-aware-lossy-method.md`。
- **DiffSynth-Engine（V2）**：compile = `compile_backend="mindie"` + `_compiled_call_impl` 原地写入 + backend 实例复用（勿重建）；Qwen-Image RoPE 实数域改写命中 `qwen_rope_pattern`；DiTCache 与 AttentionCache 互斥、CFG-on 有 shape 约束；attention 进图中性回退（不启用）。
- **LightX2V（L1）**：版本口径 = 上游合入版 **#1471** 机制（`COMPILE_BACKEND_REGISTER` /
  `hccl_eager` / rms·rope 注册表）；实测数据来自「合入前代码 + 本地镜像 #1471 等价机制」的远端
  （机制等价已验证，见归档 `{run_results_dir}/archive/lightx2v-mindiesd-case.md`）。
  compile = `use_compile:true` + `compile_backend:"mindie"` +
  `seq_p_a2a_backend:"hccl_eager"`（collective 留 eager 防 `hcom_alltoallv` 退化）；
  **量化格的准确含义（勿与已落地的原生 scheme 混读）**：上表 `W8A8_DYNAMIC` 行的 ❌ 指
  **docs `quantize()` 接口（online 动态量化）框架未接线**（仅 `dit_quantized` 预量化 ckpt 路，
  triton int8/fp8 GEMM 本机编译失败）；**`W8A8_MXFP8` 行已是 🟡**——已按**框架原生 scheme**
  `dit_quant_scheme="npu-w8a8-mxfp8"` 落地（内部直连 mindiesd `npu_quant_matmul`），
  步时下降约一成多（2× 复现）、帧 SSIM 近无损档（本组合观测，见 `lightx2v-enablement.md` §3.4）；
  mindiesd `quantize.py` W8A8_DYNAMIC 本机已具备、同 NPU 同族 harness 实证 → 该档阻塞可解
  （§11 P1，待 smoke）；缓存 ❌ = feature_caching
  对 H3 NotImplemented → 按新规 Cache 与框架对齐、mindiesd 不额外适配（原 cache_agent 接入转探针观察）
  （§11 P0）→ 时间步 24 步为现行采纳项，cache 叠加待验；SYS 跨岛并行选 head-parallel（UB 岛选
  bulk），compile × head-parallel 不兼容（Dynamo 静默回退）；TP2×SP2 实测可用但 UB 岛回退；
  RSP 无 Ring 实现（H3 SP 仅 Ulysses）。
  **跨框架差距与待补充能力 → 本表下方「跨框架待补充能力清单」+ `lightx2v-enablement.md`
  §3.4/§3.5**（2026-09 探针确认 mindiesd 侧 quantize 已具备，缺口=框架接线；Cache 不额外适配）。

### 跨框架待补充能力清单（框架侧缺口与状态；2026-09 提炼，只记能力/状态不记收益数字）

> 判据与档位定义见 §〇：**已支持** = 开关/配置即生效；**待配置** = 能力实体已具备、仅差框架接线；
> **待开发** = 能力实体不存在。本清单只登记「缺口 + 档位 + 落点 + 前置」，收益一律在任务内实测。

- **P0 缓存维度（DiTCache / AttentionCache）— 待配置**：框架侧 `feature_caching` 对 H3 显式
  NotImplemented；mindiesd `cache_agent/`（CacheAgent + DiTBlockCache + AttentionCache）已具备，
  且跨框架先例证明 **bench 侧零框架改动即可接入**（双缓存互斥、同开=fail-closed 假加速、
  计数契约 reuse/compute、CFG-on shape 约束）⇒ 「补缓存」= **接线工作**：先照先例在 bench 侧接入
  验证（注意 **compile 分段下 per-block 拦截点需先核验**），可行再 runner 级集成；与已采纳的
  时间步裁剪做叠加评估（S4-2 协议）。另：Cache 在新规下**由框架对齐承载、mindiesd 不额外适配**，
  原 mindiesd 侧接入方案转**探针观察**，不沉淀推荐姿势。
- **P1 线性层量化 seam（mindiesd `quantize()` online 路线：W8A8_DYNAMIC / W8A8_MXFP8）— 待配置**：
  能力实体（kernel + 量化线性层）在 mindiesd 已具备、同 NPU 同族 harness 已实证 ⇒ 缺口 =
  **框架侧接线**（编译前接入 online 量化），前置 = 本机 kernel smoke + compile 兼容 + 质量门禁。
  **状态更新**：已按**框架原生方案**落地（自研权重树无 `nn.Linear` ⇒ 注册原生 scheme 直连 mindiesd
  量化线性层）——**非 docs `quantize()` 接口，DYNAMIC 仍未接**。
- **P2 稀疏 / 注意力量化 seam — 待配置（eager 形态可用）**：eager 形态质量门禁通过、**可采纳**
  （质量优先选 bf16 稀疏、性能优先选 `mix` 量化路径）；**前置 = compile × rf3 的 Dynamo trace 期
  错误待解**（首步取证未完）→ 生产叠加需先解；另一路 rf_v2 在本平台的可用性**未核**。
- **P1 ACLGraph 生效核验 — 待核验**：docs 称 MindieSDBackend 内自动启用，框架 compile 流**未核验**
  ——若未生效可能为免费红利；需要 mindiesd 侧的**图捕获计数手段**才能判定。
- **P2 长序列通信重估 — 待重估**：comm masking 的收益上限 = 未重叠通信占比，而该占比**随序列变长
  上升** ⇒ 短序列档的否决结论**不迁移**到长序列档，须按目标负载重估（口径见
  `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §7）。
- **P3（硬前置 / 负证据）**：FA 量化（mindiesd FP8/MXFP8 FA 在**目标代际** **kernel 级微测可用**，
  端到端质量/收益**未验证**、框架未接线）；RSP（Ring 不支持 attn_mask，属上游限制）；
  CFG 并行（未核）。
- **移植纪律**：同一能力在不同框架的收益方向**可以相反**（例：compile 在一框架为负收益、在另一框架
  为正收益，源于架构 / 热路径差异）⇒ 本清单任何补充都必须在**目标框架本体**重验
  （kernel diff + rank0 墙钟 + 质量门禁），**不直接照搬其他框架数字**。

## 三、刷新协议（防表过期——本表硬配套）

1. **触发点**：每次模型优化进入 S0/S4/S5（训练感知——换入外部训练权重前先核该框架列是否已有
   蒸馏适配器 / VAE解码替换路径）或某特性使能失败时核对一次对应框架列；框架版本升级
   （vLLM-Omni / DiffSynth-Engine / LightX2V / mindiesd）；`docs/zh/features` 变更后
   （按新特性名对照本表行名）；新增框架经验回填（新的
   `-enablement.md` / `-notes.md` 或 §4 槽位回填）时。
2. **动作**：对照框架源码开关/env/接口 + 最小使能冒烟 → 更新单元格
   （`✅/🟡/❌/❓` + 证据码 + 版本 + 日期）；`❓` 升级为确定态须带证据码；同一框架列内出现
   不同基线的批次时标子批次码（如 V1a/V1b），勿混读。
3. **留痕**：表头按框架记录「版本 + last-checked」；状态变更在对应框架的 `-enablement.md`
   摘要段补一句（历史 case 内容已按 `.agents/README.md` §7 四分类迁移；存量 `-case.md`
   **受限保留**——登记并标注「不作为推荐加载入口」、实测数字出库 `{run_results_dir}/archive/`）。
4. **docs 联动（特性增长感知）**：mindiesd 新增/改名特性以 `docs/zh/features/*`（产品 docs，仓内直读
   真源）为准，先同步本表行名与状态，再回填 `dit-perf-opt` 侧的公开特性清单。
5. **机器化 seam 联动**：本表的状态/姿势与组合前静态判定用的机器声明
   `model-auto-optimization/scripts/feature_declarations.json` 互为印证——新增特性或改支持状态时，
   同步检查声明是否需要增行/改 seam（seam_check 语义以声明为准，本表供人读与框架列状态）。

## 四、与相邻能力边界

- mindiesd 自身 kernel/pattern 能力清单 → 槽位 S2-1（README §4）+ `docs/zh/features/*`（接口真源，仓内直读）
- 特性选档与组合 seam → `dit-perf-opt/references/combination-search.md`（组合前先按本表裁剪候选）
- 端到端质量判定 → `accuracy-gate/references/quality-gate.md` + 仓库 `evals/`
- 本表只登记「框架侧支持状态 + 姿势指针」，实测细节与回修回对应框架的 `-enablement.md`
  （开启方式/坑）与会话产物归档（数字）。

## 维护与更新

- 按 §三 协议每次刷新；docs 特性命名漂移时同步本表行名。
- 本文件被 framework-integration（Reference Files）与 model-auto-optimization（S1/S4）引用。
