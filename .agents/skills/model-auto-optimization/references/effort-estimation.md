# 模型优化耗时评估参考（Effort Model）

> 用途：model-auto-optimization §0 启动确认（时间盒/预算）、S1 融合与 S4「候选排序：低成本高概率优先」、
> 以及每阶段计划时，用本表估算「某个优化点做下来要多久、要跑多少轮」。
> 样本：
>
> - **V1（视频长任务）**：MiniMax-H3 × vLLM-Omni 0.28 / 950PR×4，2026-09-05/06 无损+有损全链复盘
>   （case 见 framework-feature-enablement/references/vllm-omni-minimax-h3-case.md；576p/60 步/118s 级）。
> - **V3（图像短任务）**：Qwen-Image-2512 × vLLM-Omni 0.28 / 950PR×1-2，2026-09-05/06
>   （case 见 framework-feature-enablement/references/vllm-omni-qwen-image-case.md；1024²/20 步/秒级）。
> - **V4（视频长任务 · cache-dit 框架本体托管链）**：MiniMax-H3 × cache-dit trunk 51979f0 × vLLM-Omni 0.26
>   / 950PR×4，2026-09-06/07（case 见 framework-feature-enablement/references/cache-dit-minimax-h3-case.md；
>   1024×576/50 步/91.4s 级 lossless；DiT 仅 USP 无 TP，e2e 不随卡数线性）。
> ⚠️ **按任务类型分档预测（视频 ≠ 图像；短任务 ≠ 长任务）**：勿用视频分钟级外推图像秒级（图像 20 步
> 单请求实测 4.3-6.2s，远小于按视频 60 步 118s 比例外推的分钟级）。区间均为**样本估计**（以远端
> 运行/等待为主），随新 case 按 §4 回填修正；不迁移机型/框架/任务类型（换平台先做 1-2 次快测 smoke
> 校准「服务启动 + 单请求 e2e」再放大预算）。

## 1. 运行成本口径（实测样本，可直接套用估算）

### 1a. 视频长任务样本（H3 576p/60 步；950PR×4 TP2×USP2，lossless 118s 起）

| 运行项 | 实测 | 备注 |
|---|---|---|
| 服务启动→health 200（4 卡 load，~135G bf16） | **≈2-5 min**（poll 20s×~6-15） | compile 档首启更长（prime）；含 text 编码器等加载 |
| 10 步 smoke 请求 | **≈17-28s/请求**（各档） | 用于候选快筛（S1/S3 矩阵/AB） |
| 60 步请求（576p，lossless 118s 起） | **30-243s/请求**（cache/量化档 30-40s，无损 118s） | 有损稳态行必须 60 步（或工作负载同款步数） |
| 768P50 请求 | 42-243s/请求 | 高分负载：每请求 ≈ ×1.8-2.1 vs 576p60 同档 |
| 单配置墙钟 ≈ | 启动 + 请求数×e2e + 采集开销 | 稳态行 2-3 请求 + 启动 ⇒ **8-15 min/配置**（60 步）；10 步档 **6-10 min/配置** |
| 整批 5 配置（768P，3 请求 + kprof/bprof） | ≈35 min | 5×7min（含采集+analyse） |
| 2-3 配置补测批（含 hq 档） | ≈10-15 min + 后处理 2-3 min | 见 09-06 cache 单点补测 |
| 零代码 AB（同栈两配置，10 步 + 单步 kprof） | ≈10-12 min + analyse ~1-2 min/配置 | 例：O6 compile×int8 重测 |
| 单步 kernel 采集+analyse | 采集含在运行；analyse ~1-3 min/配置 | OMNI_KPROF + torch_npu analyse |
| hccl 带宽等价工具（torchrun 4 rank，9 档×2 op） | ≈2-3 min/卡组 | 先 set_device 再 init |
| 质量补测（存量 mp4 抽帧+quality_compare，1 对） | ≈2-3 min/对 | 有帧则不必重跑（quality-gate.md） |

### 1b. 图像短任务样本（Qwen-Image-2512 1024²/20 步；950PR×1-2，秒级；V3 2026-09-05/06）

| 运行项 | 实测 | 备注 |
|---|---|---|
| 服务启动→health 200（1-2 卡 load，~55G bf16） | **≈1-2 min**（poll 20s×~3-5） | 39G transformer 9 分片加载 ~10-40s + 引擎 init |
| 单请求 e2e（20 步，稳态 r1-r3） | **TP1 6.05-6.21s / TP2 4.96s / TP1×USP2 4.35s**（w8a8 4.33 / +Cache 3.77s） | DiT 单步 246-310ms 主导（TP1 299ms、TP2 246ms）；warmup 首请求（含图/编译预热）~5-10s |
| 10 步 smoke 请求 | **≈2.7s/请求**（lossless） | 快筛量级秒级（视频 10 步 17-28s 的 ~1/7） |
| 21-seed 质量集（每 seed 1 请求，稳态） | ≈21×5s ≈ **2-3 min** + 启动 | 图像质量取样 = 固定 21-seed 同 seed 像素对（evals/profiles/README 约定） |
| 单配置墙钟 ≈（性能行） | 清理/探活 ~1min + 启动 ~1-2min + 3-4 请求 ~20s + kill | **≈3-5 min/配置**（20 步）——含多租户卡组探活与残留 worker 清理 |
| 单步 kernel 采集+analyse | 采集含在运行；analyse ~1-3 min | OMNI_KPROF_TARGET=qwen_image（fork hook 扩展） |
| 组合/对照批（3-6 配置 + 质量集） | ≈25-40 min | 同窗相邻对 + 21-seed 质量 |

### 1c. 视频长任务样本（cache-dit 框架 × vLLM-Omni 0.26 托管链；950PR×4；1024×576×50 步；V4 2026-09-06/07）

> cache-dit 为框架本体（trunk 51979f0），H3 由 vllm-omni 0.26 托管；DiT 仅 USP 形态（无 TP）。
> 完整数字与卡数关系见 case §3/§6；单配置 serve 墙钟 = serve 日志首尾跨度（含 load+请求+kill，
> driver 等待为上限口径）。

| 运行项 | 实测 | 备注 |
|---|---|---|
| 模型加载 | 84.82 GiB/rank ×4 ≈ **50-61s**（量化分片档 48-57s） | 每 rank 并行读盘，与卡数弱相关 |
| 服务启动→health 200 | **≈2-5 min**（load ~55s + 引擎 init） | compile 首启更长（lazy prime；kprof compile serve 实测 ~15.5 min 上限） |
| 10 步 smoke/warm 请求 | **≈18.6-38.2s**（compile smoke10 38.2s 含 lazy compile prime） | 候选快筛量级 |
| 50 步稳态请求 | lossless 87.1-98.6s；cached(R0.4/MC4) 31.5-32.8s、MC6 28.7s、MC2 42.7-43.0s；mxfp8/int8 ≈78-81s；mxfp8+Cache 25.4-32.9s；稀疏0.8 77-81.6s | curl time_total 全量实测 |
| 单配置墙钟 | 档位扫描每配置 ≈4-4.5 min（serve 首尾 4m16/4m21，含 lossless_r+cache×2）；量化/稀疏档 ≈4.1-4.3 min；计数/kstep ≈4 min；kernel 采集 2.5-15.5 min；graph dump ~23.4 min（上限） | ≈ load 1min + init 1-3min + warm 1 请求 + 稳态 n×e2e + kill |
| NPU 占用核算 | 单日主要 serve ≈4 卡 ×2-2.5h ≈ **8-10 卡·时**（上限口径） | 正式/扫描/补测多日另计 |

- **卡数 ↔ 耗时关系（勿线性外推）**：vllm-omni 0.26 DiT 仅 USP 无 TP → 单请求 e2e 与卡数无线性
  关系（步数串行 + 单并发）；卡数↑ 扩并发容量/显存余量（84.8GiB/rank < 128GB/卡）；load 与卡数
  弱相关；跨卡数对比须同并行形态（0.28 案例 TP2 2卡 36.0s vs TP2×USP2 4卡 19.1s@10 步为形态改变）。
- 后续评估预算公式：单配置墙钟 ≈ load + init(1-3min，compile 5-15min) + warm 1 请求(0.4-1.6min) +
  稳态 n×e2e + kill；NPU 成本 = 卡数 × Σ serve 墙钟。
- **组合批成本（V4 补测实测，2026-09-08）**：复核+4 个叠加组合+start0 探针+双 lossless 括窗（8 个
  serve，每 serve w10+a/b×50）≈ 4 卡 × 1.2h；每 serve ~4-6 min；稀疏正确性（kprof 单步+离线
  analyse）另 +~10 min；组合质量对（ffmpeg psnr/ssim 每对两次独立跑）~2-3 min/对——叠加矩阵按
  「serve 数 × 5 min + 质量对 × 3 min」预算。

### 1d. 视频长任务样本（同环境双树 · 显式 SDPA 基线口径；950PR×2 TP2；576p/60 步；env A 2026-09-07）

> 与前几节不同处：本样本基线 = **显式 TORCH_SDPA**（安装 editable mindiesd 的宿主默认会把
> attention 路由到 FLASH_ATTN，冻结"未加速对照"必须显式指定并核对 resolve 日志，见
> framework-feature-enablement/references/troubleshooting-vllm-omni.md §P0）；共享宿主多租户热节流
> 显著（同档 e2e 高 20-100% 的异常窗剔除后取 r1/r3 稳定对）。产物
> runs/20260907_minimax-h3_optimization/ + agentic/（run-state/stage_gate）。

| 运行项 | 实测 | 备注 |
|---|---|---|
| 服务启动→health 200（2 卡 TP2，~135G bf16 load） | **≈2-5 min** | 与 4 卡同量级（load 与卡数弱相关） |
| 60 步稳态请求（576p，显式 SDPA 基线） | 各档 e2e 跨度 **83.4-436.7s**：SDPA 436.7s（r1 444.4/r3 429.0）；FA 360.7s；mxfp8+FFN-MX 304s（Bon，unfused ~326s）；稀疏 rf_v2 0.8 222.7s；Cache 83.4s；组合(mxfp8+FFN-MX+Cache) 92.9s | 有损稳态行必须 60 步；同窗相邻对 + 反转 AB 防漂移 |
| 单配置墙钟 ≈（60 步） | 快档（cache/量化）≈5-8 min；慢档（SDPA 基线 436.7s）≈10-15 min | 启动 2-5min + 1-2 稳态请求 + kill；异常窗剔除不重复跑 |
| 质量补测（存量 mp4 抽帧 + quality_compare，1 对） | ≈2-3 min/对 | 有帧则不必重跑（quality-gate.md） |

- 本节数值口径 = 2 卡 TP2、单并发 60 步 curl 墙钟、r1/r3 稳定对、热窗剔除留痕；跨环境对比注意
  基线 backend 语义（FA 默认路由 vs 显式 SDPA 起点不同，比例不可混用）。

**外推启发（按任务类型双轨）**：

- 单请求 e2e ≈ 步数 × 单步时长 + 固定开销（VAE decode + HTTP ~0.2-0.5s，图像）；
  单步时长 ≈ f(tokens×分辨率, 并行/量化/缓存)——**先用 1-2 次快测 smoke 实测单步/单请求，再按步数放大**
  （视频 60 步 118s 的「单步 ≈2s」不可外推到图像 20 步 6s 的「单步 ≈0.3s」——任务性质不同）；
- 图像（DiT 主导、秒级）：同负载量化 int8 ≈ -13%、Cache ≈ -14%、2 卡 USP2 ≈ -12%（vs TP2）、
  组合 ≈ -24%；4-rank 并行在短任务为**负**（通信病态）；
- 视频（同 H3）：量化 int8 ≈ -15-18% 步时、稀疏 mix ≈ ×0.45、cache 命中 ≈ ×0.3-0.6（按档）；
- 上述比例供估算，正式结论仍以同窗实测为准。

**无损链路评估提速（省时手段；规则见 lossless-methodology-notes §F）**：同 §1a（10 步快测 →
dummy-run → 全步长）；图像任务快测量级为秒级（单候选分钟级可判），视频为 10s 级——时间盒按任务类型缩放。

## 2. 每优化点：投入档（实施+验证）与收益参考（H3 样本；图像参考见 V3 case）

> 投入 = 人工/机器混合耗时估算（档位）；收益参考列 = 该点在 H3（576p@60，vs 无损 118.0s）的实测收益，
> 用于「性价比」排序（低成本高概率优先）。实施类型：`开关`=现有能力使能；`fork`=三方框架补丁；
> `实现`=算子/机制开发（天级）。
> **图像短任务（V3，Qwen-Image-2512 1024²/20 步）收益参考**：并行 TP1×USP2 -30%（vs TP1）、TP2 -20%；
> 量化(w8a8) -12.7%、Cache 单点 -13.7%、量化+Cache -24%（vs TP2 4.96s）；质量 33.9-37.5 PSNR /
> 0.960-0.986 SSIM（21-seed 像素对，域远高于视频 → 阈值不迁移）；稀疏图像不可用（staying dense）；
> 4-rank 并行短任务为负（通信病态）。验证运行量按 §1b（3-5min/配置）估，投入档整体比视频行低 1 档。

| # | 优化点 | 实施类型 | 验证运行量（典型） | 估算投入 | 收益参考（H3 实测） |
|---|---|---|---|---|---|
| 1 | S0 栈补齐（装 mindiesd / 中继 fork / 权重自检） | 传输+安装 | 0 运行（import 验证） | 0.5-1.5h（wheel 现成）；重建 1h+ | 前置（无收益） |
| 2 | S1 使能 FA / 单算子（SDPA→FA） | 开关/路由 | 2 配置×10 步 + 单步 kprof | 1-2h | -11.5%@10 步 |
| 3 | 并行选型（TP/TP×USP/ring 矩阵） | 开关 | 4-7 配置（10 步快筛 + 60 步定档） | 3-6h（含 ring 不支持等排障） | USP2 -47%@10；60 步成基底 118.0s |
| 4 | Offload 解锁（DLO AllGather） | 开关 | 3-4 配置（60 步） | 1-2h | -3% 无损 + 解锁 USP4（有损档变体 85.1/52.05s） |
| 5 | compile 探针（MindIE） | fork(首建)+开关 | 2 配置×10 步 + kprof | 1-2h（首建 fork 数小时） | 负 → 回退记录（bf16/w8a8 均负） |
| 6 | 量化 w8a8（线性层 INT8 online） | 开关 | 2-3 配置×60 步 + 计数 | 1-2h | -18.3%；质量最轻（19.6/0.679） |
| 7 | 稀疏（rf_v2 逐档 + seam） | 开关 | 每档 1 配置×60 步（0.8/0.6/start_step…） | 1-3h | -23%…；0.6 非线性 → 回退教训 |
| 8 | Cache（先单点，再档位+组合） | 开关 | 单点 1-2 配置 + hq 1 配置 + 组合 2-3 配置 | 1-2h（不含最初实现） | 单点 -67.5%；组合 frontier -76.1%（28.2s） |
| 9 | 组合矩阵 + frontier/回退 | 开关 | 6-10 配置×60 步（同窗对；**必测覆盖集 [MUST] 两两 3 档 + 三元 1 档优先，再自由候选**） | 2-4h（以运行/等待为主） | frontier 28.2s；hq 35.1s（档位权衡） |
| 10 | 质量门禁/补测（绝对口径） | 后处理 | 抽帧+compare 2-3min/对；新档需重跑 | 0.5-1h/批 | 产出质量列（vs lossless 绝对值） |
| 11 | hccl 带宽（等价工具） | 脚本 | 1 组≈2-3min | ~0.5h（脚本复用后分钟级） | 通信健康判定（A.2.4） |
| 12 | 量化后融合重审（O6 类 AB + 邻接分析） | 零代码 AB | 2 配置×10 步 + analyse | ~1h/候选 | 判定采纳/回退（O6 回退记录） |
| 13 | 双报表+evidence+回填（matrix/case） | 文档 | 0 运行 | 首建 2-4h；复用模板 <1h | 强制交付（不可省） |

**候选排序经验**：先做「开关类+快测」候选（每候选 ~1-2h 可判采纳/回退，如 O6/带宽/10 步矩阵），
再做「运行重」候选（60 步稳态，按 §1 每配置 8-15min 估），最后做「实现类」（天级，需用户确认）；
一次失败/不支持（ring/compile/FA8）也要记录并计入排障预算（×1.5-2 缓冲）。

## 3. 阶段时间盒建议（样本估计；一次通过 vs 含排障）

| 阶段 | 一次通过（视频 / 图像短任务） | 含排障（常见） | 主要耗时构成 |
|---|---|---|---|
| S0 环境/权重/首跑 | 0.5-1.5h / 0.5-1.5h | 2-4h | 栈补齐、权重、首跑排障（共享宿主多租户卡组探活另计） |
| 基线冻结 + 单步 profiling | 1-2h / 0.5-1h | 2-3h | 稳态行 + analyse（图像请求秒级 → 更快） |
| 无损链（FA/并行/offload/compile） | 3-6h / 2-4h | 6-10h / 4-8h | 矩阵运行（视频 10/60 步；图像 20 步 3-5min/配置）+ 排障 |
| 有损链（量化/稀疏/cache 各档） | 3-6h / 2-4h | 6-10h / 4-8h | 每档配置 + 组合 + 21-seed 质量集（图像 2-3min/配置） |
| 质量门禁 + 组合收口 | 1-3h / 1-2h | 2-5h | 抽帧/compare 或 21-seed 像素对 + 同窗对 |
| 双报表 + evidence + 回填 | 1-2h / 1-2h | 2-4h | 文档（模板可复用；图像 case 模板已备） |
| **全链合计** | **≈1-1.5 天（视频）/ ≈4-8h（图像，秒级请求 + 3-5min/配置）** | **≈2-3 天 / ≈1-1.5 天** | 远端运行/等待占 40-60%；图像任务受共享宿主卡组扰动/探活影响占比更高 |

> 图像任务实测（V3，Qwen-Image-2512）：S0-S4 核心行 + 补测（2 卡 USP2、稀疏小档）+ 21-seed 质量
> ≈ 一轮会话 NPU 运行 ~3-4h（含多次多租户排障/探活重试）；纯远端运行成本 <1.5h（case §3b/§5）。

## 4. 校准与维护

- 每个新 case（Qwen-Image×vLLM-Omni、DiffSynth、LightX2V、cache-dit×vLLM-Omni…）在闭环后按 `.agents/README.md` §7 回填
  实际投入（运行轮次、每配置时长、排障点），修正本表区间与 §1 外推系数；**按任务类型分行登记
  （视频长任务 / 图像短任务），禁止跨类型迁移时间盒**；
- V4（cache-dit×vLLM-Omni 0.26，2026-09-06/07）已按本协议回填 §1c：与 V1（vLLM-Omni 0.28 宿主）
  同为视频长任务量级（启动 2-5 min / 每配置 4-15 min），但宿主版本与并行形态不同（0.26 DiT 仅 USP
  无 TP；cache-dit 为框架本体）——**时间盒勿跨宿主版本/并行形态迁移**；
- env A（2026-09-07，同宿主双树 + 显式 SDPA 基线口径）已按本协议回填 §1d：2 卡 TP2、60 步各档
  e2e 83.4-436.7s；**基线 backend 语义是口径项**（editable mindiesd 下默认路由 FLASH_ATTN，须显式
  TORCH_SDPA 才能冻结未加速对照），且共享宿主热窗显著——对比必须同窗相邻对 + 异常剔除；
- 首轮跨机型/框架/任务类型：先做 2 次快测 smoke 校准「服务启动 + 单请求 e2e」，再放大到全链预算；
  - 校准模板：单请求 e2e ≈ 步数 × 实测单步时长 + 固定开销；启动 ≈ f(权重体积/卡数)；
  - 图像短任务示例：TP1 单步 299ms×20 ≈ 6.0s + decode/HTTP ≈ 0.2s ⇒ 6.21s；TP2 单步 246ms ⇒ 4.96s；
- 共享机排队/他人占用不计入（先 npu-smi 查卡 + 同岛拓扑）；耗时口径均指远端同窗同卡组稳态行。
