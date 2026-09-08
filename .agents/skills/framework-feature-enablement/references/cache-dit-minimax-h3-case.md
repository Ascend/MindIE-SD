# 实例：cache-dit（框架本体 trunk）× MiniMax-H3 × vLLM-Omni 0.26 托管链（NPU 950PR）特性使能与档位实测

> 命名说明：文件名里的 `cache-dit` 是第三方框架仓库名（vipshop/cache-dit），**不是**固定特性名
> `Cache`（大写 C，见 model-auto-optimization `overview-report.md` §2.1 名词表）——本案例的
> cache-dit 是「被优化的框架本体」，Cache 特性由框架原生能力承载。
>
> 2026-09-06/07/08 实测。目标仓库：**cache-dit（vipshop/cache-dit trunk `51979f0`，fork main 已 reset 对齐）**——
> 本案例中 cache-dit 是「被优化的框架本体」，MiniMax-H3 因 diffusers 0.38 无对应 pipeline 类，由
> **vLLM-Omni 0.26** 托管（cache-dit 经 vLLM-Omni cachedit glue 消费；compile 走同栈 mindiesd `MindieSDBackend`）。
> 环境：{env_host} / 容器 {env_container}；torch 2.11.0+cpu + torch_npu 2.11.0 + vllm 0.26.0+empty +
> vllm-omni 0.26.0（editable）+ diffusers 0.38.0 + mindiesd（dev 同步重建，editable）+ cache_dit
> （trunk `51979f0`，`PYTHONPATH` 影子生效）。
> 硬件：Ascend 950PR ×4（128GB/卡，0–3 卡）；模型 {model_weight_dir}/MiniMax-H3/FL2VA（BF16，
> transformer 50 层 hidden 5376，13 分片）。
> 负载与口径：T2VA 1024×576/5s（124 帧@24fps）/50 步/seed1101/flow_shift12/audio_flow_shift3.0；
> e2e 主口径 = curl 请求墙钟中位（稳态 n≥2，排除 load/warmup）；**本文性能表述只给加速比与定性/
> 相对结论**（绝对秒数/绝对质量值在会话报表与 §6 耗时表，不入库）；同窗同卡组对比，<3% 噪声不宣称、
> 跨窗漂移 ±5%（09-08 补测窗 lossless 实测比 09-07 窗慢 ~8%，同窗对照必要）；质量 = ffmpeg psnr/ssim
> （vs 同窗 lossless 同 seed a/b 对，a/b 确定性一致）。

## 1. 环境要点（S0）

- **cache-dit 作框架本体**：本地 fork main reset 对齐 trunk（`git -c http.sslBackend=openssl fetch`；
  本地旧 NPU 提交已上游 PR #1004 合入并重构，勿再保留本地残留）；远端以
  `PYTHONPATH={repo}/cache-dit-trunksync/src` 影子生效（vllm-omni `cachedit` glue 兼容验过）。
- **mindiesd 同步重建**：`build_tik_ops` 行注释；陈旧 `build/plugin_build` cmake 缓存会漏编新源
  （先 `rm -rf`）；editable 用 `pip install -e . --no-deps --no-build-isolation`。
  ⚠️ **自研 CANN 算子部署**：mindiesd `build_ops.sh` 只把产物放到 `mindiesd/ops/vendors/*`，
  **不会自动装进运行 CANN**；运行期靠 `import mindiesd`（env.py）设置
  `ASCEND_CUSTOM_OPP_PATH={repo}/mindiesd/ops/vendors/…`——**必须先 import mindiesd 再初始化 NPU/
  建任何张量**，否则 GE 加载不到自研算子（`aclnnXxx … inferShape function does not exist`，
  如 EagleQBSA，见 §4.6）。部署校验跑对应 `tests/ops/*/…_golden.py`。
- **diffusers 0.38 无 MiniMaxH3 pipeline 类** → dummy-run harness 不可用，必须以 vllm-omni 托管；
  请求**必须带 `aspect_ratio=16:9`**（缺失 500）。
- **vllm-omni 补丁 2 处**（editable，.bak 留存，默认关）：① `regionally_compile` 加 `backend`
  参数透传；② diffusion_model_runner 在 env `OMNI_MINDIE_COMPILE=1` 时选 `MindieSDBackend` 并放开
  NPU inductor 跳过门控。
- 启动 recipe（4×950PR，0–3）：`vllm serve {model_weight_dir}/MiniMax-H3/FL2VA --omni --num-gpus 4
  --usp 4 --ring 1 --text-encoder-tp-size 4 --vae-parallel-mode tile --vae-use-tiling
  --vae-patch-parallel-size 4 --diffusion-attention-backend FLASH_ATTN`；
  多窗口墙钟漂移 ±5% → 对比须同窗交错、n≥2 中位。
- gloo 偶发 `ss1.ss_family == ss2.ss_family (10 vs 2)`（IPv4/IPv6 混用，容器 hostname 为空相关）→
  重试或强制 IPv4（`MASTER_ADDR=127.0.0.1 GLOO_SOCKET_IFNAME=lo`）后启动成功，非配置错。

## 2. 使能面速查（cache-dit × vLLM-Omni 0.26）

| 特性 | 触发 | 实测 |
|---|---|---|
| Cache（cache-backend cache_dit） | `--cache-backend cache_dit --enable-cache-dit-summary --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"residual_diff_threshold":R,"max_warmup_steps":4,"max_continuous_cached_steps":MC,"enable_taylorseer":true,"taylorseer_order":2}'` | DBCache F2/B1/R0.4/W4/MC4+TaylorSeer O2 采纳（**2.84×**）；档位扫描见 §3/§5 |
| 量化（线性层） | `--diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'` / `int8` | mxfp8/int8 接受并 serve（单点 ~1.15×）；fp8 ❓ 配置缺口（§4.5） |
| 稀疏 FA | RAINFUSION（eager，BF16）sparsity=0.8/start_step=12 | **kernel 级已确认**（见 §4.6）；e2e ≈1.15× 为该路径真实收益；**融合 op（EagleQBSA = 稀疏FA+FA量化）op 级快 ~6×，未接线**（见 §4.6） |
| kernel融合（compile） | env `OMNI_MINDIE_COMPILE=1` + `--diffusion-compile-granularity regional` → MindieSDBackend | regional 52×`MiniMaxH3DiTBlock`×4 worker 命中；无稳健收益 → **回退（默认关）**；residual_gate 误触已修（§4.1） |

## 3. 结果表（仅加速比 + 定性/相对；绝对数值在会话报表与 §6，不入库）

> 行=同窗对照（09-07 行 vs 窗 B lossless；09-08 补测行 vs 同窗 lossless，两窗差 ~8% 已注）。
> 稀疏行 = **eager rf_v3 公共 API 路径**（非 EagleQBSA 融合 op，见 §4.6）。

| 特性/组合 | 加速比 | 质量（vs lossless，定性） | 判定 |
|---|---|---|---|
| 基线（同拓扑未优化，无 DiT TP——框架未提供） | 1.00×（参照） | — | 基线 |
| **Cache** R0.4/MC4 | **2.84×** | 可接受（采纳口径） | **采纳（默认档）**；off-identity 成立 |
| Cache R0.4/MC6（速度 frontier） | 3.16× | 略降（相对 MC4） | 速度候选（视觉复核后可切部署档） |
| Cache R0.4/MC2（质量 frontier） | 2.12× | 更高（相对 MC4） | 质量优先候选 |
| 量化 mxfp8 单点 | 1.15× | 较好（优于 int8） | 单点有损（可叠加） |
| 量化 int8 单点 | ~1.15× | 一般 | 单点（质量劣于 mxfp8） |
| 量化 mxfp8 + Cache(MC4) | 3.13× | ≈Cache 单点 | 速度组合候选 |
| 稀疏 0.8/start12（eager rf_v3，复核行） | **1.15–1.25×**（同窗 1.25×） | **近无损（复现一致）** | 正确性坐实（kernel 级，见 §4.6）；单点收益有限 |
| `[补]` C1 稀疏×Cache(MC4) | **3.4–3.7×**（同窗 3.7×） | ≈Cache 单点（不额外损） | **叠加有效**：Cache 上再省 ~17% 时间 |
| `[补]` C2 mxfp8×稀疏 | **1.5–1.6×**（同窗 1.6×） | ≈mxfp8 级 | 量化×稀疏协同（快于两个单点） |
| `[补]` C3 mxfp8×稀疏×Cache | **4.1–4.5×**（同窗 4.5×） | ≈Cache 级 | **三元叠加≈最强速度组合（cache 主导）** |
| `[补]` C4 int8×稀疏×Cache | **4.2–4.5×**（同窗 4.5×） | ≈Cache 级 | ≈C3 |
| `[补]` P0 稀疏 0.8/**start0** | ~1.3× | **明显劣化（不推荐）** | start_step=12 保真旋钮必要（首 12 步 dense） |
| kernel融合（MindIE compile） | ≈0.96–0.99× | 非逐字节（图级微差） | **回退**：受控交错 A/B 相对 eager ≈-1%，3% 噪声内无稳健收益 |

## 4. 回修与坑（核心增量）

### 4.1 compile 泛型 pattern 误触（本案例最重要回修）

- **症状**：开启 compile 后每步 100 次 `residual_gate_add`（修复前开销显著），替换对象不是真实 gate。
- **定位**：DOT 图/pattern 图 dump + `kernel_details.csv` → 命中点是
  `fused_qk_norm_rope.py` RoPE 旋转段 `add(mul,mul)`（q、k 各 1 节点/block），由 `wan_residual_gate`
  **泛型 pattern 抢跑**（H3 rope pattern 未命中）。
- **处置**：mindiesd `compilation/passes/__init__.py` 移除 `enable_wan_residual_gate` /
  `enable_minimax_h3_gate` 注册（`compiliation_config.py` 默认 False；pattern 文件与 layer 保留）。
- **验证**：移除后 rga kernel=0，compile ≈ eager（同窗实测，噪声内）→ 仍无稳健收益，
  compile 维持回退（默认关）。
- **教训**：regionally 使能 compile 后必须做 kernel 级「预期融合 vs 实际命中」核对（三层证据：
  日志 → DOT 图 → kernel csv）；**泛型 pattern 会跨算子族误触**（rope-gate 抢 RoPE 旋转段）；
  命中 ≠ 收益——命中点若不是真实融合目标，收益归因即失真。

### 4.2 compile 第二请求变慢 ≠ recompile 风暴

受控交错中 compile 第 1 请求更快而第 2 请求变慢（两实例一致）→ `TORCH_LOGS=recompiles`：
**recompile 有界、guard 主变体 `rope_table is None`**（token_refiner 无 rope vs DiT 有 rope）→
非每请求重编译风暴；第二请求变慢归因噪声。

### 4.3 eager vs compile 非逐字节

compile 输出与 eager 非逐字节一致（图级数值微差）→ compile 变更**不得宣称无损**
（与 qwen-image 案例同判据）。

### 4.4 计数契约缺口（cache_dit 统计属性未填充）

cache_dit `summary.py` 暴露 `_cached_steps/_accumulated_*` 等逐步计数器，但该 trunk 在 vllm-omni
视频路由**未填充**（需 glue 补 cache_summary，改三方仓需确认）→ 用 **kernel 级计数契约兜底**：
cached 步 317 kernels（FA 54→7、Matmul 208→20、RMSNorm 110→16）vs eager 步 3157 → 每缓存步跳过
**~90%** block 计算（kernel 计数为真实参与证据，防 no-op 假加速）。

### 4.5 fp8 量化配置缺口

`TypeError: Cannot instantiate AscendFp8Config with kwargs {}`（需 ignore/quant_format 等参数）→
框架配置面缺口，留证不虚列（与报表 ❓ 一致）。

### 4.6 稀疏 FA：两条路径的本质区别（2026-09-08 补测核心）

#### 路径 1：eager rf_v3（vllm-omni RAINFUSION 后端现走路径）——「能生效但收益有限」

- 0.26 后端只调 `mindiesd.sparse_attention(sparse_type="rf_v2")`；950PR（soc_version=260 属 mindiesd
  A5 类）→ **rf_v2 被自动路由到 rf_v3**（`aclnnBlockSparseAttentionV2`，inner_precise 强制 4）。
- 配置语义：`sparsity` = 每 query block **丢弃** key block 名义比例（mindiesd `keep_len=ceil(cols×(1-sparsity))`，
  内容相关 mask：pooled q/k 相似度 softmax→topk→阈值，**每层每步重建**）；`start_step` = 前 N 步 dense；
  `skip_layers` 豁免指定 block。
- **kernel 级生效坐实**（同窗稀疏步 step20 采集）：dense FA **54→4** + `BlockSparseAttentionV2` **×50**
  （DiT self 50 站点）；token_refiner 无 video 段 → staying dense（设计使然，成本小）；host/kernel 间隙 ≈0
  （step_trace Free 5–6ms，步墙钟≈device Stage）。
- **收益边界**：mask/几何构建开销大——mask 选择逻辑本身（topk/softmax/阈值/首帧保护）≈3ms 可忽略，
  **大头是每层每步全尺寸 rearrange/pool 数据搬运**（cat/transpose/cast/mean 等 ~220ms/步，50 层重复，
  COPY/MOVE 767→2017、总 kernel 3157→5707）→ 稀疏步仅省 ~20%（1.66→1.33s），50 步中 start_step 后
  才稀疏（38/50）→ e2e ≈1.15× 真实上界；**op 级单调用该路径甚至慢于 dense FA（+18%）**。
- quality 近无损（vs lossless，同窗复现一致）与「丢 80%」并存的解释：结构感知掩码 + 5s 短片段帧间冗余
  与 prefix/first-frame 恒保留 → realized sparsity < nominal。
- 组合结论：稀疏作为**步级跳过外的"算步加速"叠加层**在 Cache 上有效（C1 ≈3.7×，质量≈Cache）；start0
  不可取（质量显著劣化）。

#### 路径 2：EagleQBSA 融合 op（稀疏FA + FA量化 = 0.28 的 "mix"）——op 级 ~6×，当前未接线

- mindiesd 自研算子（`csrc/ops/eagle_quant_block_sparse_attention` 全链 op_host/op_api/infershape；
  Q/K per-block INT8 + V per-channel FP8 + block_sparse mask，一 op 内 mask+BSA 融合）。
- **部署坑（本案例踩到）**：op 包只构建不安装进运行 CANN → GE `inferShape function does not exist`；
  修复 = `import mindiesd`（设 `ASCEND_CUSTOM_OPP_PATH`）**先于任何 NPU 张量/初始化**；golden
  （`tests/ops/eagle_quant_block_sparse_attention/…_golden.py`，EB≤1e-2）通过为校验标准。
- **op 级微基准（同 serve 几何 S=21767/block128/sp0.8，1 卡）**：dense FA 为基准 → EagleQBSA(0.8)
  **≈0.16–0.17×（快 ~6×）**；EagleQBSA 全保留 mask（只量化不稀疏）也 **≈0.6×**（INT8/FP8 计算即快，
  cos≈0.9996 量化级精度）；rf_v3 eager 路径 ≈1.17×（最慢）。
- **教训（收益归因）**：eager 公共 API 的稀疏（外部逐层 mask）与融合 op 的稀疏（mask+BSA 一体化）是
  两条不同的"稀疏"，墙钟口径不同不可混称；要「稀疏+量化」收益应优先融合 op 路径；0.26 vllm-omni
  RAINFUSION 后端无 quant/mix 接线（0.28 fork 有 `precision=mix`）→ **接线（framework-extension-dev）
  后测 e2e 组合为待决项**，op 级证据（~6×）已固化。
- **宣称纪律**：稀疏行收益必须有本步 kernel 计数证据（dense FA 数下降 + 稀疏 kernel 出现）或 op 级
  对拍（同几何同 sparsity），fail-closed；无证据不宣称「稀疏生效」。

### 4.7 通用坑速查

CRLF（上传后 `sed -i 's/\r$//'`）；嵌套引号吞参数（上传脚本执行）；日志含 ANSI/emoji → grep 前
`strings`/专用脚本；8 卡中 7 卡 Health=Alarm 系常态（用 0–3 前 npu-smi 实测可用）；容器内 /tmp
与宿主 /tmp 不同（docker cp 取文件）；质量门禁无 VLM → **inconclusive** + 并排帧存证（pair_*.png）。

## 5. 方法论（增量可复用）

- **cache 档位扫描姿势**：同窗逐档（每档同窗 lossless_r + cache×2，50 步）；单变量只动 R 或 MC
  （F2/B1/W4+TS O2 固定）；质量 = cache vs 同窗 lossless 的 ffmpeg a/b 对。**辨识主导旋钮**：
  本负载 **MC（max_continuous_cached_steps）主导**——MC6 更快（时间 -10% 量级、质量略降）、
  MC2 更慢但质量更高（时间 +34% 量级、质量 +2.8% 量级，均相对 MC4 同窗）；**R 在 0.3–0.6 平坦**
  （决策/输出一致）、0.2 过紧（少命中、变慢）→ 采纳 R0.4/MC4。
- **叠加组合（补测轮）**：cache 主导 + 稀疏/量化作"算步加速层"；行=同窗 lossless_r 括窗（批首/批尾
  各 1 次 lossless 消除窗漂移）；每 serve w10+a/b×50（n=2 稳态）；质量用 a/b 与 loss_r（确定性输出
  跨 serve 可比）；**组合 kernel 参与证据** = serve 日志 `RAINFUSION_ATTN active` 计数 + kprof 抽样步。
- **compile 收益判定**：受控交错 A/B（同卡组同日 C1→E1→C2→E2，n=4 中位）消除热漂移；<3% 噪声
  阈值内不宣称；交错同时给出 eager 对照中位。
- **kernel 级计数契约采集**：torch_npu Level1 + `analyse()` → `kernel_details.csv`；经 env hook
  （如 OMNI_KPROF，ordinal 需覆盖 warm 步数 + 目标步）在 rank0 第 N 次 `MiniMaxH3DiTModel.forward`
  采集，无需改仓库；shim 内 analyse 报 "daemon 不可解析" → **离线 analyse 独立进程跑**。
- **op 级微基准（无模型，1 卡）**：同 serve 几何（S/txt_len/latent/sparsity/head）对拍 dense FA /
  eager 稀疏路径 / 融合 op，warm+稳态取 ms；**先 import mindiesd 再建 NPU 张量**（自定义 op 注册前置）。
- **质量门禁**：视频整片段 psnr/ssim 天然偏低（混沌轨迹）→ 不设绝对阈值；视觉门无 VLM →
  inconclusive + 并排帧存证；同 seed a/b 对为可比前提（a/b 确定性一致可复验）。
- **off-identity**：`quality=lossless` 关 cache 复现基线；compile env=0 复原 eager（跨进程
  framemd5 逐帧一致）。
- **时间盒纪律**：主导旋钮辨识后收敛；未尝试候选（Fn/Bn、sp0.9/0.95、EagleQBSA 接线、fp8 修、
  逐步计数 glue）均入未尝试清单，不静默删项。

## 6. 耗时与卡数（实测；供后续评估与 effort 回填——绝对耗时按 §3 口径保留于此）

> 口径：全部 4×950PR（0–3）、单并发请求、稳态（warm/首请求单独注明）；serve 墙钟 = 远端 serve 日志
> 首尾时间戳跨度（含启动+load+请求+kill，driver 编排等待为上限口径）。

| 运行项 | 实测 | 备注 |
|---|---|---|
| 模型加载 | 84.82 GiB/rank ×4 ≈ **50–61 s**（量化分片档 54.95–66.11 GiB → 48–57 s） | 每 rank 并行读盘，与卡数弱相关 |
| 启动→health 200 | ≈2–5 min（load ~55s + 引擎 init）；compile 首启更长（含 lazy prime） | kprof compile serve 实测 ~15.5 min（含首请求 prime，上限） |
| 10 步 warm/smoke | 18.6–38.2 s（compile smoke10 38.2 s 含 lazy compile prime） | 候选快筛量级 |
| 50 步稳态请求 | lossless 87.1–98.6 s；cached 28.7–33.9 s（按档）；量化/稀疏单点 77–81 s；量化×稀疏 ~61 s；含 Cache 组合 21.8–32.1 s（稀疏×Cache 26.5、量化×稀疏×Cache 21.8–22.1、量化+Cache 29.2） | curl time_total 全量实测 |
| 单配置 serve 墙钟 | 档位/组合扫描每 serve ~4–6 min（含 w10+a/b×50+kill）；量化/稀疏档 ~4.1–4.3 min；计数/kstep ~4 min；kernel 采集档 2.5–15.5 min；graph dump ~23.4 min（上限） | ≈ load 1min + init 1–3min + warm 1 请求 + 稳态 n×e2e + kill |
| NPU 占用核算 | 单日（09-07）主要 serve ≈ 4 卡 × 2–2.5 h ≈ **8–10 卡·时**（上限口径）；09-08 补测（复核+4 组合+start0+双 lossless）≈ 4 卡 × 1.2 h | 正式/扫描/补测多日另计 |

**卡数 ↔ 耗时关系（注意，勿线性外推）**：

- 本栈 vllm-omni 0.26 DiT 仅 **USP 形态（无 TP）**：单请求 e2e 与「卡数」无线性关系（步数串行 +
  单并发）；卡数↑ 主要扩**并发容量/显存余量**（84.82 GiB/rank < 128 GB/卡），对单请求 e2e 无承诺。
- 模型加载 ~55 s 为每 rank 并行读盘，与卡数弱相关（磁盘/网络读速主导）。
- 跨卡数对比必须**同并行形态**（0.28 案例：TP2(2卡) 与 TP2×USP2(4卡) 同负载差异为并行形态改变，
  非纯卡数效应）；本任务无 2 卡同形态数据 → 不宣称。
- 后续评估建议：单配置墙钟预算 ≈ load 1min + 引擎/图 init 1–3min（compile 首启 5–15min）+
  warm 1 请求（0.4–1.6min 视档）+ 稳态 n×e2e + kill/清理；NPU 成本 = 卡数 × Σ serve 墙钟。

## 7. 结论与建议部署

- **采纳（默认）**：Cache R0.4/MC4（2.84×，质量可接受，kernel 计数契约成立）。
- **速度组合候选（09-08）**：三元 mxfp8×稀疏×Cache / int8×稀疏×Cache（4.1–4.5×，质量≈Cache 级）与
  sparse×Cache（3.4–3.7×）——Cache 主导，稀疏在 Cache 上再省 ~17% 且不额外损质量。
- **稀疏单点**：eager rf_v3 路径 ≈1.15–1.25× 近无损（正确性已坐实），仅质量优先场景可用；start0 不可取。
- **EagleQBSA（稀疏FA+FA量化）**：op 级 ~6× 证据已固化、部署坑已修复；0.26 未接线 → 接线测 e2e
  组合为待决项（framework-extension-dev，需确认）。
- compile 维持回退（默认关）；residual_gate 泛型误触已修（注册移除、代码保留）。
- 计数契约：逐步 computed/cached 属性未填充 → kernel 级契约兜底已补齐；改 vllm-omni glue 出
  cache_summary 属三方仓改动，需确认后做。
- 报表/证据为会话产物（overview/detail/final/evidence + consolidated_summary + 帧/mp4 + kprof csv +
  graphdump，不入库）。
