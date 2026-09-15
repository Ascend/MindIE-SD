# cache-dit（框架本体 trunk）× vLLM-Omni：特性开启方式（框架差异记录）

> 定位：本文件只放 **cache-dit 作「被优化的框架本体」、由 vLLM-Omni 0.26 托管**这条链上的**特有**内容——
> 开启方式（命令 / 开关 / 日志契约）、框架侧前置与坑、特性开关面板、与其它框架 / 版本的差异。
> **通用方法与判定纪律**见
> `model-auto-optimization/references/lossless-methodology-notes.md`（无损·计算/通信方法）、
> `dit-perf-opt/references/combination-search.md`（有损组合协议）、
> `accuracy-gate/references/quality-gate.md`（质量门禁）；
> **能力支持面**（✅/🟡/❌/❓ 与证据码）见 `framework-support-matrix.md`。
> **本文件不写绝对耗时与绝对质量分值**（二者只在本次环境成立，不可迁移）；**要写**大致加速比
> （量级/约数）与质量变化度（降幅/差值）；
> 实测记录与产物坐标见 §6。
> 与 `vllm-omni-enablement.md` 为**同级并列**：后者管 **vLLM-Omni 0.28 自身**的免训练特性开启方式，
> 本文件管 **cache-dit 作框架本体 + vLLM-Omni 0.26 托管**这条链的开启方式（0.26 与 0.28 的接线差异见
> §4），两者不互相抄。
> 路径占位符：`{model_weight_dir}`（权重根）、`{run_results_dir}`（运行产物根）、
> `{env_host}` / `{env_container}`（运行环境）。
> 本文件的方向性结论一律是该框架 × 该模型 × 该规模的**观测**，不是执行序（换框架 / 模型 / 规模须重判）。

## 1. 画像与版本边界

### 1.1 画像速答（迁移清单）

- **命名说明（先读）**：`cache-dit` 是第三方框架仓库名（`vipshop/cache-dit`），**不是**固定特性名
  `Cache`（大写 C，见 model-auto-optimization `overview-report.md` §2.1 名词表）——本链的 cache-dit 是
  **「被优化的框架本体」**，`Cache` 特性由该框架的原生能力承载。
- **托管链**：MiniMax-H3 在 diffusers 0.38 **无对应 pipeline 类** ⇒ dummy-run harness 不可用，必须以
  **vLLM-Omni 0.26** 托管（cache-dit 经 vLLM-Omni 的 `cachedit` glue 被消费）；compile 走同栈
  mindiesd `MindieSDBackend`。
- **并行面**：本栈 vllm-omni 0.26 DiT **仅 USP 形态（无 TP）** ⇒ 单请求 e2e 与「卡数」无线性关系
  （步数串行 + 单并发）；卡数↑ 主要扩**并发容量 / 显存余量**，对单请求 e2e 无承诺。
- **请求契约**：请求**必须带 `aspect_ratio=16:9`**（缺失 500）。

### 1.2 版本边界与依赖前置

- 版本锚点：**cache-dit trunk `51979f0`**（fork main 已 reset 对齐）× **vllm-omni 0.26.0（editable）** +
  vllm 0.26.0+empty + diffusers 0.38.0 + torch 2.11.0+cpu / torch_npu 2.11.0 + mindiesd（dev 同步重建，
  editable）；硬件 Ascend 950PR ×4（128 GB/卡）；模型 `{model_weight_dir}/MiniMax-H3/FL2VA`（BF16；
  transformer 50 层 / hidden 5376，13 分片）。**使能结论只在该框架版本 + 该模型 + 该环境成立**。
- cache-dit 侧：本地 fork main reset 对齐 trunk（`git -c http.sslBackend=openssl fetch`）；本地旧 NPU
  提交已上游 PR **#1004** 合入并重构 → **勿再保留本地残留**；远端以
  `PYTHONPATH={repo}/cache-dit-trunksync/src` **影子生效**（vllm-omni `cachedit` glue 兼容验过）。
- mindiesd 同步重建：`build_tik_ops` 行注释；陈旧 `build/plugin_build` cmake 缓存会漏编新源（先
  `rm -rf`）；editable 用 `pip install -e . --no-deps --no-build-isolation`。
- **vllm-omni 补丁 2 处**（`[探针]`，见 §5）：`regionally_compile` 加 `backend` 参数透传；
  `diffusion_model_runner` 在 env 门控下选 `MindieSDBackend`。

## 2. 启动与并行前置

### 2.1 启动 recipe（4×950PR，0–3 卡）

```bash
vllm serve {model_weight_dir}/MiniMax-H3/FL2VA --omni --num-gpus 4 \
  --usp 4 --ring 1 --text-encoder-tp-size 4 \
  --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 4 \
  --diffusion-attention-backend FLASH_ATTN
```

- 负载与口径（迁移时的对照基准）：T2VA 1024×576/5s（124 帧@24 fps）/50 步；e2e 主口径 = curl 请求
  墙钟中位（稳态 n≥2，排除 load / warmup）；质量 = ffmpeg psnr/ssim（**vs 同窗 lossless 同 seed a/b 对**，
  a/b 确定性一致可复验）。
- **口径纪律**：同窗同卡组对比；**<3% 噪声不宣称**；多窗口墙钟有漂移（同配置不同窗口可达数个百分点）
  ⇒ 对比须**同窗交错、n≥2 中位**；跨窗绝对值不可直接比。

### 2.2 自研算子部署前置（顺序敏感，硬约束）

- mindiesd `build_ops.sh` 只把产物放到 `mindiesd/ops/vendors/*`，**不会自动装进运行 CANN**；运行期靠
  `import mindiesd`（`env.py`）设置 `ASCEND_CUSTOM_OPP_PATH={repo}/mindiesd/ops/vendors/…`。
- ⚠️ **必须先 `import mindiesd` 再初始化 NPU / 建任何张量**，否则 GE 加载不到自研算子
  （`aclnnXxx … inferShape function does not exist`，如 EagleQBSA）。
- 部署校验：跑对应 `tests/ops/*/…_golden.py`（可见性 → 走的是哪一个 → 数值的**判据单点**在
  `../../operator-dev/references/custom-op-runtime-deploy-verify.md`；本文件 §2.2 只给该框架的顺序约束）。

### 2.3 启动偶发（非配置错）

- gloo 偶发 `ss1.ss_family == ss2.ss_family (10 vs 2)`（IPv4/IPv6 混用，容器 hostname 为空相关）→
  重试即可；持续出现则强制 IPv4（`MASTER_ADDR=127.0.0.1` + `GLOO_SOCKET_IFNAME=lo`）。

## 3. 特性开关面板（开关 / 日志与计数契约 / 坑）

### 3.1 Cache（`--cache-backend cache_dit`）

- 开关与参数面：

  ```bash
  --cache-backend cache_dit --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"residual_diff_threshold":R,
                   "max_warmup_steps":4,"max_continuous_cached_steps":MC,
                   "enable_taylorseer":true,"taylorseer_order":2}'
  ```

- 机制：Cache **不改变单 forward 的 kernel 组成**，作用是**步级跳过 forward** ⇒ kernel 级证据对
  Cache 不适用，判据是端到端 e2e + 质量档位。
- **档位扫描（本组合观测）**：主导旋钮是 **MC（`max_continuous_cached_steps`）**——MC 调大更快、
  质量略降；MC 调小更慢但质量更高（相对 MC4 同窗：MC2 时间 **+34% 量级**、质量 **+3% 量级**；
  MC6 时间 -10% 量级、质量略降）。**R（`residual_diff_threshold`）在 0.3–0.6 平坦**（决策 / 输出
  一致），R 过紧（如 0.2）则少命中并变慢。**采纳方向 = R0.4 / MC4 为默认部署档**（速度 frontier 与
  质量 frontier 各为两端备选）；**换框架 / 模型 / 规模须重判**。
- **计数契约缺口（坑）**：cache_dit `summary.py` 暴露 `_cached_steps` / `_accumulated_*` 等逐步计数器，
  但该 trunk 在 vLLM-Omni **视频路由未填充**（需 glue 补 cache_summary；改三方仓须确认）⇒ 用
  **kernel 级计数契约兜底**：cached 步的 kernel 计数约为 eager 步的**十分之一量级**（FA / MatMul /
  RMSNorm 三类同时骤降）⇒ 每缓存步跳过**约九成** block 计算。kernel 计数是**真实参与证据**，
  防 no-op 假加速。
- off-identity：以 `quality=lossless` 关 cache 复现基线（跨进程 framemd5 逐帧一致）。

### 3.2 量化（线性层 `mxfp8` / `int8`）

- 开关：`--diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'`（int8 为 `method:"int8"`）；
  两档均被接受并可 serve。
- 采纳方向（**本组合观测**）：两档都是**单点有损且可叠加**的维度；`mxfp8` 质量优于 `int8`（同链
  对拍定性结论）。量化是**跨 seam 可叠加**的常备维度（协同强度与单点排序随负载规模变化，见
  `combination-search.md`）。
- **fp8 配置缺口（留证不虚列）**：`TypeError: Cannot instantiate AscendFp8Config with kwargs {}`
  （需 ignore / quant_format 等参数）⇒ 框架配置面缺口，按 ❓ 登记、**不虚列收益**。

### 3.3 稀疏 FA（`RAINFUSION` 后端）——两条路径，路径区分先于收益判定

- 开关：`--diffusion-attention-backend RAINFUSION`（eager，BF16）+ `sparsity` / `start_step` /
  `skip_layers`。

**路径 1：eager `rf_v3`（0.26 后端现走路径）——「能生效但收益有限」**

- 路由事实：0.26 后端只调 `mindiesd.sparse_attention(sparse_type="rf_v2")`；950PR（`soc_version=260`，
  属 mindiesd A5 类）→ **`rf_v2` 被自动路由到 `rf_v3`**（`aclnnBlockSparseAttentionV2`，
  `inner_precise` 强制 4）。
- 配置语义：`sparsity` = 每 query block **丢弃** key block 的**名义**比例（mindiesd
  `keep_len=ceil(cols×(1-sparsity))`；内容相关 mask：pooled q/k 相似度 softmax→topk→阈值，
  **每层每步重建**）；`start_step` = 前 N 步 dense；`skip_layers` 豁免指定 block。
- 生效证据：dense FA kernel 计数骤降 + `BlockSparseAttentionV2` 按 DiT self 站点数出现（kernel 级
  坐实）；token_refiner 无 video 段 → **staying dense**（设计使然，成本小）；host/kernel 间隙可忽略。
- **收益边界（本组合观测）**：mask 选择逻辑本身（topk/softmax/阈值/首帧保护）开销可忽略，
  **大头是每层每步的全尺寸 rearrange/pool 数据搬运**（COPY/MOVE 计数随之翻倍，随层数线性放大）⇒
  稀疏步只省下**约两成步时**，且 `start_step` 之前的步仍 dense ⇒ **端到端收益被大幅摊薄**；
  **op 级单调用该路径甚至略慢于 dense FA**。
- 质量与「丢 80%」并存的解释：结构感知掩码 + 短视频片段帧间冗余 + prefix / first-frame 恒保留 ⇒
  **realized sparsity < nominal**（同窗复现，质量近无损）。

**路径 2：EagleQBSA 融合 op（= 0.28 的 `mix`：稀疏 FA + FA 量化）——op 级收益显著但本链未接线**

- 能力实体：mindiesd 自研算子（`csrc/ops/eagle_quant_block_sparse_attention` 全链
  op_host/op_api/infershape；Q/K per-block INT8 + V per-channel FP8 + block_sparse mask，
  **一 op 内 mask+BSA 融合**）。
- **接线边界（关键）**：0.26 vLLM-Omni 的 RAINFUSION 后端**无 quant / mix 接线**（0.28 fork 有
  `precision=mix`）⇒ 本链只能用 eager 路径；融合 op 的收益只能先在 **op 级微基准**拿到
  （同几何对拍 dense FA，op 级显著更快；只量化不稀疏的变体亦有明显加速，量化精度在该 op 的量化级
  容差内）。**接线（框架侧结构性开发，见 `../SKILL.md` §2 分支 B）后测 e2e 组合为待决项**。
- 部署前置见 §2.2（`import mindiesd` 顺序）；golden（`tests/ops/eagle_quant_block_sparse_attention/…`）
  通过为校验标准。

#### 组合方向与宣称纪律

- 组合方向（**本组合观测**）：稀疏作为**步级跳过之外的「算步加速」叠加层**在 Cache 之上有效
  （叠加后质量仍≈Cache 单点、不额外损）；**`start_step=0` 不可取**（质量显著劣化）——
  「前 N 步 dense」的保真旋钮是必要的，`start_step=12` 型配置才可采纳。
- **宣称纪律（fail-closed）**：稀疏行的收益**必须有本步 kernel 计数证据**（dense FA 计数下降 +
  稀疏 kernel 出现）**或 op 级对拍**（同几何同 sparsity）；**无证据不宣称「稀疏生效」**。

### 3.4 kernel 融合（`torch.compile` / MindieSDBackend）

- 接入：env `OMNI_MINDIE_COMPILE=1` + `--diffusion-compile-granularity regional` → 逐
  `MiniMaxH3DiTBlock` 编译（多 worker 命中）。通用接线姿势（「需赋值」陷阱、backend 单例、collective
  留 eager、图形态差异）见 `vllm-omni-enablement.md` §3.5 与本技能正文 §1.6「接入姿势」。
- 判定方向（**本组合观测**）：**受控交错 A/B**（同卡组同日，C1→E1→C2→E2，n≥4 取中位，消除热漂移）
  相对 eager **在噪声阈值内无稳健收益** ⇒ **回退（默认关）**，不做半开状态。
- ⚠️ **eager 与 compile 输出非逐字节**（图级数值微差）⇒ compile 变更**不得宣称无损**。
- ⚠️ **第二请求变慢 ≠ recompile 风暴**：`TORCH_LOGS=recompiles` 显示 recompile 有界，guard 主变体为
  `rope_table is None`（token_refiner 无 rope vs DiT 有 rope）⇒ 非每请求重编译；第二请求变慢归因噪声。

### 3.5 计数契约与真实性核验（本链）

- Cache：逐步 computed/cached 属性未填充 → **kernel 级契约兜底**（§3.1）。
- 稀疏：本步 kernel 计数或 op 级对拍（§3.3 宣称纪律）。
- **组合参与证据** = serve 日志 RAINFUSION active 计数 + kprof 抽样步（serve 日志与 kprof 产物同源）。
- 单步 kernel 采集：用 env 门控 hook（如 `OMNI_KPROF` 类；ordinal 须覆盖 warm 步数 + 目标步）在 rank0
  第 N 次 `MiniMaxH3DiTModel.forward` 采集，**无需改仓库**；shim 内直接 `analyse` 会报
  「daemon 不可解析」⇒ **离线 analyse 用独立进程跑**（采集方法见 `profiling-collect`）。

## 4. 与其它框架 / 版本的差异（迁移对照）

| 维度 | vLLM-Omni 0.28（见 `vllm-omni-enablement.md`） | cache-dit trunk × vLLM-Omni 0.26（本文件） |
|---|---|---|
| 被优化对象 | vLLM-Omni 自身托管链上的免训练特性面 | **cache-dit（第三方框架本体，PYTHONPATH 影子上游 trunk）经 vLLM-Omni `cachedit` glue 消费** |
| 并行形态 | TP / USP / TP×USP 多形态可选 | **仅 USP（无 TP）** ⇒ 卡数↑ 只扩并发容量 / 显存余量 |
| 稀疏接线 | `RAINFUSION_ATTN`；0.28 fork 另有 `precision=mix`（稀疏 FA + FA 量化融合 op） | 0.26 后端**无 quant / mix 接线** ⇒ 只能走 eager 路径，融合 op 收益仅 op 级 |
| Cache 参数面 | `--cache-backend cache_dit --cache-config`（DBCache 参数面同源） | 同一参数面；但 cache-dit 是**被优化框架本体**（trunk 影子生效，参数面与统计属性随 trunk 版本） |
| compile | kernel 级为负 → 默认关（另有 FFN-MX / mxfp8 档 compile 融合的正向闭环，见矩阵与 `vllm-omni-enablement.md`） | 本链 compile 回退（无稳健收益）；**泛型 pattern 误触**已修（§5） |
| 请求契约 | HTTP 服务（curl `/v1/images/generations` 等） | 同 HTTP 服务；**必须带 `aspect_ratio=16:9`**（缺失 500） |

- 特性**分类**跨框架相似（留在方法与矩阵），**开启方式**按框架不同（本节 + §2/§3）。

## 5. 回修与坑（`[探针]` 标注）

> 以下前两条为**未合入上游 / 本地 fork 改动**（env 门控或仓内改动）⇒ 按本仓「经验 vs 探针」判定为
> **探针**，不作为推荐姿势与报表宣称；由探针发现的 **durable 约束**（部署顺序、命中判据）按经验处理。

1. `[探针]` **vLLM-Omni 补丁 2 处**（editable，`.bak` 留存，**默认关**）：① `regionally_compile`
   增加 `backend` 参数透传；② `diffusion_model_runner` 在 env `OMNI_MINDIE_COMPILE=1` 时选
   `MindieSDBackend` 并放开 NPU inductor 跳过门控。
2. `[探针]` **泛型 pattern 跨算子族误触回修（本链最重要回修）**：mindiesd
   `compilation/passes/__init__.py` 移除 `enable_wan_residual_gate` / `enable_minimax_h3_gate` 注册
   （`compiliation_config.py` 默认 False；pattern 文件与 layer 保留）。症状 = 开启 compile 后**每步上百次
   `residual_gate_add`**、替换对象不是真实 gate；定位 = DOT 图 / pattern dump + `kernel_details.csv`
   → 命中点是 `fused_qk_norm_rope.py` 的 RoPE 旋转段 `add(mul,mul)`（q、k 各 1 节点/block），
   被 `wan_residual_gate` **泛型 pattern 抢跑**（H3 rope pattern 未命中）；移除后 rga kernel 计数归零、
   compile ≈ eager（同窗实测，噪声内）。（合入上游状态本案例未记录 ⇒ 按探针处理。）
3. `[探针]` **单 forward kernel 采集 hook**（env 门控，见 §3.5 与 `profiling-collect`）：零仓库改动的
   运行时 shim；目标模型可扩展（换 env 指向其它 DiT 模型类的 forward）。

教训（durable）：regionally 使能 compile 后**必须做 kernel 级「预期融合 vs 实际命中」核对**
（三层证据：日志 → DOT 图 → kernel csv）；**命中 ≠ 收益**——命中点若不是真实融合目标，收益归因即失真。

## 6. 产物坐标指针（实测记录不入 skills）

- 运行产物根 `{run_results_dir}/`：各档 `*.mp4`、serve 日志（后端解析 / pattern / compile / 阶段计时）、
  `quality*/`（ffmpeg a/b 对）、`frames*/`、`kprof_*/…/ASCEND_PROFILER_OUTPUT/`（单步 kernel 与
  step_trace）、graphdump、报表与 evidence。
- **本次迁移归档**：`{run_results_dir}/archive/cache-dit-minimax-h3-case.md` —— 本链的**绝对耗时、
  绝对加速比与质量数值**原文；只在原环境成立，**不可跨模型 / 框架 / 规模 / 窗口引用**。
- 通信采集明细（USP2 全链路通信分布、shim 字节口径与 CANN 二次验证）：会话报表
  `comm_analysis_usp2.md` / `comm_cann_verify.md`；**采集方法**见 `profiling-collect` 与
  `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §6。
- 支持矩阵：本链**能力面**按 `framework-support-matrix.md` 对应格核对，**开启方式**指回本文件。

## 7. 维护与更新

- 触发（版本）：cache-dit 离开 trunk `51979f0`、或 vllm-omni 由 0.26.0 换版时，§1.2 的版本锚点、§1.1「DiT 仅 USP 形态（无 TP）」与 §3.1 的 Cache 档位结论（`--cache-backend cache_dit` / `--cache-config` 的 MC / R）整节失效——本文「使能结论只在该框架版本 + 该模型 + 该环境成立」。
- 触发（缺口补齐 / 框架侧接线）：§3.1 的 cache_summary 计数缺口、§3.2 的 fp8 配置缺口、§3.3 路径 2「融合 op 未接线」（0.28 fork 的 `precision=mix`）任一被框架侧补上，或 §2.2 的自研算子部署顺序（先 `import mindiesd`）被改动时，对应小节与 `framework-support-matrix.md` 对应格须一并刷新。
- 触发（探针）：§5 的 vllm-omni 补丁 2 处（`regionally_compile` 的 `backend` 透传、`diffusion_model_runner` 选 `MindieSDBackend`）与泛型 pattern 误触回修均属 `[探针]`，重装 editable 或合入上游后须重新确认是否仍在、是否仍需本地保留。
- 复核：换版后按 §3.5 计数契约复核——同窗（同卡组、n≥2 中位）复采一次单步 `kernel_details.csv`，确认 cached 步的 FA / MatMul / RMSNorm 计数仍相对 eager 步骤降、且 serve 日志 `RAINFUSION` active 计数仍在；再按 §3.1 用 `quality=lossless` 关 cache 复现基线（跨进程 framemd5 逐帧一致），确认 off-identity 未破。
