# 实例：MiniMax-H3 × vLLM-Omni 0.28 无损+有损全特性叠加

> 2026-09-05 实测。目标仓库：vllm-omni（0.28.0 时代 e305afba + NPU fork 补丁）、MindIE-SD（dev-skills 549bc60 重建）。
> 环境：env B {env_host} / 容器 {env_container} / venv /opt/omni28 =
> torch 2.13.0+cpu + torch_npu 2.13.0.rc1 + vllm 0.28.0+empty + triton-ascend 3.2.1 + vllm-omni 0.28.0 + mindiesd dev。
> 硬件：Ascend 950PR ×4（128GB/卡，0-3 卡）；模型 {model_weight_dir}/MiniMax-H3/FL2VA（BF16，T2VA 5s @1024×576@24fps + AAC32k）。
> 请求口径：prompt/seed=1101/flow_shift=12/audio_shift=3.0；墙钟 steady r2/r3；同卡组同步数同 seed 对比。

## 1. 环境搭建（S0）要点

- venv `--system-site-packages` 复用系统 torch 2.11 作回退；torch2.13+torch_npu2.13rc1（PyPI，CANN 9.1.0-950 OK）优先。
- vllm-0.28.0 sdist 走 env A→本地→env B 中转（pythonhosted/aliyun 均慢/缺）；mindiesd 仅 `build_py + pip install -e --no-deps`（不装依赖）。
- mindiesd requirements 锁 torch 2.9、setup 插件变体仅到 torch210 —— **对 torch 2.13 编译实测可行**（fixed 模式）。
- CRLF：归档解包后 sed 清洗；build_tik_ops 注释（`s|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# ...|`）。

## 2. 使能面速查（mindiesd 进 venv 后自动路由，0.28 disable-vllm-ascend）

| 特性 | 触发 | 实测 |
|---|---|---|
| FLASH_ATTN | backend 未配置且 `find_spec("mindiesd")`（platform.py:156-165） | SDPA → FA 单点为正（@10 步 TP2，约一成） |
| 单算子替换 | norm/rope/adaln 层 CustomOp dispatch（fast_layernorm / rotary_position_embedding / layernorm_scale_shift）；RMSNorm eager 即 `npu_rms_norm` | 与 FA 同开 |
| 编码器 patch | NPU 平台 init_diffusion_model_runner_runtime（fused RoPE + GQA-SDPA + packed GEMM+npu_swiglu） | 文本编码亚秒量级 |
| MINDIE_SD_FA_TYPE | mindiesd manual 分支枚举 {prompt_flash_attn, fused_attn_score, ascend_laser_attention}；vllm-omni 只比 ==ascend_laser_attention（flash_attn.py:564） | 950PR 未 AB（勿在Ascend 950PR&950DT系列产品设） |

## 3. 无损叠加结果（10 步稳态；60 步括号）

- S1 FLASH_ATTN（TP2 2卡）单点为正；S3 **TP2×USP2（4卡）= 最优无损形态**（60 步稳态同样保持最优）。
- **坑**：单 stage TP2×USP2（dit_world=4）需 `--text-encoder-tp-size 4`，否则编码器子组 `cpu_group is None` assert（pipeline_minimax_h3._build_text_encoder_group 只建 ranks=[0,tp) 子组，非组内 rank 的 GroupCoordinator assert）。
- 确定性：同配置逐字节一致；**跨拓扑（2卡 vs 4卡）≠ 逐字节**（同 seed 跨拓扑 PSNR/SSIM 显著劣化 @10 步）→ 无损对比必须在同一并行配置内。

## 4. 有损叠加（最优无损拓扑上；60 步 steady，质量=21 采样帧 vs lossless）

| 组合 | e2e（相对 lossless FA） | 加速 vs lossless FA | 质量变化度 vs lossless |
|---|---|---|---|
| lossless FA | 参照 | 1.00×（参照） | — |
| +INT8(online dynamic) | 约降两成 | 约 1.2 倍 | SSIM 降约三成 |
| +稀疏 rf_v2 0.8(bf16) | 约降两成半 | 约 1.3 倍 | SSIM 降约四成半 |
| +稀疏(bf16)+INT8 | 约降三成 | 约 1.4 倍 | SSIM 降约四成半 |
| +稀疏(**EagleQBSA mix**)+INT8 | 约降三成半 | 约 1.6 倍 | SSIM 降约四成半 |
| +稀疏(bf16)+INT8+**Cache-DiT** | **约降七成** | **三倍多（vs env A 基线 SDPA 为十倍量级）** | SSIM 降约五成 |
| 备选 TP1×USP4+INT8 | 约降三成 | 约 1.4 倍 | SSIM 降约三成 |
| 备选 TP1×USP4+INT8+mix | 约降五成半 | 约 2.3 倍 | SSIM 降约五成 |

（原始读数见 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`）

### S4-2 组合搜索增量（2026-09-05，combination-search 协议首案例校准；60 步 steady；质量=21 采样帧 vs lossless）

| # | 组合（TP2×USP2 基底） | seam | steady | 质量变化度 vs lossless | 结论 |
|---|---|---|---|---|---|
| F4 | int8+稀疏 bf16 0.8+Cache（默认） | precision×attention×step | 约降七成（历史窗） | SSIM 降约五成 | frontier |
| **F5** | int8+稀疏 **mix(EagleQBSA)** 0.8+Cache（默认） | attention 同 seam 取最强档 | **降幅最大（本窗最快）** | SSIM 降约五成（较 F4 再低约 0.02） | **新 frontier（本窗内最快；质量较 F4 略降）** |
| F5hq | F5 + Cache 高保真（max_continuous_cached_steps=1, threshold 0.12） | step_decision 收紧 | 降幅小于 F5 | SSIM 降约五成（优于 F5） | 质量优先档（墙钟明显上升换 SSIM 回升） |
| **F6（补测 09-06）** | **单独 Cache**（lossless 基底，默认档；无 INT8/稀疏——单点隔离） | 单点行 | 约降近七成 | **SSIM 降约三成半** | cache 单点 = 最轻损维度；与 768P cache 单点降幅交叉一致（hq 档 SSIM 降约两成半） |
| 层回退探针 | int8+稀疏 bf16 0.8 start_step=2 | step 窗口前移 | 约降三成半（本窗） | SSIM 降约四成半 | start_step 保真≈ss0（同窗），收益有限 |
| 层回退探针 | int8+稀疏 bf16 **0.6** | attention 降档 | 约降三成（本窗） | SSIM 降约五成 | **质量非线性**（0.6 档 SSIM 反低于 0.8 档）→ 稀疏档须逐档端到端验证，勿按密度外推 |

（原始读数见 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`）

- 计数契约（真实性核验）：RAINFUSION 日志含 `staying dense`（无 video 段 role 走 dense 兜底）；Cache-DiT 有
  Parallelism/Quantization config 缺失告警但缓存增益实存；INT8 层宽超限自动回退见日志。
- 跨窗口漂移声明：不同窗口绝对值可比性受热/同机负载影响（同档历史窗与本窗差异可达一成量级）——
  **frontier/回退结论以同窗相邻对为准**（本窗 mixcache 快于 hq 档）。
- 命名与绝对质量口径（补测批 09-06 落地，规则见 model-auto-optimization `overview-report.md` §2.1–§2.3/§4）：
  特性名固定名词 + `量化(w8a8/f8/w8a8f8)` 修饰符（w8a8=线性层 8bit；f8=注意力侧 8bit——本任务 mix 的 int8
  仅覆盖稀疏块路径，dense FA 兜底 bf16，表述须写明覆盖范围；全注意力 FA 8bit 未实现标 ❓ 不虚列）；
  质量一律 vs 同构 lossless 绝对值（USP4 行已由交叉值改为绝对值）。
- **量化后融合重审实例（w8a8，对应 SKILL S4 纪律 6 / methodology-notes §D）**：使能 INT8
  online 后单步 kernel 序列基本等长——原 MatMul 几近全部被 **`DynamicQuantV2` → `QuantBatchMatmulV3`**
  接管（每 DiT block 恰 5 个量化 GEMM：qkv/out/fc1-merged/down/adaln），
  零新增布局搬运（MOVE/COPY 计数不变）、无独立 dequant → **GEMM 级融合已到位**，新机会集中在
  DQ 上游 epilogue（norm/SwiGlu）、A 侧动态量化并入 GEMM、FA 路径布局、量化域贯通、通信重叠与
  新序列 compile 重测（O1–O7 完整清单与证据见归档 `H3_w8a8_fusion_analysis.md`；**报表归属：实施后
  的量化使能融合收益按 overview-report.md §2.4 归入 `量化(w8a8)` 行、拆分在量化子表展示，
  当前未实施项标 ❓ 不计入行收益**）；**通信面（对应
  SKILL S4 纪律 6 / methodology-notes §E）**：+INT8 后单步 Comm(未重叠) 耗时不变但占比
  上升（由约六分之一升至约五分之一；叠加 mix 后约三成，Overlapped=0）→ 量化后须按新占比重估掩盖空间并审视量化/压缩通信
  （TP allreduce 部分和、USP 注意力 K/V 交换低精度传输）；候选实现若属框架结构性缺口走
  `../SKILL.md` §2 分支 B，不静默改三方框架。

- 质量门禁（evals）：同卡组同 seed 冻结 lossless60 基线；定量 psnr/ssim + 锐度 lapvar；视觉判卷 inconclusive（无 VLM）→ 并排产物存证；off-identity 成立。
- 质量阈值首个案例已回填 `MindIE-SD/evals/profiles/minimax-h3.toml`。

## 5. 使能回修（fork 改动，均验证）

1. **RAINFUSION_ATTN 500 修复**：rainfusion_attn.py 增加
   `supports_packed_mask_free` classmethod（=True，同 FlashAttention 契约）——否则 H3 packed [real,pad] 构建
   attn_mask 且 RainFusion 拒 mask（layer.py:498-502 raise "does not support attn_mask"）。
   ⚠️ 必须是 classmethod 而非类属性（调用方 `backend.supports_packed_mask_free()` → 属性会报 `'bool' object is not callable`）。
2. **MindIE compile 注入（S1 融合链）**：interface.py 默认方法 + diffusion_model_runner.py 平台分发 + NPU platform 实现
   （env `OMNI_MINDIE_COMPILE=1` 门控、backend 单例、regional 粒度）——全部 block 编译成功、pattern 注册、输出逐字节一致，
   但 **kernel 级验证为负**（见 §6）→ 默认关。
3. **通用 kernel 采集 hook**（NPU platform，env `OMNI_KPROF/OMNI_KPROF_AFTER/OMNI_KPROF_DIR`）：rank0 第 N 次
   `MiniMaxH3DiTModel.forward` 用 torch_npu.profiler(Level1)+tensorboard_trace_handler 包一次 forward →
   之后 `from torch_npu.profiler.profiler import analyse(dir)` 离线聚合出
   `ASCEND_PROFILER_OUTPUT/{kernel_details,step_trace_time,...}`（msprof --export 是采集工具不适用；torch profiler
   wrapper 的 NPU 分支不产 csv；必须 analyse）。

## 6. 无损·计算（融合）方法论与实测

**流程（回填 skill 的通用动作）**：① 用代码或 profiling 的**算子执行序**给出「可融合 kernel 列表」；
② 对照 **mindiesd + CANN 融合能力**（npu_rms_norm/npu_rotary_mul/npu_swiglu/FA/fused qk-norm-rope 等）识别融合位置与
**需额外准备的融合算子**；③ 对每个候选做**独立验证**（kernel diff / 墙钟）辨识是否真生效。

- 实测：H3 DiT 热路径 eager 已被 mindiesd/CANN 单算子覆盖（每步按站点数出现 RmsNorm / RotaryV2 / FA、编码器 swiglu；站点计数见归档）。
- MindIE compile 单步 kernel diff：kernel 数与 Copy/Move 数上升、Computing 与步墙钟抬高，
  输出不变 → **未产生收益反增开销 → 不采纳（默认关）**；结论：真正热路径 eager 已融合，
  剩余 split/silu/cat/index 等候选融合收益不足以抵消 compile 图内拷贝/调度开销（10 步墙钟中性一致）。
- VAE 等耗时：10 步 e2e 中 diffuse(DiT) 占八成多、余量为非 DiT 段；60 步非 DiT 段升至数十秒量级
  （VAE tile decode 帧数相关 + 多 rank 交接）。

## 7. 无损·通信方法论与实测

- 并行候选需在**框架能力范围**逐一使能比较再选型：TP2×USP2（选定）与 TP4×USP1（相当）为无损候选中最快一档；
  TP1×USP4 仅 INT8 档可行；Ring2（TP2×USP1×ring2）框架不可用
  （Ring 不支持 attn_mask，mask_sp_padding 开关无效）→ 序列并行选 USP。
- **profiling 找优化空间**（step_trace_time 单步；随并行策略不同）：

  | 策略 | 步 wall | Computing | Comm(未重叠) | Comm 占比 |
  |---|---|---|---|---|
  | TP2×USP2（lossless） | 参照 | 参照 | 参照 | **约三成（三者最高）** |
  | TP4×USP1（lossless） | 略低 | 更高 | 更低 | 约两成 |
  | TP1×USP4（int8） | 最低 | 最低 | 最低 | 约一成 |

  → **Overlapped=0（框架零重叠）**；未重叠通信占比 USP2>TP4>USP4；vllm-omni 0.28 无 comm-stream 掩盖 →
  实现候选参照 mindiesd/parallel + LightX2V `hccl_eager`；掩盖收益上限随策略为步时的一到三成，本尺寸 compute-bound，先长视频重测再投入。
- **特性叠加的 kernel 演化**（单 forward，576p）：lossless FA 态下 MatMul / FA / Copy 各按站点数出现 →
  +INT8：QuantBatchMatmulV3+DynamicQuantV2 接管 MatMul（被替代链计数归零，Quant 按站点数出现），步墙钟降约一成半 →
  +mix 稀疏：EagleQuantBlockSparseAttention 承接 attention（+FA dense 兜底），小 kernel/copy 计数增长但
  Computing 再降、步墙钟降约三成；Cache 不改单 forward kernel（作用=步级跳过 forward）。
  （原始计数见 `{run_results_dir}/archive/vllm-omni-minimax-h3-case.md`）
- **内存受限解锁并行**（用户经验）：当更优并行策略因显存不可行（如 H3 BF16 单 rank 全量
  超出单卡容量，无法 TP1×USP4）时，**先在无损阶段尝试 offload 类特性降显存解锁**，而非直接放弃或跳到有损：
  - vllm-omni：`--enable-distributed-layerwise-offload`（DLO，默认 AllGather 路径：host 存 1/DP + H2D/AllGather
    重叠）在 4 卡上解锁 **TP1×USP4 BF16 无损** —— 与 TP2×USP2 相当（较高步数略优，约 3% 量级）；
    日志：`Distributed layer-wise offloading enabled on ... blocks ...
    dp_size=1, sp_size=4`。**DLO no-AllGather（rank-local H2D）慢数倍（host-bound）→ 选 AllGather 路径**。
  - mindiesd `enable_offload`；PyTorch FSDP/CPU-offload 语义开关（按框架命名查）。
  - ⚠️ 互斥/副作用：FastH3 拒绝任何 offload；普通 `--enable-layerwise-offload` 会触发 OOM killer（用 DLO）。
- **有损完成后复查被显存卡住的通信组合**：量化（INT8 online）降显存后解锁 TP1×USP4
  （较 USP2-int8 约降一成）与 +EagleQBSA mix（降幅更大）；即每个量化档落地后回跑「并行×显存余量」候选。

## 8. 证据落点

- 远端 {run_results_dir}/：各档 `*.mp4`、`*_serve.log`（后端解析/pattern/compile/stage 计时）、
  `quality*/quality60*/qualitycmp*/qualitycurve*.json`、`frames*/frames60*/frames_cmp*/frames_curve/`、
  `kprof_kbf_{eager,compile}/…/ASCEND_PROFILER_OUTPUT/`。
- 本地会话产物目录（`{run_results_dir}`）：H3_omni_tuning_analysis.md、H3_TUNING_REPORT_2026-09-05.md（§4b-4e 矩阵）、h3_frames/montage_*.png、h3_lossless60_r2.mp4。

## 9. 维护与更新

- **触发（口径失效即整表作废）**：§3/§4 的加速比与质量变化度只在开篇声明的口径内成立——
  2026-09-05、env B、vLLM-Omni 0.28.0（`e305afba` + NPU fork 补丁）、MiniMax-H3-FL2VA 576p 10/60 步、
  同卡组同步数同 seed；框架版本升级、环境/依赖换档（torch 2.13 / CANN 9.1.0 档）或基线变更后，
  本表数字不得沿用，须按 `vllm-omni-enablement.md` 重采。
- **触发（探针与开关面）**：§5 三条 fork 回修（RAINFUSION `supports_packed_mask_free` classmethod、
  `OMNI_MINDIE_COMPILE` compile 注入、`OMNI_KPROF/OMNI_KPROF_AFTER/OMNI_KPROF_DIR` 采集 hook）
  合入上游后按探针处置改写；§2 被点名的触发面（`find_spec("mindiesd")` 自动路由、`MINDIE_SD_FA_TYPE`、
  `--diffusion-quantization-config` 的 `method:int8`、`RAINFUSION_ATTN`、`cache_dit`、
  `--enable-distributed-layerwise-offload`）改名 / 移除时，本节与矩阵 V1 列（含子批次 V1a/V1b）同批核对。
- **复核方法**：按 §4「计数契约」那条做最小复核即判本节结论是否仍成立——同卡组同 seed 冻结一次
  lossless 基线后跑目标档，确认日志仍出现 `staying dense`（无 video 段 role 兜底）/ Cache-DiT 配置缺失
  告警 / INT8 层宽超限回退，并确认 `kernel_details.csv` 里 `DynamicQuantV2` → `QuantBatchMatmulV3`
  仍接管 MatMul；任一取证取不到 ⇒ 该档（含其 SSIM 变化度）作废，不得沿用。
