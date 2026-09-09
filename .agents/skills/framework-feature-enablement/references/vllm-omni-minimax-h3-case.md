# 实例：MiniMax-H3 × vLLM-Omni 0.28（NPU 950PR）无损+有损全特性叠加

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
| FLASH_ATTN | backend 未配置且 `find_spec("mindiesd")`（platform.py:156-165） | SDPA 40.7s → FA 36.0s @10步 TP2（-11.5%） |
| 单算子替换 | norm/rope/adaln 层 CustomOp dispatch（fast_layernorm / rotary_position_embedding / layernorm_scale_shift）；RMSNorm eager 即 `npu_rms_norm` | 与 FA 同开 |
| 编码器 patch | NPU 平台 init_diffusion_model_runner_runtime（fused RoPE + GQA-SDPA + packed GEMM+npu_swiglu） | 文本编码 <0.1s |
| MINDIE_SD_FA_TYPE | mindiesd manual 分支枚举 {prompt_flash_attn, fused_attn_score, ascend_laser_attention}；vllm-omni 只比 ==ascend_laser_attention（flash_attn.py:564） | 950PR 未 AB（勿在 950PR/950DT 设） |

## 3. 无损叠加结果（10 步稳态；60 步括号）

- S1 FLASH_ATTN（TP2 2卡）36.0s；S3 **TP2×USP2（4卡）= 最优无损 19.1s（60: 118.0s）**。
- **坑**：单 stage TP2×USP2（dit_world=4）需 `--text-encoder-tp-size 4`，否则编码器子组 `cpu_group is None` assert（pipeline_minimax_h3._build_text_encoder_group 只建 ranks=[0,tp) 子组，非组内 rank 的 GroupCoordinator assert）。
- 确定性：同配置逐字节一致；**跨拓扑（2卡 vs 4卡）≠ 逐字节**（同 seed PSNR 21.6/SSIM 0.75 @10）→ 无损对比必须在同一并行配置内。

## 4. 有损叠加（最优无损拓扑上；60 步 steady，质量=21 采样帧 vs lossless）

| 组合 | e2e | 加速 vs 118.0s | PSNR/SSIM vs lossless |
|---|---|---|---|
| lossless FA | 118.0s | 1.00× | — |
| +INT8(online dynamic) | 96.4s | 1.22× | 19.6/0.679 |
| +稀疏 rf_v2 0.8(bf16) | 90.7s | 1.30× | 16.5/0.542 |
| +稀疏(bf16)+INT8 | 85.7s | 1.38× | 16.7/0.542 |
| +稀疏(**EagleQBSA mix**)+INT8 | 75.1s | 1.57× | 16.8/0.553 |
| +稀疏(bf16)+INT8+**Cache-DiT** | **32.3s** | **3.65×（vs env A SDPA 344.9s = 10.7×）** | 16.0/0.513 |
| 备选 TP1×USP4+INT8 | 85.1s | 1.39× | vs lossless 19.43/0.692 |
| 备选 TP1×USP4+INT8+mix | 52.05s | 2.27× | vs lossless 15.97/0.523 |

### S4-2 组合搜索增量（2026-09-05，combination-search 协议首案例校准；60 步 steady；质量=21 采样帧 vs lossless）

| # | 组合（TP2×USP2 基底） | seam | steady | 质量 vs lossless | 结论 |
|---|---|---|---|---|---|
| F4 | int8+稀疏 bf16 0.8+Cache（默认） | precision×attention×step | 32.3s（历史窗） | 16.0/0.513 | frontier |
| **F5** | int8+稀疏 **mix(EagleQBSA)** 0.8+Cache（默认） | attention 同 seam 取最强档 | **28.2s** | 15.7/0.496 | **新 frontier（本窗内最快；质量 -3%SSIM vs F4）** |
| F5hq | F5 + Cache 高保真（max_continuous_cached_steps=1, threshold 0.12） | step_decision 收紧 | 35.1s | 16.0/0.510 | 质量优先档（+24%墙钟换 +1.4%SSIM） |
| **F6（补测 09-06）** | **单独 Cache**（lossless 基底，默认档；无 INT8/稀疏——单点隔离） | 单点行 | **38.4s** | **19.33/0.665** | cache 单点 = 最轻损维度；与 768P cache 单点 -66.7% 交叉一致（hq 档 49.6s / 21.69/0.763） |
| 层回退探针 | int8+稀疏 bf16 0.8 start_step=2 | step 窗口前移 | 76.3s（本窗） | 16.7/0.544 | start_step 保真≈ss0（同窗），收益有限 |
| 层回退探针 | int8+稀疏 bf16 **0.6** | attention 降档 | 80.9s（本窗） | 15.5/0.489 | **质量非线性**（0.6 PSNR 反低于 0.8）→ 稀疏档须逐档端到端验证，勿按密度外推 |

- 计数契约（真实性核验）：RAINFUSION 日志含 `staying dense`（无 video 段 role 走 dense 兜底）；Cache-DiT 有
  Parallelism/Quantization config 缺失告警但缓存增益实存（35.8→28.2s）；INT8 层宽超限自动回退见日志。
- 跨窗口漂移声明：不同窗口绝对值可比性受热/同机负载影响（如 bf16-ss0 历史 85.7s vs 本窗 ss2 76.3s）——
  **frontier/回退结论以同窗相邻对为准**（本窗 mixcache 28.2 vs hq 35.1）。
- 命名与绝对质量口径（补测批 09-06 落地，规则见 model-auto-optimization `overview-report.md` §2.1–§2.3/§4）：
  特性名固定名词 + `量化(w8a8/f8/w8a8f8)` 修饰符（w8a8=线性层 8bit；f8=注意力侧 8bit——本任务 mix 的 int8
  仅覆盖稀疏块路径，dense FA 兜底 bf16，表述须写明覆盖范围；全注意力 FA 8bit 未实现标 ❓ 不虚列）；
  质量一律 vs 同构 lossless 绝对值（USP4 行已由交叉值改为绝对值）。
- **量化后融合重审实例（w8a8，2026-09-06，对应 SKILL S4 纪律 6 / methodology-notes §D）**：使能 INT8
  online 后单步 kernel 序列 2625→2885 行——266 MatMul → 6 遗留 + **260 对 `DynamicQuantV2`(53ms/1.3%)→
  `QuantBatchMatmulV3`(654ms/15.7%)**（每 DiT block 恰 5 个量化 GEMM：qkv/out/fc1-merged/down/adaln），
  零新增布局搬运（MOVE/COPY 971 不变）、无独立 dequant → **GEMM 级融合已到位**，新机会集中在
  DQ 上游 epilogue（norm/SwiGlu）、A 侧动态量化并入 GEMM、FA 路径布局、量化域贯通、通信重叠与
  新序列 compile 重测（O1–O7 完整清单与证据见归档 `H3_w8a8_fusion_analysis.md`；**报表归属：实施后
  的量化使能融合收益按 overview-report.md §2.4 归入 `量化(w8a8)` 行、拆分在量化子表展示，
  当前未实施项标 ❓ 不计入行收益**）；**通信面（对应
  SKILL S4 纪律 6 / methodology-notes §E）**：+INT8 后单步 Comm(未重叠) 0.81s 不变但占比
  16.6%→19.6%（+mix 30.8%，Overlapped=0）→ 量化后须按新占比重估掩盖空间并审视量化/压缩通信
  （TP allreduce 部分和、USP 注意力 K/V 交换低精度传输）；候选实现若属框架结构性缺口走
  framework-extension-dev，不静默改三方框架。

- 质量门禁（evals）：同卡组同 seed 冻结 lossless60 基线；定量 psnr/ssim + 锐度 lapvar；视觉判卷 inconclusive（无 VLM）→ 并排产物存证；off-identity 成立。
- 质量阈值首个案例已校准（语义见 quality-gate.md；运行时 profile 由 `evals/scripts/gen_profile.py` 生成到 `runs/{task_id}/profiles/`，不入库）。

## 5. 使能回修（fork 改动，均验证）

1. **RAINFUSION_ATTN 500 修复**：rainfusion_attn.py 增加
   `supports_packed_mask_free` classmethod（=True，同 FlashAttention 契约）——否则 H3 packed [real,pad] 构建
   attn_mask 且 RainFusion 拒 mask（layer.py:498-502 raise "does not support attn_mask"）。
   ⚠️ 必须是 classmethod 而非类属性（调用方 `backend.supports_packed_mask_free()` → 属性会报 `'bool' object is not callable`）。
2. **MindIE compile 注入（S1 融合链）**：interface.py 默认方法 + diffusion_model_runner.py 平台分发 + NPU platform 实现
   （env `OMNI_MINDIE_COMPILE=1` 门控、backend 单例、regional 粒度）——52 block 编译成功、pattern 注册、输出逐字节一致，
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

- 实测：H3 DiT 热路径 eager 已被 mindiesd/CANN 单算子覆盖（RmsNorm n=210/step、RotaryV2、FA 54/step、编码器 swiglu）。
- MindIE compile 单步 kernel diff：kernel 2625→2925、Copy/Move 569→1069、Computing 1268→1341ms/step、
  wall 1765→1829ms/step，输出不变 → **未产生收益反增开销 → 不采纳（默认关）**；结论：真正热路径 eager 已融合，
  剩余 split/silu/cat/index 等候选融合收益不足以抵消 compile 图内拷贝/调度开销（10 步墙钟中性一致）。
- VAE 等耗时：10 步 e2e 19.0s = diffuse(DiT) 15.84s(83%) + ~3.2s；60 步非 DiT ~25s（VAE tile decode 帧数相关 + 多 rank 交接）。

## 7. 无损·通信方法论与实测

- 并行候选需在**框架能力范围**逐一使能比较再选型：TP2×USP2 19.1s@10/118s@60（选定）；
  TP4×USP1 19.05s@10（相当）；TP1×USP4 仅 INT8 可行 15.0s@10/85.1s@60；Ring2（TP2×USP1×ring2）框架不可用
  （Ring 不支持 attn_mask，mask_sp_padding 开关无效）→ 序列并行选 USP。
- **profiling 找优化空间**（step_trace_time 单步；随并行策略不同）：

  | 策略 | 步 wall(us) | Computing | Comm(未重叠) | Comm 占比 |
  |---|---|---|---|---|
  | TP2×USP2（lossless） | 1,765,014 | 1,268,104 | 485,087 | 27.5% |
  | TP4×USP1（lossless） | 1,704,631 | 1,394,750 | 297,507 | 17.5% |
  | TP1×USP4（int8） | 1,242,253 | 1,100,018 | 131,031 | 10.5% |

  → **Overlapped=0（框架零重叠）**；未重叠通信占比 USP2>TP4>USP4；vllm-omni 0.28 无 comm-stream 掩盖 →
  实现候选参照 mindiesd/parallel + LightX2V `hccl_eager`；掩盖收益上限随策略 10-27% 步时，本尺寸 compute-bound，先长视频重测再投入。
- **特性叠加的 kernel 演化**（单 forward，576p）：lossless FA 2625 kernel/步（FA 54、MatMul 266、Copy 470、wall 1765ms）→
  +INT8：QuantBatchMatmulV3+DynamicQuantV2 接管 MatMul（266→6，Quant 520），wall 1476ms(-16%) →
  +mix 稀疏：EagleQuantBlockSparseAttention 承接 attention（+FA dense 兜底），小 kernel/copy 增多（Copy 470→1270）但
  Computing 再降、wall 1233ms(-30%)；Cache 不改单 forward kernel（作用=步级跳过 forward）。
- **内存受限解锁并行**（用户经验，2026-09-05 已实证）：当更优并行策略因显存不可行（如 H3 BF16 单 rank 全量
  135G > 128GB，无法 TP1×USP4）时，**先在无损阶段尝试 offload 类特性降显存解锁**，而非直接放弃或跳到有损：
  - vllm-omni 实测：`--enable-distributed-layerwise-offload`（DLO，默认 AllGather 路径：host 存 1/DP + H2D/AllGather
    重叠）在 4×950PR 上解锁 **TP1×USP4 BF16 无损** —— 10 步 19.2s（≈ TP2×USP2 19.1s）、60 步 **114.4s
    （-3% vs TP2×USP2 118.0s）**；日志：`Distributed layer-wise offloading enabled on 52 blocks ...
    dp_size=1, sp_size=4`。**DLO no-AllGather（rank-local H2D）慢 ~5.7×（109s@10 步，host-bound）→ 选 AllGather 路径**。
  - mindiesd `enable_offload`；PyTorch FSDP/CPU-offload 语义开关（按框架命名查）。
  - ⚠️ 互斥/副作用：FastH3 拒绝任何 offload；950PR 上普通 `--enable-layerwise-offload` 会触发 OOM killer（用 DLO）。
- **有损完成后复查被显存卡住的通信组合**（已实证）：量化（INT8 online）降显存后解锁 TP1×USP4 = 85.1s@60
  （-11.7% vs USP2-int8）与 +EagleQBSA mix = 52.05s@60；即每个量化档落地后回跑「并行×显存余量」候选。

## 8. 证据落点

- 远端 {run_results_dir}/：各档 `*.mp4`、`*_serve.log`（后端解析/pattern/compile/stage 计时）、
  `quality*/quality60*/qualitycmp*/qualitycurve*.json`、`frames*/frames60*/frames_cmp*/frames_curve/`、
  `kprof_kbf_{eager,compile}/…/ASCEND_PROFILER_OUTPUT/`。
- 本地 D:\framework：H3_omni_tuning_analysis.md、H3_TUNING_REPORT_2026-09-05.md（§4b-4e 矩阵）、h3_frames/montage_*.png、h3_lossless60_r2.mp4。
