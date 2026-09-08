# 实例：Qwen-Image-2512 × vLLM-Omni 0.28（NPU 950PR）无损+有损优化（图像模型首个 case）

> 2026-09-05 实测。仓库：vllm-omni（0.28.0 e305afba + NPU fork 补丁，env A 与 env B fork md5 一致）、
> MindIE-SD（dev，mindiesd 3.1.0 editable，env B 中继）。
> 环境：env A {env_host} / 容器 {env_container} / venv /opt/omni28 = torch 2.13.0+cpu + torch_npu 2.13.0.rc1 +
> vllm 0.28.0+empty（triton gluon 守卫 env 级补丁）+ vllm-omni 0.28.0 + mindiesd dev（中继自 env B）。
> 硬件：Ascend 950PR（128GB/卡；**共享多租户宿主**，NPU 占用动态，卡组逐档 16GB 探活）。
> 模型：{model_weight_dir}/Qwen-Image-2512（diffusers；QwenImagePipeline，1024²，20 步，official prompt，seed 42）。
> 对比口径：唯一基线 TP1 FLASH_ATTN 6.21s（S0 冻结）；无损主栈 TP2 4.96s（有损叠加基底）；
> 质量 21 同 seed 像素对 vs 同构 lossless（图像模型以固定 seed 集代替视频 21 采样帧）。

## 1. 环境要点（S0）

- vllm-omni editable 指向 vllm-omni-028；fork 补丁 = 4 文件（rainfusion supports_packed_mask_free /
  diffusion_model_runner + interface MindIE compile 注入 / npu platform 实现 + OMNI_KPROF/BPROF）。
- mindiesd editable 从 env B `/opt/omni28` 布局打包中继（finder/.pth/dist-info + 包树同名落位），
  Requires-Dist 仅 torch/torch_npu → 同 venv 即 import 成功。
- vllm CLI 导入链需 `triton.experimental.gluon` 守卫（triton-ascend 3.2.0 无）→ 重写容器内 vllm
  `triton_utils/__init__.py`（H3 同款 env 级补丁；非 fork 内容）。
- **多进程残留坑**：`pkill -f 'vllm serve'` 只杀主进程，`vLLM-Omni::DiffusionWorker` 子进程存活并持卡
  （各 ~54GB）→ 每档切换必须 `pkill -9 -f 'vLLM-Omni::DiffusionWorker'` + 16GB 分配探活确认。

## 2. 使能面速查（mindiesd 进 venv 自动路由 + 显式开关）

| 特性 | 触发 | 实测（qwen-image 1024²/20 步） |
|---|---|---|
| FLASH_ATTN | backend 未配置且 `find_spec("mindiesd")` | TP1 基线即 FA：FlashAttentionScoreV4 240/步 45.8ms（15.3% step） |
| 单算子 | AdaLayerNormV2/RotaryPositionEmbeddingV2/GeluV2 | 240/240/120 每步；rope 实数域 fused（qwen_image rope_utils 命中） |
| 量化(w8a8) | `--diffusion-quantization-config '{"transformer":{"method":"int8","activation_scheme":"dynamic"}}'` | **-12.7%**（4.33s）；DQ+QuantBatchMatmul 480/步；质量 37.5/0.986 |
| Cache | `--cache-backend cache_dit --enable-cache-dit-summary` | **单点 -13.7%**（4.28s；th=0.24/warmup=4/max_cont=3）；质量 35.7/0.965 |
| 稀疏 rf_v2 | `--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{…}}'` | 路由 ✓ 但 **staying dense：图像层无 qkv_layout/BSND video 段 → 零稀疏（输出=lossless 逐字节）**；**0.8 与 0.4（<60% 小档，用户建议补测）两档均 staying dense——与档位无关** |
| MindIE compile | `OMNI_MINDIE_COMPILE=1`（platform 注入） | 60 modules 编译成功但**输出非逐字节**（compiled FA 数值/seed 语义）+ 仅 ~-4% → 默认关 |

## 3. 无损结果（1024²/20 步；同窗相邻对）

- TP1 基线 6.05-6.21s（跨卡 0/1/4/6 输出 md5 全等 = 逐字节确定性）；DiT 单步 299.4ms（Computing 286.8ms）。
- **TP2 = 4.96-5.37s**（-11%~-20%）：单步 245.9ms = Computing 153.7 + Comm(未重叠) 76.7（31.2%，
  Overlapped=0）。TP4 / TP2×USP2 = 27.8s（4-rank 逐层通信病态）→ 回退。
- **TP1×USP2（2 卡，--usp 2）= 4.35s**（A/B/A 复现 4.34-4.36s；vs TP2 -12%、vs TP1 -30%）→ **2 卡无损
  最优**（日志 `Applying sequence parallelism to QwenImageTransformer2DModel (sp_size=2, mode=ulysses)`；
  r1 偶发慢请求 ~6.7-20.5s = 宿主抖动注记；step_trace ❓ 待补）。**教训：并行候选矩阵勿漏 2 卡 USP 形态
  （TP1×USP2）；TP2 与 USP2 收益结构不同（TP2=240 allreduce/步 per-call 开销主导；USP=少次大交换）。
  拓扑事实：0-3 / 4-7 各为 UB 岛（跨岛 SYS）；同岛亦受他户干扰（{6,7} 在他户占 NPU4 时 51-56s 反例）→
  同窗同干净岛复现为准。**
- compile 探针：kernel 行 5555→4055、步时 -4.4%，GEMM 180.4ms 不变；输出 d96e9a2c vs eager 1c702651
  （非无损）→ **回退（默认关）**；torch_npu 告警 compiled FA 设随机种子时结果可与 eager 不同。
- 跨拓扑输出 ≠ 逐字节（预期）；同拓扑同配置 r1==r2==r3 逐字节。

## 3b. 实际运行耗时 vs 预期（用户要求记录；2026-09-05）

- 图像 20 步短任务推理为**秒级**：实测稳态 4.34-6.21s/请求（TP1×USP2 4.35 / TP2 4.96 / w8a8+Cache 3.77），
  warmup 首请求（含编译/图预热）~5-10s；远小于按视频 60 步 118s 步数比例外推的分钟级预期。
- 参照（别机/别栈，仅量级参照不混比）：外部参考基线 1 卡 BF16 7.67s / 2 卡 2.70s；
  本机旧 0.26 栈手册 ~4.8s@8 卡 TP8。请求耗时由 DiT 单步 246-310ms × 20 步主导。

## 4. 有损结果（叠加 TP2 基底 4.96s；质量 21 seeds vs lossless）

| 组合 | e2e | Δvs 主栈 | PSNR/SSIM | 计数 |
|---|---|---|---|---|
| lossless TP2 | 4.96s | — | — | FA 240/MatMul 606/步 |
| 量化(w8a8) | 4.33s | -12.7% | 37.5/0.986 | DQ+QBMM 480/步 |
| Cache 单点 | 4.28s | -13.7% | 35.7/0.965 | DBCacheConfig；步跳过随图（seed 间 4.29-4.54s 双峰） |
| **量化(w8a8)+Cache** | **3.77s** | **-24.0%** | **33.9/0.960** | 叠加（最强组合对照行；**基底=TP2**，USP2×有损交叉 ❓ P1） |
| 稀疏 rf_v2 0.8 / 0.4 | =lossless | 0%（staying dense） | 输出=lossless 逐字节 | 不可用证据日志（档位无关） |
| 10 步 lossless | 2.67s | — | 决策记录 | 20 步固定主档 |

- 质量门禁：quantitative 登记通过（图像同 seed 像素对比远高于视频档 H3 19.6/0.679——无轨迹混沌）；
  **visual_artifact inconclusive（无 VLM）→ 不宣称质量通过**，并排 montage 存证；off-identity 成立。
- **量化后融合重审（S4 纪律⑥D/E）**：w8a8 单步 Computing 153.7→122.9ms(-20%)，Copy 11.9ms 不变
  （GEMM 级量化已到位，无新增搬运）；DQ 480/步上产邻接 GEMM → epilogue 融合机会 ❓；
  **通信占比 31.2%→36.8%（GEMM 加速致 comm 占比升）**，Overlapped=0（框架无 comm-stream，
  结构性缺口 → framework-extension-dev 范围确认外，机会项 ❓ P1）。

## 5. 回修与坑

1. kprof hook 目标扩展：fork `_install_kprof` 硬编码 MiniMaxH3DiTModel → 加 `OMNI_KPROF_TARGET=qwen_image`
   （QwenImageTransformer2DModel；env 门控，默认不变）——平台级 hook 扩展姿势（H3 case 同源方法）。
2. 多租户共享宿主：NPU 逐窗被占（他人训练随时起停）→ 每档前 pkill 残留 worker + 16GB 分配探活；
   NPU1 曾被他户占满致 TP2 首个 OOM 误判为显存不足（实为宿主占用）。
3. Cache 叠加告警 `Can't find Parallelism/Quantization Config for QwenImageTransformerBlock` = 配置缺失
   告警但缓存增益实存（-13.7%/组合 -24%）——同 H3 经验，勿据此判定失效。
4. JSON 类 CLI 参数经多层 shell 会丢引号（`invalid loads value`）→ 一律 JSON 文件 + 驱动读取嵌入
   （`file:` 前缀），不经 argv。

## 6. 方法论（对 H3 case 的增量）

- **图像模型质量口径**：视频用 21 采样帧（时间轴），图像用**固定 21-seed 同 seed 像素对**（确定性
  单点 → 多 seed 取样均化）；图像 INT8/缓存质量显著高于视频档（无混沌轨迹）——有损阈值/语义不可
  从视频案例迁移。
- **稀疏适用性判据前置**：rainfusion rf_v2 需要层声明 `qkv_layout='BSND'`（视频轴定位）——图像 2D
  tokens 无视频段 → **使能前先查 qkv_layout/BSND**，避免白跑 dense 兜底档（fail-closed：staying dense
  日志 + 输出逐字节=lossless 判定「未生效」，不虚列收益）。**稀疏档位按任务类型分档：视频可大稀疏
  （0.8 实测）；图像若未来框架提供 2D/BSND 稀疏路径，应从 <60%（0.4-0.5）小档起试**（2026-09-05
  用户建议落地：0.4 实测仍 staying dense，当前 vllm-omni 图像无路径）。
- **短任务并行判据**：20 步图像 DiT 为短 compute 任务 → 4-rank（TP4/USP2）逐层通信开销 >> GEMM 分摊
  收益（27.8s vs 2 卡 4.4-5.0s）；**2 卡形态下 TP1×USP2（4.35s）< TP2（4.96s）**——Ulysses 少次大交换
  vs TP2 240 allreduce/步的 per-call 开销结构差异；并行候选矩阵须覆盖 2 卡 USP（TP1×USP2），勿只测
  TP2 与 4-rank 组合；拓扑选型先查 UB 岛（npu-smi -t topo），同岛选卡 + 实测前探活。

## 7. 结论与证据

- 采纳：并行(TP1×USP2，2 卡) 4.35s（-30% vs TP1）；并行(TP2) 4.96s（S4 有损基底）；
  量化(w8a8) 4.33s；Cache 4.28s；量化(w8a8)+Cache **3.77s = 1.65× vs TP1 基线**（-24% vs TP2 基底）；
  质量 33.9-37.5 PSNR / 0.960-0.986 SSIM（visual 判卷待 VLM/人工）。
- 回退/不可用：compile（输出非无损）、TP4/TP2×USP2 4-rank（通信病态）、稀疏 rf_v2（图像不可用，
  0.4/0.8 档位无关，名义行）。
- **待办（P1，诚实标注不混分母）**：无损最优已改判 TP1×USP2 → 有损组合与质量基线按纪律须在 USP2
  基底复核重采；TP1×USP2 step_trace/带宽。
- 证据：runs/20260906_qwen-image-2512_optimization/（overview/detail/final/evidence/manifest +
  s0-s4_evidence.md + frames/ + kernel csv）；远端 `{run_results_dir}`；
  profile 回填 `evals/profiles/qwen-image-2512.toml`（V3 校准）；support-matrix 证据码 V3。
