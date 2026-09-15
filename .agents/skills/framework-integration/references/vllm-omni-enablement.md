# vLLM-Omni：特性开启方式与框架差异（0.28）

> 内容索引：§1 框架画像与版本边界 → §2 启动与并行（含前置）→ §3 特性开关面板（开关 / 日志契约 / 坑）
> → §4 与其它框架的差异（快速迁移对照）→ §5 回修与坑（`[探针]` 标注）→ §6 产物坐标指针。
>
> 定位：本文件只放 **vLLM-Omni 侧特有**的内容——开启方式（命令 / 开关 / 日志契约）、框架侧前置与坑、
> 特性开关面板、与其它框架的差异对照。**通用方法与判定纪律**见
> `model-auto-optimization/references/lossless-methodology-notes.md`（无损·计算/通信方法）、
> `dit-perf-opt/references/combination-search.md`（有损组合协议）、
> `accuracy-gate/references/quality-gate.md`（质量门禁）、
> `train-aware-lossy-method.md`（训练感知有损方法）；**能力支持面**（✅/🟡/❌/❓ 与证据码 V1/V3/V4）
> 见 `framework-support-matrix.md`。
> **本文件不写绝对耗时与绝对质量分值**（只在本次环境成立，不可迁移）；**要写**大致加速比
> （量级/约数）与质量变化度（相对基线的差值/降幅）——纪律见 `.agents/README.md` §7「数字纪律」；
> 实测记录与产物坐标见 §6。
> 与 `vllm-omni-train-aware-enablement.md` 为**同级并列**：后者专管**训练感知**（换入外部训练权重）的开启方式与
> 该模型侧契约，本文件管**免训练特性**的开启方式，两者不互相抄。
> 路径占位符：`{model_weight_dir}`（权重根）、`{run_results_dir}`（运行产物根）、
> `{env_host}` / `{env_container}`（运行环境）。
> 本文件的方向性结论一律是该框架 × 该模型 × 该规模的**观测**，不是执行序（换框架 / 模型 / 规模须重判）。

## 1. 框架画像与版本边界

### 1.1 画像速答（迁移清单）

| 检查项 | vLLM-Omni 实况 |
|---|---|
| 算子注册机制 | 无 `rms_type`/`rope_type` 抽象接口（模型层 torch 算子）→ 走 compile 接入 |
| 并行实现 | `--usp N --ring 1 --text-encoder-tp-size N`（USP/Ring 序列并行，多卡 a2a/ring 通信） |
| compile 图形态 | 对整个 transformer 编译（`torch.compile(pipe.transformer, backend=...)`），含 a2a/ring collective → **有 LightX2V 同款 a2a 退化风险** |
| 动态 shape | 推理分辨率固定 → 静态；但 CFG / 并行分支可能有动态维度，需验证 |
| 长序列通信 | USP 多卡通信占比随序列变长升高（同 LightX2V 长视频现象） |
| 通信掩盖机制 | 0.28 **无 comm-stream 掩盖**（单步 `Overlapped=0`）⇒ 掩盖收益上限 = 未重叠通信占比（§2.4） |
| 接入 / 验证入口 | HTTP 服务（`curl /v1/images/generations` 等），非 CLI 推理 |

**结论**：vLLM-Omni 与 LightX2V 相似度更高（都是多卡序列并行 + 整个 transformer 编译）：
**a2a 留 eager 的修复很可能同样必要**（须按 §1.3 的验证回路实测确认）。

### 1.2 版本边界与依赖前置

- 版本锚点：**vllm-omni 0.28.0 时代**（commit `e305afba` + NPU fork 补丁）；实测托管
  MiniMax-H3-FL2VA / Qwen-Image-2512 等扩散模型——**使能结论只在该框架版本 + 该模型 + 该环境成立**。
- **fork 补丁面（4 文件，均为探针，见 §5）**：rainfusion `supports_packed_mask_free` /
  `diffusion_model_runner` + `interface.py` 的 MindIE compile 注入 / NPU platform 实现 + kprof hook。
- 依赖前置：venv 用 `--system-site-packages` 复用系统 torch 作回退；torch 2.13 + torch_npu 2.13
  （PyPI，CANN 9.1.0 档可用）优先。mindiesd requirements 锁 torch 2.9、setup 插件变体仅到 torch210
  —— **对 torch 2.13 编译实测可行**（fixed 模式）。
- 装法坑：vllm sdist 走「外部环境 → 本地 → 运行环境」中转（公共源慢 / 缺）；mindiesd 只用
  `build_py` + `pip install -e --no-deps`（不装依赖），或按 editable 布局（finder / `.pth` / dist-info +
  包树同名落位）中继，`Requires-Dist` 仅 torch/torch_npu 时同 venv 即可 import 成功。
- 归档解包后需清洗 CRLF；`build_tik_ops` 调用行需注释掉
  （`s|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# ...|`）。
- vllm CLI 导入链需 `triton.experimental.gluon` 守卫（triton-ascend 3.2.0 无）→ 重写容器内 vllm
  `triton_utils/__init__.py`（**env 级补丁，非 fork 内容**）。

### 1.3 使能判断口径（与 dummy run 对比 + 三层证据）

- 使能集合与 dummy run 一致（pattern 匹配基于图结构，不依赖层数 / 权重）⇒ 用 dummy run 验证使能、
  用真实权重做耗时（同 `dummy-run` 技能）。
- 三层证据（可靠性递增）：`MINDIE_LOG_LEVEL=DEBUG` 日志（⚠️ 2048 字符截断陷阱）→ `graph_log_url`
  DOT 图（需 pydot）→ `kernel_details.csv`（融合 kernel 的实际执行次数）。

## 2. 启动与并行（含前置）

### 2.1 启动姿势（多卡服务）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve {model_weight_dir}/Qwen-Image-2512 \
  --omni --host 0.0.0.0 --port 8091 --trust-remote-code \
  --num-gpus 8 --tensor-parallel-size 8 \
  --vae-use-tiling --vae-patch-parallel-size 8
```

**启动自检与端点选型**（分支 A「框架侧验证」骨架的该框架落地；通用骨架见 `../SKILL.md` §1.3）：

```bash
curl -s http://127.0.0.1:8091/health          # 期望 200 OK
curl -s http://127.0.0.1:8091/v1/models       # 期望列出模型
curl -s http://127.0.0.1:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"{model_weight_dir}/Qwen-Image-2512","prompt":"a red teapot","size":"1024x1024","num_inference_steps":20,"seed":42}'
# 响应含 b64_json（PNG）：解码后 file 校验应为 "PNG image data, 1024 x 1024"
```

- **Edit / I2I 类**（`QwenImageEditPlusPipeline`，如 Qwen-Image-Edit-2511）**必须走 multipart**，
  否则 500 `Missing preprocess images`：

```bash
curl -s -m 600 -X POST "http://127.0.0.1:8091/v1/images/edits" \
  -F "model={model_weight_dir}/Qwen-Image-Edit-2511" -F "image=@/tmp/test_input.png" \
  -F "prompt=Convert this landscape to a watercolor painting style" \
  -F "size=1024x1024" -F "num_inference_steps=20" -F "seed=42" -o edit_response.json
```

- 端点选型速查（同 `troubleshooting-vllm-omni.md` §E3）：`QwenImagePipeline` → `/v1/images/generations`
  （仅 prompt）；`QwenImageEditPlusPipeline` → `/v1/images/edits`（multipart + image 文件）。
- 模型切换：`pkill -9 -f "vllm serve"`（并确认 `DiffusionWorker` 子进程已清，见下条）后用新权重
  路径重启同一命令。
- 权重落位按 `env-install/references/weights-prep.md` §2.2「落位约定」
  （`{model_weight_dir}/{模型名}/{任务变体}/`；模型名之下不再加厂商/组织层）。

- `--text-encoder-tp-size`（本组合观测，2026-09-13 按形态细分；**换并行形态 = 重判**）：
  - **序列并行（USP）形态**（如 TP2×USP2 → `dit_world=4`）：须与 DiT world 对齐，否则编码器子组
    `cpu_group is None` assert（`pipeline_*._build_text_encoder_group` 只建 `ranks=[0,tp)` 子组，
    非组内 rank 的 `GroupCoordinator` assert）。
  - **纯 TP 形态（无序列并行，`usp=1`）**：**该形态在本组合下整体不可用，与 tetp 取值无关**——
    两种编码器模式各失败在不同位置：
    - `--text-encoder-tp-size` 对齐到 DiT world（`dit_world=2`）→ 更早失败，视觉输入 seam：
      `RuntimeError: pixel_values and image_grid_thw must be provided together`（`encoder.py` fail-closed；
      DiT 拿不到有效输入，`diffuse` 亚秒返回、不出视频）。
    - 改用框架默认 `1`（`_SingleRankEncoderGroup` 占位组；日志 `text_encoder_tp_size=1, vision replicated`）
      → 编码器能跑，但更晚在 DiT 输入构造处失败：`_build_denoise_inputs` →
      `IndexError: tuple index out of range`。
    ⇒ **根因是「无序列并行」这一形态本身，不是 `--text-encoder-tp-size` 的取值**；不要把 tetp 当该形态的
    修复开关。该形态在本框架 × 本模型 × 本 t2va 负载下**直接否决**（换负载/框架版本须重测）。
  - **归因纪律（本组合教训）**：只按「第一个貌似合理的根因」收口会给出错误结论——本例先归因到 tetp，
    补了「换 tetp 再跑一次」的对照臂后才发现两个模式都失败、真因是形态。**至少有对照臂才允许把
    「形态不可用」与「参数配错」区分开。** 另：历史该模型在 576p/5s 档曾有纯 TP 形态可跑通的记录，
    与本轮 768P/15s 负载**不可比**，故不据此推翻或采信，只作参照。
- **多进程残留坑**：`pkill -f 'vllm serve'` 只杀主进程，`vLLM-Omni::DiffusionWorker` 子进程存活并持续
  持卡 → 每档切换必须杀干净子进程，并用小分配探活确认卡真的空出来。
  ⚠️ 共享宿主/共享容器上**禁止按进程名批量 pkill**（会误杀同宿主其他并行实验的 serve）——
  只按自己启动的 **PID 树**递归 kill，并在启动前确认目标端口空闲。

### 2.2 并行形态：能力范围与前置

- **并行候选必须在框架能力范围内逐一使能实测比较**，不能只按理论通信量选型：
  - `Ring` 形态框架不可用（**Ring 不支持 attn_mask**，`mask_sp_padding` 开关无效）→ 序列并行选 **USP**；
  - `TP1×USP2`（2 卡 Ulysses）在短任务（少步 / 小分辨率）下优于 `TP2` → **候选矩阵勿漏 2 卡 USP 形态**；
  - 4-rank 形态（`TP4` / `TP2×USP2`）在短 compute 任务下逐层通信开销 >> GEMM 分摊收益 → 回退。
- 无损形态的采纳方向（**本组合观测**）：H3-FL2VA 侧选 `TP2×USP2`（`TP4×USP1` 与之相当）为最优无损
  形态；`TP1×USP4` 仅量化档可行（BF16 全量驻留超单卡容量）。Qwen-Image 侧短任务选 2 卡 `TP1×USP2`，
  `TP2` 作有损叠加基底。**换模型 / 换规模 / 换框架须重判**。
- **CP 形态的可用面（本组合观测）**：复合「AllGather-KV × Ulysses」（`--usp k --allgather-degree m`）
  可跑通、且可叠加分块掩盖（逐字节精确）；**环状形态在本框架不可用于稀疏**（对 `ring_degree>1`
  硬拒，且其路径会绕过注意力后端 ⇒ 稀疏**静默失效**）；4 卡 `usp=2, allgather-degree=2` 在本负载下
  起不来（HTTP 5xx，未定位）⇒ **未跑通的形态不进候选矩阵**。形态之间怎么选（GQA / 跨岛带宽 /
  形态自带计算的条件化判据）见 `dit-parallel-opt/references/parallel-form-selection-method.md`。
- 拓扑选型属跨框架共性方法：先查 UB 岛（`npu-smi info -t topo`）同岛选卡 + 实测前探活，详见
  `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md`。

### 2.3 显存不足时用 offload 解锁并行（无损阶段）

- 更优并行策略因显存不可行时（如单 rank 全量权重超单卡容量），**先在无损阶段试 offload 类特性降显存**，
  而不是直接放弃或跳到有损；有损档（量化）落地后再回跑被显存卡住的并行组合。

| 开关 | 路径与要点 |
|---|---|
| `--enable-distributed-layerwise-offload`（DLO） | 默认 AllGather 路径：host 存 `1/DP` + H2D/AllGather 重叠；日志判据 `Distributed layer-wise offloading enabled on <N> blocks ... dp_size=…, sp_size=…`。**选 AllGather 路径**：no-AllGather（rank-local H2D）明显更慢、host-bound |
| `--enable-layerwise-offload` | ⚠️ 950PR 上会触发 OOM killer → 用 DLO，不用普通 layerwise |
| 互斥 / 副作用 | ⚠️ FastH3 拒绝任何 offload |

- mindiesd 侧 `enable_offload`、PyTorch FSDP/CPU-offload 语义开关同族（按框架命名查）；方法序列见
  `lossless-methodology-notes.md` §B。

### 2.4 通信面方向（本组合观测）

- **未重叠通信占比随并行策略改变**：同模型同步数下，「Comm(未重叠) 占比」呈 `USP2 > TP4 > USP4(int8)`
  的排序；框架无 comm-stream 掩盖（`Overlapped=0`）⇒ **掩盖收益上限 = 未重叠通信占比**，须按策略重估。
  本尺寸为 compute-bound，方向是先长序列重测再决定是否投入掩盖实现（实现参照 mindiesd/parallel +
  LightX2V `hccl_eager`；属框架结构性缺口 → `../SKILL.md` §2 分支 B）。
- **量化后通信须重审**：量化使 compute 缩短、comm 基本不变 ⇒ comm 占比抬升（掩盖空间变大），并出现
  低精度传输候选（TP allreduce 的部分和、USP 注意力 K/V 交换）；方法见
  `lossless-methodology-notes.md` §E。

## 3. 特性开关面板（开关 / 日志契约 / 坑）

### 3.1 自动路由面（mindiesd 进 venv 后自动生效）

| 特性 | 触发 | 说明 |
|---|---|---|
| FLASH_ATTN | backend 未配置且 `find_spec("mindiesd")`（`platform.py`） | 单点收益为正（本组合观测） |
| 单算子替换 | norm / rope / adaln 层 CustomOp dispatch（`fast_layernorm` / `rotary_position_embedding` / `layernorm_scale_shift`）；RMSNorm eager 即 `npu_rms_norm` | 与 FA 同开；图像侧另见 AdaLayerNormV2 / RotaryPositionEmbeddingV2 / GeluV2 |
| 编码器 patch | NPU 平台 `init_diffusion_model_runner_runtime`（fused RoPE + GQA-SDPA + packed GEMM + `npu_swiglu`） | 文本编码阶段非瓶颈（占比可忽略） |
| `MINDIE_SD_FA_TYPE` | mindiesd manual 分支枚举 `{prompt_flash_attn, fused_attn_score, ascend_laser_attention}`；vllm-omni 只比 `==ascend_laser_attention`（`flash_attn.py`） | ⚠️ 950PR / 950DT 场景**勿设置**（未做 AB） |

- 热路径 eager 已被 mindiesd/CANN 单算子覆盖时，**再叠 compile 的 kernel 级结果为负**（拷贝 / 调度开销
  增、输出不变）⇒ 不采纳；剩余融合机会集中在 **DQ 两侧**与注意力路径布局（方法见
  `lossless-methodology-notes.md` §A/§D，实测记录见 §6 归档）。
- **基线冻结必须显式指定注意力后端，禁止依赖默认路由**：装了 editable mindiesd 的宿主平台默认把
  diffusion attention 路由到 FLASH_ATTN；若要对齐"未加速对照"（如 SDPA），须显式传
  `--diffusion-attention-config '{"default":{"backend":"TORCH_SDPA"}}'` 并在服务日志确认 resolve 行，
  同时记录输出 md5 / off-identity——口径冻结以「显式配置 + 日志 + 输出」三方一致为准，不靠"以为没装
  mindiesd 就默认 SDPA"（编排层口径见 `model-auto-optimization/**`）。

### 3.2 量化（`w8a8` / int8 online）

- 开关：`--diffusion-quantization-config '{"transformer":{"method":"int8","activation_scheme":"dynamic"}}'`；
  mxfp8 档为 `method:mxfp8`（i8 域 ≠ MX e4m3，**不能混用**）。
- 日志契约：`DynamicQuantV2 → QuantBatchMatmulV3` 接管 MatMul（每 DiT block 5 个量化 GEMM：
  qkv / out / fc1(merged) / down / adaln），**零新增布局搬运** ⇒ GEMM 级量化融合已到位。
- 坑：**层宽超限会自动回退**（见日志）；量化只覆盖稀疏块路径时 dense FA 兜底 bf16 —— 对外表述必须
  写明覆盖范围；框架无全注意力 FA 8bit 接线 ⇒ 含全注意力 f8 的组合登记 ❓，不虚列收益。
- 采纳方向（**本组合观测**）：单点收益为正、且与稀疏 / 缓存跨 seam 可叠 ⇒ 作为有损组合的常备维度。
  **量级与质量变化度**：视频档单点约 1.2 倍量级、vs lossless 的 SSIM 降约三成；图像档单点约一成多、
  **质量基本无感**（SSIM 降约 0.01）——**两档不可跨域引用**
  （协同强度与单点排序随负载规模变化，须在本任务口径下测，见 `combination-search.md`）。

### 3.3 稀疏（`RAINFUSION_ATTN` / rf_v2）

- 开关：`--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{…}}}'`。
- **使能前置（几何判据，fail-closed）**：RAINFUSION 需要层声明 `qkv_layout='BSND'`（视频轴定位）；
  图像 2D tokens 无视频段 → 运行期 **staying dense**（日志出现 `staying dense`）+ 输出与 lossless
  **逐字节相同** ⇒ 判「未生效」，**不虚列收益**；该现象与稀疏档位无关（大小档皆同）。
- 退出条件同理：**稀疏的 `end_step` 语义是"末 N 步保留 dense"，不是"从第 N 步起稀疏"**——
  误设成全程步数会让稀疏全程 staying-dense；改参数前先核源码 / 文档语义，改完用输出对比验证确实参与。
- 稀疏档位**逐档端到端验证**：质量随稀疏度**非线性**（低密度档的定量指标可低于高密度档）→ 禁止按密度外推。
- 稀疏「路径区分」先于收益判定：eager 公共 API（外部逐层 mask，如 rf_v3）与融合算子（op 内 mask+BSA
  一体，如 EagleQBSA）是两条不同实现，墙钟口径不可混称；同 seam 只留最强档（见 `combination-search.md` seam 表）。
- **与序列并行（CP）叠加的前置（本组合观测）**：稀疏选块需要**全量 K/V** ⇒ 形态必须是
  「先汇聚、后稀疏」（AllGather-KV 满足；环状不满足且路径绕过后端 ⇒ 静默失效）；窗口全局偏移按
  「完整 KV 长度 − 本 rank 逻辑 Q 长度」算（无 joint/拼接张量为 0，**不是** `k_len − q_len`）；
  每 rank 的 KV 分片长度须是稀疏/量化块的整数倍；选块掩码是 **per-head** 的、不可跨 head 分片复用。
  契约、验收判据与失效模式表见 `dit-parallel-opt/references/cp-sparse-combination-method.md`。
- **生效判定不得只看收益**：叠加臂与 lossless 档**逐字节相同**（或日志出现 `staying dense`）⇒
  判「**未生效**」而非「无损」，该组合不得登记收益。

### 3.4 Cache（`cache_dit`）

- 开关：`--cache-backend cache_dit --enable-cache-dit-summary`；参数面 `DBCacheConfig`
  （阈值 / warmup / `max_continuous_cached_steps`）。
- 机制：Cache **不改变单 forward 的 kernel 组成**，作用是**步级跳过 forward** ⇒ kernel 级证据对 Cache
  不适用，判据是端到端 e2e + 质量档位。
- 日志契约坑：`Can't find Parallelism/Quantization Config for <Block>` 是**配置缺失告警**，缓存增益实存
  → 勿据此判失效。
- 采纳方向（**本组合观测**）：Cache 是**同域内最轻损**的有损维度（默认档质量即可接受），故常作为
  组合基底之一（视频档单点降幅居首；图像档单点约一成多、SSIM 降约 0.04）；
  收紧窗口参数可换质量优先档。少步 / 训练感知档的 Cache 可能**结构性失效**（预热步数吃满步数预算）
  —— 判据与处置见 `vllm-omni-train-aware-enablement.md` §5（字节级等价，不记作「收益小」）。

### 3.5 `torch.compile`（MindieSDBackend）接入点

```python
from mindiesd.compilation import MindieSDBackend
compiled = torch.compile(pipe.transformer, backend=MindieSDBackend())
```

- ⚠️ 与 DiffSynth-Engine 同款陷阱：`torch.compile(module, backend=...)` 需**赋值**
  （`module._compiled_call_impl = ...` 或 `nn.Module.compile()` 等价），不赋值不生效（pattern 0 命中）。
- 注入姿势：`interface.py` 默认方法 + `diffusion_model_runner.py` 平台分发 + NPU platform 实现，
  env `OMNI_MINDIE_COMPILE=1` 门控、backend 单例、regional 粒度（fork，见 §5）。
- 已观测结论：编译成功、pattern 注册，但 **kernel 级验证为负**（拷贝 / 调度开销增、输出不变）；
  图像侧另因 compiled FA 数值 / seed 语义导致**输出非逐字节** ⇒ **默认关**（回退，非「收益小」）。

### 3.6 计数契约与真实性核验（日志判据）

- **逐字节确定性**：同拓扑同配置多次运行输出逐字节一致；**跨拓扑输出 ≠ 逐字节** ⇒ 无损对比必须在同一
  并行配置内做，跨拓扑数值只能作参考。
- 「未生效」的判据是**字节级等价 + 计数缺失**（`staying dense`、输出 md5 相同），不是「收益小」。
- 采集口径：日志 2048 截断会误判 0 命中 → 以 DOT 图 / `kernel_details.csv` 为准（§1.3）。
- CLI 坑：JSON 类参数经多层 shell 会丢引号（`invalid loads value`）→ 一律 **JSON 文件 + 驱动读取嵌入**
  （`file:` 前缀），不经 argv。

## 4. 与其它框架的差异（快速迁移对照）

| 维度 | LightX2V | DiffSynth-Engine | vLLM-Omni |
|---|---|---|---|
| 并行 | USP4（a2a 在 block 内） | 单卡（无通信） | USP/Ring 多卡（collective 在 transformer 内） |
| compile 粒度 | 逐 block | 逐 submodule | 整个 transformer |
| a2a 留 eager | 必须（否则 collective 进图退化为变长 a2a、**单次耗时放大数十倍**；留 eager 后转正收益——均本组合观测） | 不需要 | 很可能必须（须实测） |
| 运行时注册表接入 | 有接口（rms/rope） | 无 | 无 |
| 验证入口 | torchrun + CLI | pipeline CLI | HTTP 服务（curl `/v1/images/generations`） |

按 LightX2V 经验推断的 vLLM-Omni 修复面（**推断，未全部实测**，接入必须按验证回路确认）：

| 修复 | LightX2V 必要性 | vLLM-Omni 预期 |
|---|---|---|
| a2a/ring collective 留 eager（`torch._dynamo.disable`） | 必须（不退化为变长 a2a 才转正收益；本组合观测） | **很可能必须**（USP 多卡 + 整个 transformer 编译） |
| backend 实例复用 | 必须 | 必须（跨框架通用） |
| pattern 匹配实际图形态 | 必须（chunk 双 split） | 按模型图 dump 验证（各模型不同） |

- **profiling 采集差异**：多卡服务 + HTTP 入口 ⇒ 采集补丁需挂在 transformer forward 上（而非 CLI 推理），
  warmup 用服务预热请求代替。
- 特性**分类**跨框架相似（留在方法与矩阵），**开启方式**按框架不同（本节 + §2/§3）。

## 5. 回修与坑（`[探针]` 标注）

> 以下均为**框架 fork 改动**：未合入上游 / env 门控 / 默认关 ⇒ 按本仓「经验 vs 探针」判定为**探针**，
> 不作为推荐姿势与报表宣称；其中由探针发现的 **durable 约束**（框架支持状态、部署顺序、命名纪律）
> 按经验处理（如 §2.1 的 `--text-encoder-tp-size` 契约、§3.3 的几何前置判据）。

1. `[探针]` **RAINFUSION mask-free 契约修复**：`rainfusion_attn.py` 增加 `supports_packed_mask_free`
   **classmethod**（=True，同 FlashAttention 契约）——否则 H3 packed `[real,pad]` 会构建 `attn_mask`
   且 RainFusion 拒 mask（`layer.py` raise "does not support attn_mask"）。
   ⚠️ 必须是 classmethod 而非类属性（调用方 `backend.supports_packed_mask_free()` → 属性会报
   `'bool' object is not callable`）。
2. `[探针]` **MindIE compile 注入**：`interface.py` 默认方法 + `diffusion_model_runner.py` 平台分发 +
   NPU platform 实现（env `OMNI_MINDIE_COMPILE=1` 门控、backend 单例、regional 粒度）→ **默认关**
   （kernel 级验证为负 / 图像侧输出非逐字节，见 §3.5）。
3. `[探针]` **通用单 step kernel 采集 hook**（NPU platform，env `OMNI_KPROF` / `OMNI_KPROF_AFTER` /
   `OMNI_KPROF_DIR`）：rank0 第 N 次 `*DiTModel.forward` 用 `torch_npu.profiler`(Level1) +
   `tensorboard_trace_handler` 包一次 forward，之后 `from torch_npu.profiler.profiler import analyse(dir)`
   离线聚合出 `ASCEND_PROFILER_OUTPUT/{kernel_details,step_trace_time,...}.csv`
   （msprof `--export` 是采集工具不适用；torch profiler wrapper 的 NPU 分支不产 csv；**必须 analyse**）。
   目标模型扩展：新增 env `OMNI_KPROF_TARGET`（如图像侧指向 `QwenImageTransformer2DModel`），
   门控默认行为不变——平台级 hook 扩展姿势。
4. **多租户共享宿主**（环境约束，非探针）：NPU 逐窗被占（他人训练随时起停）→ 每档前杀残留 worker +
   小分配探活；曾出现「首个 OOM 被误判为显存不足」，实为宿主占用。

## 6. 产物坐标指针（实测记录不入 skills）

- 运行产物根 `{run_results_dir}/`：各档 `*.mp4`、`*_serve.log`（后端解析 / pattern / compile / stage
  计时）、`quality*/`、`frames*/`、`kprof_*/…/ASCEND_PROFILER_OUTPUT/`（单步 kernel 与 step_trace）、
  报表与 `evidence.json`。
- 权重坐标：`{model_weight_dir}/MiniMax-H3/FL2VA`（BF16 主体）、`{model_weight_dir}/Qwen-Image-2512`
  （diffusers 布局，`QwenImagePipeline`）。
- 闭环产物目录：`runs/20260906_qwen-image-2512_optimization/`（overview / detail / final / evidence /
  manifest + `s0-s4_evidence.md` + frames + kernel csv）；运行时质量 profile 由
  **仓库根** `evals/scripts/gen_profile.py`（相对本文件 `../../../../evals/scripts/gen_profile.py`）
  生成到 `runs/{task_id}/profiles/`（**不入库**）。
- **本次迁移归档（原 vllm-omni 系列 case 的绝对数字）**：会话产物目录 `{run_results_dir}/archive/` 下
  `vllm-omni-case.md` / `vllm-omni-minimax-h3-case.md` / `vllm-omni-qwen-image-case.md` —— 只在原环境
  成立，**不可跨模型 / 框架 / 规模 / 窗口引用**。
- 支持矩阵证据码 V1（MiniMax-H3-FL2VA）/ V3（Qwen-Image-2512）的**能力面**留在
  `framework-support-matrix.md`，**开启方式**指回本文件。

## 7. 维护与更新

- **触发（版本边界）**：跨过 §1.2 的版本锚点（vllm-omni 0.28.0 时代 commit `e305afba` + NPU fork 补丁）
  时，§1.2–§3 的依赖前置、开关名与坑整段重核；§5 三条 `[探针]`（RAINFUSION `supports_packed_mask_free`
  classmethod、`OMNI_MINDIE_COMPILE=1` 的 compile 注入、`OMNI_KPROF*` 采集 hook）任一**合入上游 / 解除
  env 门控**后，须从探针段移出并按 `../SKILL.md` §2.4 改档（不得留「以防万一」的旧标注）。
- **触发（被点名对象改名 / 移除）**：§3 开关面板与 §2 并行前提里的具体对象是本节维护清单——
  `--diffusion-quantization-config` 的 `method:int8` / `method:mxfp8`、`--diffusion-attention-config` 的
  `RAINFUSION_ATTN` 与其 `qkv_layout='BSND'` 几何前置及 `end_step` 语义、`--cache-backend cache_dit`
  与 `DBCacheConfig`、`MINDIE_SD_FA_TYPE`、`--enable-distributed-layerwise-offload`、
  `--usp` / `--ring` / `--allgather-degree` / `--text-encoder-tp-size` 契约：任一项在框架侧改名 / 移除，
  对应小节与 §4 对照表重写。
- **触发（能力面刷新）**：V1 / V3 / V4 证据码与格状态只在 `framework-support-matrix.md` 更新
  （含其列 last-checked）；本文件不复写状态，只留开启方式与坑，两处不得各写一份状态。
- **复核方法**：按 §1.3 + §3.6 做一次最小核对即判本文结论是否仍成立——同一拓扑同一配置跑两遍确认输出
  逐字节（跨拓扑输出本就非逐字节，不作数），再以 `graph_log_url` DOT 图 / `kernel_details.csv` 的融合
  kernel 计数判命中（**勿用会被 2048 字符截断的日志判 0 命中**），并检查日志有无 `staying dense` 或
  `Can't find Parallelism/Quantization Config` 类兜底证据（二者均按「未生效 / 配置缺失」而非「收益小」读）。
