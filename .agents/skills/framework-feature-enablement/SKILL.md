---
name: framework-feature-enablement
compatibility: vLLM-Omni / DiffSynth-Engine / LightX2V / diffusers 等第三方推理框架；远端昇腾容器 + CANN；mindiesd 已安装（安装见 env-install）
description: >
  三方框架语境下的特性使能与验证：把特性（融合算子 / FA / 稀疏 / 量化）在 vLLM-Omni / DiffSynth-Engine / LightX2V 等框架侧使能并验证。覆盖运行时算子接入
  （rms/rope 注册表替换）、compile 融入（_compiled_call_impl/backend 复用/a2a 留 eager/图形态差异）、
  特性开关开启验证、使能异常定位回修（mindiesd vs 框架适配侧）。
  当用户要确认量化/稀疏/Cache 开关生效、把 mindiesd 接入三方框架、排查融合算子未命中或使能
  异常、或问某融合算子在外框架是否生效时使用；即使用户只说"起 vllm 服务""模型在 vllm 里跑不通"
  或把 mindiesd 编译接到其他框架而未说框架名，也应触发。纯安装/权重走 env-install，纯采集/分析走
  profiling-collect / profiling-analyze。由 model-auto-optimization 的 S1（融合接入）/S4（有损使能）阶段调用，
  亦由 dev-workflow 的框架接入/验证场景指引加载。
---

# 三方框架特性使能与验证

## 定位与分工

| 相邻能力 | 分工 |
|---|---|
| env-install | 环境安装与权重准备（本技能的前置） |
| profiling-collect / profiling-analyze | 证据侧：采集与 kernel diff / 三层证据 |
| compilation-dev / operator-dev | 实现侧：新增/适配 pattern、算子本体 |
| dummy-run | 快速验证载体：融合可行性、使能集合预判 |
| **本技能** | **使能决策 + 框架侧接入/开关执行 + 生效验证 + 异常回修** |

前置：环境安装与权重就绪已完成（见 env-install：`import mindiesd` 成功、三方框架装好、
模型权重可用且经其确认）。

## 使能与验证回路

```text
① 使能评估 → ② 路径决策 → ③ 使能异常定位与回修
```

### ① 使能评估

- 目标特性需要 mindiesd 的什么 kernel/pattern？（量化/稀疏/缓存开关、融合 pattern、单算子接口）
- 框架侧接入点是什么？（配置注册表、特性开关、compile 入口、图形态）
- 特性来源是 mindiesd 能力还是框架原生？（决定查 mindiesd-features 还是框架文档）

### ② 路径决策

```text
a) 有现成融合/单算子 → 开启并验证生效
   姿势：特性开关 / 注册表替换 / _compiled_call_impl 赋值（见 §compile 融入）
   → 三层证据确认（图命中 → kernel diff → 墙钟，证据走 profiling-analyze）

b) 融合不支持（无 pattern / 图形态不匹配 / 无抽象接口）
   → 接入 mindiesd 算子（runtime 替换，见 §运行时算子接入）
     或走 compile（新增/适配 pattern，图形态差异加变体）
   → 需要实现改动时指向 compilation-dev / operator-dev
   → 融合可行性先用 dummy-run 快速验证（pattern 命中不依赖层数/权重），
     再在真实权重下 kernel diff + 墙钟复验
```

- 使能判断用 dummy run 即可（pattern 匹配基于图结构）；使能集合与真实权重一致。
- 单次结果不迁移：换框架/并行配置必须按回路重新验证。

### ③ 使能异常定位与回修

使用 FA、稀疏等算子时使能异常（不生效 / 精度不符 / 报错 / 性能劣化）：

1. 定位**算子执行差异**：图命中 vs 运行期实际 kernel（`graph_log_url` DOT 图 → `kernel_details.csv`）、
   数值/精度差异、shape/布局差异（如 4D attention 张量使 `residual_gate_add` 运行期 fallback）。
2. 判定根因：
   - **mindiesd 侧**：pattern 实现错误 / 注册缺失 / kernel 语义不符 → 指向 compilation-dev / operator-dev 修复；
   - **mindiesd 自研算子部署侧**：自定义 CANN 算子（如 EagleQBSA）产物只在 repo
     `mindiesd/ops/vendors/*`，运行期靠 `import mindiesd`（env.py 设 `ASCEND_CUSTOM_OPP_PATH`）——
     **必须先 import mindiesd 再初始化 NPU/建张量**，否则 `aclnnXxx … inferShape function does not
     exist`（部署/顺序问题，非参数问题；golden 校验，详见 cache-dit-minimax-h3-case §4.6）；
   - **框架适配侧**：调用姿势（`_compiled_call_impl` 未赋值）、图形态与 pattern 期望不符、开关未生效、
     Dynamo 重编译（backend 实例未复用）→ 本技能内修复框架侧适配。
3. 修复后复验：重新采集 + kernel diff + 墙钟（profiling-collect → profiling-analyze），确认根因消除。

## 框架侧验证（已部署模型，真实权重）

### 服务启动（vLLM-Omni 示例，950PR 8 卡）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve {model_dir}/Qwen-Image-2512 \
  --omni --host 0.0.0.0 --port 8091 --trust-remote-code \
  --num-gpus 8 --tensor-parallel-size 8 \
  --vae-use-tiling --vae-patch-parallel-size 8
```

验证：

```bash
curl -s http://127.0.0.1:8091/health                    # 期望 200 OK
curl -s http://127.0.0.1:8091/v1/models                  # 期望列出模型
curl -s http://127.0.0.1:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"{model_dir}/Qwen-Image-2512","prompt":"a red teapot","size":"1024x1024","num_inference_steps":20,"seed":42}'
# 响应含 b64_json（PNG），解码后 file 校验应为 "PNG image data, 1024 x 1024"
```

启动已知坑（详见 `references/troubleshooting-vllm-omni.md` §E2）：

- `ImportError: libxcb.so.1` → opencv-python 依赖 X11 库 → `dnf install -y libxcb xcb-util* libX11 ... mesa-libGL`
- 权重分片缺失 → `ValueError: ... weights were not initialized from checkpoint` →
  对照 `*.safetensors.index.json` 逐个核对分片，缺失的从 hf-mirror 补下载
- 950PR 上**不要设置** `MINDIE_SD_FA_TYPE`（该变量不适用于 950PR/950DT）

### Edit 类模型验证（Qwen-Image-Edit-2511 等）

Edit/I2I 模型（`QwenImageEditPlusPipeline`）必须用 `/v1/images/edits`（multipart），
不能走 `/v1/images/generations`（会 500 Missing preprocess images）：

```bash
# 1) 造一张输入图（或直接用现有图片）
python - <<'PYEOF'
from PIL import Image, ImageDraw
img = Image.new('RGB', (1024, 1024), (135, 206, 250))
d = ImageDraw.Draw(img)
d.rectangle([200, 300, 824, 900], fill=(34, 139, 34))
img.save('/tmp/test_input.png')
PYEOF

# 2) multipart 编辑请求
curl -s -m 600 -X POST "http://127.0.0.1:8091/v1/images/edits" \
  -F "model={model_weight_dir}/qwen/Qwen/Qwen-Image-Edit-2511" \
  -F "image=@/tmp/test_input.png" \
  -F "prompt=Convert this landscape to a watercolor painting style" \
  -F "size=1024x1024" -F "num_inference_steps=20" -F "seed=42" \
  -o edit_response.json
# 响应含 b64_json（PNG），解码后 file 校验应为 "PNG image data, 1024 x 1024"
```

> 端点选型：`QwenImagePipeline`（Qwen-Image / Qwen-Image-2512）→ `/v1/images/generations`（仅 prompt）；
> `QwenImageEditPlusPipeline`（Qwen-Image-Edit-2511）→ `/v1/images/edits`（multipart + image 文件）。
> 模型切换：`pkill -9 -f "vllm serve"` 后用新权重路径重启同一命令即可。

### 特性开关生效验证

按框架入口执行，重点检查特性叠加是否生效：

- 量化开关 → 权重精度是否符合预期（950PR 默认 `--quantization mxfp8`）
- 稀疏开关 → sparsity 参数是否生效
- Cache 开关 → 缓存命中日志有无

验证通过标准：

| 检查项 | 标准 |
|--------|------|
| 推理无异常 | 无 `RuntimeError` / `OOM` / `CANN error` |
| 输出合法 | shape > 0，非全零输出 |
| 显存正常 | 峰值 < 物理显存 90% |
| 特性叠加 | 量化/稀疏/Cache 开关生效 |

**计数契约（真实性核验，防 no-op 假加速）**：开关「生效」以外，宣称收益前还须有技术
真实参与的**运行期计数**并写入产物——缓存复用次数、稀疏/量化 kernel 调用数、融合 kernel
实际执行次数等；可设 fail-closed 断言（如「预期 dense N / sparse M / actual kernel M」，
不符即运行契约失败）。与 model-auto-optimization「声明纪律与真实性核验」配合，
计数口径记入 artifact-layout 产物（technique_counters）。

## 运行时算子接入（有抽象接口）

LightX2V 等框架若提供 `rms_type` / `rope_type` 类注册表接口：把算子替换为 mindiesd 单算子
（如 `npu_rms_norm`、`npu_rotary_mul`），收益参考实测（rms -6.2%、rope 合计约 -10%）；
无接口框架走 compile（见下）。详细方法见各框架 case。

## compile 融入（无抽象接口 / 图形态不匹配）

以 DiffSynth-Engine（Qwen-Image）为例的 3 处适配（详见 `references/diffsynth-engine-notes.md`）：

| 位置 | 改动 |
|---|---|
| `configs/base.py` + `args.py` | `PipelineConfig.compile_backend="mindie"` + `--compile-backend` CLI |
| `pipelines/base.py` | `compile_transformer_blocks` 对 `_repeated_blocks` 逐个 `submodule._compiled_call_impl = torch.compile(submodule._call_impl, backend=MindieSDBackend(), fullgraph=False)` |
| 模型层 | Qwen-Image RoPE 改实数域等价（命中 `qwen_rope_pattern`）；text encoder key 归一化（transformers>=5.x 布局） |

跨框架通用姿势：

- ⚠️ `torch.compile(submodule, backend=...)` 不赋值不生效，必须写入 `_compiled_call_impl`
  （等价 `nn.Module.compile()`）。未赋值时 warmup 与 eager 一致、pattern 0 命中。
- **backend 实例复用**：每次新建 `MindieSDBackend()` → Dynamo `BACKEND_MATCH failure`
  反复重编译直至 `recompile_limit` 后静默退回 eager；须缓存单实例。
- **compile backend 经平台注册表注入（推荐形态）**：不要在框架核心硬编码第三方包名。
  LightX2V 合入版（#1471）以 `COMPILE_BACKEND_REGISTER`（merge 平台级
  `PLATFORM_COMPILE_BACKEND_REGISTER`）解析 `compile_backend`——平台侧注册懒工厂
  （`compilation/ascend_npu/mindie.py`），core 只查注册表、未知名 **raise**、不静默回退；
  实例仍须缓存单份。
- **collective 留 eager（防 a2a 退化）**：多卡序列并行（USP/Ring）下若把 collective 编进图，
  `split_sizes` 可能被推断错导致 `hcom_alltoallv` 退化（200-270ms/次 vs 固定 alltoall ~4ms）；
  用 `torch._dynamo.disable`/`torch.compiler.disable` 把 collective 留在 eager（graph-break），
  计算段继续融合。
  - ⚠️ **优先框架原生可选机制，不要全局改共享 collective 类**：LightX2V 合入版提供平台注册
    后端 `seq_p_a2a_backend="hccl_eager"`（`ops/a2a/ascend_npu/`，`@torch.compiler.disable`），
    common 零改动、其它平台不受影响。全局钉 disable 到 `TorchUlyssesA2A.exchange` 的版本
    曾被评审收敛为"平台注册 + 配置可选"（LightX2V #1471 真实案例）。
- **图形态差异**：同一算子在不同框架 Dynamo 展开形态不同（如 `chunk(2)` 在 diffusers 是单
  split 双 getitem、在 LightX2V 是两个独立 split 节点）→ pattern 需加变体；诊断用
  `graph.print_readable()` 对比 pattern 期望与真实图。

融合算子使能判断（三层证据，可靠性递增）：

```text
MINDIE_LOG_LEVEL=DEBUG 日志 → graph_log_url DOT 图 → kernel_details.csv
```

- ⚠️ 日志 2048 字符截断会误判 0 命中 → 以 DOT 图 / kernel_details.csv 为准
- ⚠️ 图命中 ≠ 运行期全部生效：`residual_gate_add` 对 4D attention 张量运行期 fallback，
  需看融合 kernel 实际执行次数
- 不做耗时比较：dummy 与真实权重的耗时差异由层数主导，耗时评估必须真实权重 + kernel diff
  （证据走 profiling-collect → profiling-analyze）
- ⚠️ **稀疏/融合的"路径区分"先于收益判定**：eager 公共 API（外部逐层 mask，如 rf_v3）与融合算子
  （op 内 mask+BSA 一体，如 EagleQBSA）是两条不同实现，墙钟/收益口径不可混称；先 op 级微基准
  （同几何对拍 dense vs 目标路径，`import mindiesd` 前置，warm+稳态取 ms）确认 kernel 价值，再接线
  e2e 复验（方法见 profiling-analyze heuristics「Host/Kernel 与数据搬运归因」，案例
  cache-dit-minimax-h3-case §4.6/§5）

## 案例与实测

| 参考 | 加载时机 |
|---|---|
| `references/troubleshooting-vllm-omni.md` | vLLM-Omni 构建/启动/运行期异常（§E1/E2/E3） |
| `references/diffsynth-engine-notes.md` | DiffSynth-Engine compile 接入细节（部署要点见 env-install） |
| `references/diffsynth-engine-case.md` | DiffSynth-Engine 性能/使能验证与实测结论（§6 复核：无损近 floor、attention 进图中性回退、DiTBlockCache/AttentionCache 有损使能矩阵、双缓存互斥、CFG-on shape 约束） |
| `references/cache-enablement-pattern.md` | 三方框架用 mindiesd 库现有 CacheAgent/DiTBlockCache/AttentionCache 做有损缓存的通用姿势（接入三要素 + 验证三件套 + 实测档位 + 坑速查） |
| `references/lightx2v-mindiesd-case.md` | LightX2V 接入 mindiesd 完整案例（runtime 接入 + compile 修复 + kernel diff；已对齐合入版 #1471：`hccl_eager` / `COMPILE_BACKEND_REGISTER`；含框架侧结构性实现——**跨侧，开发姿势/流程见 framework-extension-dev**） |
| `references/vllm-omni-case.md` | vLLM-Omni 托管模型 + mindiesd 的性能使能方法 |
| `references/vllm-omni-minimax-h3-case.md` | MiniMax-H3 × vLLM-Omni 0.28（NPU 950PR）全特性叠加案例：S1 自动生效面 / compile kernel 级验证（负面结论 + 单步 kernel 采集 hook 方法）/ RAINFUSION mask-free 回修 / 无损·有损组合数字与质量门禁 / 并行选型与通信分析；含框架侧 fork 实现记录——**跨侧，开发见 framework-extension-dev** |
| `references/cache-dit-minimax-h3-case.md` | **cache-dit（框架本体 trunk）× MiniMax-H3 × vLLM-Omni 0.26**（NPU 950PR）：cache-backend cache_dit 使能 + DBCache 参数面（F/B/R/W/MC+TaylorSeer）与档位扫描（**MC 主导、R 0.3–0.6 平坦**）；量化 mxfp8/int8 单点与 fp8 配置缺口；稀疏 FA 两条路径——**eager rf_v3（外部逐层 mask，kernel 级已确认，1.15–1.25× 近无损）vs EagleQBSA 融合 op（稀疏FA+FA量化，op 级快 ~6×，含自研算子部署坑 `ASCEND_CUSTOM_OPP_PATH` 前置）**；MindieSDBackend compile 接线与**泛型 pattern 误触回修**；受控交错 A/B 判定无收益回退；kernel 级计数契约兜底；**叠加组合实测（sparse×Cache 3.4–3.7×、量化×稀疏×Cache 4.1–4.5×，start0 质量劣化）**；含耗时与卡数关系（4 卡，e2e 不随卡数线性） |
| `references/vllm-omni-qwen-image-case.md` | **Qwen-Image-2512（图像）× vLLM-Omni 0.28**（NPU 950PR）首个图像模型 case：自动路由核验（FA/AdaLN/RoPE/GELU eager 覆盖）/ compile 输出非无损否决 / TP2 采纳 + 4-rank 短任务通信病态 / 量化(w8a8)+Cache 组合 -24%（质量 33.9/0.960）/ **rf_v2 图像不可用判据（qkv_layout/BSND 前置）+ 图像质量域口径（21-seed 像素对）** |

> 单次结论不迁移：案例中"某修复必要/有效/无效"仅在该框架 + 并行配置下成立；
> 换框架/换配置必须按「使能与验证回路」重新验证。

## 故障排查

- 通用部署问题（SSH、docker exec、CRLF、环境依赖）→ `env-install/references/troubleshooting-env.md`
- vLLM-Omni 构建/启动/运行期异常 → 见上表 `references/troubleshooting-vllm-omni.md` 行

## Reference Files

- `references/framework-support-matrix.md` — 加载时机: 使能某特性前查框架支持面 / 使能失败排查 /
  S4 组合前裁剪候选 / 框架或 docs 版本升级后刷新（特性命名以 docs/zh/features 为准）
- 各框架案例/笔记/接入姿势（加载时机与摘要）→ 单点登记于上方「案例与实测」表，勿重复登记
- 环境安装与权重准备 → `env-install/SKILL.md`
- 证据（采集/kernel diff）→ `profiling-collect/SKILL.md`、`profiling-analyze/SKILL.md`
- 实现层改动（pattern/算子）→ `compilation-dev/SKILL.md`、`operator-dev/SKILL.md`
- 快速验证 → `dummy-run/SKILL.md`

## 维护与更新

当 vllm / vllm-ascend / vllm-omni 版本或框架接入姿势变化、新增框架使能/回修经验、
或融合 pattern 使能集合变化时，按 dev-workflow 的复盘流程更新本 skill；框架特有实测结论
写入对应 case 文件，正文只保留通用回路与姿势。

> 体积提示：本 skill 是能力层中负载最重者（接近半编排）。若继续膨胀，优先把框架特定章节
> 下沉为 `references/{framework}-{variant}.md`（渐进披露），正文只保留通用回路与决策姿势。
> 边界判据：本技能 = 开启/使用框架**已存在**的特性（开关/接口/机制已具备，含为使能的最小 glue 修复）；
> 框架**未支持**特性的从无到有开发（comm-stream masking / 缓存框架 / 稀疏/量化消费者等）→
> `framework-extension-dev`（经 model-auto-optimization §0 确认）。
> 跨侧重叠：特性落地常需 mindiesd 能力 + 框架接入点两侧配合（合作界面随框架而异）。分工按
> **代码落点定侧**——mindiesd 仓部分归 dev-workflow + compilation-dev / operator-dev / aclgraph-dev；
> 框架仓部分归本技能（接线）或 framework-extension-dev（结构性）；对接联调在本技能验证回路收口
> （计数契约 + 三层证据），框架侧接口差异记录进 framework-support-matrix。
