# DiffSynth-Engine：MindIE-SD compile 接入与使能判断（框架差异记录）

> 定位：本文件只放 **DiffSynth-Engine**（`diffsynth_engine`，Qwen-Image 等扩散模型的独立推理框架）
> 这条链上的**特有**内容——框架画像与版本边界、部署与启动前置、compile 接入与使能判断、
> 与其它框架的差异、回修与坑。**本文件由原 `diffsynth-engine-case.md`（性能/使能视角）与
> `diffsynth-engine-notes.md`（部署/接入细节）合并而成**，两文件的互补关系到此收口，不再互相指引。
> **通用方法与判定纪律**见 `framework-integration/SKILL.md` 正文「融合算子使能判断（三层证据）」段、
> `pattern-dev/references/fusion-enablement-notes.md`（融合开关治理）、
> `model-auto-optimization/references/lossless-methodology-notes.md`（无损·计算/通信方法）、
> `accuracy-gate/references/quality-gate.md`（质量门禁）、
> `references/cache-enablement-pattern.md`（三方框架用库现有 CacheAgent 做有损缓存的通用姿势）；
> **能力支持面**（✅/🟡/❌/❓ 与证据码 V2）见 `references/framework-support-matrix.md`。
> **本文件不写绝对耗时与绝对质量分值**（二者只在本次环境成立，不可迁移）；**要写**大致加速比
> （量级/约数）与质量变化度（降幅/差值）：
> 迁移前的绝对数字原文归档于 `{run_results_dir}/archive/diffsynth-engine-case.md`，
> 结构条目留档（该文件无绝对性能数字）见 `{run_results_dir}/archive/diffsynth-engine-notes.md`
> —— 两者只在原环境成立，**不可跨模型 / 框架 / 规模 / 窗口引用**。
> 路径占位符：`{model_weight_dir}`（权重根）、`{run_results_dir}`（运行产物根）、`{user}`（远端账号占位）。
> 本文件的方向性结论一律是该框架 × 该模型 × 该规模的**观测**（标注「本组合观测」），不是执行序
> （换框架 / 模型 / 负载规模须重判）。

## 1. 画像与版本边界

### 1.1 画像速答（迁移清单）

| 检查项 | DiffSynth-Engine 实况 |
|---|---|
| 算子注册机制 | 无 `rms_type` / `rope_type` 类抽象接口（模型层硬编码 torch 算子）→ 走 compile 接入 |
| 并行实现 | 单卡 / 无 Ulysses（`_repeated_blocks` 逐个编译），无 a2a → 无「a2a 留 eager」问题 |
| compile 图形态 | `Pipeline.compile_transformer_blocks` 对 submodule 编译；chunk/split 展开形态与 dummy 类似 |
| 动态 shape | 序列 / 分辨率固定（1024²），无 `Sym` 符号 → dynamic 参数无 guard 负担 |
| 长序列通信 | 单卡无通信；融合收益不被通信稀释 |

**结论**：DiffSynth-Engine 与 LightX2V 差异大（无注册表接口、无 a2a）——LightX2V 的**运行时注册表接入路径
不适用**，必须走 **compile**（LightX2V 的 a2a 修复也不需要）。

### 1.2 版本边界与依赖前置

- **独立 Python 包**：`diffsynth_engine` 与 MindIE-SD 部署方式不同（见 §2）。
- **依赖锁冲突（决定 `--no-deps`）**：该框架 pyproject 锁定 `transformers==4.57.6` / `diffusers==0.36.0`，
  而容器内是 `transformers 5.14.1` / `diffusers 0.38.0`（vllm-omni / mindiesd 依赖）。
- mindiesd 侧以 **compile 工作区**源码路径注入（见 §2.1），不改容器内已装的 mindiesd。
- 使能结论只在**该框架版本 + 该模型（Qwen-Image，真实权重 60 层 / 1024²）+ 该容器环境**成立。

## 2. 部署与启动前置

### 2.1 部署步骤（远端容器）

| 步骤 | 命令 / 要点 |
|---|---|
| 增量传输 | 复用 remote-access 的 SSH 连接复用 + env-install 的增量上传（排除 `.git` / `__pycache__` / `tests`），目标 `/home/{user}/code/DiffSynth-Engine` |
| 安装 | `cd /home/{user}/code/DiffSynth-Engine && pip install -e . --no-deps` |
| 激活 mindiesd | 脚本内 `sys.path.insert(0, "/home/{user}/code/mindie-sd-compile")`（compile 工作区）——**避免用 pip 替换容器内已装的 mindiesd**（替换会波及 vllm-omni 等其他容器共享依赖） |

### 2.2 已知坑（安装前必读）

- **setuptools-scm 无 `.git` 报错**：远端源码包无 `.git` 时 `pip install -e .` 报
  `LookupError: setuptools-scm was unable to detect version`。
  解决：`export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIFFSYNTH_ENGINE=1.0.0`。
- **`--no-deps` 的原因**：该框架 pyproject 锁定的 transformers / diffusers 版本低于容器内
  vllm-omni / mindiesd 所需版本（见 §1.2）⇒ 带依赖安装会把容器环境降级破坏，
  所以一律 `--no-deps`，随后用框架自有的 API 兼容检查确认可导入。

## 3. 特性开关面板

### 3.1 compile 接入（`compile_backend` + `_compiled_call_impl` 原地写入）

- 配置入口：
  - `diffsynth_engine/configs/base.py`：`PipelineConfig` 增加 `compile_backend: str = "inductor"`
    （可选 `"mindie"`）；
  - `diffsynth_engine/args.py`：`--compile-backend {inductor,mindie}` CLI。
- 接入点：该框架的 compile 入口是 `Pipeline.compile_transformer_blocks`（对 `_repeated_blocks`
  逐个 `submodule.compile()`）。**必须等价于 `nn.Module.compile()`：编译 `_call_impl` 并原地写入
  `_compiled_call_impl`**：

```python
if backend == "mindie":
    from mindiesd.compilation import MindieSDBackend
    compile_backend = MindieSDBackend()
...
for submodule in model.modules():
    if submodule.__class__.__name__ in repeated_blocks:
        if compile_backend is not None:
            # ⚠️ 必须等价于 nn.Module.compile()：编译 `_call_impl` 并原地写入 `_compiled_call_impl`
            submodule._compiled_call_impl = torch.compile(
                submodule._call_impl, backend=compile_backend, fullgraph=False
            )
        else:
            submodule.compile()
```

- ⚠️ **关键陷阱：`torch.compile(submodule, backend=...)` 不赋值不生效**。`torch.compile` 返回包装对象，
  直接调用不修改原模块；必须写入 `submodule._compiled_call_impl`（与 `nn.Module.compile()` 内部实现一致）。
  实测教训：第一次接入时直接 `torch.compile(submodule, backend=MindieSDBackend())` 未赋值
  → warmup 时间与 eager 完全一致、pattern 0 命中 → 排查到赋值问题后修正。
- **backend 实例复用**：`MindieSDBackend()` 单实例复用（勿逐 submodule 重建）——防 BACKEND_MATCH
  重编译；该结论**跨框架通用**（见 §4）。

### 3.2 命中 pattern 的模型层前置（Qwen-Image）

- **RoPE 实数域改写（命中 `qwen_rope_pattern` 的前提）**：该框架 `apply_rotary_emb_qwen(use_real=False)`
  原实现用复数域 `x_rotated = torch.view_as_complex(x.float() ...)`。要命中 `qwen_rope_pattern`，
  应改写成实数域等价形式（与 dummy run 的 compute_precision `_rewrite_apply_rotary_emb_qwen` 完全一致）：

```python
xr, xi = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
cos = freqs_cis.real.unsqueeze(1).to(x.dtype)
sin = freqs_cis.imag.unsqueeze(1).to(x.dtype)
out_real = xr * cos - xi * sin
out_imag = xr * sin + xi * cos
x_out = torch.stack([out_real, out_imag], dim=-1).flatten(3)
```

  不改写的后果：① 图内 fp32 复数岛 → 非真 bf16 图；② `qwen_rope_pattern` 不命中。

- **text encoder key 归一化（正确性前提，非性能）**：容器 `transformers>=5.x` 的
  `Qwen2_5_VLForConditionalGeneration` 结构调整（`visual.*` → `model.visual.*`；
  `model.layers/embed_tokens/rotary_emb/norm.*` → `model.language_model.*`），而权重文件是 4.x 布局
  → `load_state_dict(strict=True)` 失败。解决：在 pipeline 加载时注入 key_mapping：

```python
key_mapping = {
    "^visual\\.": "model.visual.",
    "^model\\.layers\\.": "model.language_model.layers.",
    "^model\\.embed_tokens\\.": "model.language_model.embed_tokens.",
    "^model\\.rotary_emb\\.": "model.language_model.rotary_emb.",
    "^model\\.norm\\.": "model.language_model.norm.",
}
```

### 3.3 使能判断：三层证据（可靠性递增）

| 层级 | 方法 | 证据 |
|---|---|---|
| 1. 日志 | `MINDIE_LOG_LEVEL=DEBUG` | `PatternMatchPass replace N` |
| 2. 图 dump | `CompilationConfig.graph_log_url={dir}` | DOT 图出现 `npu_rotary_mul` / `npu_rms_norm` 等 |
| 3. kernel | torch_npu profiler → `kernel_details.csv` | 融合 kernel 出现 + 原始 kernel 消失 |

使能集合一致的判定标准：`kernel_details.csv` 中 `compile_only` 的融合 kernel
（`residual_gate_add_kernel` / `RotaryPositionEmbeddingV2` / `AdaLayerNormV2` / `FastGelu`）出现，
且对应的原始算子（`GeluV2`、逐元素 `Mul`/`Stack` 链）消失。

- **使能判断不依赖层数 / 权重**：pattern 匹配基于图结构，dummy run（随机权重 2 层）与真实权重
  （60 层）的融合 kernel **使能集合完全一致** → 用 dummy run 即可验证「外部框架下 pattern 是否使能」，
  真实权重直接复用该结论（耗时评估则必须真实权重 + kernel diff，见 §5.1）。
- 该框架 chain 的 **backend 实例复用**与 **collective 是否留 eager** 两条姿势按框架能力面判定，
  不照搬 LightX2V（见 §4）。

### 3.4 使能集合与收益结构（本组合观测）

| 融合 pattern | 融合 kernel（kernel_details.csv） | dummy run | DiffSynth-Engine |
|---|---|---|---|
| qwen_rope | `RotaryPositionEmbeddingV2` / `npu_rotary_mul` | ✅ | ✅ |
| qk_norm RMSNorm | `npu_rms_norm`（kernel 名 RmsNorm） | ✅ | ✅ |
| 残差 gate | `residual_gate_add_kernel` | ✅ | ✅（仅 3D 站点） |
| 调制 | `AdaLayerNormV2` | ✅ | ✅ |
| GELU | `FastGelu` | ✅ | ✅ |

- **收益结构（比例关系，必须留存）**：MatMul / addmm 占 kernel 总耗时**约三分之二**、FA **约一成半**
  ——两者都不参与 pattern 融合 ⇒ **无损融合的绝对上限就是「norm / rope / 激活 / 元素级」区段**
  （本模型为**个位数百分比量级**）。
- **单点收益排序**（顺序是方法，具体占比须现场实测；读数见归档）：
  **qwen_rope（最大单项）> adaln×3 > fast_gelu / rms_norm ≈ wan_residual_gate**；
  **逐项均净正、无负贡献** ⇒ 无损融合已近 floor（本组合观测）。
- **收益来源单一**：单卡无通信 ⇒ 端到端收益**完全来自算子融合**，没有 LightX2V 那条链的
  通信重叠红利；kernel diff 可见融合 kernel 出现且逐元素 `Mul`/`Stack`/`Cat` 大幅减少。
- **端到端方向**：真实权重（1024²，latent 输出）下 compile 相对 eager 有**稳定正收益**
  （warmup 后稳态测量，个位数百分比量级），输出正确性保持（latent 与 eager 高相似、
  VAE 解码图像肉眼一致）。
- **口径纪律（判定阈值）**：全部为同窗口同卡组、中位数口径；**<3% 视为噪声不宣称**。
  热卡 / 同机其他 vllm 服务争用会把 per-step 抬到 **2 倍以上量级**（失真）——**基线必须冷卡 +
  同窗口 + 中位数**，否则收益方向都可能判反。
- ⚠️ 上述结论仅在「该框架 + Qwen-Image + 该容器环境 + 该分辨率」成立；换配置须按
  「使能与验证回路」重验（dummy 只判使能，不判收益）。

### 3.5 有损使能（S4）——用库现有能力，未新增框架功能

- **框架能力边界（先读）**：DSE 框架本身**没有有损特性**——量化只匹配 `nn.Linear`（DSE 主算力为
  自定义 TP linear）；稀疏**无抽象接口**。⇒ 按用户指示**不向框架添加任何新特性**，有损验证全部使用
  mindiesd **库现有 CacheAgent 能力**（`DiTBlockCache` / `AttentionCache`）在 **bench 脚本侧**接入
  （DSE 源码零改动，`git status` + 远端 grep 双核实）。通用接入姿势（三要素 + 验证三件套）见
  `cache-enablement-pattern.md`，本文只记该链特有方向。
- **档位方向（本组合观测）**：CFG-off 单调用路径下扫描 DiTBlockCache 参数面——
  **收益最大档 = `[15,60) ss2 iv2`**（约三成）；**质量最优档 = `[30,60) ss4 iv2`**（复用步长更小，
  收益约一成）；**AttentionCache `[30,60) ss2 iv2`** 是粒度更细的质量最优方法（收益约两成）。
  **质量变化度**（vs 同在线基线）：收益最大档 latent cos 降约 0.01、像素偏离约 2%（mean-abs 口径）；
  质量最优档 cos 降约 0.001、像素偏离约 1% ⇒ **收益与质量同向拉开档位**。
  计数契约 `reuse` / `compute`
  与期望一致（与「复用步数 × block 数」吻合，**非 no-op**）。
- **无损 / 有损可独立叠加**：off-identity 复核显示 eager + DiT cache 与 eager + AttentionCache 的
  latent 偏差与 compile + 同档一致 ⇒ **两个方向互不抵消**；cache=0 档可复现基线（disable 恢复）。
- **双缓存叠加 = fail-closed（两方法互斥）**：`DiTBlockCache` + `AttentionCache` 同时使能时墙钟
  **看似更快但 latent 明显偏离**（低于接受线）→ **假加速**；根因是 DiT 复用步跳过 block 使内层
  AttentionCache 计数失步。两缓存方法**只能二选一**（docs 建议 DiTCache 优先、AttentionCache 备选）。
- **CFG-on 生产路径 + DiTBlockCache = shape 冲突**：pos / neg 文本长度不同 → 缓存 delta 形状不匹配
  （库语义约束：cache 假定每 denoise step 单次调用 shape 恒定），**非 bug**；需按分支独立 agent /
  text padding / batch-concat（框架集成项，**未做**）。
- **方向选择**：收益优先取 DiT `[15,60) ss2 iv2`；质量敏感取 AttentionCache 或 DiT `ss4` 档；
  双缓存不做叠加。**换框架 / 模型 / 规模须重判**。

## 4. 与其它框架的差异（照搬清单）

| 做法 | LightX2V（有效） | DiffSynth-Engine（是否需要） |
|---|---|---|
| a2a 留 eager（`torch._dynamo.disable`） | ✅ 必须（Ulysses a2a 在 block 内） | ❌ 不需要（无 a2a，单卡） |
| backend 实例复用 | ✅ 必须 | ✅ 同样必须（防 BACKEND_MATCH 重编译） |
| swiglu 双 split 变体 | ✅ 必须（`chunk(2)` 展开成双 split） | ❌ 按实际图形态验证（Qwen 无 swiglu） |
| 运行时注册表接入（rms_type / rope_type） | ✅ 有接口 | ❌ 无接口，走 compile |

**结论**：**backend 实例复用是跨框架通用**；**a2a 与 pattern 形态是框架特定**，必须各自验证。
三框架并列对照（含 vLLM-Omni）见 `vllm-omni-enablement.md` §4。

## 5. 回修与坑（`[探针]` 标注）

> §3.1 / §3.2 的改动均为**框架侧 fork 改动**（未合入上游，`compile_backend` 默认仍为 `inductor`）
> ⇒ 按本仓「经验 vs 探针」判定为**探针**，不作为推荐姿势与报表宣称；由探针发现的 **durable 约束**
> （部署顺序、使能判据、`_compiled_call_impl` 原地写入）按经验处理。本节以下为使用侧陷阱（经验）。

### 5.1 三个陷阱

1. **日志 2048 截断**：`MINDIE_LOG_LEVEL=DEBUG` 的 graph 行被 `MAX_LOG_STRING_LEN=2048` 截断
   → grep 搜不到融合 kernel 会**误判 0 命中**。以 `graph_log_url` 落盘 DOT（需 `pip install pydot`，
   否则 `FXGraphDrawer requires the pydot package`）或 `kernel_details.csv` 为准，二者不受截断影响。
2. **图命中 ≠ 运行期全部生效**：`residual_gate_add` 对 4D attention 张量运行期 fallback
   （日志 `fallback (ndim)`）→ 必须看 `kernel_details.csv` 中融合 kernel 的**实际执行次数**
   （3D 站点执行融合 kernel、4D 站点 fallback 到原生），不能只看图命中。
3. **不做耗时比较**：dummy（2 层 / 小参数量）与真实权重（60 层）的耗时差异由**层数**主导，
   dummy 编译时融合收益被新增 kernel（广播 / 填充等）淹没 → 「dummy 反而更慢」是**假象**。
   耗时评估必须真实权重 + kernel diff（证据走 profiling-collect → profiling-analyze）。

### 5.2 `residual_gate_add` 4D fallback（噪音，非致命）

- 现象：text-stream rope（动态 s31）的 `out_imag = xr*sin + xi*cos` add 链被通用 `x + y*gate`
  pattern 误匹配（`qwen_rope` 只吃掉 image 静态 rope），运行期 4D `[B,S,H,D//2]` 触发
  triton 3D-only fallback + **每调用一次 `print(flush=True)`** → 每次推理约**二百余次 python dispatch**
  与日志刷屏（墙钟影响在**数个百分点以内量级**）。
- 处置方向（按需）：`qwen_rope` pattern 补 text 动态 rope 变体（吃回子图）**优先于**给
  `residual_gate_add` 补 4D kernel；或 fallback print 限流。当前影响小，优先级中低。
- 与 LightX2V / MiniMax-H3 的同类误匹配根因（通用 pattern 抢跑）见
  `pattern-dev/references/benefit-rootcause-guide.md` §3 R4。

### 5.3 attention 进图编译：实测中性 → 回退（本组合观测）

- 姿势：`USPAttention.forward` 的 `@torch.compiler.disable` 仅在 SP 多卡路径保留
  （collective 留 eager），单卡路径放行进图。
- 结果：kernel 级**仅边界 copy 微减**（噪声内），FA 本体 kernel 不变（eager 与图内均为同一
  fused FA）；交错 A/B 墙钟**中性偏慢**。
- ⚠️ **附加风险**：`npu_fusion_attention` 被 compile 后 seed / offset 可能被优化（**在目标代际上用最小复现探针确认**），
  固定 seed 时结果可能偏离 eager（torch_npu 官方警告）⇒ 编译变更**不得宣称无损**。
- **结论（方向选择）**：**单卡 attention 进图无收益且引入 seed 语义风险 → 回退（不启用）**；
  FA 区段的收益应找 **FA 算子侧 / 布局侧**，不是「把它编进 compile 图」。
  该方向已登记于 `framework-support-matrix.md`（V2 行）。

## 6. 产物坐标指针（实测记录不入 skills）

- 运行产物根 `{run_results_dir}/`：bench 脚本（有损 CacheAgent 接入点，DSE 源码零改动）、
  `quality*/`（latent cos / 像素对拍）、`graphdump/`（`graph_log_url` 落盘 DOT）、
  `kernel_details.csv`（使能与 kernel diff 证据）。
- 权重坐标：`{model_weight_dir}/Qwen-Image`（diffusers 布局；真实权重 60 层 / 1024²）。
- **本次迁移归档（原 DSE 两文件的绝对数字）**：`{run_results_dir}/archive/diffsynth-engine-case.md`
  （§5 实测结论 + §6 复核与扩展结论的**原文**，含绝对耗时 / 加速比 / 质量数值）与
  `{run_results_dir}/archive/diffsynth-engine-notes.md`（结构条目留档；该文件**无绝对性能数字**）
  —— 只在原环境成立，**不可跨模型 / 框架 / 规模 / 窗口引用**，也不得读作该组合的预期值。
- 支持矩阵证据码 **V2** 的**能力面**留在 `framework-support-matrix.md`，**开启方式**指回本文件。
- 逐步耗时 / 显存 / kernel 明细属内部测量记录，不入库（见 `.agents` 治理与隐私约定）。

## 7. 维护与更新

- 触发（框架 / 依赖）：DiffSynth-Engine 或其 pyproject 锁定的 transformers / diffusers 版本、容器内 vllm-omni / mindiesd 所需版本变化时，§1.2 的依赖锁冲突与 §2.2 的 `pip install -e . --no-deps` + `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIFFSYNTH_ENGINE` 前置须重核；若 `PipelineConfig.compile_backend` 的上游默认值不再是 `inductor`（§5 的探针合入），§3.1 接入点与 §5 的探针标注须改写。
- 触发（使能集合 / 模型层前置）：`qwen_rope_pattern`、`residual_gate_add`、`AdaLayerNormV2`、`FastGelu` 等 pattern 的注册或命中形态变化，或 §3.2 的 RoPE 实数域改写 / text encoder key_mapping 所依赖的 transformers 布局（`visual.*` → `model.visual.*` 一类迁移）变化时，§3.3 三层证据与 §3.4 使能集合表、`framework-support-matrix.md` 的 V2 行须重测。
- 触发（有损接线）：§3.5 的档位来自 mindiesd 库现有 CacheAgent（`DiTBlockCache` / `AttentionCache`）在 bench 脚本侧的接入（DSE 源码零改动）——库侧 cache 接口、计数契约（reuse / compute）或双缓存互斥语义变化时，该节档位方向与「CFG-on 生产路径 shape 冲突」结论须重跑。
- 复核：最小核对 = dummy run（随机权重 2 层）复现 §3.3 的使能集合（pattern 匹配基于图结构、不依赖层数 / 权重），再在真实权重（60 层 / 1024²）上确认 `kernel_details.csv` 里 `residual_gate_add_kernel` / `RotaryPositionEmbeddingV2` / `AdaLayerNormV2` / `FastGelu` 实际执行、且对应原始算子（`GeluV2`、逐元素 `Mul`/`Stack` 链）消失——日志 2048 截断会误判 0 命中，须以 DOT 图或 kernel csv 为准。
