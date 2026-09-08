---
name: dummy-run
compatibility: diffusers, transformers, modelscope（离线 config 可选）, NPU 设备
description: 简化模型代码的快速验证载体：用随机权重/精简代码在 NPU 上快速验证模型架构
             兼容性、算子先接入与融合可行性（产物：架构兼容性 + 基线/快验输出），无需真实权重。
             已部署模型的框架侧验证见 framework-feature-enablement，环境安装见 env-install。
             当用户需要确认模型能否跑通、评估参数量与显存占用、或在算子接入/融合优化前
             做快速验证时使用此 skill；即使用户只提到"帮我试试这个模型能不能跑"而未说
             dummy run，也应触发。由 model-auto-optimization 的 S0/S1（融合/编译快验）阶段调用，
             亦由 dev-workflow 的模型验证阶段指引加载。
---

# Dummy Run 验证（模型架构兼容性）

使用随机权重在昇腾 NPU 上快速验证模型架构能否正确推理，无需下载真实权重。

## 前置确认

验证前必须明确以下信息：

| 确认项 | 说明 | 示例 |
|--------|------|------|
| **模型名称 + 规格** | 完整的模型标识 | `FLUX.1-dev`, `Wan2.2-T2V-14B` |
| **框架** | 推理框架 | `vLLM Omni` / `Cache DiT + diffusers` / `魔乐社区` |
| **是否有真实权重** | 权重状态 | 仅配置文件 / 已下载完整权重 |
| **依赖清单** | 需要的 Python 包 | `diffusers`, `transformers`, `sentencepiece` |
| **部署状态** | env-install 是否已完成 | 已安装 / 未安装（需先执行 env-install） |

## 路径判断

```text
├─ 无真实权重 → §A Dummy Run 构造验证
└─ 有真实权重 → framework-feature-enablement 框架侧验证（需 env-install 完成）
```

> 环境部署问题见 env-install。NPU OOM 处理见 env-install 故障排查。

---

## §A Dummy Run 构造验证

使用随机权重构造模型，快速验证架构兼容性，无需下载几十 GB 的真实权重。

### A1 适用场景

- 在 NPU 上验证模型架构兼容性，无需下载真实权重
- 评估模型参数量、显存占用、推理耗时
- 先验证能跑通，再决定是否下载完整权重

### A2 手动逐组件构造

```python
# Transformer / VAE / Scheduler：从 diffusers 加载 config → 随机权重
transformer_cfg = FluxTransformer2DModel.load_config(config_dir, subfolder="transformer")
transformer = FluxTransformer2DModel.from_config(transformer_cfg, torch_dtype=torch.bfloat16)

# Text Encoder：从 transformers 加载 config → 随机权重
clip_cfg = CLIPTextConfig.from_pretrained(clip_dir)
text_encoder = CLIPTextModel(clip_cfg).to(torch.bfloat16)

# Tokenizer：需要真实词表文件（KB 级），通过公开 repo 下载
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)

# 组装 Pipeline
pipe = FluxPipeline(
    scheduler=scheduler, vae=vae,
    text_encoder=text_encoder, tokenizer=tokenizer,
    transformer=transformer, ...
)
```

配置文件可通过 modelscope 离线下载（KB 级），命令行 `--config_cache` 指定路径，无需 `HF_TOKEN`。
两种构造方式的适用场景与已知陷阱详见 references/construction-methods.md。

### A3 优化技巧

| 优化项 | 方式 | 效果 |
|---|---|---|
| 减少 Transformer block 数 | `transformer_cfg["num_layers"] = 2` | 参数量大幅降低 |
| 关闭 CFG | `guidance_scale=1.0` | Transformer forward 减半 |
| 跳过 VAE decode | `output_type="latent"` | 跳过 ~70% 推理耗时 |
| 配置缓存 | `snapshot_download(model_id, local_files_only=True)` | 免联网，秒级启动 |
| Warmup + Timed 分离 | 先 warmup 不计时，再 timed 计时 | 排除 JIT 冷启动 |

> `num_layers` 裁剪仅适用于支持动态层数的模型。

### A4 验证结果示例

```text
transformer params:       14.29 B
Total params:             34.38 B
Estimated memory (bf16):  64.0 GB

[CPU offload mode]
Build time:               359.1 s
Inference time:           122.1 s (2 steps, 5 frames)
Peak NPU memory:          18.74 GB
Verification:             PASSED
```

### A5 常见问题（构造专属）

| 问题 | 原因 | 解决 |
|---|---|---|
| `AttributeError: 'list' object has no attribute '__module__'` | Pipeline.from_config() tokenizer bug | 手动逐组件构造 |
| `TransformersModel has no attribute 'from_config'` | transformers 版本不支持 | `ModelClass(config)` |
| `CLIPConfig has no attribute 'hidden_size'` | AutoConfig 返回错误类型 | `CLIPTextConfig.from_pretrained()` |
| `NotImplementedError: Cannot copy out of meta tensor` | meta→to_empty 后 buffer 残留 | 降级为 CPU 直接构造 |

> 环境问题（`diffusers` 缺失、`sentencepiece` 缺失）见 env-install 故障排查表（troubleshooting-env.md）。

### A6 量化与精度模式（`--quant`，统一接口）

各 `*_infer.py`（wan / minimax / qwen / flux）统一使用 `--quant {fp32|bf16|w8a8}`（默认 `bf16`，
替代原 `--compute-precision`；hunyuan_image3 暂未接入）：

| 模式 | 行为 |
|---|---|
| `bf16` | 模型级 bf16 计算精度：权重 cast + `.float()` 精度岛源码级改写（compile 图真正 bf16） |
| `w8a8` | W8A8 在线量化（Matmul-only）on bf16 基座；**格式按 NPUDevice 自动选择**：A5（950PR）→ **MXFP8**，A2/A3（910B/910C）→ **INT8**（`W8A8_DYNAMIC`） |
| `fp32` | 原 fp32 计算 |

量化范围（kernel 实证）：只有 `nn.Linear` 被替换为在线量化 Linear（`npu_quant_matmul` /
`npu_dynamic_mx_quant`），GroupMatmul 仅走 MoE 路径（dummy 无 MoE 不触发），FA 不量化；
**其余向量运算（norm/rotary/attention/gate 等）保持 bf16**，图中无 fp32 计算节点。

共享能力位于 `examples/dummy_run/model/common/`（按职责分类）：
`precision.py`（bf16/fp32 机制）、`compile_patches.py`（dropout/pos_embed 性能补丁）、
`quantization.py`（w8a8 设备感知 + `apply_w8a8_quant`）。
**完整模块职责/API/接入方式见 `references/model-common.md`。**

**性能基线**（950PR，2 层 dummy，transformer Timed，compile vs eager）：

| 模式 | 结论 |
|---|---|
| bf16 compile | 全面小幅加速（-7% ~ -22%） |
| w8a8 compile | 修复量化层 guard bug 后全面加速（-7% ~ -30%），且优于 bf16 compile |

**diffusers 0.40 全模型 wall（w8a8，A310-50，三模型并行一模型一卡，2026-09-06）**：

| 模型 | eager (ms) | compile (ms) | 收益 |
|---|---:|---:|---:|
| Wan2.2-T2V (dummy) | 951.49 | 841.76 | **-11.5%** |
| MiniMax-H3 (dummy，FFN 融合 compile pattern 承载) | 21.77 | 15.65 | **-28.1%** |
| FLUX.1-dev (dummy) | 17.2 | 15.14 | **-12.0%** |

> 并行复现口径：每模型独立卡、eager→compile 同卡先后；`PYTHONPATH=/tmp/dif040_site`
> （diffusers 0.40 隔离安装，MiniMax-H3 需要）；qwen_image 在 0.40 broken 排除。
> MiniMax-H3 的 mm_swiglu_mxquant 融合**不接入 eager**（dummy 无 layer-route patch），
> 由 compile 侧 pattern `enable_minimax_h3_ffn_fusion`（默认 True）在图编译期命中承载；
> kernel 级 §C 双报表见 `references/compile-ab-report-template.md` §3
> 与 `tmp/mmx_w8a8/dummy_ab_reports_mmx_fusion_default_on.md`。
>
> ⚠️ 历史教训：w8a8/mxfp8 compile 曾比 eager 慢 11~229×，根因是量化层 forward 内就地改
> `self.bias` dtype 导致 Dynamo guard 每次失败重编译（~1.8s/次）。已修复（局部变量）。
> 遇到 compile 远慢于 eager 先跑 `TORCH_LOGS=recompiles`，详见 compilation-dev §4。

---

## 算子先接入 / 融合可行性快验

在真实权重接入（framework-feature-enablement）与融合优化（model-auto-optimization S1 融合）之前，用本载体做两件事：

- **算子先接入**：把候选 mindiesd 单算子（如 npu_rms_norm / npu_rotary_mul）先替换进 dummy 的模型代码，验证调用姿势与输出一致性（配合 framework-feature-enablement 的运行时接入姿势）；
- **融合可行性快验**：pattern 命中基于图结构、不依赖层数与权重 → 在 dummy（num_layers=2）上开启 compile/开关，用图 dump 看候选 pattern 是否命中实际图形态；命中后再到真实权重做 kernel diff + 墙钟复验（profiling-collect → profiling-analyze）。

快验清单（每步留证据）：

1. 小层数 dummy 跑通（eager 基线）
2. 开启目标使能项（runtime 替换或 compile，姿势见 framework-feature-enablement）
3. 图命中确认（graph.print_readable / pattern dump；需新增 pattern 时指向 compilation-dev）
4. 输出一致性（dummy 内前后对比）
5. 结论（可行/不可行 + 原因）交 S1 融合决策；不可行不进入真实权重试错

> 注意：dummy 的耗时不代表真实权重（层数主导），快验只回答「能不能命中/姿势对不对」；收益评估必须回到真实权重。

---

## 阶段耗时与显存追踪

推理耗时与显存的精确追踪使用 _PhaseTimer 类，详见 references/phase-timer.md。

> 需要更细粒度的 profiling 数据（kernel_details.csv, trace_view.json）时使用 profiling-collect。

---

## §B' 并行跑多个 dummy 模型（profile 目录隔离）

同一 host 并行验证多个模型（一模型一 NPU 卡）时，**必须隔离 profiling 输出目录**：
所有 `*_infer.py` 的 `--profile` 默认写 `./profile_l1`（CWD 内），并行 worker 共用会互相
覆盖 → kernel_details 缺失/损坏（实测：并行采集 wan csv=None、flux 被拖慢）。

- 每个 `*_infer.py` 支持 `--profile-dir {dir}`，或环境变量 `DUMMY_PROFILE_DIR={dir}`
  （模块级 `PROFILE_DIR = os.environ.get("DUMMY_PROFILE_DIR", "./profile_l1")`）。
- 并行时按 模型×配置 给独立目录：`--profile-dir {work}/profiles/{model}_{eager|compile}`；
  采集完再按模型回传/聚合，不要事后从共享 `profile_l1` 猜归属。
- 纯 wall timed（无 `--profile`）不写 profile_l1，无此冲突；kernel 级报表才需要隔离目录。
- 例：`examples/dummy_run/README.md` CLI 表已登记 `--profile-dir`；远端并行 AB 脚本
  （allmodels_ab.py / mmx_v040_ab.py）按此隔离。qwen_image dummy 在 diffusers 0.40
  broken（`QwenEmbedRope._compute_video_freqs` 对 str device 调 `.type`），0.40 全模型
  AB 排除之。

---

## §C compile vs 非 compile（eager）对比：强制双报表

**强制要求（每次比较 compile 与非 compile 的收益时必须产出，缺一不可）**：
同一配置分别跑 eager（非 compile）与 compile（开启 MindieSDBackend + 融合开关），各采一份
profile（各 `*_infer.py` 用 `--profile`；并行时加 `--profile-dir` 隔离，见 §B'），在结果中附
下列两张表（markdown 表格）。口径统一如下：

- **数据窗口**：eager/compile 各一次 transformer timed + 对应 `kernel_details.csv` 聚合；
  "融合"指 compile 相对 eager 引入的融合单元（pattern/自研算子：npu_rms_norm、npu_rotary_mul、
  gather_scale_shift、gather_residual_gate、swiglu/FFN fused op、mm_swiglu_mxquant 等）；
  非融合的 GEMM/FA 不拆行，只出现在 detail 的执行序里作为"未融合"行。
- **Block 耗时基准**（两张表的收益分母）：**非 compile（eager）下该模型 transformer block
  单步耗时**（= eager timed transformer / 步数；无法拆步时用 eager transformer timed）。
  相对收益一律 = (融合后耗时 − 融合前耗时) / eager block 耗时 × 100%。
- **未实现填充**：尚未实现的融合单元行——"是否完成融合=N（预期可融合：{依据}）"、
  "融合后耗时=融合前耗时"、"收益=0"（预期值不得虚填）。
- 站点→kernel 归属需结合图 dump（融合后节点）与 eager 分解链做映射，避免仅按 kernel 名猜测。

### 总览表（overview：按融合算子一行）

| 序号 | 融合算子 | 融合前组成（eager 被替代链） | 是否完成融合 | 融合前耗时 | 融合后耗时 | 相对融合前 block 耗时的收益 |
|---:|---|---|:--:|---:|---:|---:|
| 1 | npu_rms_norm | Pow+Mean+Rsqrt+Mul/Add 分解链 | Y | 1.20ms | 0.46ms | -3.3% |

### 明细表（detail：按 block 算子执行序列行）

| 算子归属（Attn/FFN(MoE)） | 融合后所属算子 | 未融合时的组成 | 融合后性能 | 未融合性能 | 相对未 compile block 耗时的收益 |
|---|---|---|---:|---:|---:|
| FFN | mm_swiglu_mxquant（mm+swiglu+mxquant） | DxQ→Qmm([S,2F])→swiglu→DxQ→Qmm(out) | 1.85ms | 2.03ms | -1.1% |

- detail 的"融合后所属算子"：该行在 compile 图里的算子名（原算子未变则填原名）；
  "未融合时的组成"：eager 下该站点由哪些 kernel 组成（原算子未变则填单一算子名）。
- 每行性能为该站点/算子覆盖 kernel 的耗时合计（kernel_details 聚合）；
  "未融合性能"对已完成融合的行 = eager 侧被替代链合计；收益 = (未融合性能−融合后性能)/eager block 耗时。

示例（MiniMax-H3 w8a8 fusion-on 口径）见 `references/compile-ab-report-template.md`；
融合收益归属分析（GEMM 不变、小 kernel 融合）见
`../compilation-dev/references/pattern-dev-notes.md` §5。
0.40 全模型双报表（wan/flux kernel 级 + minimax wall/kernel）：
`tmp/mmx_w8a8/dummy_ab_reports_v040.md`、`dummy_ab_reports_mmx_fusion_default_on.md`。

## Reference Files

- `references/construction-methods.md` — 加载时机: 涉及模型构造方式选择或 gated model 配置文件获取时
- ⏱️ `references/phase-timer.md` — 加载时机: 需要精确追踪推理各阶段耗时与显存时
- 📊 `references/compile-ab-report-template.md` — 加载时机: 做 compile vs 非 compile 对比、需要双报表模板与填写示例时（§C 强制项）
- `references/minimax-h3-notes.md` — 加载时机: 涉及 MiniMax-H3 算子语义（npu_swiglu 等）、真实图形态或模型级验证时
- `references/model-common.md` — 加载时机: 涉及 `--quant` 模式、`model/common` 共享模块职责/API、或 w8a8 设备映射时

## 维护与更新

当遇到新模型的兼容性问题、框架版本升级导致验证方式变化、
或发现新的 NPU 算子兼容性问题时，按 dev-workflow 的复盘流程更新本 skill。
