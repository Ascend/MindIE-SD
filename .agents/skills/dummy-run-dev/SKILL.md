---
name: dummy-run-dev
compatibility: diffusers, transformers, modelscope（离线 config 可选）, NPU 设备
description: Dummy Run 模型验证：使用随机权重（无需真实权重）在 NPU 上快速验证新模型
             架构兼容性，评估参数量、显存占用与推理耗时。已部署模型的框架侧验证见
             framework-integration，部署见 ascend-deploy。
             当用户需要确认模型能否跑通、评估参数量和显存占用、
             查验各组件构造耗时分布时使用此 skill。
             即使用户只提到"帮我试试这个模型能不能跑"而未说 dummy run，也应触发。
             由 dev-workflow 的验证阶段触发。
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
| **部署状态** | ascend-deploy 是否已完成 | 已部署 / 未部署（需先执行 ascend-deploy） |

## 路径判断

```text
├─ 无真实权重 → §A Dummy Run 构造验证
└─ 有真实权重 → framework-integration §1 部署验证（需 ascend-deploy 完成）
```

> 环境部署问题见 ascend-deploy。NPU OOM 处理见 ascend-deploy §2 Step 7。

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

> 环境问题（`diffusers` 缺失、`sentencepiece` 缺失）见 ascend-deploy 故障排查表。

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

> ⚠️ 历史教训：w8a8/mxfp8 compile 曾比 eager 慢 11~229×，根因是量化层 forward 内就地改
> `self.bias` dtype 导致 Dynamo guard 每次失败重编译（~1.8s/次）。已修复（局部变量）。
> 遇到 compile 远慢于 eager 先跑 `TORCH_LOGS=recompiles`，详见 compilation-dev §4。

---

## §B 部署验证 → framework-integration

> 已部署模型的框架侧验证（vLLM Omni / Cache DiT + diffusers / 魔乐社区）已迁出，
> 见 `framework-integration` skill §1（含 vLLM-Omni 全栈部署 §2）。

---

## 阶段耗时与显存追踪

推理耗时与显存的精确追踪使用 _PhaseTimer 类，详见 references/phase-timer.md。

> 需要更细粒度的 profiling 数据（kernel_details.csv, trace_view.json）时使用 profiling-collection。

## Reference Files

- 📦 `references/construction-methods.md` — 加载时机: 涉及模型构造方式选择或 gated model 配置文件获取时
- ⏱️ `references/phase-timer.md` — 加载时机: 需要精确追踪推理各阶段耗时与显存时
- 📝 `references/minimax-h3-notes.md` — 加载时机: 涉及 MiniMax-H3 算子语义（npu_swiglu 等）、真实图形态或模型级验证时
- 🧩 `references/model-common.md` — 加载时机: 涉及 `--quant` 模式、`model/common` 共享模块职责/API、或 w8a8 设备映射时

## 维护与更新

当遇到新模型的兼容性问题、框架版本升级导致验证方式变化、
或发现新的 NPU 算子兼容性问题时，按 dev-workflow 的复盘流程更新本 skill。
