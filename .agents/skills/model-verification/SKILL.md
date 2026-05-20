---
name: model-verification
description: 在 NPU 上验证模型推理的正确性和兼容性。
             §A Dummy Run：使用随机权重快速验证新模型架构兼容性（无需真实权重）。
             §B 部署验证：验证已部署模型的推理正确性（依赖 ascend-deploy 完成部署）。
             产出粗粒度时序数据可传递给 performance-analysis 或 profiling-collection。
             当用户需要确认模型能否跑通、评估参数量和显存占用、
             查验各组件构造耗时分布时使用此 skill。
             即使用户只提到"帮我试试这个模型能不能跑"而未说 dummy run，也应触发。
             由 dev-workflow 的验证阶段触发。
---

# 模型验证

验证模型在昇腾 NPU 上能否正确推理。

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
└─ 有真实权重 → §B 部署验证（需 ascend-deploy 完成）
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

---

## §B 部署验证

> 前置条件：模型已通过 ascend-deploy 部署到 NPU 设备，`import mindiesd` 成功。

验证已部署模型在真实权重下的推理正确性。按框架选择验证方法：

### B1 vLLM Omni 部署

```bash
# 1. 检查服务状态
curl http://localhost:8000/health

# 2. 发送 1 次推理请求
curl http://localhost:8000/generate -H "Content-Type: application/json" \
    -d '{"prompt": "test", "max_tokens": 1}'

# 验证: HTTP 200 + 输出非空
```

### B2 Cache DiT + diffusers 部署

```python
# from_pretrained 加载已部署模型
pipe = FluxPipeline.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
).to("npu")

# 跑 1 步推理
output = pipe("test prompt", num_inference_steps=1)

# 验证: 无异常、无 OOM、output.images[0] shape 合法
print(f"Output shape: {output.images[0].size}")
```

### B3 魔乐社区部署

按社区指定入口执行，重点检查特性叠加是否生效：

- 量化开关 → 权重精度是否符合预期
- 稀疏开关 → sparsity 参数是否生效
- Cache 开关 → 缓存命中日志有无

### B4 验证通过标准

| 检查项 | 标准 |
|--------|------|
| 推理无异常 | 无 `RuntimeError` / `OOM` / `CUDA error` |
| 输出合法 | shape > 0，非全零输出 |
| 显存正常 | 峰值 < 物理显存 90% |
| 特性叠加 | 量化/稀疏/Cache 开关生效 |

---

## 阶段耗时与显存追踪

推理耗时与显存的精确追踪使用 _PhaseTimer 类，详见 references/phase-timer.md。

> 需要更细粒度的 profiling 数据（kernel_details.csv, trace_view.json）时使用 profiling-collection。

## Reference Files

- 📦 `references/construction-methods.md` — 加载时机: 涉及模型构造方式选择或 gated model 配置文件获取时
- ⏱️ `references/phase-timer.md` — 加载时机: 需要精确追踪推理各阶段耗时与显存时

## 维护与更新

当遇到新模型的兼容性问题、框架版本升级导致验证方式变化、
或发现新的 NPU 算子兼容性问题时，按 dev-workflow 的复盘流程更新本 skill。
