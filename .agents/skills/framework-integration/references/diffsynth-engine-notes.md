# DiffSynth-Engine 外部框架适配经验（MindIE-SD compile 接入）

> 本文档记录把 MindIE-SD 的 `MindieSDBackend` 编译接入 **DiffSynth-Engine**
> （`diffsynth_engine`，Qwen-Image 等扩散模型的独立推理框架）时的完整经验：
> 部署、compile 适配、模型层改写、以及**融合算子使能量的判断方法**。
> 与 dummy-run-dev（随机权重 2 层）的对比只用于**使能判断**，不做耗时比较
> （耗时评估必须以真实权重 + kernel diff 为准，见 compilation-dev Phase 6）。

---

## 1. 部署（远端容器）

DiffSynth-Engine 是独立 Python 包，与 MindIE-SD 部署方式不同：

| 步骤 | 命令 / 要点 |
|---|---|
| 增量传输 | 复用 ascend-deploy 的 SSH 连接复用 + 增量上传（排除 `.git`/`__pycache__`/`tests`），目标 `/home/<user>/code/DiffSynth-Engine` |
| 安装 | `cd /home/<user>/code/DiffSynth-Engine && pip install -e . --no-deps` |
| 激活 mindiesd | 脚本内 `sys.path.insert(0, "/home/<user>/code/mindie-sd-compile")`（compile 工作区），避免用 pip 替换容器内已装的 mindiesd（替换会波及 vllm-omni 等其他容器共享依赖） |

### 已知坑

- **setuptools-scm 无 .git 报错**：远端源码包无 `.git` 时
  `pip install -e .` 报 `LookupError: setuptools-scm was unable to detect version`。
  解决：`export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIFFSYNTH_ENGINE=1.0.0`。
- **--no-deps 的原因**：DiffSynth-Engine 的 pyproject 锁定 `transformers==4.57.6` /
  `diffusers==0.36.0`，而容器内是 `transformers 5.14.1` / `diffusers 0.38.0`
  （vllm-omni/mindiesd 依赖）。带依赖安装会把容器环境降级破坏，
  所以一律 `--no-deps`，随后用框架自有的 API 兼容检查确认可导入。

---

## 2. compile 接入（MindieSDBackend）

DiffSynth-Engine 的 compile 入口是 `Pipeline.compile_transformer_blocks`（对 `_repeated_blocks`
逐个 `submodule.compile()`）。接入 MindIE-SD 需改 3 处：

### 2.1 配置入口

- `diffsynth_engine/configs/base.py`：`PipelineConfig` 增加 `compile_backend: str = "inductor"`（可选 `"mindie"`）
- `diffsynth_engine/args.py`：`--compile-backend {inductor,mindie}` CLI

### 2.2 compile_transformer_blocks 接入 MindieSDBackend

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

> ⚠️ **关键陷阱：`torch.compile(submodule, backend=...)` 不赋值不生效**。
> `torch.compile` 返回包装对象，直接调用不修改原模块；必须写入
> `submodule._compiled_call_impl`（与 `nn.Module.compile()` 内部实现一致）。
> 实测教训：第一次接入时直接 `torch.compile(submodule, backend=MindieSDBackend())`
> 未赋值 → warmup 时间与 eager 完全一致、pattern 0 命中 → 排查到赋值问题后修正。

### 2.3 Qwen-Image 模型层改写（命中 pattern 的前提）

- **RoPE 实数域改写**：DiffSynth-Engine 的 `apply_rotary_emb_qwen(use_real=False)` 原实现用
  复数域 `x_rotated = torch.view_as_complex(x.float()...)`。要命中 `qwen_rope_pattern`，
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

- **text encoder key 归一化**：容器 transformers>=5.x 的 `Qwen2_5_VLForConditionalGeneration`
  结构调整（`visual.*`→`model.visual.*`；`model.layers/embed_tokens/rotary_emb/norm.*`→
  `model.language_model.*`），而权重文件是 4.x 布局 → `load_state_dict(strict=True)` 失败。
  解决：在 pipeline 加载时注入 key_mapping：

  ```python
  key_mapping = {
      "^visual\\.": "model.visual.",
      "^model\\.layers\\.": "model.language_model.layers.",
      "^model\\.embed_tokens\\.": "model.language_model.embed_tokens.",
      "^model\\.rotary_emb\\.": "model.language_model.rotary_emb.",
      "^model\\.norm\\.": "model.language_model.norm.",
  }
  ```

---

## 3. 融合算子使能量的判断（与 dummy-run 对比的核心）

> 目标：确认外部框架下**哪些 pattern 使能、融合成哪些 kernel**。
> 只判断使能集合，不比耗时（耗时结论需真实权重 + kernel diff）。

### 3.1 判断方法（三层证据，可靠性递增）

| 层级 | 方法 | 证据 |
|---|---|---|
| 1. 日志 | `MINDIE_LOG_LEVEL=DEBUG` | `PatternMatchPass replace N` |
| 2. 图 dump | `CompilationConfig.graph_log_url=<dir>` | DOT 图出现 `npu_rotary_mul` / `npu_rms_norm` 等 |
| 3. kernel | torch_npu profiler → `kernel_details.csv` | 融合 kernel 出现 + 原始 kernel 消失 |

> ⚠️ **日志 2048 字符截断陷阱**：MINDIE 日志 `MAX_LOG_STRING_LEN=2048`，
> 长 graph dump 行被截断，grep 日志搜不到融合 kernel 会**误判"0 命中"**。
> 以 `CompilationConfig.graph_log_url` 落盘的 DOT 文件（需 `pip install pydot`，
> 否则 `FXGraphDrawer requires the pydot package`）或 kernel_details.csv 为最终依据，
> 它们不受 2048 截断影响。

### 3.2 使能集合与 dummy run 一致（实测结论）

DiffSynth-Engine（真实权重 60 层 / 1024²）与 dummy run（随机权重 2 层）的
**融合 kernel 使能集合完全一致**：

| 融合 pattern | 融合 kernel（kernel_details.csv） | dummy run | DiffSynth-Engine |
|---|---|---|---|
| qwen_rope | `RotaryPositionEmbeddingV2` / `npu_rotary_mul` | ✅ | ✅ |
| qk_norm RMSNorm | `npu_rms_norm`（kernel 名 RmsNorm） | ✅ | ✅ |
| 残差 gate | `residual_gate_add_kernel` | ✅ | ✅ |
| 调制 | `AdaLayerNormV2` | ✅ | ✅ |
| GELU | `FastGelu` | ✅ | ✅ |

判断要点：

- 使能判断**不依赖层数/权重**：pattern 匹配基于图结构，dummy（2 层）与真实（60 层）
  命中集合一致 → 用 dummy run 即可验证"外部框架下 pattern 是否使能"。
- **图命中 ≠ 运行期全部生效**：`residual_gate_add` 对 4D attention 张量运行期
  fallback（日志 `fallback (ndim)`）。使能判断要同时看 kernel_details.csv 中
  融合 kernel 的**实际执行次数**（3D 站点执行融合 kernel，4D 站点 fallback 到原生）。
- 使能集合一致的判定标准：kernel_details.csv 中 `compile_only` 的融合 kernel
  （`residual_gate_add_kernel` / `RotaryPositionEmbeddingV2` / `AdaLayerNormV2` /
  `FastGelu`）出现，且对应的原始算子（`GeluV2`、逐元素 `Mul/Stack` 链）消失。

### 3.3 不做耗时比较的原因

dummy run（2 层 / 0.72B）与真实权重（60 层 / 20B）的耗时差异由**层数**主导：
融合收益在小模型上会被编译图新增 kernel（广播/填充等）淹没，导致
"dummy 编译反而更慢"的假象；真实权重下融合收益随层数放大才体现为正收益。
因此**融合算子使能判断用 dummy run 即可，耗时评估必须以真实权重 + kernel diff 为准**
（详见 compilation-dev Phase 6，不在本文档重复耗时结论）。

---

## 维护与更新

当 DiffSynth-Engine 或类似外部框架（如 Z-Image / Wan 在 diffsynth_engine 中的接入）的
部署/compile 接入方式变化、或发现新的使能判断陷阱时，按 dev-workflow 的复盘流程更新本文件。
