# 实例：vLLM-Omni 托管扩散模型 + MindIE-SD（性能/使能视角）

> ⚠️ **本文件是 vLLM-Omni 特定实例**。全栈构建/启动排障见
> `framework-feature-enablement/references/troubleshooting-vllm-omni.md`（互补，不重复）；
> 本文聚焦**性能使能方法 + 使能判断**，按 SKILL.md §6.2 迁移清单组织。
> vLLM-Omni 托管 Qwen-Image / Wan2.2 / MiniMax-H3 等，**使能结论在对应模型 + 环境成立**。

## 1. 框架画像（§6.2 迁移清单速答）

| 检查项 | vLLM-Omni 实况 |
|---|---|
| 算子注册机制 | 无 `rms_type`/`rope_type` 抽象接口（模型层 torch 算子）→ 走 compile 接入 |
| 并行实现 | `--usp N --ring 1 --text-encoder-tp-size N`（USP/Ring 序列并行，多卡 a2a/ring 通信） |
| compile 图形态 | 对整个 transformer 编译（`torch.compile(pipe.transformer, backend=...)`），含 a2a/ring collective → **有 LightX2V 同款 a2a 退化风险** |
| 动态 shape | 推理分辨率固定 → 静态；但 CFG/并行分支可能有动态维度，需验证 |
| 长序列通信 | USP 多卡通信占比随序列变长升高（同 LightX2V 15s 现象） |

**结论**：vLLM-Omni 与 LightX2V 相似度更高（都是多卡序列并行 + 整个 transformer 编译）：
**a2a 留 eager 的修复很可能同样必要**（需按 §6 协议验证）。

## 2. 使能方法（多卡服务 + compile）

### 2.1 启动（950PR 8 卡示例，详见 framework-feature-enablement 服务启动与验证章节）

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

### 2.2 compile 接入点

```python
from mindiesd.compilation import MindieSDBackend
compiled = torch.compile(pipe.transformer, backend=MindieSDBackend())
```

> ⚠️ 与 DiffSynth-Engine 同款陷阱：`torch.compile(module, backend=...)` 需
> **赋值**（`module._compiled_call_impl = ...` 或 `nn.Module.compile()` 等价），
> 不赋值不生效（pattern 0 命中）。

### 2.3 预期必要的修复（按 LightX2V 经验推断，需验证）

| 修复 | LightX2V 必要性 | vLLM-Omni 预期 |
|---|---|---|
| a2a/ring collective 留 eager（`torch._dynamo.disable`） | 必须（+50% → 正收益） | **很可能必须**（USP 多卡 + 整个 transformer 编译） |
| backend 实例复用 | 必须 | 必须（跨框架通用） |
| pattern 匹配实际图形态 | 必须（chunk 双 split） | 按模型图 dump 验证（Qwen/Wan/MiniMax 各不同） |

> ⚠️ 以上为**推断**，vLLM-Omni 场景尚未实测；接入时必须按 §6 协议验证。

## 3. 使能判断（与 dummy run 对比）

同 DiffSynth-Engine §3：使能集合与 dummy run 一致（pattern 匹配基于图结构，
不依赖层数/权重），用 dummy run 验证使能、真实权重做耗时。

三层证据：`MINDIE_LOG_LEVEL=DEBUG` 日志（⚠️ 2048 截断陷阱）→ `graph_log_url` DOT
（需 pydot）→ kernel_details.csv（融合 kernel 实际执行次数）。

## 4. 与 LightX2V / DiffSynth-Engine 的差异

| 维度 | LightX2V | DiffSynth-Engine | vLLM-Omni |
|---|---|---|---|
| 并行 | USP4（a2a 在 block 内） | 单卡（无通信） | USP/Ring 多卡（collective 在 transformer 内） |
| compile 粒度 | 逐 block | 逐 submodule | 整个 transformer |
| a2a 留 eager | 必须 | 不需要 | 很可能必须（待验证） |
| 运行时注册表接入 | 有接口（rms/rope） | 无 | 无 |
| 验证入口 | torchrun + CLI | pipeline CLI | HTTP 服务（curl /v1/images/generations） |

> vLLM-Omni 的 profiling 采集：多卡服务 + HTTP 入口 → 采集补丁需挂在
> transformer forward 上（而非 CLI 推理），warmup 用服务预热请求代替。

## 5. 收益预期（框架特定，需实测）

- 使能集合与 dummy 一致 → 融合 kernel 生效（rope/rms/gate/adaln/gelu，按模型）
- 多卡通信占比高 → a2a 留 eager 的**通信重叠红利**可能显著（LightX2V 通信 -35% 同源）
- 真实权重耗时：需完整推理 + rank0 口径墙钟 + compare_traces.py（本文不给出数字，
  该环境未完成 vLLM-Omni 真实权重墙钟对比）
