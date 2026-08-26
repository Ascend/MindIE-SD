---
name: aclgraph-dev
compatibility: torch_npu（torch.npu.NPUGraph / torch.npu.graph / graph_pool_handle）、已安装 mindiesd；静态 shape 推理场景
description: NPU 图批量下发能力（aclgraph / aclgraph_ex 家族）的开发与调优。覆盖
             mindiesd 的 aclgraph_backend：NPUGraph 静态 capture、全局 graph pool、
             lazy capture、专用 copy stream + event 管线、shape/dtype 校验、max_entries 驱逐。
             当用户需要减少 host launch 开销、静态 shape 大 batch 场景加速、
             或排查 NPUGraph replay 输入不匹配问题时使用此 skill。
             即使用户只提到"批量下发"或"graph capture"而未说 aclgraph，也应触发。
             Inductor/default 后端（aot_autograd + codegen + Copy 消减）见 compilation-dev。
---

# ACLGraph 批量下发

## 边界：与 compilation-dev 的分工

```text
compilation-dev（pattern matcher + Inductor）      aclgraph-dev（批量下发）
├─ PatternBase / register_replacement              ├─ torch.npu.NPUGraph 静态 capture
├─ 三段注册 / 单测 / mismatch 调试                 ├─ graph pool / lazy capture / replay
├─ default 后端: aot_autograd + Inductor codegen   ├─ 专用 copy stream + event 管线
├─ functionalization → _to_copy → InplaceCopy      └─ shape/dtype 校验 + max_entries 驱逐
└─ Copy 消减（default 路径）
```

两者通过 `CompilationConfig` 的 `aclgraph_only` / `aclgraph_with_compile` 开关联动
（见 `mindiesd/compilation/mindie_sd_backend.py` 的选路逻辑）。

## 机制（事实源: mindiesd/compilation/aclgraph_backend.py）

NPUGraph 批量下发的本质：把一次推理的算子序列**静态捕获**成图，之后仅需 replay，
省去每步的 host 端算子 launch（Python enqueue / AclExec 调用）。适用条件是**输入
shape/dtype 稳定**（变化会触发重新 capture 或校验失败）。

核心对象：

- `torch.npu.NPUGraph()` + `torch.npu.graph(npu_graph=aclgraph, pool=pool)`：捕获执行体
- `torch.npu.graph_pool_handle()`：全局 graph 内存池，避免每次 capture 重新分配
- capture 期间 patch `gc.collect` / `torch.npu.empty_cache` 为空操作，防止捕获中内存被回收
- `_ACLGraphEntry` 按 **input_shape 缓存**（`entries: dict[shape, entry]`），同 shape 复用图

## 配置开关（事实源: mindiesd/compilation/compiliation_config.py）

| 开关 | 默认 | 含义 |
|---|---|---|
| `aclgraph_only` | False | 跳过编译（不跑 pattern/Inductor），直接对原始图做 NPUGraph capture |
| `aclgraph_with_compile` | False | 先走 `MindieSDBackend.compile()`（pattern 融合等）再 capture |
| `aclgraph_lazy_capture` | False | 首次调用时才 capture（capture 用 `detach()` 共享存储）；False 时用 `detach().clone()` 稳定缓冲 |
| `aclgraph_max_entries` | 0 | 缓存的图条目上限，>0 时按 FIFO 驱逐最旧条目 |
| `safe_output_mode` | True | replay 输出 clone 一份再返回（防输出被调用方原地修改污染图内 buffer） |

`npu_graph_available` 自动检测：`torch.npu.NPUGraph` 与 `torch.npu.graph` 均存在才为 True；
不可用时 aclgraph 开关自动失效回退 default。

## 使用流程

```python
from mindiesd.compilation import MindieSDBackend, CompilationConfig

# 场景 A: 纯批量下发（不做 pattern 编译）
CompilationConfig.aclgraph_only = True

# 场景 B: pattern 编译 + 批量下发（推荐，pattern 收益与 launch 收益叠加）
CompilationConfig.aclgraph_with_compile = True

torch.compile(model, backend=MindieSDBackend())
```

选路逻辑（`mindie_sd_backend.py`）：

```text
aclgraph_with_compile && npu_graph_available → compile() 后 aclgraph
aclgraph_only && npu_graph_available        → 直接 aclgraph（跳过 compile）
else                                       → default（aot_autograd + Inductor）
```

## 关键行为与坑（代码实证）

1. **输入校验（D1）**：replay 时逐个对比 `static_buf` 与 `new_inp` 的 shape/dtype，
   不一致直接 `RuntimeError: ACLGraph input mismatch at position i`。
   动态 shape 模型必须关掉 aclgraph 或保证 shape 稳定。
2. **data_ptr 跳过（C1）**：`static_buf.data_ptr() == new_inp.data_ptr()` 时跳过 copy
   （同一存储直接复用，零拷贝）。
3. **异步 copy（C3）**：需要拷贝的输入走**专用 copy stream**（`torch.npu.Stream`）批量
   `copy_`，再 `record_event` + 默认流 `wait_event`，与 capture 内计算重叠。
   copy 前（A1）默认流先 `synchronize()` 保证输入就绪。
4. **地址漂移告警（D2）**：DEBUG 级别日志会对 `input_addresses` 与本次输入 `data_ptr()`
   不一致打 warning——调用方用不同存储复用图时提示先拷入 static buffer。
5. **safe_output_mode**：默认 True 返回 clone；输出被外部原地修改时不污染图内 buffer。
6. **静态 shape 要求**：shape 变化 → 触发新 capture（首次开销大）或校验失败。大 batch /
   固定分辨率推理收益最大；动态 seq/分辨率场景慎用。

## 与 torch_npu 生态的对应

`aclgraph` / `aclgraph_ex` 是 torch_npu / torchair 生态中的图批量下发后端家族
（名称随版本变化）。MindIE-SD 的接入点是上述 `aclgraph_*` 开关 + `create_aclgraph_backend()`，
本 skill 只描述本仓已实现的能力；torch_npu 侧新后端名称以远端环境实测为准。

## 维护与更新

当 `aclgraph_backend.py` / `compiliation_config.py` 的 capture 行为、开关或校验逻辑变化，
或 torch_npu NPUGraph API 升级时，按 dev-workflow 的复盘流程更新本 skill。
