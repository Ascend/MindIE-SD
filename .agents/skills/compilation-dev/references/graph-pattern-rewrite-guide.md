# Graph Pattern 手动改写融合（mkldnn_fusion.py 范式）

> **本文件解决**：trace 式 `register_replacement` pattern 无法匹配的图——尤其是
> pattern 中间夹着**动态 shape 节点**（view/reshape 的目标尺寸随 batch/token 变化）
> 或需要按运行时 meta 精细校验后手动改图的场景。
>
> 出处：`torch/_inductor/fx_passes/mkldnn_fusion.py`（`_recover_linear`、
> `linear_bias_pattern`、`register_graph_pattern`/`GraphPatternEntry`）。
> 本仓案例 1：MiniMax-H3 w8a8 FFN hidden 融合（`mm_swiglu_mxquant`），
> 2026-09 真图命中 3/3、位级一致、-33%（案例脉络见 `pattern-dev-notes.md` §5.2）。
> 案例 2：FLUX/Wan/Qwen `mm_gelu_mxquant`（同范式 4/2/3 站点命中；**fp8 量化级近似非位级**）
> ——kernel/工程细节见 operator-dev `../operator-dev/references/mmgelu-flux-wan-qwen-case.md`，
> 全链编排见 `../operator-dev/references/catlass-ffn-fusion-guide.md`。
> 案例 3（vLLM-Omni 跨框架变体，2026-09-06）：H3 FFN hidden 在 vLLM-Omni 0.28 真图是
> `Qmm → torch_npu.npu_swiglu（单融合 op，无 split/silu/mul）→ DxQ → Qmm`，权重为
> GEMM-ready `(K,N)` + scale `(c,N,2)`（无独立 transpose 节点）——同融合 op 需**第二
> pattern 变体** + C++ `AdaptLayoutCached` 布局自适应（检测 `w.size(0)==k` 缓存 `w^T`，
> 免 row-swap），命中 52/52、60 步 e2e -5.8%；另踩坑：裸 aclrtLaunch kernel 与异步
> 转置拷贝跨流竞态（AI Core 507015）→ launch 前 `aclrtSynchronizeStream`；量化级近似
> 非位级（单层 rel 0.36%）。证据 runs/20260906_minimax-h3_mmxfp8_compile_fusion/evidence/。

## 1. 何时该走这条路（路径选择）

| Pattern 形态 | 路径 | 说明 |
|---|---|---|
| trace 式可表达（含 `nn.Module` 权重作 pattern 输入） | `register_replacement`（PatternBase） | mindie 主流路径；weight 收进 `inputs()`/`pattern()`/`replacement()` 参数，freeze 前命中（参考 `rms_norm_pattern.py`） |
| **pattern 中间含动态 shape 节点（view 尺寸随 S/batch 变）** | **GraphPatternEntry + 手动改写 handler（本文件）** | trace 式会把动态常量固化 → 永不命中；`Ignored()` 可通配 |

> ⛔ 自定义 FX Graph Pass（手写 graph traversal）已废弃删除（曾见
> `custom-graph-pass-guide.md`），不得使用；需手动改图一律走本文件的 GraphPatternEntry。

**典型失败信号**：pattern 注册成功、单元测试通过，但 compile kernel csv 中融合
kernel 数 = 0；pre-pattern dump 显示目标链里夹着 `view([1, S, 2F])` 之类尺寸随
batch 变化的节点，且同一图内不同 site 的 S 不同。

## 2. 机制速览

- 注册入口：`torch._inductor.pattern_matcher.GraphPatternEntry(pattern=..., extra_check=..., handler=...).register(pattern_pass)`。
- handler 签名：`(match, *args, **kwargs)`——匹配成功后由 pass 循环调用，**手动改写图**，
  不要求返回 replacement 表达式（这正是它比 `register_replacement` 灵活、且能绕开
  replacement-trace 限制的原因）。
- `match.output_node()`：pattern 树的根（目标链最末端节点）；`match.graph`：所在 fx graph。

## 3. 四条硬规则（每一条都是真实踩坑实证）

### 3.1 pattern 只写 args（全 `Arg()`），**一个 kwargs 都不要写**

`CallFunction` 的 kwargs 匹配会把 pattern 列出的 kwargs 与真实 node kwargs 逐键比较；
即使值写成 `Ignored()`/`KeywordArg`，一旦结构对不上（真实多了 `bias`、或 kwarg 值
类型不同）就整链失败。实测：Qmm 节点 `Arg×3 + kwargs` matched 0，`Arg×3`（无 kwargs）
matched 28。

```python
# ✅ 正确：只列位置参数
hidden = CallFunction(torch_npu.npu_quant_matmul, Arg(), Arg(), Arg())
# ❌ 错误：列任何 kwargs（即便 Ignored）都可能致不匹配
hidden = CallFunction(torch_npu.npu_quant_matmul, Arg(), Arg(), Arg(),
                      scale_dtype=Ignored(), pertoken_scale=KeywordArg("x_scale"), ...)
```

需要 kwargs（如 `pertoken_scale`、`output_dtype`）时，在 **handler 里从 matched node
自取**（`node.kwargs.get("pertoken_scale")`），pattern 侧不碰。

### 3.2 动态 shape 参数用 `Ignored()` 通配

真实 view 目标尺寸 `[1, S, 2F]` 中 S 动态（同一图 S=1 与 S=3967 并存）；`Ignored()`
匹配任意常量参数 → view 尺寸不参与匹配。`view` 节点本身仍要写出来（链结构必须连续）：

```python
h3 = CallFunction(torch.ops.aten.view.default, hidden, Ignored())  # [1, S, 2F] 任意 S
```

### 3.3 共享子节点：复用同一 PatternExpr 实例 + `_users=MULTIPLE`

真实图里 `split` 的输出被两个 `getitem` 消费（hidden 半 / gate 半），`npu_dynamic_mx_quant`
的输出也被两个 `getitem` 消费（fp8 / scale）。若 pattern 里写两个独立的
`CallFunction(operator.getitem, parts, 0/1)`，会生成两个不同的 pattern 节点 → 匹配器
要求对应两个真实节点 → 失败。**必须把共享节点定义为同一个实例并放宽 users**：

```python
from torch._inductor.pattern_matcher import CallFunction, Arg, Ignored, MULTIPLE
parts = CallFunction(aten.split.Tensor, h3, f, -1, _users=MULTIPLE)   # 一个实例
g0 = CallFunction(operator.getitem, parts, 0, _users=MULTIPLE)        # 消费同一 parts
g1 = CallFunction(operator.getitem, parts, 1, _users=MULTIPLE)
silu = CallFunction(aten.silu.default, g1)
act3 = CallFunction(aten.mul.Tensor, g0, silu)
...
dq = CallFunction(torch_npu.npu_dynamic_mx_quant, act2, _users=MULTIPLE)
aq = CallFunction(operator.getitem, dq, 0)
```

（`_users=2` 亦可，`MULTIPLE` 更稳。）

### 3.4 handler 手动改图：从 output_node 反向走 producer 链

匹配成功后 handler 拿到的不是绑定好的叶子，而是 `match`；链节点要从
`match.output_node()` 沿 `.args[0]`/`.args[i]` 反向追溯（mkldnn `_recover_linear`
同款手法）。改写 = 在原链末端前插入 fused 调用 → `replace_all_uses_with` → 逐个
erase 不再有 user 的链节点：

```python
def handler(match, *args, **kwargs):
    graph = match.graph
    out_node = match.output_node()          # out-proj Qmm
    aq_node = out_node.args[0]              # getitem(dq, 0)
    dq_node = aq_node.args[0]               # npu_dynamic_mx_quant(act2)
    act2_node = dq_node.args[0]             # view(act3, [S, F])
    act3_node = act2_node.args[0]           # mul(hid, silu(gate))
    # mul 两个 operand 之一接 silu —— 分辨 hid/gate 顺序
    a, b = act3_node.args
    silu_node = a if a.target == aten.silu.default else b
    hid_node = b if a.target == aten.silu.default else a
    gate_node = silu_node.args[0]           # getitem(parts, 1)
    parts_node = gate_node.args[0]          # split(h3, F, -1)
    h3_node = parts_node.args[0]            # view(hidden, [1, S, 2F])
    hidden_node = h3_node.args[0]           # hidden Qmm(x1, w1ᵀ, wsᵀ)
    x1_node, w1_node, w1s_node = hidden_node.args[0], hidden_node.args[1].args[0], hidden_node.args[2].args[0]
    x_scale_node = hidden_node.kwargs.get("pertoken_scale")   # kwargs 在 handler 里取
    w2_node, w2s_node = out_node.args[1].args[0], out_node.args[2].args[0]

    with graph.inserting_before(out_node):
        fused = graph.call_function(torch.ops.mindiesd.mm_swiglu_mxquant,
                                    (x1_node, w1_node, w1s_node, x_scale_node, aic_num))
        out_t = graph.call_function(operator.getitem, (fused, 0))
        out_s = graph.call_function(operator.getitem, (fused, 1))
        # ... 重建 out-proj Qmm（transpose 权重 + pertoken_scale=out_s）...
    out_node.replace_all_uses_with(repl)
    for node in (out_node, aq_node, dq_node, act2_node, act3_node, silu_node,
                 hid_node, gate_node, parts_node, h3_node, hidden_node):
        if node.users:
            continue
        graph.erase_node(node)
```

> 多返回值 fused op 必须 `getitem(fused, 0/1)` 拆出再喂下游；带 int 入参的 op
> （如 `aic_num`）在 `graph.call_function` 里直接给常量。

## 4. 注册与接入

```python
from torch._inductor.pattern_matcher import GraphPatternEntry
entry = GraphPatternEntry(pattern=pat, extra_check=lambda m: True, handler=handler)
entry.register(pattern_pass)          # pattern_pass = patterns.pattern_pass
```

- 注册进 mindie 的 `patterns.pattern_pass`（`passes/register_pattern_to_pass.py` 的
  `patterns` 单例）后，即被 `PatternMatchPass.__call__` 的 while 循环驱动。
- 建议封装为 `register_xxx_graph_entries(pattern_pass)` 由 `passes/__init__.py` 在
  对应 fusion config 开启时调用；注册前先做 op 可用性 gate（op 未编入则跳过，import 安全）。
- **多级融合注册序**：更强融合（整链）先注册、先消费子图；弱融合（如单点 swiglu）
  在该融合开启时**不要注册**（否则弱融合先吞掉子图 → 强融合无 site 可匹配）。

## 5. 调试方法论（真实图注入 probe + 前缀逐级隔离）

> ⚠️ **不要用本地 symbolic_trace / 单独 make_fx 图验证命中**——probe 图形态与真实
> compile 图不同（symbolic_trace 产 `call_method view`/`torch.split`；真实图是
> `call_function aten.view.default`/`OpOverload split.Tensor`；本地 make_fx 还会把
> silu 分解成 neg/exp/add/div）。**必须在 MindieSDBackend 的
> `apply_pattern_match_passes` 前 monkey-patch 注入测试 pass，打在真实 compile 图上。**

### 5.1 真实图注入 probe（模板）

```python
orig = MindieSDBackend.apply_pattern_match_passes.__func__

def patched(cls, graph, inputs):
    cnt = probe_pass.apply(graph)   # probe_pass 在真实 pre-pattern 图上跑
    print(f"PROBE_MATCHED {cnt}", flush=True)
    return orig(cls, graph, inputs)

MindieSDBackend.apply_pattern_match_passes = classmethod(patched)
```

前置条件：跑 compile 前把与 fusion 竞争的弱 pattern 关闭（如
`enable_minimax_h3_swiglu=False`、fusion off），保证目标链保持原始形态可被 probe。

### 5.2 前缀逐级隔离（定位首个不匹配节点）

把链拆成渐进变长的一组 pattern，逐个注册并打印匹配数，第一个掉到 0 的就是断点：

```python
p1 = Qmm(...)                                    # 单节点
p2 = view(p1, Ignored())                         # +view
p3 = split(p2, F, -1)                            # +split
p4 = mul(getitem(p3,0), silu(getitem(p3,1)))     # +激活（注意共享节点 MULTIPLE）
...
```

实测序列（MiniMax FFN）：p1 matched 28 → p2 25 → p3 3 → p4（独立 getitem 实例）0 →
**修正共享节点为同一实例后 p4/p5/p6/p7 全 3**。断点瞬间暴露"共享节点实例"问题，
免去整链盲试。

### 5.3 命中判定

- **注册成功 ≠ 命中**。最终判定看 compile `kernel_details.csv` 中融合 kernel 出现
  （如 `GroupedMxMatmulSliceMSwigluMxQuant`）且计数 = 期望站点数。
- 通用脚本：`../scripts/probe_real_graph_pattern.py`（注入 probe）、
  `../scripts/isolate_pattern_prefix.py`（前缀隔离）、`../scripts/check_fusion_hit.py`
  （kernel csv 命中判定）。

## 6. 数值核验（图改写语义无损）

手动改图后必须验证改写前后数值一致（融合是等价替换）：

- 同 seed、同进程：先跑 eager（无融合）拿基线 latents，再跑 compile（融合）拿输出，
  全元素对比：`mean_rel = (|a-b|/|b|).mean()` 与 `max_abs`，达标线按路径等价性定（见下）。
- **路径完全等价 → 位级断言 0.0**：改写前后数值路径一致才可（h3 `mm_swiglu` 特例：输入
  量化与激活路径一致）；一般无量化改写链用 `2^-7` 量级阈值。
- **量化链的位级是特例不是默认**：kernel 计算路径与参考不等价时（`mm_gelu`：kernel fp32
  激活 vs torch bf16 参考路径、Qmm bias 数值路径差异）→ **fp8 量化级近似**（字节一致 98%+、
  解码 rel≈1e-3），按量化级容差 + 模型级质量门验收，勿套"应 0"；判别法/数字见 operator-dev
  `../operator-dev/references/mmgelu-flux-wan-qwen-case.md` §2。
- 通用脚本：`../scripts/numeric_check_eager_compile.py`。

## 维护与更新

当 GraphPatternEntry/`register_graph_pattern` API 或 pass 循环行为随 torch 版本变化时
更新本文件；新案例（不同 target/更多 kwargs 需求）按 §3 规则校验后补充到 §5 踩坑。
