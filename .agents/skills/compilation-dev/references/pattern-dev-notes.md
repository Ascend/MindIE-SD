# Pattern 编译开发规范

> **本文件是 `compilation-dev` 的补充细节**：Pattern 开发的全生命周期
> 统一由 `compilation-dev` SKILL.md 覆盖，本文件承载注册机制层面的
> 易错细节（ABCMeta isinstance 陷阱、去重注册等）。
>
> **调试与验证**: pattern 实现后若在模型 graph 上未命中，路由到 `compilation-dev` SKILL.md 进行定位和修复。

## 1. Pattern 注册机制

### ABCMeta isinstance 陷阱

`PatternBase` 继承自 `ABC`（有 ABCMeta 元类），其 `isinstance` 会对实现抽象方法的子类返回 `True`。判断实例 vs 类时必须额外排除 `type`：

```python
# 错误：isinstance(SomePatternClass, PatternBase) → True（ABCMeta 行为）
# 正确：
if not isinstance(pat, type) and isinstance(pat, PatternBase):
    # 实例路径
else:
    # 类路径
```

### 去重注册

- 使用模块级 `_registered_pattern_names: set[str]` 记录已注册 pattern
- 测试 setUp 须同时清理该集合和 `patterns.pattern_replacements`

### 外接逻辑融合优先于接口照搬

从外部项目（如 vllm-ascend）引入融合逻辑时：

- **逻辑层**：直接采用外部项目的 pattern 形状、replacement 目标、kernel 调用方式
- **接口层**：保持本地框架约定（如 `PatternBase` 的 `@staticmethod` 接口），不照搬外部项目的实例方法风格
- **参数桥接**：外部项目通过构造函数注入的参数（如 `scale`、`dtype`），用工厂函数 `create(dtype, scale)` 通过闭包注入，对齐本地既有模式

```python
# 正例：工厂函数 + 闭包桥接
def create(dtype, scale=1.0):
    class MulAddPattern(PatternBase):
        @staticmethod
        def pattern(x, y):
            return x * scale + y        # scale 来自闭包
        @staticmethod
        def replacement(x, y):
            return muls_add(x, y, scale) # kernel 融合
    return MulAddPattern

# 反例：照搬外部项目的实例方法风格
class MulAddPattern(PatternBase):
    def __init__(self, scale): ...
    def get_pattern(self): ...  # 与本地 PatternBase 接口冲突
```

## 2. 测试文件组织

```text
tests/
├── compilation/
│   ├── test_bench_utils.py          # benchmark 公共函数
│   ├── test_backend.py              # 后端集成测试（仅正确性）
│   ├── test_pattern_registration.py # 注册机制测试
│   ├── patterns/
│   │   ├── test_gelu_pattern.py
│   │   ├── test_rmsnorm_pattern.py
│   │   ├── test_rope_pattern.py
│   │   ├── test_adalayernorm_pattern.py
│   │   └── test_xxx_pattern.py      # 新增 pattern 测试
│   └── regression/
│       └── test_xxx_regression.py   # 模型级回归测试
├── layers/
│   ├── test_muls_add.py             # kernel 独立单元测试
│   ├── test_rope.py
│   └── test_rmsnorm.py
```

## 3. 双层测试原则

每个融合 kernel 必须同时具备两层测试：

| 层级 | 路径 | 覆盖内容 | 断言标准 |
|---|---|---|---|
| **kernel 层** | `tests/layers/test_xxx.py` | dtype/shape/scale 组合、边界值（scale=0/1/-1）、inplace 安全性、device/dtype 保真性、多次调用一致性 | `torch.allclose(atol=...)` 按 dtype 分档：float32=1e-5, float16=1e-2, bfloat16=1e-1 |
| **pattern 层** | `tests/compilation/patterns/test_xxx.py` | `torch.compile` + `MindieSDBackend` 全链路：pattern 是否触发、replacement 是否生效、输出正确性 | `cosine_similarity > 2^-7`，**不强制耗时断言**（除非张量足够大） |

### kernel 层测试模板

```python
class TestMulsAdd(unittest.TestCase):
    def test_basic_result_float32(self):
        x = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        y = torch.randn(4, 4096, dtype=torch.float32, device="npu")
        result = muls_add(x, y, 1.5)
        expected = x * 1.5 + y
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_scale_variants(self):
        for scale in [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0]:
            result = muls_add(x, y, scale)
            expected = x * scale + y
            self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_no_inplace_modification(self):
        x_orig = x.clone(); y_orig = y.clone()
        _ = muls_add(x, y, 1.5)
        self.assertTrue(torch.equal(x, x_orig))
        self.assertTrue(torch.equal(y, y_orig))

    def test_dtype_preservation_bfloat16(self):
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        result = muls_add(x, y, 1.0)
        self.assertEqual(result.dtype, torch.bfloat16)
```

## 4. 易错细节：forward 内就地修改模块状态破坏 compile guard

**反模式**（实测：`mindiesd/quantization/layer.py` 的量化 Linear，W8A8/W4A4 系列）：

```python
# ❌ forward 内就地修改模块状态 —— torch.compile 反模式
def quant_matmul(self, x):
    if self.bias.dtype != torch.float32:
        self.bias = self.bias.to(torch.float32)   # 每次调用都改变模块状态
    ...
    output = torch_npu.npu_quant_matmul(..., bias=self.bias, ...)

# ✅ 用局部变量，不 mutate 模块属性（fp32 精度保留）
def quant_matmul(self, x):
    bias = self.bias.to(torch.float32) if self.bias.dtype != torch.float32 else self.bias
    ...
    output = torch_npu.npu_quant_matmul(..., bias=bias, ...)
```

**后果**：Dynamo guard 记录的是 trace 时的模块状态（如 bias=bf16），forward 把 bias 改成 fp32 后，
下一次调用 guard 失败 → **每次执行都触发一次完整重编译**（Dynamo trace + Inductor codegen +
triton JIT ≈ 1.8s），compile 比 eager 慢 10~200×，且 kernel profile 呈极端 host-bound
（wall 1.8s 中 kernel 仅 ~17ms，单个大设备空闲间隙）。

**诊断**：

```shell
# 1. kernel_details.csv: wall_ms / kernel_sum_ms >> 10 且 Wait Time 高、单个大间隙
#    → 先怀疑重编译，不要直接归因 kernel 慢
# 2. 确认重编译与 guard 失败原因
TORCH_LOGS=recompiles python xxx_infer.py --compile ... 2>&1 | grep -E "Recompiling|guard failure"
```

**规则**：

- 任何算子层/模块的 `forward` **禁止就地修改模块属性**（`self.xxx = ...`）；需要 dtype 转换用局部变量
- 模块状态（参数/buffer 的 dtype、shape、值）必须在 `__init__` 固定，Dynamo guard 才能稳定
- compile 性能异常（compile 远慢于 eager）先跑 `TORCH_LOGS=recompiles` 排除重编译，再进入 kernel 分析

**同类案例：被 trace 的 python 包装读写模块级 dict/缓存**（2026-09 catlass 融合 op 实测）：
`fused_op(x)` 薄包装内 `cache.get(device)`（首调写入模块级 `_cache`）→ Dynamo 对该 dict
生成 `___dict_contains(...)` guard，写入后每次调用 guard 失败 → **每次调用完整重编译
（实测 ~1.6 s/call，eager 正常、compile 慢 ~100×）**；`TORCH_LOGS=recompiles` 报
`___dict_contains(0, G[...]._cache)` guard 失败。修复：trace 路径**无状态直查**（设备属性/
常量在 trace 期求值一次，不写任何模块级可变对象）。

## 5. 自定义/plugin 算子接入 compile 图与前后核验

（承接 §4 重编译主题；catlass kernel 集成完整方案见
`../operator-dev/references/catlass-kernel-integration.md`。）

- **自定义/plugin op 必须注册 fake 才会被 Dynamo 当不透明单节点**（不展开/不分解）：
  `register_mindie_fake_op(name)` 条件注册（op 未编译进插件时跳过，import 不炸；调用时给
  清晰报错）；fake 输出用 `x1.new_empty(...)` **跟随输入设备**——返回 `device="meta"` 会与
  真实 npu buffer 混设备，Dynamo fake 传播报
  `Unhandled FakeTensor Device Propagation ... found two different devices meta, npu:0`。
- 同命名空间二次注册：插件 C++ 已 `TORCH_LIBRARY(mindiesd)` 时，测试/兜底 stub 必须用
  `torch.library.Library("mindiesd", "FRAGMENT")`（`"DEF"` 等价重复 TORCH_LIBRARY 直接报错）；
  stub 前先 import 真实算子模块，避免同 op 双 fake。
- **compile 前后收益核验（kernel 级）**：同一配置分别 eager 与 compile 采集
  `kernel_details.csv`，按算子家族聚合对比——
  - GEMM/FA 类（QuantMatmulV5/FA/MatMulV3/融合 GEMM）在 compile 下**不变**（compile 不改
    GEMM 内核），占 compile kernel_sum ~80%+；
  - 收益来自小 kernel 链被融合/消减（rmsnorm/adaln/gate 的 aten 分解碎片 Mul/Pow/Add/Mean/
    Slice/Index/InplaceCopy vs 换来的自研 kernel RmsNorm/RotaryV2/gather 等，net 收益通常
    数个 ms）；
  - compile 慢先排除重编译（§4），再进 kernel 归因。MiniMax-H3 w8a8 实测数字指针：
    `tmp/mmx_w8a8/mmx_h3_w8a8_ffn_fusion_analysis.md`、`tmp/mmx_w8a8/h3_w8a8_optimization_headroom.md`。

### 5.2 真实案例：动态 shape 节点让 trace 式 pattern 永不命中 → GraphPatternEntry 手动改写

**场景**（2026-09 MiniMax-H3 w8a8 FFN hidden 融合 `mm_swiglu_mxquant`）：compile
侧 fusion pattern 默认开启但**长期不命中**（compile kernel csv fused=0，FFN 站点只被
triton swiglu 吃掉）。完整经验与规则沉淀在
`references/graph-pattern-rewrite-guide.md`（本文只留案例脉络与结论）。

**真实图形态（pre-pattern dump）**：

```text
Qmm(x1 fp8, w1ᵀ, wsᵀ, pertoken_scale=x_scale) [S, 2F]
  → view([1, S, 2F]) → split(F) → silu → mul → view([S,-1])
  → npu_dynamic_mx_quant → Qmm(out-proj)
```

两个 view 是 diffusers SwiGLU 的 shape-noise；**S 动态**（同图 site1 S=1、site2/3
S=3967）。trace 式 pattern（PatternBase/register_replacement）把 view 目标尺寸固化为
example 常量 → 永不命中。旁证：triton swiglu pattern 能命中是因它的输入就是 3D
`[1,S,2F]`（view 在 pattern 外），不含 Qmm 前缀。

**试错链（全死路，教训如下）**：

| 方案 | 结果 | 死因 |
|---|---|---|
| 全局 fold（redundant pass 折叠 view+split） | 反复崩 | fold 改 split 输入 rank 后 swiglu pattern 的 3D 假设在真实 shape 重放时崩（`split got 1`）；误伤 attn split(5376) |
| fusion pattern 加 3D-view 变体（pattern 内显式 view） | 不命中 | trace 时 `-1` 被具体化 → view 常量 S 失配 |
| 手写 `search_fn_pattern`（`Ignored()` 通配 view） | 不可落地 | torch 内部接口（仅 `register_lowering_pattern` 1 处用）；`check_fn` 无条件要求 `match.kwargs` 含 search_fn 全部参数名，手写 `Arg()` 收集进 args → 深链必抛 `Not all inputs to pattern found in match.kwargs` |
| 本地 symbolic_trace / 独立 make_fx probe | 误导多轮 | probe 图形态与真实 compile 图不同（symbolic_trace 产 `call_method view`/`torch.split`；真实图是 `call_function aten.view.default`/OpOverload `split.Tensor`；make_fx 还把 silu 分解成 neg/exp/add/div）→ probe matched 0 不代表真实现状 |

**正解（mkldnn_fusion.py 范式，已真图命中 3/3）**：改用 `GraphPatternEntry` + 手动改写
handler——四条硬规则（全 `Arg()` 叶子一个 kwargs 都不写 / 动态 view 尺寸 `Ignored()` /
共享子节点同一实例 + `_users=MULTIPLE` / handler 从 `match.output_node()` 反向沿 producer
链改图并逐个 erase 无 user 节点）见 `references/graph-pattern-rewrite-guide.md` §3-§4，本文不复述。
注册接入同文 §4：封装 `register_xxx_graph_entries(pattern_pass)` 由 `passes/__init__.py`
在 fusion config 开启时调用，**fusion on 时跳过弱融合（triton swiglu）注册**（否则弱融合
先吞子图、强融合无 site）。

**验证**：compile kernel csv fused ×3（= FFN 站点数）；同 seed eager vs compile
latents 位级一致（mean_rel=0.0）；transformer 23.4 → 15.75ms（-33%）。

**方法论沉淀**：调试必须在真实 compile 图上注入 probe（monkey-patch
`MindieSDBackend.apply_pattern_match_passes` 前置跑测试 pass），并用**前缀逐级隔离**
（p1 Qmm→p2 +view→p3 +split→p4 激活…）定位首个不匹配节点。通用脚本：
`../scripts/probe_real_graph_pattern.py`、`../scripts/isolate_pattern_prefix.py`、
`../scripts/check_fusion_hit.py`、`../scripts/numeric_check_eager_compile.py`。

## 维护与更新

当PatternBase/注册框架行为变化时，按 dev-workflow 的复盘流程更新本文件。
