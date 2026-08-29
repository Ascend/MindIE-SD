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

## 维护与更新

当PatternBase/注册框架行为变化时，按 dev-workflow 的复盘流程更新本文件。
