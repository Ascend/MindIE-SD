# 单元测试模板

pattern 开发中需要的测试文件组织、双层测试原则和代码模板。

---

## 1. 测试文件组织

```text
tests/
├── compilation/
│   ├── test_bench_utils.py          # benchmark 公共函数
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
│   └── test_rmsnorm.py
```

---

## 2. 双层测试原则

每个融合 kernel 必须同时具备两层测试：

| 层级 | 路径 | 覆盖内容 | 断言标准 |
|---|---|---|---|
| **kernel 层** | `tests/layers/test_xxx.py` | dtype/shape/scale 组合、边界值、inplace 安全性、device/dtype 保真性、多次调用一致性 | `torch.allclose(atol=...)` 按 dtype 分档 |
| **pattern 层** | `tests/compilation/patterns/test_xxx.py` | `torch.compile` + `MindieSDBackend` 全链路：pattern 是否触发、replacement 是否生效、输出正确性 | `cosine_similarity > 2^-7`，不强制耗时断言 |

---

## 3. Kernel 层测试模板

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

    def test_dtype_preservation_bfloat16(self):
        x = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        y = torch.randn(4, 4096, dtype=torch.bfloat16, device="npu")
        result = muls_add(x, y, 1.0)
        self.assertEqual(result.dtype, torch.bfloat16)
```

---

## 4. Pattern 层测试模板

```python
@unittest.skipIf(os.environ.get("MINDIE_TEST_MODE", "ALL") == "CPU",
                 "Skip NPU-dependent tests")
class TestXxxPatternCase(unittest.TestCase):

    def _run_and_compare(self, model, args):
        compiled = torch.compile(model, backend=MindieSDBackend())
        compiled(*args)
        torch.npu.synchronize()

        t_c = benchmark(compiled, args)
        t_o = benchmark(model, args)

        out_c = compiled(*args).reshape(1, -1).float()
        out_o = model(*args).reshape(1, -1).float()
        cos_sim = torch.cosine_similarity(out_c, out_o)[0].item()
        return cos_sim, t_c, t_o

    def test_xxx_bf16(self):
        model = XxxPatternModel()   # forward 与 pattern() 完全一致
        x = torch.randn(1, 4096, 128, dtype=torch.bfloat16, device="npu")
        weight = torch.randn(128, dtype=torch.bfloat16, device="npu")
        cos_sim, _, _ = self._run_and_compare(model, (x, weight))
        self.assertGreater(cos_sim, 2 ** -7)
```

**关键注意事项**:

- **单元测试通过 ≠ pattern 命中了模型**。测试 model 与 pattern 共享相同代码 → 必然匹配。真正的匹配验证需要通过全模型 profiling + kernel diff 确认（Phase 6）。
- 若 pattern 涉及 `nn.Module` 参数（weight 来自 `get_attr`），即使单元测试通过，全模型也可能静默失败。参见 `mismatch-catalog.md` 类型 7。

---

## 维护与更新

当新增 pattern 类型或发现现有测试覆盖不足时更新此文件。
