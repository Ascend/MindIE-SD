# 在线量化契约：算法分派、量化范围、图节点与布局互通

> **作用域**：本文件的结论来自**指定设备代际 + 指定软件栈 + 指定模型**的实测与源码核对，
> 换任一维（芯片代际 / CANN·torch_npu 版本 / 模型）都必须复测后再采信（复测纪律见 SKILL §0）。
> **分工**：本文件回答「在线量化到底改了什么、落到哪些算子、出什么图节点」；
> 编码公式/舍入/scale 粒度/退化块规则见 SKILL §一/§二；「该不该开量化、开哪一档」见 `dit-perf-opt`；
> 融合 pattern 侧的图形态与命中见 `pattern-dev`。

## 1. 算法分派契约（算法由设备决定，不由档位名决定）

在线 W8A8 档在载体侧只有一个入口，**具体算法按设备代际自动分派**——**代际 → 实际算法的对应不写死在本表**，
**代际 → 实际算法没有单一真源文档**（`docs/zh/features/quantization.md` 只给档位语义）；须用 `npu-smi info` 确认目标代际后**现场取证**（框架侧代码/日志 + 量化节点计数）：

- 已复现的两种算法与数值契约：**W8A8_MXFP8**（e8m0 块尺度 + e4m3 payload → SKILL §一）、
  **W8A8_DYNAMIC（INT8）**（→ SKILL §二）；本案例所处的代际落在哪一条见会话产物归档。

- **判据**：任何"w8a8"结论都必须写清**实际算法**——两种算法的编码公式、scale 粒度、退化块规则
  完全不同（见 SKILL §一/§二），档位名相同 ≠ 数值契约相同，也 **≠ 可跨代际引用**。
- 入口允许显式强制算法（`algorithm` 参数）；历史脚本使用的「强制 MXFP8 兼容别名」与
  「按设备自动分派」是两种语义，引用时不得混用。
- **证据等级**：INT8（`W8A8_DYNAMIC`）路径**弱**（仅有载体侧记录）——
  在目标代际上真机复测前，不得把它写成已验证结论。

## 2. 量化范围契约（哪些算子会被替换）

- 被替换对象 = **`nn.Linear`（Matmul 类）**；分组 Matmul（GroupMatmul）只走 MoE 路径，
  非 MoE 图不触发。
- **FA / 注意力后端不量化**；norm / rotary / attention 内部 / gate 等**向量运算保持基座精度**。
- **图级契约（可判定）**：结果图中**不应出现高精度计算节点**（fp32 / int32 计算输入）——
  验证方式是遍历编译图标记高精度计算节点，不是"看起来没问题"。
- **计数契约**：落地后以「量化命中的 Linear 数 + 残留 `nn.Linear` 数」证明覆盖；
  残留不为 0 时必须逐条给出豁免原因（白名单/探测失败层），不得静默。

## 3. 接口契约（载体侧统一入口）

```text
apply_w8a8_quant(pipe, attrs, dtype, fallback_layers, algorithm=None)
```

| 参数 | 契约 |
|---|---|
| `attrs` | 被量化的子模块名（如 `transformer`）；未列入的模块不动 |
| `dtype` | 基座精度：在线量化**跑在 bf16 基座**上，量化只作用在 Matmul |
| `fallback_layers` | **必须豁免**的层。典型场景：靠 `next(iter(parameters()))` 探测 dtype 的 time embedder——量化后参数为空会 `StopIteration`；豁免按算法枚举指定（如 `W16A16`） |
| `algorithm=None` | 按设备自动分派（§1）；显式传入则强制该算法 |

配套接口：`report_quant_layers`（出 §2 的计数证据）、`_align_bias_dtype`（量化层 bias 对齐兜底）。

## 4. 与 compile 的耦合契约（guard 稳定性）

量化层 forward **禁止就地修改模块状态**（反例：`self.bias = self.bias.to(torch.float32)`）：
Dynamo guard 记录的是 trace 时的模块状态，每次调用改变它 ⇒ 每次调用完整重编译，
compile 反比 eager 慢若干数量级（现象与诊断单点见
`../../pattern-dev/references/pattern-dev-notes.md` §4）。本契约要求：**bias / dtype 处理用局部变量，
模块状态在 `__init__` 固定**，并把该不变式写进注释（破坏它会产生静默的极端 host-bound）。

## 5. w8a8 编译图节点集（torch_npu API 事实）

- 每个量化 Linear 展开为：`npu_dynamic_mx_quant`（MXFP8 路径：激活沿 k 方向分块、块尺度 32，
  出 fp8e4m3 payload + e8m0 scale）+ `npu_quant_matmul`（QuantBatchMatmulV3 / V5 家族）出 bf16。
- **判据用途**：宣称"量化档已生效"时以**这些节点的出现与计数**为准（图节点证据），
  不以开关声明或日志为准。
- **语义等级**：MXFP8 输出是**量化级近似**（误差为 e4m3 payload 的量化分散度量级），
  不是位级等价；"逐字节对拍"只在同算法同路径下成立。

## 6. 与 catlass 例程的布局互通（torch ↔ catlass）

- `npu_dynamic_mx_quant` 的 scale 布局（激活 `[m, ceil(k/32)/2, 2]`；权重按 k 量化时
  `[n, ceil(k/32)/2, 2]`，e8m0 字节）与 catlass MXFP8 例程输入布局**字节一致，可直接喂入**。
  **实测范围有限**（一档 `m=128 / k=512 / N=2048`），其它 shape 未验证 ⇒ 换 shape 先重测布局兼容。
- 解码对拍相对误差为**量化级量级**（约 2%，属 e4m3 量化分散度而非布局错位）。
- **语义核验纪律**：以 fp32 真实输出例程（或量化输出**解码**对拍）作语义基准；
  **勿只依赖量化输出的自比对**——其敏感性未经证实，不得据此下"语义正确"结论。
- 布局细节单点见 SKILL §1.2：`[T, ceil(c/2), 2]` 是**形状标注而非置换**，不要假设需要 deswizzle。

## 7. 与相邻技能的分工

- **选档语义**（该不该开、开哪一档、设备代际 → 算法映射）→ `dit-perf-opt`
  （`../../dit-perf-opt/references/quant-tier-device-mapping.md`）；
- **图形态 / 融合 pattern 命中 / GraphPatternEntry 接线** → `pattern-dev`；
- **融合 kernel 的算子实现与 DSL 选型** → `operator-dev`；
- **量化前移的通信字节账与并行作用域** → `dit-parallel-opt`（账目算法见 SKILL §四·B）。

## 维护与更新

设备代际 → 算法映射、豁免层清单、w8a8 图节点集、catlass 布局互通范围变化时更新本文件；
每次更新同时标注**实测范围**与**证据等级**（强 / 弱 / 未复测）。
