---
name: compilation-dev
compatibility: torch + torch_npu NPU 环境、已安装 mindiesd、CANN（集成验证需 NPU）
description: >
  MindIE-SD Pattern matcher 与 Inductor/default 后端开发能力：Pattern 创建/注册/调试、Copy 算子
  消减（default 路径）全生命周期；供 model-auto-optimization 的 S1 kernel 融合（接入/compile）替换场景指向（框架侧使能
  不支持、需新增/适配 pattern 时）。批量下发见 aclgraph-dev，算子本体开发见 operator-dev。
  当用户需要写/改/调试 pattern、处理融合不命中、做 Copy/InplaceCopy 消减或注册 replacement 时
  使用；即使用户只说"pattern 不生效""copy 算子变多了""compile 没融合"而未说全称，也应触发
  （触发词：pattern / compile / Copy / InplaceCopy / fusion / register_replacement）。
  由 dev-workflow 的 pattern/compile 开发场景路由触发，亦由 model-auto-optimization 的 S1 kernel 融合（接入/compile）替换场景触发。
---

# 编译开发（Pattern + Copy 消减 + 后端选择）

## 生命周期

```text
Phase 1: 模型分析 → 提取每个原语的完整代码 + 判断参数来源
Phase 2: 创建 Pattern → 先检查现有 pattern → 决定路径:
  ├─ functional API / nn.Module 参数均走 register_replacement（weight 作 pattern 输入，
  │   freeze 前窗口命中；**禁止自定义 Graph Pass**，见下）
  └─ pattern 中间夹动态 shape 节点（view 随 S/batch 变）→ GraphPatternEntry（graph-pattern-rewrite-guide.md）
Phase 3: 三段注册
Phase 4: 单元验证 → ⚠️ 仅验证代码正确性，不等同于全模型命中
Phase 5: Debug Mismatch → graph dump → 逐节点对齐 → 修正/回退
Phase 6: 集成验证 → kernel diff 确认 → 全模型回归
Phase 7: Copy 消减 → 检测 ViewCopy 翻倍 → 后端选择
```

每阶段有明确的工具和产出物。Phase 5 可回路到 Phase 2（当 `register_replacement` 框架性失败时）。

---

## 行为约束（复盘硬纪律：本 skill 开发中多次反复修正提炼，先读后做）

以下纪律来自 MiniMax-H3 w8a8 FFN fusion（`mm_swiglu_mxquant`，2026-09，案例见
`pattern-dev-notes.md` §5.2）中**多次反复修正**的过程教训——每条都对应一次真实返工，违反即返工：

1. **接入载体：eager 试验可以，compile 验证不得前端绕开**。新开发的算子可先
   eager 接入做效果试验；但**验证 compile 时，必须让融合真实发生在 compile 图内**
   （kernel csv 出现融合 kernel、同 seed eager vs compile 位级对比通过）——禁止在前端
   （推理/dummy 脚本、wrapper、环境开关）绕开或替换 compile 路径来"通过"验证。
   eager 侧辅助代码只是试验手段，验证口径始终是 compile 图真实命中。
2. **命名：内容驱动 + pattern 命名一致 + 全链路一次完成**。融合算子**按其融合内容
   确定名字**（如 `mm_swiglu_mxquant` = MM + swiglu + MX 量化，不是随意临时名）；
   对外标识首次落码前与用户确认命名，落码后**全链路一次完成**（torch op / python
   wrapper / config key / kernel 名 / 报告 / skills）并 grep 旧名清零、重新核验；
   **pattern 侧命名（pattern 文件、工厂类、注册函数、config flag）必须与算子本身
   名字相符**，禁止 pattern 用与算子无关的别名。
3. **开关：默认不额外添加**。新融合**默认开启、不设开关**（配置面即维护成本，
   已加开关再拆是返工）；确需开关/默认关时先与用户确认再落码。
   确需开关时**只收敛到 `CompilationConfig` 一个 flag（默认 True 见证据）**，禁止在
   dummy/infer 另设 env/arg 开关；细则/现状清单见 `references/fusion-enablement-notes.md`。
4. **验证载体：基于 dummy run 直接验证，无需额外构图**。dummy run 运行快，
   compile pattern 的命中与形态验证**直接基于 dummy run 的真实 compile 图**
   （monkey-patch `apply_pattern_match_passes` 注入 probe，见
   `graph-pattern-rewrite-guide.md` §5）；**不另建独立测试图**——本地 symbolic_trace /
   单独 make_fx 构图形态与真实图不同，其命中结论不得驱动方案选择。
5. **数值核验先行**。任何图改写/融合**宣称完成前**先跑同 seed eager vs compile 对比确证
   等价，再采 kernel diff 报收益；核验数字入报告。**位级（0.0）仅当改写前后数值路径完全
   等价**（h3 `mm_swiglu` 特例）；含量化融合常为 **fp8 量化级近似**（字节一致 98%+、
   rel≈1e-3），按量化级容差 + 模型级质量门验收并标注"非位级"——细则见
   `references/graph-pattern-rewrite-guide.md` §6。

---

## 验收与推进纪律（先验收后推进，出口证据化）

> 与 model-auto-optimization 的 run-state/自验证回执同构但更轻：能力层不引入运行状态文件，
> 证据行写入实施记录 / 案例文件（`pattern-dev-notes.md` 或复盘归档）即可。

- **拒收语义**：每 Phase 收尾把「证据行」落记录后再宣称该 Phase 完成；缺证据、或证据与
  产物矛盾 → 不宣称完成、不进入收益宣称；不自行审查代码替代验证。
- **证据行格式（机械可查，与编排层自验证回执一致）**：

```text
- Phase {n} 证据: <支撑脚本 / 日志路径>
- 命中: <PatternMatchPass replace N / check_fusion_hit 输出 / kernel csv 融合 kernel 计数>
- 数值核验: <同 seed eager vs compile 结论；位级 或 fp8 量化级标注>
- 收益判定: <采纳 / 回退 + 依据>；开关状态: <默认 True / 单一 config flag>
```

- **Phase 3 出口**：三段注册文件 + 注册生效证据（`__all__` / registry / config flag 可 grep
  到）；注册不可达不得进入 Phase 4。
- **Phase 4 出口**：测试通过 + 记录「单测通过 ≠ 全模型命中，命中留待 Phase 6 kernel diff」。
- **Phase 6 出口**：三层顺序不可跳——命中（replace N / `scripts/check_fusion_hit.py` 计数）→
  kernel diff（融合 kernel 出现 + 原始消失）→ 全模型回归；**三层全过 + 数值核验后才允许宣称
  「融合完成 / 收益成立」**；命中但无计数/无墙钟收益 → 按证据回退或标注不宣称。
- **Phase 7 出口**：Copy 消减前后 kernel csv 对比记录（`scripts/analyze_copy_kernels.py` 输出）。
- **断点接力**：路径切换（Phase 5→2、Phase 4 失败回退）、收益判定等关键节点，先落一条
  「当前 Phase + 已确认决策 + 未决问题」状态行再继续；上下文压缩后从记录重建。

---

## Phase 1: 分析模型结构

在 diffusers 源码中定位目标模型的 RMSNorm / RoPE / AdaLN 的实际实现代码，
提取完整代码片段作为 pattern 和测试 model 的依据。

**产出物**: 每个原语的完整代码片段。**额外记录**: 判断参数来源——`nn.Module` 参数（`self.weight`）为 `get_attr`，
函数输入为 `placeholder`。该判断影响 pattern 签名设计（weight 收进 pattern 输入参数），**不影响
路径选择**——路径只有 register_replacement 与 GraphPatternEntry 两条（自定义 Graph Pass 已禁止）。

**候选判定行（优化点识别 gate，进入 Phase 2 前补齐）**: 对每个候选 pattern 记录三要素——
① 是否复用现有 pattern（复用决策与理由）；② 期望收益类型（kernel 数减少 / 搬运消除 / 后端
选择）；③ Phase 2 路径预判（register_replacement / GraphPatternEntry）。
明确不做/暂缓的候选连同理由一并记录，**不静默跳过优化点识别**；Phase 6 用候选判定行核对
收益归属（预期与实际不符按证据回退）。

---

## Phase 2: 创建 Pattern + 路径选择

**原则**: 始终创建新文件（非侵入式），不修改现有 pattern 文件。
先检查现有 pattern 是否匹配（复用决策），Phase 5 确认不匹配后才新建。

**路径选择**: 由 Phase 1 的「参数来源」决定：

| 目标算子参数来源 / Pattern 形态 | Pattern 路径 | 执行位置 |
|-----------------|-------------|---------|
| functional API (`F.rms_norm` 等) | `register_replacement` | `register_pattern_to_pass` |
| `nn.Module` (`self.weight` 等) | `register_replacement`（weight 作为 pattern/replacement 的**输入参数**，非 get_attr） | `register_pattern_to_pass` |
| pattern 中间含**动态 shape 节点**（view 尺寸随 S/batch 变）、或需按运行时 meta 精细校验后手动改图 | **GraphPatternEntry + 手动改写 handler**（mkldnn_fusion.py 范式） | `register_xxx_graph_entries(pattern_pass)` |

- **register_replacement 路径**: 创建 `PatternBase` 子类（工厂+闭包），注册到 `pattern_registry`。
  代码模板和融合 op 速查见 `references/pattern-templates.md`。
  **weight 等 `nn.Module` 参数如何表达**：把 weight 放进 `inputs()` 与 `pattern()/replacement()` 的
  参数里（meta tensor 输入，freeze 前窗口命中，见 `rms_norm_pattern.py`），**不是**写自定义 Graph Pass。
- **GraphPatternEntry 路径**: 当 trace 式 pattern 把动态常量固化而永不命中时选用
  （典型：`view([1,S,2F])` 的 S 随 batch 变且同图并存多个 S）。手写 CallFunction 树
  （全 Arg 叶子、view 尺寸 Ignored、共享子节点复用同一实例 + `_users=MULTIPLE`）+
  handler 从 `match.output_node()` 反向走 producer 链手动改图。完整规则/代码骨架/
  调试方法见 `references/graph-pattern-rewrite-guide.md`，真图命中案例见 `pattern-dev-notes.md` §5.2。

> ⛔ **禁止：自定义 FX Graph Pass（手写 graph traversal pass）**。曾以
> `references/custom-graph-pass-guide.md` 记录"get_attr 权重无法用 register_replacement → 手写
> 遍历 FX graph 节点改图"，该方案**已废弃删除**：torch 2.11 下 `nn.Module` 权重在 freeze 前仍是
> placeholder，用 `register_replacement` 双参数 pattern（weight 作输入）即可命中，无需绕过
> pattern matcher；确需手动改图的场景走 **GraphPatternEntry**（pattern matcher 原生 API）。
> 不得新建手写 FX graph traversal pass（含在 `PatternMatchPass` 里加 `_rewrite_*` 方法遍历
> `graph.nodes` 的做法）。

---

## Phase 3: 三段注册

**总是 3 个文件**（全部是代码追加）：

1. `patterns/__init__.py` — `__all__` + `from .xxx_pattern import XxxPatternGroup`
2. `passes/__init__.py` — `pattern_registry` 字典追加
3. `compiliation_config.py` — `FusionPatterns` dataclass 追加 `enable_xxx: bool = True`

命名规范: config key 使用 `enable_{model}_{op}` 格式。检查清单见 `references/registration-checklist.md`。

---

## Phase 4: 单元验证

Test Model 的 forward 与 Phase 1 提取的代码完全一致。
验证标准: `cosine_similarity(compiled, original) > 2^-7`。

> ⚠️ **单元测试通过 ≠ pattern 命中了模型**。测试 model 与 pattern 共享相同代码 → 必然匹配。
> 全模型匹配需 Phase 6 的 kernel diff 最终确认。
> 若 pattern 漏了 `nn.Module` 权重（weight 未收进 pattern 输入参数，真实图为 get_attr 形态）——
> 单元测试仍通过但全模型静默失败，见 mismatch 类型 7。

测试组织与模板见 `references/test-templates.md`。

---

## Phase 5: Debug Mismatch

若 Phase 4 通过但全模型未命中：

1. Dump 模型 traced FX graph → 保存为文本
2. 定位目标算子子图 → 逐节点对齐 pattern → 定位第一个不匹配节点
3. 对照 mismatch 类型修正

### 路径 A: 修正 `register_replacement` pattern（类型 1-6）

修正 pattern 代码使之与模型 graph 一致，重新部署。详见 `references/mismatch-catalog.md` 类型 1-6。

### 路径 B: 修正参数来源（类型 7 —— placeholder vs get_attr）

当 mismatch 原因为 `placeholder` vs `get_attr`（mismatch 类型 7），**不要实现自定义
FX graph traversal pass（已禁止，见 Phase 2 ⛔）**。正确修法：

1. **weight 作 pattern 输入参数**：把 `nn.Module` 的 weight/bias 放进 `inputs()` 与
   `pattern()/replacement()` 参数（meta tensor 输入），freeze 前窗口可命中——
   torch 2.11 下权重 freeze 前是 placeholder 而非 get_attr（参考
   `patterns/rms_norm_pattern.py` / `patterns/minimax_h3_rmsnorm_pattern.py`）。
2. **若确需手动改图**（如 pattern 中间夹动态 shape 节点）→ 走 GraphPatternEntry
   （见 `references/graph-pattern-rewrite-guide.md`），**不手写 pass**。

操作指南: `references/graph-comparison-guide.md`。

---

## Phase 6: 集成验证

依次通过三层验证，**顺序不可跳**:

| 层级 | 方法 | 确认内容 |
|------|------|---------|
| 1. Pattern 命中 | graph dump + 日志 | `PatternMatchPass replace N` 增加 |
| 2. Fusion kernel | profiling → `kernel_details.csv` | 融合 kernel 出现 + 原始 kernel 消失 |
| 3. 全模型回归 | `dummy run --compile` | 推理正常完成且无 crash |

**Kernel 级 diff 验证**（比 pattern 日志更可靠）:

1. 分别采集 eager 和 compile 的全模型 profiling（`--profile`）
2. 从 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 按 kernel 名称聚合耗时
3. 对比 `eager_only`（被融合的原始算子）和 `compile_only`（新增的融合 kernel）
4. 同名 kernel 耗时差排序 → 定位编译开销

一键执行（`scripts/compare_profiles.py`，含算子族聚合 + 逐 kernel delta）：

```bash
python scripts/compare_profiles.py --eager eager/kernel_details.csv --compile compile/kernel_details.csv
```

**Kernel 名称确认**:

- RMSNorm → `rms_norm` / `RmsNorm`
- RoPE → `npu_rotary_mul` / `RotaryMul`
- AdaLN → `adaln` / `adln`
- GELU → `FastGelu`

---

## Phase 7: Copy 算子消减（default/Inductor 路径）

Pattern 匹配成功后检查 Copy 算子膨胀。`default` 后端的 `aot_autograd` functionalization
将 view/reshape 转为 `_to_copy` → Inductor codegen → `InplaceCopy` NPU kernel。

**检测**: 在 `kernel_details.csv` 中搜索 `InplaceCopy/ViewCopy/TensorMove/StridedSlice`。
同时对比 eager vs compile 的同名 kernel 耗时差，定位膨胀源。

一键执行（`scripts/analyze_copy_kernels.py`，含 Copy 统计 + Top kernel + 前后算子归因）：

```bash
python scripts/analyze_copy_kernels.py --csv compile/kernel_details.csv --label compile
```

**消减方案**（本仓可实现）:

- **方案 A** (推荐): 修复 pattern 匹配，减少 functionalization 引入的 Copy 膨胀
- **方案 B**: 混合模式（VAE eager + transformer compile）
- **方案 C**: 静态 shape / 大 batch 场景改用 aclgraph 批量下发（见 aclgraph-dev）

> torchair_ge / npugraph_ex 作为后端选项**在本仓未实现**（`CompilationConfig` 无
> `backend_mode` 及对应常量），历史上记录的"四后端对比"不适用于本仓，勿按旧文档执行。

完整流程见 `references/copy-elimination-guide.md`。

---

## Reference Files

| 文件 | 加载时机 |
|------|---------|
| `references/pattern-templates.md` | Phase 2: 创建 pattern 代码模板 + 融合 op 速查 |
| `references/pattern-dev-notes.md` | Phase 2/4: 注册机制易错细节（ABCMeta isinstance 陷阱、去重注册）；Phase 5/集成: guard 破坏→重编译排查（含 trace 路径读写模块级 dict 案例）、自定义/plugin 算子接入 compile 图（fake/FRAGMENT/无状态）、compile 前后 kernel 级收益核验 |
| `references/graph-pattern-rewrite-guide.md` | Phase 2/5: GraphPatternEntry 手动改写融合（mkldnn_fusion.py 范式）——动态 shape 节点（view 随 S 变）无法用 trace 式 pattern 匹配时的正解：全 Arg 叶子/Ignored view/共享节点 MULTIPLE/handler 手动改图 + 真实图注入 probe 与前缀隔离调试 |
| `references/registration-checklist.md` | Phase 3: 注册核对清单 |
| `references/test-templates.md` | Phase 4: 测试组织 + 双层测试模板 |
| `references/mismatch-catalog.md` | Phase 5: 7 类 mismatch 目录 |
| `references/graph-comparison-guide.md` | Phase 5: Graph dump + 节点对齐方法 |
| `references/copy-elimination-guide.md` | Phase 7: Copy 消减全流程（default 路径） |
| `references/benchmark-guide.md` | Phase 6/7: 计时方法（L2-flush 放计时区外、warm/cold 双档） |
| `references/benefit-rootcause-guide.md` | Phase 6: pattern 命中后验证收益（kernel diff → 逐 pass AB → R1-R5 根因目录） |
| `references/fusion-enablement-notes.md` | 融合使能/开关治理：R1-R7 开关细则（唯一 compilation-config 命名空间/默认值纪律/多载体共享 flag）+ FFN 融合现状清单 + 使能经验（真实图 seam→layer-route 载体、计数契约、fp8 级数值标注） |

## Bundled Scripts

- `scripts/compare_profiles.py` — Phase 6: eager vs compile kernel diff（算子族聚合 + 逐 kernel delta）
- `scripts/cmp_kernels.py` — Phase 6: 算子类 count/耗时轻量对比
- `scripts/analyze_copy_kernels.py` — Phase 7: Copy 膨胀检测（统计 + Top kernel + 前后算子归因）
- `scripts/check_fusion_hit.py` — Phase 6/graph-pattern: 融合命中判定（kernel csv 中融合 kernel 计数 vs 期望站点数，可选 eager csv 应 0）
- `scripts/probe_real_graph_pattern.py` — graph-pattern 调试: 真实 compile 图注入 probe（在 apply_pattern_match_passes 前打测试 pattern 并计数）——模板，替换模型构建段使用
- `scripts/isolate_pattern_prefix.py` — graph-pattern 调试: 前缀逐级隔离（渐进变长链定位首个不匹配节点）——模板
- `scripts/numeric_check_eager_compile.py` — graph-pattern 验证: 同 seed eager vs compile 输出位级对比（图改写语义无损核验）——模板

## 维护与更新

当发现新的 mismatch 类型、Copy 膨胀场景或后端支持矩阵变化时更新本 skill。
各 reference 文件的更新触发条件见各自的"维护与更新"章节。
