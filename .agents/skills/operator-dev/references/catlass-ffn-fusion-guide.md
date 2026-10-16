# 只读 catlass 融合算子开发与真图使能（MindIE-SD 编排）

> 场景：在 MindIE-SD 内把「量化 matmul + 激活(swiglu/gelu) epilogue + 输出 MX 量化」做成
> 融合算子并在 compile 图真实命中。案例：MiniMax-H3 `mm_swiglu_mxquant`（真图命中、位级一致；
> 计数与日期见会话产物归档 `{run_results_dir}/archive/`）与 FLUX/Wan/Qwen `mm_gelu_mxquant`（GraphPatternEntry 真图；站点计数/收益排序见
> pattern-dev `fusion-enablement-notes.md` §3，集成侧细节见 `mindiesd-fusion-notes.md` §7）。
> 机制细节不在此重复：pattern/graph-entry 写法 → pattern-dev `graph-pattern-rewrite-guide.md`；
> ASC 混编/构建/torch custom op → `catlass-kernel-integration.md`；开关治理/载体 → pattern-dev
> `fusion-enablement-notes.md`。本文给编排与决策（案例细节一律走对应 case 归属文件指针：实测记录
> 在会话产物目录、经验在 `mindiesd-fusion-notes.md`，不重复）。

## 0. 定位

六段流水线，每段有产物与验收；可跳段、别跳验收：

```text
P1 定语义与边界 -> P2 vendored kernel 开发 -> P3 standalone 数值/计时
P4 mindiesd 集成 -> P5 compile 真图使能 -> P6 验证、报告与开关治理
```

立场（都来自真实返工）：

- **catlass 库本体只读、0 改动**：它只是 include 依赖。新 kernel 头 vendored 进
  `csrc/ops/{op}/include/catlass/...`，csrc/CMakeLists 将该 include 置于
  `MINDIESD_CATLASS_HOME/include` 之前；命名唯一，不与 catlass 原生同名（防 include 遮蔽）。
- **dummy/推理前端禁止 monkeypatch 融合**：验证口径始终是 compile 图内 fused 节点计数。
  前端 layer-route 只用于最早期的效果试探，不作交付载体。
- **trace 式 register_replacement 匹配不了真实图**：真实 diffusers FFN 链带动态 shape 的
  `view/reshape`（尺寸随 S 变）+ `_to_copy`（functionalization 产物）噪声；正解是
  GraphPatternEntry（mkldnn 范式）手写改写，别在 canonical pattern 上死磕。
- **适用边界**：本文面向**含 matmul（CV）融合**——mm + 激活(swiglu/gelu)/量化 epilogue；
  **无 matmul 的纯 vector（VV）融合**（swiglu/gate/激活+量化等 elementwise）推荐 triton 链
  （skill-map §1.1），勿把 catlass 流程套到 VV 任务。
- **开关收敛到 compilation-config**：一融合一 flag、默认开启、唯一命名空间（见 §6）。

## 1. P1 定语义与边界

1. 用 dummy run（`--quant w8a8 --compile --profile`）kernel 序确认目标链真实形态。同族差异：
   h3 = [S,2F]+swiglu（宽度减半、需行重排）；flux/wan/qwen = [S,F]+gelu（无减半、无重排，
   需能容忍 `reshape`）；激活语义按模型代码核实。
2. 找语义锚点：ops-nn `matmul/quant_matmul_activation_quant`（某设备档：fp8 MX mm + gelu + 输出 MX
   量化；档位与代际的映射见 `../../dit-perf-opt/references/quant-tier-device-mapping.md`，不在本文件写死）——部署 CANN 未注册该 aclnn 时，其文档仍是公式/布局权威参考；catlass 65
   `mm+swiglu+mxquant` 给出底座与三段指数域量化（`ComputeMaxExp/ComputeScale/QuantToFp8`）。
3. 核实真实模型量化事实再开工：FFN 上/下投影 **bias 是否为 0**（flux 实测全 0；非零才需要
   kernel bias 支持）；输出 scale 布局与 torch `npu_dynamic_mx_quant` 字节一致即零转换直供。
4. 验收口径前置：设备档（MXFP8 档需自研；INT8 档常可直接用现成 torch 算子
   `npu_quant_matmul_gelu`——档位映射现场查（设备属性 / `npu-smi`））、目标线（≥1%/step 等）、报告模板（dummy-run §C）。

## 2. P2 vendored kernel 开发

- 能复用就不重写：mx 量化尾三段静态函数可经 `TileSwigluAndMxQuant::ComputeMaxExp/...` 直接复用；
  激活用对齐公式（gelu 恒等式 `x·σ(1.59577·(x+0.044715·x³))`，与 FastGelu/torch tanh 对齐）。
- 三头落位 vendored 路径：`gemm/kernel`（kernel 类）、`epilogue/block`（block epilogue）、
  `epilogue/tile`（激活+量化 tile），保持 `#include <catlass/...>` 相对名。
- 可选 bias（仅模型 bias≠0）：epilogue 在激活前按列加。GM→UB 装载用
  `AscendC::DataCopy(Local, Global, uint32 元素数)`（count=**元素数**非 32B 块），同步用
  **MTE2_V**（DataCopy 是 MTE2；MTE3_V 会竞态）；bias 的 UB 位置（VECIN/VECOUT/VECCALC）
  以实际编译+对拍为准。API 错用史与"列加没加对"判别法见 `mindiesd-fusion-notes.md` §7.2。

### 2.1 落码前先做**片上资源预算**（三行算术，能省掉一次重写）

融合改动常常要把**两份中间结果同时留在片上**（两个 L0C 累加器）或**让某个操作数常驻**（A 留在 L1）。
这两件事都受**硬容量**约束，且**事先可算**——本仓实测两次教训（一次是方案被否、一次是白写一版 kernel）：

```text
L0C 需容纳    累加器份数 × L1_TILE_M × L0_TILE_N × 4 B        ≤  L0C 容量
L1  需容纳    A 的全部 strip 数 × L1_TILE_M × L1_TILE_K × 1 B  ≤  L1 容量
```

- **累加器**：本仓交付形状 `L0_TILE_N = 256` 下两份 fp32 累加器需 `2 × 128 KiB = 256 KiB`，
  而**现场取到的 L0C 容量正好被这两份占满**（容量按设备属性现取），**没有余量留给 L0C↔UB 的 fixpipe 管线**
  ⇒ **在该形状下交错不可能**；把 `L0_TILE_N` 降到 128（`2 × 64 KiB`）才装得下。
  即"交错"这个动作**要求先改 N 粒度**，而那是 **host 侧 tile 形状**改动，不是 kernel 头文件能解决的。
- **操作数常驻**：本仓 A 的足迹（`strip 数 × L1_TILE_M × L1_TILE_K × 1 B`）**远超现场取到的 L1 容量**
  ⇒ **一次最多常驻少数 strip**，"让 A 常驻 L1 以免二次搬运"这条捷径**不成立**——
  两个 K 循环仍然各自从 GM 流全部 strip。**唯一可行的是在一个 K 循环内逐 strip 共享**，
  也就是必须走交错形态。

**纪律**：先算这两行，再决定改动形态；**"看起来低风险"的方案往往在容量上就不存在**。
把算出来的数字写进方案（维护者据此判断可行性），别等编译器/运行期报错才发现。
**前置关系（强制）**：这两行算术**同时是收益判定阶段的前置关**——`fusion-scope-analyze` 出
"建议融合"之前必须先算过（判据单点在
`../../fusion-scope-analyze/references/fusion-benefit-method.md` §8 最后一条）；本文件是它的**修法侧**。

## 3. P3 standalone 数值与计时

- 对拍对象 = torch 参考链（同量化输入 → Qmm → 激活 → DxQ），比 fp8 输出与 scale：字节一致率
  98%+/解码 rel≈1e-3 即通过（fp8 量化级）；位级一致是特例（h3 swiglu），别默认要求。
- 判别"没加/加错列/精度差"：用**常数/ramp bias**（列不敏感）+ 逐列误差统计，别在随机 bias 里
  猜列错位（fp8 边界翻转会放大随机噪声）——证据与数字见 `mindiesd-fusion-notes.md` §7.2。
- 计时（同窗同卡）：fused vs 现役链；**预期形态**是 `fused ≈ Qmm 耗时`，收益 = 被吸收的小 kernel
  （激活 + 输出 DxQ）及其 HBM 往返。这是**待验证的预期，不是保证**：融合核自身的核芯效率可能低于
  被替换的 Qmm（头号嫌疑：输出拆半 ⇒ 同一归约轴跑两遍、操作数装载重复；或 tile 被累加器共存挤小，
  见本节末）。
- **实测收益打平或为负时，禁止直接判"融合无收益"**：先按
  `../../fusion-scope-analyze/references/fusion-benefit-method.md` **§10** 做**等功忙计数对照**
  ——等 MAC 数下比核芯忙计数与搬运忙计数，并把 `duration − 核芯忙计数` 取作 epilogue 增量。
  判为"缺陷在融合核自身"即回到 §2 的 kernel 侧修（本技能承接）；判为"结构性受限"才可关闭该项。
  同窗必须带一个**未触碰算子的负对照**；邻近算子**在管道占比不变的前提下**变慢属频率效应，另行报账。

## 4. P4 mindiesd 集成

按 h3 `mm_swiglu_mxquant` 模板镜像：`csrc/ops/{op}/{op}.cpp`（ASC 宿主 extern "C"）→
`csrc/plugin/{op}.{h,cpp}` → `register_ops.cpp`（守卫宏）→ `csrc/CMakeLists.txt`
（ASC 混编 + vendored include 在前）→ `mindiesd/layers/{op}.py`（wrapper+fake，op 存在性守卫）。
坑：同 basename 的 ops 版与 plugin 版 `{op}.cpp` 上传/同步互相覆盖（ops 文件混入
`#include "{op}.h"` 致编译失败）——用不同暂存名上传并核对。

## 5. P5 compile 真图使能（GraphPatternEntry）

1. 真实 compile 图 probe（monkeypatch `apply_pattern_match_passes` 抓 post-pattern 图）确认
   节点形态（reshape 尺寸、激活是 aten gelu 还是 `npu_fast_gelu`、bias 节点）。canonical miss 常见，
   不据此否定融合。
2. pattern 文件写 GraphPatternEntry + handler（同 h3 `register_ffn_fusion_graph_entries`）：
   - pattern 树：`Qmm(Arg×3) → reshape(Ignored) → 激活(npu_fast_gelu 优先，另注册 aten.gelu 变体)
     → reshape(Ignored) → DxQ(_users=MULTIPLE) → getitem(0, _users=MULTIPLE) → Qmm(Arg×3)`；
   - handler：从 `match.output_node()`（out-proj Qmm）反向走 producer 取
     x1/w1/w1_scale/x_scale/bias，插 fused 调用 + 重建 out-proj Qmm（kwargs 从原节点复制，
     别写死 bias=None）；
   - `passes/__init__.py` 在 config flag 开且 op 就绪时注册（try/except 不打断 compile setup）。
3. 验收 = 计数契约：fused 实例数 == 命中站点数、被替代 kernel 消失；off（flag False）恢复原链。
4. 非 FeedForward 形态站点（flux single-block 的 proj_mlp+act_mlp→concat→proj_out、wan refiner）
   不命中 → 报告按 §C 未实现行标注（N+依据+收益 0），不虚填。

## 6. 开关治理

默认开启不设开关是首选；确需开关只收敛到
`mindiesd/compilation/compiliation_config.py::CompilationConfig`（一融合一
`enable_{scope}_{feature}`，默认 True 需真实命中背书；关闭=置 False 同 flag 复现 AB）。
禁止 dummy/推理入口独立 env/arg 开关；多载体（graph entry/pattern）共享 flag；
`MINDIESD_CATLASS_HOME` 等构建开关非运行开关。细则/清单：pattern-dev
`fusion-enablement-notes.md`。

## 7. 验证、报告与复盘

三层证据（图命中 → kernel diff/计数 → 墙钟同窗同卡）；报告按 dummy-run §C 双表 + 与 h3 同族
先例对照 + post-enable 六面复核；数值 fp8 级近似要写明"非位级"并标注真机质量门待办。
新坑/新决策按本 skill 维护节回填。

## Reference Files

| 文件 | 加载时机 |
|------|---------|
| `mindiesd-fusion-notes.md` §7 | P1-P7 实战对照（真实图链/bias=0/装载坑/工程坑；本站案例的集成侧要点在该节） |
| `catlass-kernel-integration.md` | P4：ASC 混编/CMake/链接/torch custom op 细节 |
| `../../pattern-dev/references/graph-pattern-rewrite-guide.md` | P5：GraphPatternEntry 四条硬规则与 handler 写法 |
| `../../pattern-dev/references/fusion-enablement-notes.md` | §1/§2 开关治理 R1-R7 与现状清单；§4 compile 阶段适配动作（阶段 2） |
| `../../pattern-dev/references/pattern-dev-notes.md` §5 | P5/P7：compile 前后 kernel 级收益核验 |
| `../../dummy-run/references/compile-ab-report-template.md` | P7：§C 双表模板 |
| 外部（语义参考）：ops-nn `quant_matmul_activation_quant` docs | P1 公式/布局锚点 |

## 8. 维护与更新

- **触发条件**：① catlass 版本或 vendored 落位变化（`csrc/ops/{op}/include/catlass/...` 与
  `MINDIESD_CATLASS_HOME/include` 的先后、`TileSwigluAndMxQuant::ComputeMaxExp/ComputeScale/QuantToFp8`
  的可复用性），含 catlass（**外部库**，非本仓文件）`scripts/build.sh` 复用 build 目录时的全清重建前提；
  ② CANN Ascend C 语义变化（`AscendC::DataCopy` 的 count 单位、MTE2_V 同步、`DataCopyPad` 参数）；
  ③ §1.2 的语义锚点 ops-nn `quant_matmul_activation_quant` 公式/注册状态变化；④ §6 的
  `enable_{scope}_{feature}` flag 与 `passes/__init__.py` 注册路径、或目标模型 FFN 形态（h3 swiglu
  与 flux/wan/qwen gelu）变化。
- **复核方法**：对 §1.1 的 dummy run（`--quant w8a8 --compile --profile`）重跑一次，核对 probe 到的
  FFN 站点链与 §1/§5 描述一致、且 fused 实例数 == 命中站点数（计数契约）；不符即改对应小节。
