# 融合算子使能与开关治理（compilation 侧）

> 场景：融合算子/pattern 使能入口与"开关"设置。与 SKILL.md「行为约束 #3（默认开启、不设开关）」呼应：
> 默认开启不设开关是首选；**确需开关时必须收敛到 `CompilationConfig`，禁止散落独立开关**。
> 案例：MiniMax-H3 `mm_swiglu_mxquant`（2026-09 前）；FLUX/Wan/Qwen-Image `mm_gelu_mxquant`
> （2026-09 本仓：kernel+layer/pattern+使能+开关收敛全链路，报告 `{run_results_dir}/`
> `fusion_enable_report.md`、`phase1_3_delivery_report.md`）。
> §4 = **compile 阶段的适配动作**：由 `framework-integration` SKILL §② 的**阶段 2**
> （API 接入验证通过后）指向本文件；阶段 1（API / runtime 注入）的方法与顺序纪律在该 SKILL §②。

## 0. 原则

0.1 **默认开启、不加开关**（SKILL.md 行为约束 #3 原文）：新融合默认生效；开关是配置面维护成本，
已加再拆是返工。
0.2 确需开关（跨阶段默认关/回退/AB 对照）→ **只在 `mindiesd/compilation/compiliation_config.py`
的 `CompilationConfig`（融合类在 `fusion_patterns`）留一个 flag**，其余层（dummy/infer/pattern 文件/
layer-route helper/测试）一律**读它**，不设第二渠道。

## 1. 开关细则（R）

- **R1 唯一命名空间**：运行期使能开关只存在于 `CompilationConfig`（`fusion_patterns` 等分组）。
  禁止在 dummy/推理入口新增独立 env/arg 开关；临时验证 env/脚本开关用后即弃，不得进入常开代码。
- **R2 一融合一 flag**：命名 `enable_{scope}_{feature}`（如 `enable_flux_wan_ffn_gelu_fusion`，
  覆盖同形态模型时在注释写明适用模型）。字段注释必须含：语义（融合链/算子）、载体
  （pattern / layer-route / …）、依赖算子与编入条件（catlass op 等）、默认值依据、关闭方法
  （=置 False；关闭不得需要第二个开关）。
- **R3 默认值纪律**：验收达标（功能正确 + 计数契约 + AB ≥ 目标线 + off-identity）→ 默认 True，
  注释记录依据/日期/AB；未达标 → 默认 False。默认 True 必须可一键置 False 复现
  （off-identity/AB 对照），复用同一 flag。
- **R4 多载体共享 flag**：同一融合同时存在 pattern 与 layer-route 等载体时共享同一 flag，
  任何载体不得独立默认/独立开关（历史教训：mm_gelu 曾 pattern flag(False) 与 dummy env(默认 1)
  两套并存造成默认漂移，已收敛）。
- **R5 无效使能不进默认路径**：默认 True 的使能必须真实生效（Phase 6 kernel 计数契约验证）。
  canonical pattern 真实图未命中（seam）不算生效——未打通前不得以"默认开但无效"交付；改用
  命中的载体（如 layer-route）时在 flag/pattern 注释标注 carrier 与命中验证结果（含 seam 记录）。
- **R6 评审守则**：新增开关需说明：解决的问题、替代方案（能否复用已有 flag/自动判定）、
  默认值与证据、删除项；发现漂移/冗余开关 → 收敛或删除后再合入。
- **R7 构建开关非运行开关**：`MINDIESD_CATLASS_HOME` 等只控制 op 是否编入 .so
  （未编入时 pattern 组不注册 / layer 调用报错 / import 安全），不是运行期使能开关，文档注明即可。

## 2. 现状清单（FFN 量化融合，均为 compilation-config 唯一开关）

| 融合算子 | flag（唯一） | 默认 | 载体/说明 |
|---|---|---|---|
| `mm_swiglu_mxquant`（MiniMax-H3 FFN） | `enable_minimax_h3_ffn_fusion` | True | compile GraphPatternEntry 真图命中（mkldnn 范式，register_ffn_fusion_graph_entries）；register_replacement canonical 子线备用 |
| `mm_gelu_mxquant`（FLUX/Wan/Qwen FFN） | `enable_flux_wan_ffn_gelu_fusion` | True | **compile GraphPatternEntry 真图命中**（register_ffn_gelu_fusion_graph_entries：Qmm→reshape(Ignored)→npu_fast_gelu→reshape(Ignored)→DxQ→Qmm，handler 反向改图，bias 原样直传）；dummy 前端零融合代码（layer-route 已撤） |

- 其余既有融合（rms/rope/adaln/gate 等）同为 `fusion_patterns` pattern flag，无独立开关、
  默认 True（`enable_minimax_h3_norm_rope` 已随算子整体移除）。强弱融合互斥靠**注册序**：
  `enable_minimax_h3_ffn_fusion` 先于 `enable_minimax_h3_swiglu`、
  `enable_flux_wan_ffn_gelu_fusion` 先于 `enable_fast_gelu`（fusion 链含其子图）。
- 运行期外：构建开关 `MINDIESD_CATLASS_HOME`（R7）。

## 3. 融合使能经验（compilation 视角，2026-09 实录）

1. **canonical pattern ≠ 必命中**：register_replacement 匹配规范扁平链；真实 compile 图若带
   shape-noise（view/`_to_copy` 等，见 Phase 7 的 functionalization 产物）会整链 miss。
   新增 FFN 类融合先做真实图 probe（`scripts/probe_real_graph_pattern.py`）确认，勿以
   symbolic/make_fx 构图命中代替（行为约束 #4）。
2. **命中验证 = kernel 计数契约**：以 compile kernel csv 中融合 kernel 实例数 == FFN 站点数、
   被替代 kernel（gelu/DxQ/up-Qmm）消失为准（`scripts/check_fusion_hit.py`）；图命中日志不替代。
3. **真实图命中载体 = GraphPatternEntry（mkldnn 范式），不用前端 layer-route**：真实 diffusers
   FFN 链带 aten.reshape 动态 shape 噪声，trace 式 register_replacement 无法匹配（S 固化）；
   正解 = 手写 CallFunction 树（全 Arg 叶子 + reshape 尺寸 `Ignored()` + DxQ `_users=MULTIPLE`），
   由 handler 从 output_node 反向走 producer 链手动改图（h3 `register_ffn_fusion_graph_entries`、
   本次 `register_ffn_gelu_fusion_graph_entries` 同款，2026-09 实测 flux 4/wan 2/qwen 3 站点命中）。
   dummy/推理前端**禁止** monkeypatch 融合（SKILL 行为约束 #1）：layer-route 曾作验证载体，
   已撤；compile 验证一律以 compile 图内 fused 节点计数为准（`scripts/check_fusion_hit.py`）。
4. **真实模型 bias 需实查**：flux FFN up/down Linear bias 实测全 0，kernel 可选 bias 支持按
   通用性保留（语义 Qmm(bias) 同列向：常数/ramp 对拍高度一致、zeros-bias ≡ no-bias 在 fp8
   量化级一致，真机风险 nil）；bias 装载/判别法等 kernel 侧细节见 operator-dev
   `../../operator-dev/references/mindiesd-fusion-notes.md` §7.2（本文不复述）。
5. **catlass 库只读复用（0 改动）**：自定义 kernel 头 vendored 进 mindiesd
   `csrc/ops/{op}/include/catlass/...`，csrc/CMakeLists 将其 include 置于 `MINDIESD_CATLASS_HOME/include`
   之前；catlass 树不新增/不修改任何文件（示例/头/CMake 均回滚）。
6. **结构适配边界按 §C 未实现行标注**：FeedForward 形态（net[0].proj up + net[2] down）可泛型
   覆盖（flux joint ff/ff_context、qwen img/txt_mlp）；**该形态三个模型的单点收益排序 =
   FLUX（约 6%）> Qwen（约 4%）> Wan（约 1%，接近零；本组合观测，Wan 的 FFN 占比与 kernel
   形态不同）**；
   非此形态（flux single-block 的
   proj_mlp+act_mlp→concat→共享 proj_out、wan refiner 等）不套用，标注"未实现 N + 依据 + 收益 0"。
7. **命名/报告全链路一致**（行为约束 #2）：算子名（`mm_gelu_mxquant`）与 pattern 文件/工厂/
   config flag/报告保持一致；报告按 dummy-run §C 双表与 post-enable 六面复核口径产出。

## 4. compile 阶段的适配动作（阶段 2；API 验证通过后再做）

> 入口：`framework-integration` SKILL §②「两阶段顺序纪律」的**阶段 2**——阶段 1
> （API / runtime 注入）**接口可行 + 同 seed 数值对拍通过**后，才做图级适配。
> 若阶段 1 的单点收益已达标、或本条链判定为负收益，则**不做 compile 适配**，把回退记入迭代表
> （即停在 API 阶段，判据见该 SKILL §②「何时停在 API 阶段即可」）。

顺序 1→7 不可跳（出口证据行格式见 SKILL「验收与推进纪律」）：

1. **候选判定行**（入口先落一行）：复用哪个既有 pattern 族 / 期望收益类型（kernel 数减少 /
   搬运消除 / 后端选择）/ 路径预判（register_replacement / GraphPatternEntry）。
2. **载体选择**（SKILL Phase 2 路径表）：functional API 与 `nn.Module` 权重（freeze 前作 pattern
   输入）→ `register_replacement`；pattern 中间夹动态 shape 节点、或需按 meta 精细校验 →
   GraphPatternEntry + handler 手动改图。**禁止手写 FX graph traversal pass**。
3. **接线**：register_replacement 走三段注册（阶段与出口见 SKILL Phase 3；**三个文件清单与命名
   规范单点**见 `registration-checklist.md`，本文件不复述）；GraphPatternEntry 走
   `register_xxx_graph_entries(pattern_pass)` + flag 仍在 `FusionPatterns` 声明。
4. **真图命中**（勿用 symbolic / make_fx 构图代替）：`scripts/probe_real_graph_pattern.py` 在真实
   compile 图内打 probe 计数；阶段 1 若改过模型层（如 RoPE 实数域改写），此处须核对它确实是
   命中的**前置**、而不是被图形态噪声吃掉。
5. **数值核验**：同 seed eager vs compile 对拍（`scripts/numeric_check_eager_compile.py`）——
   位级（改写前后数值路径完全等价）或 fp8 量化级近似（须标注「非位级」+ 模型级质量门）。
6. **计数契约**：`scripts/check_fusion_hit.py` 核对融合 kernel 实例数 == 站点数、被替代 kernel
   消失（图命中日志不替代）。
7. **收益判定与开关治理**：受控交错 A/B（噪声阈值 <3%）判采纳 / 回退；开关按 §1 R1-R7——
   唯一 `CompilationConfig` 命名空间、默认 True 需命中 + AB 证据、净负直接移除不留死代码。

与阶段 1 的分界（两条最容易写错）：

- **阶段 1 的产物是阶段 2 的输入，不是替代品**：eager 单算子已覆盖的站点不必再进图（重复承载只会
  加拷贝 / 调度开销）；阶段 2 的目标是阶段 1 表达不了的部分——整链融合与图级拷贝消除。
- **阶段 2 不得在前端绕开 compile 来"通过"验证**：eager 侧辅助代码只是试验手段，验证口径始终是
  compile 图内真实命中（SKILL 行为约束 #1）。

## 维护与更新

新增融合或调整默认值 → 同步 §2 清单；发现新的开关漂移/载体冲突 → 复盘后并入 §1/§3；
删除已收敛的旧 env/开关 → 在 R6 记录。
