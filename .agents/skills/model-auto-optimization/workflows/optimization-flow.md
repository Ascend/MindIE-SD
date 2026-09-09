# 模型自动优化执行流程（workflow）

> 由 `model-auto-optimization/SKILL.md` 入口分流后 Read 本文件，按阶段模板与门禁严格执行；
> 本文件只承载「流程怎么推进」，各阶段支撑技能的知识点仍在对应 SKILL.md / references 单点维护。

## 运行状态锚点（先读后做）

- 本任务流程状态唯一真相源 = 运行状态文件 run-state（位置与写法见
  `references/run-state.md`，默认 `{工作目录}/agentic/run-state.md`，不存在先按模板创建）。
- 进入新阶段 / 收到执行回执 / 做裁决前，先读 run-state 推进表；每个阶段收尾即时回写
  （状态 + 验收证据路径），不延后到收尾（上下文压缩后可从此重建）。
- 阶段推进门禁：每阶段收尾必跑 `python scripts/stage_gate.py --stage {Sn} --run-dir <…>`，
  error=0 才允许进入下一阶段或宣称闭环；未过不得推进、不得宣称完成。

## 触发链速查（本文件 = 业务 L2；机制契约归 L1）

> 三层定位：本 workflow 是**模型自动优化业务线的 L2**——阶段化执行 + 单特性策略深挖；机制
> 契约（run-state/stage_gate/迭代表/双报表/声明纪律/必测组合覆盖）属 **L1 编排协议**（见
> SKILL「强制流程」与 references/run-state.md），本文件只引用不重复。

任务**必触发链路**（不可跳过，跳过即门禁/交付缺失）：

```text
① 入口分流（L1 SKILL） → ② Read 本文件（L2 必读，不得绕过）
→ ③ 建 run-state（推进表 + 迭代表 + 特性覆盖清单）
→ ④ 覆盖清单：固定特性全集逐项触发判定（做/分析后做/不做+理由，闭环前无未裁决）
→ ⑤ 逐阶段五步 + stage_gate.py --stage {Sn}（S0→S1 融合→S3 并行→S4 有损→S5 训练感知(可选)→close；S2 已并入 S1）
→ ⑥ 闭环：overview/detail 双报表（§C 候选治理 + §E 质量证据）→ --stage close
```

- **有损路径必测组合（L1 契约，detail-report §C.2 禁区）**：`量化×稀疏` / `稀疏×Cache` /
  `量化×Cache` / 三元 `Cache+量化(w8a8f8)+稀疏`——每行必须实测或登记豁免（带证据），不得以
  预算/排序跳入"未尝试"。
- **条件触发**：S0 `env-install`+`dummy-run`（环境未就绪）；S1 融合 `framework-feature-enablement`
  与 profiling 回路、compilation/operator（使能/融合/开发）；S3 `parallelism-strategy`（多卡）；S4 `combination-search` /
  `seam_check` / `post-enable-review` / 质量门禁（quality-gate + `runs/{task_id}/profiles/{model}.toml`（gen_profile.py 生成不入库） +
  quality_compare 现算）。
- **既有 case 文件非必触发**：framework×model 专属 case（vllm-omni-*/lightx2v-*）仅在同
  框架×同模型时命中；必触发的是上表机制与模板（机制在 L1/L2，判据在 L3）。

## 执行波次与并发（单特性 ∥ → 组合后置）

- **波次 1 · 单特性并行**：覆盖清单判「做」的每个单特性（`kernel融合` 内容 / `并行` / `Cache` /
  `量化` / `稀疏` / `时间步优化`…）作为**独立特性任务，可多 agent 并行**（资源允许：同卡组互斥
  调度、`evidence/{task_id}/{stage}/{feature}/` 隔离、迭代表 round 全局唯一、编排者单写——见
  dispatch-templates「单点特性并行执行」）；每个特性走「特性候选选择链」收敛到采纳档
  （迭代表 retain/签名），产出**特性级过程记录**（迭代表 + detail 分节 + §E 该特性质量证据）。
- **波次 2 · 组合后置**：组合（含 [MUST] 必测两两 + 三元 `Cache+量化+稀疏`）**在单特性采纳档
  收敛后再启动**——否则探索路径爆炸；多个组合可**多 agent 并发**；同 seam 互斥组合不并行
  （seam_check 裁定，取最强档顺序化）。
- **收口（L1 总览）**：编排者把各特性/组合反馈数据录入 `overview_report.md`（覆盖收口表：每个
  特性 + 必测组合均有行与单值）；特性内部方案/过程**不进总览**（留在 detail/迭代表/§E）——
  两者不同、缺一不可（见 SKILL「L1 收口」）。

## 阶段推进规则（通用，对 S0/S1/S3/S4/S5 与闭环一致生效；S2 已并入 S1）

1. 每阶段五步：方案确认 → 执行（实施/采集）→ 自验证 → 验收 gate → 阶段总结回写。
2. **方案确认点**：实施类阶段开始前，把候选集/方案/回退策略用结构化提问向用户逐条确认，
   结论记录进 run-state「决策与轮次」；用户未确认不得实施。
3. **拒收语义**：执行者（或 subagent）回执缺自验证记录、或记录与 evidence 矛盾 → 拒绝接受，
   列出缺失项要求补齐后重交；编排者不得用自行审查替代验证。
4. **FAIL 循环上限 5 轮**：验收 FAIL → 定位修复 → 重验，同一阶段实施/复核合计超 5 轮 →
   回退该阶段改动（或关闭开关）→ 向用户报告阻塞点请求决策，结论记入 run-state。
5. 单变量与同口径：一次只改一个变量；对比用同窗口同卡组、同口径（见 SKILL「声明纪律」）。
6. 阶段通过且用户确认后，再进入下一阶段；涉及本仓代码改动按 dev-workflow 子任务与提交流程。
7. **假说先行、一次一候选（探索型阶段 S1/S3/S4 内部迭代）**：任何候选/档位/尝试先写
   run-state「候选与迭代表」一行（假说 + 预期），再实施；不得并行盲试多个候选。单候选失败
   不结束迭代——记拒绝签名（crash / implementation-wrong / degenerate / dominated /
   out-of-scope / no-gain）后换下一假说；structured negative 是候选素材不是终点。
8. **预算与收尾**：迭代表记录 `预算: max_rounds / 已用`（时间盒见 references/effort-estimation.md）；
   预算耗尽或目标达成 → 带 frontier 收尾（terminal_pending_review），把未尝试候选保留为清单
   交用户选档（进 detail-report §C），不静默丢弃、不留无限调优。
9. **契约变更即新版本**：改变基线/拓扑/口径/特性组合/产物结构（含跨任务复用结论）→ 视为新版本：
   关闭旧 claims，重跑 smoke + formal 且质量门禁重过后方可再宣称（见 SKILL「声明纪律」）；
   有损项质量对照的 golden 锚 = 同 seed 冻结 baseline 帧/输出（沿 quality-gate.md，不跨负载迁移）。

## 特性覆盖清单（触发判定前置 · 防漏机制）

- 任务启动（SKILL §0 启动确认）时在 run-state 建「特性覆盖清单」：对**固定特性全集**
  （`kernel融合` / `并行` / `Cache` / `量化({修饰符})` / `稀疏` / `时间步优化` / 训练感知组可选）+
  framework-support-matrix 预扫缺口，逐项给触发判定：`做（目标档/经验档）` / `分析后做` /
  `不做（理由：无对应瓶颈 / 预期收益小（整 block <0.5%）/ 框架不支持）`；分析过程中按证据
  更新（记 round）。
- 判定是显式的：不做/暂缓必须带理由，禁止静默不触达某特性方向。
- **档位排序**：每能力按 framework×模型组合标档位——`已支持` / `待配置` / `待开发` / `上界`
  （图例见 framework-support-matrix §〇）：`已支持 / 待配置` 先做（直接用或接线）；`待开发`
  （无支持需完整实现，如 operator-dev/extension-dev）先估成本并经 §0 补齐策略用户确认再投入；
  `上界` 只作天花板诊断不宣称（定义预留，待人工填写）。
- 闭环前复核清单**无"未裁决"项**（见「闭环复验」），否则视为交付缺失。

## 标准回路（采集 → 分析 → 方案 → 复验；S1 融合分析 / S4 / 闭环共用）

1. 采集：`profiling-collect`（自家脚本 scripts/collect_profile.py；三方框架入口用补丁 + torchrun）
   → ASCEND_PROFILER_OUTPUT；Warmup 在 profiler 外（默认 5 步，compile ≥10）。
2. 分析：`profiling-analyze`（scripts/analyze_trace.py 5 层管道 → 瓶颈/融合候选）。
3. 方案：`performance-optimization`（mindiesd-features.md 选档）；框架侧开关/使能由
   framework-feature-enablement 执行；差距 <3% 视为噪声（阈值单点维护于 performance-optimization）。
4. 复验：重新采集 + scripts/compare_traces.py kernel diff + 锚点行实测 e2e（同窗口同卡组，
   rank0/p50）；中间行可用首步耗时辅助对比（见下）。

**步数口径（e2e 主口径：锚点行实测，中间行可估算；首步耗时辅助 · 强制，见 overview-report §1/§1.1）**：

- **性能主口径 = 单个完整请求墙钟（e2e，s/请求）**：`基线（base）` / `最终推荐` / `三元组合`
  三行（报表结论锚点）必须**实测 e2e**（禁止估算）；完整请求墙钟随步数**非线性**变化
  （固定开销 + VAE tile 解码 + 热效应），不同步数行不按 e2e 等比例换算/直接比较。
- **中间行 e2e 可估算并标 `[估算]`**：`行 e2e[估算] = base DiT 耗时 × (行首步耗时 / base 首步
  耗时) + (VAE + 其他固定耗时)`（§1.1 公式，同步数前提）；少步/估算只用于铺中间链与筛选方向，
  **结论锚点不估算**。
- **首步耗时（warmup 后第 1 步, s/step）= 辅助口径**：步数无关、跨 run 可比——用于少步性能
  识别（1 预热 + 1 采集，profiling-collect「少步快速采集经验」）与中间行 e2e 估算输入；
  需要质量分析的有损行与最终叠加/组合行 = 全量步数实测（质量与 e2e 同次运行）。

停止条件：目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈（定义在 performance-optimization，此处不重复）。

## 阶段 S0：环境准备

- 支撑技能：`env-install`（含 `remote-access` 工具）、`dummy-run`。
- 入口动作：mindiesd + 三方框架安装/确认 → 权重确认与下载（下载前先确认远端是否已有）→
  dummy run 出基线输出。
- 方案确认点：目标模型/框架/部署拓扑/时间盒（SKILL §0 启动确认未做则先补）。
- 自验证回执需含：import mindiesd 成功证据、权重无 `.incomplete` 证据、dummy-run 输出路径。
- 验收证据：evidence/S0 下 import 记录 + 权重校验记录 + dummy-run 输出。
- 门禁：`stage_gate.py --stage S0`；经验回填槽位 S0-1。

## 阶段 S1：无损 · kernel 融合优化（接入 + 融合统一链 · 原 S1/S2 合并）

> 本阶段 = "kernel 融合优化"的统一流程：**先 profiling 分析，再识别机会点与收益，然后按框架
> 能力判定接入方式（API/compile）、必要时开发算子，最后才决定是否 compile**——不再分
> "算子接入"与"融合优化"两个独立阶段。令牌 S2 已移除（并入 S1；stage_gate STAGES 亦不含 S2）。

1. **Profiling 分析算子序**：profiling-collect → profiling-analyze（5 层管道 / kernel 执行序），
   明确各算子/分解链耗时与执行序（走标准回路；**少步识别收益**：如 1 步预热 + 1 步采集即可，
   见标准回路「步数口径」）。
2. **识别可融合机会点**：至少覆盖 **rope、norm**（RMSNorm/RoPE/AdaLN 等）；并识别是否还有其它
   优化空间（数据搬运/Copy/图形态/后端选择），候选先入迭代表（假说先行，推进规则 7/8）。
3. **收益分析**：对候选做融合前后**算子执行序/耗时收益评估**（候选 → 执行序对比 → 收益判定；
   少步下用首步耗时（辅助口径）同口径识别收益，见标准回路「步数口径」）；
   **首步 block 占比 <0.5% = 收益小可不执行**（签名 no-gain，进 A.1.3/§C）；≥0.5% 才进入采纳
   评估（三层证据复验；<3% 噪声不宣称；**最终采纳行全量步数实测 e2e，作为结论锚点**）。
4. **接入方式判定**：查框架是否支持算子接入（是否有抽象接口）：
   - 有抽象接口（rope/norm 类）→ 通过 **mindiesd 接口接入**（framework-feature-enablement
     运行时注册表替换 = `kernel融合` 特性 · API 接入方式）；
   - 无抽象接口 → 尝试 **compile 接入**（pattern / GraphPatternEntry，compilation-dev）。
   - 若融合算子本身**不存在** → **operator-dev 开发**（路由外部 cannbot + 本仓经验）；
     若算子已存在 → 直接接入。
5. **初成验证与 compile 决策**：算子初次开发完成可**先经 API 接入确认可用**；随后**依据框架
   本身的设计选择是否 compile**（非一刀切）。
6. **compile 快速验证载体**：为便于 compile 开发，先把模型代码做成 **dummy run**；compile 基于
   dummy run 快速验证（命中/形态/数值核验），**通过后再到真实框架上验证**（compile 验证口径 =
   compile 图真实命中，禁止前端绕开）。
7. 归属与记录：内容属 `kernel融合` 特性（子项/方法 compile 或 API 接入进说明列与细分 A.1，
   不派生特性名）；每候选迭代表 retain/reject + 签名。

- 支撑技能：`profiling-collect`、`profiling-analyze`、`framework-feature-enablement`、
  `compilation-dev`、`operator-dev`、`dummy-run`。
- 方案确认点：候选集排序与时间盒（effort-estimation）向用户确认；接入方式（API/compile）与
  "是否 compile"按上述判定链与用户确认后再实施。
- 验收证据：三层证据（图命中 → kernel diff → 锚点行实测 e2e）+ 数值核验 + 精度结论；API 接入项另
  见 kernel diff（分解链消失 / 单算子出现）。
- 门禁：`stage_gate.py --stage S1`（迭代表含每候选 gate 证据与 retain/reject 裁决）。

## 阶段 S3：无损 · 并行通信

- 支撑技能：`parallelism-strategy`、`profiling-collect`、`profiling-analyze`（带宽/掩盖测量）。
- **卡组前提（强制）**：选卡必须先满足 parallelism-strategy「卡组拓扑规则」（2 卡 ∈ {0-1,2-3,4-5,6-7} /
  4 卡 ∈ {0-3,4-7} / 8 卡 = 0-7）；非法卡组不采纳；卡组变更 = 契约变更，锚点行须新卡组重测。
- 执行逻辑（按序推进）：
  1. **卡组健康与拓扑核验**：`npu-smi -t topo` + `npu-smi info`（Health/Alarm/占用）；多卡
     4 步 smoke 验证 per-step 时长（43s/步 = SIGKILL 残留态，须驱动复位/换组）；
  2. **带宽探针**：先测**不同卡数**下的通信带宽（如 4 / 8 / 16 卡；hccl/集合通信带宽基准，
     同拓扑同窗、固定 rank 口径；等价工具 torchrun + torch_npu 姿势见
     parallelism-strategy ascend-topology-bandwidth-diag §3）；
  3. **拓扑决策**：若明显发现 **4 卡带宽高于 8 卡** → **仅在 4 卡内做 USP**、两个 4 卡之间做
     **CP**（USP4CP2 场景；该场景的**稀疏须依赖并行稀疏**）——候选与理由入迭代表/说明列；
  4. **默认序**：优先 **USP** → 再看是否有 **CFG** → 然后 **CP**（allgather KV、Q 切分）；
  5. **通算掩盖**：通信方案选定后**同步做 compute/comm 掩盖**（step_trace 拆分：
     compute / comm(未重叠) / free，Overlapped 可用性），**识别掩盖率**并记录。
- 方案确认点：卡组拓扑结论 + 带宽探针结论 + 并行候选（USP/CFG/CP、拓扑、USP4CP2 等）向用户确认。
- 执行：few-step + 多 rank 验证特性正确开启与掩盖率（少量 step 快测；采纳项回真实 serve 同窗
  复验，并行拓扑/通信不在快测覆盖内）。
- **资源受限诚实标注**：合法卡组/多卡不可用导致带宽探针或掩盖率未测 → evidence 显式记 ❓ +
  原因 + 复用历史证据的显式引用（`../runs/{旧task}/evidence/...` + 「复用历史非本轮实测」），
  **禁止缺省 S3 分析即标 done**（2026-09-08 教训：S3 曾用旧 evidence 残留蒙混通过门禁）。
- 验收证据：卡组拓扑核验 + 固定 rank 口径通信下降 + 掩盖率识别（或 ❓ 说明）+ 输出一致 + 墙钟。
- 门禁：`stage_gate.py --stage S3 --task-id {task_id}`（证据须在本任务 evidence/{task_id}/S3/）。

## 阶段 S4：有损优化

- 支撑技能：`performance-optimization`（特性选档）、`framework-feature-enablement`（开关）、
  `benchmark-dev`（稀疏度/量化档选型证据）。
- 方案确认点：目标档（经验默认 8bit + 80% 稀疏 + Cache）与组合边界向用户确认；确认内容含
  **组合覆盖清单必测集**（两两 + 三元 Cache+量化+稀疏，量化档以 `w8a8f8` 为代表档参与组合），
  缺维度或必测降级豁免须在此一并说明。
- **量化档位链（必须按序推进）**：`w8a8` → `w8a8f8` → `w4a4f8` →（可选）`w4a4f4`。
  档位链上每一档：框架实现则使能并测；**框架未实现则不使能**（标注，不虚列）；
  `w4a4f4` 为可选档——框架支持才尝试。**规格标识统一用 `w8a8(mxfp8)` 式**（修饰符 + 括注具体
  规格：类型 mxFP8/int8、FA 路径 FP8RotateQuantFA/框架案例等），入迭代表与报表说明列。
- **量化选型来源（按序）**：优先从**框架推荐**中选择格式与方法；框架无推荐时，次之从
  **mindiesd docs（quantization.md/features）** 中选择。
- **Cache / 时间步优化：与框架能力看齐**——按框架推荐/自带能力（cache_dit 等）使用其配置档，
  **mindiesd 不做额外适配**（不引入 cache_agent 兜底路径）；时间步=免训练步数裁剪（换权重归
  S5 训练感知）。
- **稀疏**：稀疏本身与**量化叠加（指 FA 量化 f8）** 后会产生**新算子**（稀疏+FA 量化融合）——
  以 **mindiesd 算子性能**为分析基准；若框架未接入该算子，则**本地 benchmark 测试**
  （benchmark-dev/mindie_bench）后，**接入优选算子与对应 mask 算法**；**质量可通过调整稀疏度
  决定**（稀疏度-质量曲线为选档依据）。经验起扫：图像 0.6 / 视频 0.8 起步；图像无 2D 路径时
  staying-dense 必须 fail-closed（不虚报命中）。
- **组合试验规则**：
  - 量化参与组合时**必须包括 `w8a8f8` 与其他特性的组合**，在若干候选组合中**选择最高性能的
    组合**作为该量化档组合代表；
  - **Cache 与时间步优化二选一**（同一次组合/同一条对比行只选其一，不叠加）；
  - [MUST] 必测两两 + 三元沿用（豁免需原因+证据，先于自由候选；迭代表 [MUST] 行 close 前无未裁决）。
- **试验记录（强制表格化）**：每一条试验信息都必须在表格中记录（迭代表 + detail §B 各子表 +
  §E），候选分类直接写 **`算子(稀疏度xx%)`** 式（如 `rf_v2(80%)`、`EagleQBSA(80%)`），
  量化行带规格标注（`量化(w8a8f8)` + 括注规格），不得只写无规格的裸名。
  **每行登记实测步数**：性能选型/识别行可少步（注明「识别步数」，如 1 步），质量与采纳候选行
  全量步数；**少步识别 → 全量叠加核验**：少步只筛方向，最终叠加/采纳必须全量步数实测且与
  同口径全量基线比（见 overview-report §1.1、标准回路「步数口径」）。
  **回退必须全量实测**：cache/稀疏/量化发生回退（降档/错开窗口/换实现/整维回退后仍保留的档）时，
  回退后最终档必须全量步数实测 e2e 识别性能加速（回退档=潜在采纳锚点，禁止少步/估算宣称，
  见 combination-search「层回退策略」）。
- **特性候选选择链（L2 纵向：对每个判定「做」的特性，深挖到最佳策略）**：经验档起扫 → 单变量
  扫描/曲线（benchmark-dev）→ 当前算子/载体不达标则**回退到替代算子/载体**（如稀疏 rf_v2 →
  ada_bsa、量化档按档位链推进/降档、compile ↔ API 换载体）→ 与其它已采纳特性 seam 组合试叠
  （[MUST] 必测行优先）→ 质量/计数门禁 → 迭代表 retain/签名 → **收口该特性最佳策略**（采纳档进
  overview/detail 对应行 + §E 质量证据登记）。判据/选项在 L3（support-matrix 档位、
  mindiesd-features、profiles `[domain]`），**执行链与回退裁决在本 L2 + 迭代表**（一次一候选、
  带签名）。
- 执行纪律：单特性开启 + 精度校验 → 组合试验按 `performance-optimization/references/
  combination-search.md`（seam 静态判定 / 单变量叠加 / frontier 保留 / 层回退）→ 每档落地后
  **post-enable-review 六面复核**（references/post-enable-review.md）→ 无损项叠加复核；
  每档/每组合先入迭代表（推进规则 7/8），否决必须带签名。
- **组合覆盖清单（S4-2 必测，防静默缺行）**：凡 `量化`/`稀疏`/`Cache` 中 ≥2 个维度有通过
  quality gate 的单点 frontier，S4-2 必须在迭代表建「[MUST] 必测行」：跨 seam 两两全测
  （量化×稀疏、稀疏×Cache、量化×Cache，**量化以 w8a8f8 为代表档**）+ **三元
  Cache+量化(w8a8f8)+稀疏 强制实测**（Cache 与时间步二选一前提下，选 Cache 或时间步其一进
  必测集；仅当某维度在 framework×模型下不可用时才可降级「豁免」并登记原因与证据）。必测行先于
  自由候选执行；覆盖清单收口 = 无未裁决 [MUST] 行（模板与豁免语义见 combination-search.md
  「必测组合覆盖集」）。
- 质量门禁：有损项端到端质量判定（quality-gate.md + 仓库 evals/），pass/fail 结论登记进
  run-state 备注；对照锚 = 同 seed 冻结 baseline 帧（不跨负载迁移阈值）。
- **量化后融合复审（L1 调度，量化方案最终确认后必做）**：量化档定稿后，由 **L1 调度**一次
  「基于量化后 kernel 序列的融合机会复审」——回 S1 融合链审视量化后新出现的融合机会
  （如量化 GEMM + epilogue、量化激活融入等），候选入迭代表并按 0.5% 阈值 + 三层证据评估；
  有收益 → 实现并**收益归因到量化行**（§2.4，不单列无损行）；无新机会 → 迭代表记录
  no-gain/不适用。复审由 L1 在阶段间调度（非 L2 自行决定是否做）。
- 验收证据：单特性精度报告 + 组合矩阵（含 [MUST] 覆盖清单，无未裁决必测行）+ 回退名单 +
  每档使能复核记录（迭代表/§B/§E 表格化，命名 `算子(稀疏度xx%)` 与量化规格标注）。
- 门禁：`stage_gate.py --stage S4`（迭代表含每档 gate 证据与裁决，[MUST] 行含裁决或豁免理由）。

## 阶段 S5：训练感知优化（预留分支 · 少步蒸馏为基础，可叠 SLA / QAT）

> **暂无案例**：本节为分支骨架，供首次真实训练感知案例回填时按骨架落地并接线
> （质量门禁沿用 S4 的生成质量域协议；经验/坑按 case 回填规范沉淀到能力技能 references 与
> 槽位 S5-1）。在案例回填前：S5 只登记不宣称（不产出可对外收益声明）。

- **定位与基础**：训练感知优化**独立为 S5**——以「无损优化收敛（S1 融合 / S3 并行）+ 少步蒸馏
  （训练感知基础）」为前提，进一步考虑**叠加 SLA 与 QAT**（语义与组合边界待案例定义）。
- **边界**：与免训练组（S4）区分——训练感知 = 换/精调模型权重（少步蒸馏模型、可训练稀疏、
  QAT），命名归报表「训练感知优化」行组（overview-report §3c）；质量-速度权衡以
  「相对全步数/原始模型」的质量基线表述，不跨域套用 S4 阈值。
- **执行骨架（案例回填后逐条细化）**：
  1. 前置：无损（S1/S3）+ 少步蒸馏 baseline 跑通（同 seed 质量锚）；
  2. **少步蒸馏权重来源**：若**框架本身支持 4 步 / 8 步蒸馏权重** → 下载并尝试
     （**推荐从 modelscope 社区下载权重**）；若框架**本身不支持蒸馏权重 → 不处理**（跳过
     少步蒸馏子项，不额外适配）；
  3. 候选：少步蒸馏档（4/8 步）→ 叠 SLA → 叠 QAT（逐档单变量、迭代表假说先行/签名/预算）；
  4. 质量-速度权衡登记：每档给相对原始模型质量 + 墙钟；计数契约/off-identity 照常；
  5. 与 S4 有损维叠加的 seam 语义（如 QAT×量化档）待案例界定后入 combination-search 声明。
- 支撑技能：预留（训练侧流程/能力待 case 回填；仓库能力技能按需补充）。
- 验收证据（骨架）：S5 baseline + 每候选质量-速度权衡表 + 迭代表裁决。
- 门禁：`stage_gate.py --stage S5`（案例回填后细化验收点；无案例跳过本阶段，run-state 不登记 S5 行即可）。

## 闭环复验（close）

- **特性覆盖清单复核（前置）**：run-state「特性覆盖清单」无未裁决项（每特性有 做/不做 结论
  与理由可查），未裁决先补判再进 close gate。
- **组合覆盖清单复核（前置，S4-2）**：凡触发必测集的任务，run-state 迭代表 [MUST] 行无
  「未裁决」——每行有 已测（retain/reject + 证据）或 豁免（**形态穷尽清单** + 原因 + 证据，
  缺清单视为豁免无效）；缺失即视为组合覆盖缺口，先补齐（补测或合规豁免）再进 close gate。
- **报表结构 lint（前置，机器校验）**：`scripts/report_lint.py {overview_report.md}` error=0
  （总览表 8 列/枚举/锚点禁估算/质量列非空——防「列要求不符」类缺口，2026-09-08 起强制）。
- **profile 强校验（前置，机器校验）**：`evals/scripts/check_profile.py --model {model}
  --task-dir runs/{task_id}_{model}_optimization` error=0——profile 必须由 gen_profile.py 在
  S0 冻结基线后生成到 runs/{task_id}/profiles/（**仓库不存具体模型 profile**）；缺 profile /
  字段占位 / 仓库内现具体 profile → 不得 close。
- **强制交付**：优化总览报表 `overview_report.md`（基线 = TP×N 多卡未优化的单个完整推理请求
  墙钟；三层级行组 + 每特性组合搜索子表；模板与规则见 references/overview-report.md）+
  优化细分报表 `detail_report.md`（references/detail-report.md）——缺任一视为未闭环；框架未提供
  特性必须明确标注「未提供」并给证据。
- 执行：同窗口同卡组复现 → 收益口径一致核对 → 双报表 + final_report + evidence.json 归档
  （references/artifact-layout.md）→ 槽位/经验回填（.agents/README.md §7 回填规范）。
- 门禁：`stage_gate.py --stage close --task-id {task_id}`（推进表该行声明路径必须包含
  overview_report.md 与 detail_report.md；close 自动跑 report_lint + check_profile，见 stage_gate
  `_run_close_tools`）。

## 阶段反馈与收尾

- 流程/技能缺口（描述不清、缺约束、参考过时）→ 收尾前记入 run-state 反馈节，随复盘按
  dev-workflow §6.3/§6.4 回填，不进入对外 final_report。
- workflow 或阶段清单变化 → 同步 SKILL.md「强制流程」指针、scripts/stage_gate.py 的 STAGE
  常量与 references/run-state.md。
