---
name: model-auto-optimization
compatibility: 无额外工具依赖（负责编排其余能力技能）；前置：目标模型/框架代码与 NPU 环境（见 S0）
description: >
  多模态模型自动优化流程总入口（编排层）：把一个三方框架托管的扩散模型从环境准备到无损/有损
  优化自动跑完。任务判定后按 S0 环境准备、S1 kernel 融合优化（原 S1/S2 合并）、S3 并行通信、
  S4 有损优化、S5 训练感知（预留，少步蒸馏 + SLA/QAT）、闭环复验路由到能力技能（env-install /
  framework-integration / profiling-collect / profiling-analyze / performance-optimization /
  dit-parallel-opt / dummy-run 等），检查各阶段产物与验收并回填经验槽位。
  当用户需要对具体模型做框架接入、无损/有损加速、并行调优或确认收益时使用本入口；即使用户只提
  模型名加"优化/加速/跑通/采profile"而未说框架或阶段，也应由本入口判定路由。仓库代码开发类任务
  请走 dev-workflow。流程执行/验收按 workflows/optimization-flow.md，run-state + stage_gate 门禁。
---
# 模型自动优化总入口（编排层 · S0–S4 + 闭环）

## 职责

1. **任务判定**：区分模型优化（本入口）与仓库开发（dev-workflow）。
2. **阶段路由与执行**：入口分流后按 `workflows/optimization-flow.md` 阶段模板推进，路由到对应
   能力技能执行；编排层不直接改代码。
3. **运行状态维护**：本任务推进状态唯一真相源 = run-state（`references/run-state.md`，
   默认 `{工作目录}/agentic/run-state.md`）；每阶段收尾即时回写推进表并跑
   `scripts/stage_gate.py` 门禁，error=0 才进入下一阶段。
4. **产物与验收检查**：每阶段核对验收点（细节在能力技能与 workflow 阶段模板内单点维护）。
5. **证据记录与经验回填**：保留基线/单变量/证据链（kernel diff + 锚点行实测 e2e / 首步耗时辅助
   口径）；新经验按
   槽位登记回填（.agents/README.md §2/§4），不另起炉灶。

## L1 收口（总览 vs 特性报表 · 三层定位）

> 本技能 = **L1 编排入口**：定序、交付件契约、机制所有权与**总览收口**；业务 L2 =
> `workflows/optimization-flow.md`（阶段化执行 + 单特性策略深挖）；L3 = 能力技能。

- **总览报表（overview_report.md）由 L1 维护 = 覆盖收口表**：确保**每个特性及必测组合都有行与
  数据**（含单点行、叠加链、最强组合、必测 [MUST] 两两+三元）；数据由 L2 各特性/组合策略
  **反馈录入**，L1 不自测、只核对覆盖与口径（性能/质量/步数列同基线、无空值）。
  **每行登记实测步数（「步数」列必填）**：性能识别行可为少步（如 1 步，同口径少步对比），
  质量/最终叠加行全量步数——少步识别收益、最终叠加全量（见 optimization-flow 标准回路
  「步数口径」与 overview-report §1.1）。
- **L1 阶段间调度（编排职责）**：关键横切动作由 L1 在阶段间调度——如**量化方案最终确认后，
  L1 调度「量化后融合复审」**（基于量化后序列回 S1 融合链审视新融合机会，见
  optimization-flow S4）；类似地，跨阶段复核（量化后回 S1/S3）也由 L1 统一安排，而非由
  L2 特性策略自行决定。
- **特性自身的报表**（detail 分节 + 迭代表过程 + §E 质量证据）体现**该特性内部的方案选择与
  过程记录**（候选→扫描→回退→采纳档→证据）；两者不同、缺一不可（close 校验双报表）。
- 执行波次与并发见 optimization-flow「执行波次与并发」：单特性可并行，组合后置于单特性收敛后。

## 强制流程（先读后做，不绕过）

- 入口分流后必须 **Read `workflows/optimization-flow.md`**，按其中阶段模板/确认点/验收 gate/
  闭环强制交付执行；不得绕过 workflow 直接实现或直接宣称完成。
- 任务开始先建 run-state；每阶段收尾：更新推进表（status=done + 验收证据路径）→
  `python scripts/stage_gate.py --stage {Sn} --run-dir {工作目录}/agentic` → error=0 才进入
  下一阶段或宣称闭环（未过不得推进）。
- 执行回执/自验证格式与 subagent 派发模板见 `workflows/references/dispatch-templates.md`：
  回执缺自验证或与 evidence 矛盾 → 拒收重交；验收 FAIL 修复合计上限 5 轮 → 回退该阶段改动并
  向用户报告阻塞点。

## 任务判定

```text
修改 MindIE-SD 代码（pattern/算子/图下发/测试/文档/提交）→ dev-workflow（侧轨）
对一个模型做接入/加速/量化稀疏缓存/并行/性能确认                 → 本入口（编排层）
├─ 环境/权重/dummy run 未就绪 → S0（env-install + dummy-run）
├─ kernel 融合优化（profiling→机会点(rope/norm 等)→收益→接入/开发→compile 决策）→ S1
│    （framework-integration + profiling 回路 + pattern-dev/operator-dev + dummy-run）
├─ 多卡并行/通信掩盖/拓扑选型 → S3（dit-parallel-opt）
├─ 量化/稀疏/缓存/精度组合 → S4（performance-optimization + framework-integration）
├─ 训练感知（DiT/模型级换外部训练权重：少步蒸馏，叠 SLA/QAT）→ S5（案例已回填，见 S5 槽位）
├─ 非 DiT 段（VAE 全部 + host 固定开销；阶段账占比 ≥10% 才启动）→ S6（VAE 模块 / host 模块）
└─ 整体收益确认/产物归档 → 闭环复验（标准回路）
```

## 0. 任务启动确认（口径与触发判定前置）

任务判定为主轴、进入 S0 前先做一次启动确认（防中途返工与越界改动）：

1. **预扫支持面与缺口**：按 `framework-integration/references/framework-support-matrix.md`
   与目标框架版本，列出预期关键优化路径与缺口清单；用 `references/effort-estimation.md` 估时间盒。
2. **与用户一次性确认并登记**（记录进 run-state「任务与口径」+「特性覆盖清单」，
   旁注 final_report）：
   - 目标与验收：无损/有损档位、目标加速、质量门与时间盒；
   - 特性覆盖清单：对固定特性全集 + 预扫缺口逐项给触发判定——`做（目标/经验档）` /
     `分析后做` / `不做（无瓶颈 / 预期收益小（整 block <0.5%）/ 框架不支持）`，理由可查
     （清单模板见 run-state.md，闭环前复核无未裁决项）；
   - 缺口补齐策略：① **框架侧特性没落地** → 统一走 `framework-integration`
     （原 `framework-feature-enablement` + `framework-extension-dev` 已合并为本技能）
     ——框架已有 → 分支 A 开关使能；框架自身结构性缺口 → 分支 B 补齐开发（先确认改动归属与是否接受改
     三方框架仓 / 合入上游；跨侧特性按代码落点拆两侧子任务）；② fork 补丁 / monkey-patch（钉版本）；
     ③ 绕过；④ 仅记录不补；
   - 实现边界：mindiesd 仓内改动 → dev-workflow 子任务；编排层不直接改代码。
3. **中途新缺口**：回补一次确认，不静默切换补齐方式。

## 阶段路由表

| 阶段 | 目标 | 支撑能力技能 | 产物 | 验收点 | 槽位 |
|------|------|-------------|------|--------|------|
| S0 环境准备 | 环境安装（mindiesd + 三方框架）、权重确认/下载、基线跑通 | `env-install`（含 `remote-access` 工具）、`dummy-run` | 环境清单 + 权重校验 + 基线输出 | `import mindiesd` 成功；权重无 `.incomplete`（下载前先确认环境是否已有）；dummy-run 出结果 | S0-1 |
| S1 DiT·融合 | profiling→识别融合机会（≥rope/norm 等，含其它优化空间）→执行序收益→按框架能力接入（抽象接口=API 接入 / 无接口=compile；算子不存在→operator-dev 开发→初成先 API 接入确认→依框架设计决定 compile；dummy-run 快验→框架验证） | `framework-integration`、`profiling-collect`、`profiling-analyze`、`pattern-dev`、`operator-dev`、`dummy-run` | 算子执行序 + 融合清单 + 三层证据 | 三层证据 + 数值核验 + 精度；候选执行序收益（整 block <0.5% 可不执行）；compile 口径=真图命中（禁前端绕开） | S1-1、S2-1 |
| S3 DiT·并行 | 带宽探针（4/8/16 卡等）→ 4 卡带宽显著高则 USP4CP2（并行稀疏）；默认 USP→CFG→CP（allgather KV/Q 切分）→ 通算掩盖 | `dit-parallel-opt`、`profiling-collect`、`profiling-analyze` | 并行方案 + 多 rank 证据 + 掩盖率 | 固定 rank 口径通信下降；掩盖率识别；特性正确开启；输出一致；候选整体耗时评估 + 时间预算内收敛 | S3-1、S3-2 |
| S4 DiT·有损 | 缓存/稀疏/量化逐个开启 + 精度校验 + 组合试验（允许层回退） | `performance-optimization`、`framework-integration`、`benchmark-dev` | 单特性精度报告 + 组合矩阵（含 [MUST] 覆盖清单）+ 回退名单 + **每档使能复核记录（post-enable-review）** | 目标档（8bit + 80% 稀疏 + 缓存）；墙钟 + 精度双指标；含无损项叠加复核；kernel 序列变化特性（量化等）使能后按 **post-enable-review 六面复核**；S4-2 组合必测集收口（两两 + 三元 Cache+量化+稀疏，无未裁决 [MUST] 行，见 combination-search.md） | S4-1、S4-2 |
| S5 DiT·训练感知 | 换入**外部训练权重**换速度并承担质量代价：少步蒸馏适配器（DiT/模型级），以无损收敛（S1/S3）为前置，可叠 SLA / QAT。**VAE 侧一律归 S6**（含换解码器） | `framework-integration`、`performance-optimization` | 质量-速度权衡记录（相对原始模型）+ **两条并列部署建议（保画质档 / 预览档）** | 前置契约先验（步数语义 = denoiser 评估次数 ≠ sigma 点数；权重装载计数防 no-op 假加速）；叠加行全量步数实测 | S5-1 |
| **S6 VAE + host** | 非 DiT 段优化：**VAE 全部归本阶段**（计算 + 通信），内含手段链——**无损优先**（分片 / 等价 / tile / 编译）→ 达不到目标时启用**换权重**（有损；含换入外部训练的小型自编码器替换原生解码器）；host 段做固定开销优化（交付搬运 + 装载预热，**不含并发/吞吐**） | `vae-opt`、`host-opt`（本阶段专属模块） | VAE 方案 + 一致性证据；host 固定开销账 | **进入条件**：MAO 阶段账测得 `非DiT-解码段` / `非DiT-host段` 占比 **≥10%**（门限口径见 `references/bottleneck-labels.md`，**须带步数档**）；未达则在推进表登记 `skipped`（带理由）；换权重档须与原生解码器同 latent 对拍 SSIM + 灰度相关 + 钳位占比并声明画质档位；换解码器前先核参考框架契约 | — |
| 闭环复验 | 端到端收益确认 + 产物归档 + 槽位回填 | 标准回路（见下） | 复验报告 + 回填记录 + **优化总览报表 + 优化细分报表** | 同窗口同卡组复现；收益口径一致；输出总览/细分报表 | — |

## 标准回路（采集 → 分析 → 方案 → 复验）

> 执行步骤与各阶段用法见 `workflows/optimization-flow.md`「标准回路」；命令/脚本细节在
> profiling-collect / profiling-analyze / performance-optimization 单点维护，此处只留指针与约束。

- Warmup 必须在 profiler 外（默认 5 步，compile 场景 ≥10）；差距 <3% 视为噪声（阈值单点维护）。
- 停止条件：目标达成 / 噪声范围 / 外部瓶颈 / 硬件瓶颈（定义在 performance-optimization）。
- **强制交付**：闭环必须输出**优化总览报表**（`overview_report.md`，基线 = TP×N 多卡未优化的
  单个完整推理请求墙钟（e2e 实测）、三层级行组 + 每特性组合搜索数据、**每行「e2e + 步数 +
  首步耗时」列必填**）与**优化细分报表**
  （`detail_report.md`）——双报表缺任一视为未闭环（模板/规则见 references/overview-report.md
  与 references/detail-report.md，随 final_report/evidence.json 归档）；框架未提供特性须明确标注
  「框架未提供」并给证据。本入口在闭环收尾以 `stage_gate.py --stage close` 校验声明与归档。
  **e2e 主口径：base/最终推荐/三元组合必须实测；中间行可估算标 `[估算]`；首步耗时辅助；
  质量行与最终叠加必须全量步数实测**（见 optimization-flow 标准回路「步数口径」）。

## 评估纪律总纲（无损 / 有损）

> 执行细节与每阶段确认点/验收 gate 在 `workflows/optimization-flow.md`；这里只保留跨阶段判据。

- **无损（S1 融合 / S3 并行）**：候选集 + 尝试 + **整体耗时评估**（性能主口径 = e2e 完整请求
  墙钟：锚点行实测；识别/中间行可用首步耗时辅助或 `[估算]`，见 optimization-flow 标准回路
  「步数口径」）；限时收敛，预算到点带证据收尾；采纳/回退以三层证据判定（图命中 ≠ 运行期生效，
  kernel diff 为准、锚点行实测 e2e 为最终），命中无收益按证据回退；快测（few-step / dummy-run）
  只给方向性结论，采纳项必须回真实 serve 全步长同窗复验（口径边界见 lossless-methodology-notes §F）。
- **有损（S4）**：精度/质量看端到端（全量步数多次复现），性能主口径 = e2e 完整请求墙钟（锚点
  实测 / 中间行 `[估算]`），首步耗时（warmup 后第 1 步）作辅助口径跨步数对比——少步只筛方向，
  质量行/最终叠加全量步数实测；**cache/稀疏/量化发生回退时，回退后最终档必须全量步数实测 e2e
  识别性能加速**（回退档=潜在采纳锚点，禁止少步/估算宣称，见 combination-search「层回退策略」）；
  单算子数据只作方向参考；**量化按档位链推进** `w8a8(mxfp8 式标注) → w8a8f8 → w4a4f8 →（可选）w4a4f4`（框架未实现
  档不使能；选型来源：框架推荐优先 → mindiesd docs 次之）；**稀疏 FA 先扫稀疏度-性能曲线**
  （benchmark-dev / mindie_bench）再端到端，质量由稀疏度调节；**稀疏×FA 量化(f8) 会产生新算子**
  → 按 mindiesd 算子性能分析，框架未接入则本地 benchmark 后接入优选算子与 mask 算法；无损项
  叠加复核（被稀释/失效 → 回退并记录）；组合试验按
  `dit-perf-opt/references/combination-search.md`（seam 静态判定 / 单变量叠加 / frontier 保留 / 层回退 +
  **必测组合覆盖集 [MUST]**：两两全测 + 三元 `Cache+量化(w8a8f8)+稀疏` 强制，防静默缺行——
  **量化组合以 w8a8f8 为代表档、Cache 与时间步优化二选一**）；**试验记录表格化**：分类用
  `算子(稀疏度xx%)`（如 rf_v2(80%)）与量化规格标注（w8a8(mxfp8) 式）；
  **Cache/时间步与框架能力看齐（mindiesd 不做额外适配）**；
  **改变 kernel 序列/数据口径/显存/精度域的特性（量化/稀疏档/缓存/并行拓扑/compile/offload）
  落地后按 references/post-enable-review.md 六面复核并留痕**（融合 / 通信 / 显存并行 / 质量口径 /
  组合 / 计数；量化后需按新序列重走 S1（融合）/S3（并行）回路并重审通信量化可能）。
- **训练感知（S5）**：独立阶段——以无损收敛（S1/S3）+ 少步蒸馏为基础，叠加 SLA/QAT
  时逐档给「相对原始模型」的质量-速度权衡；**少步蒸馏权重：框架支持 4/8 步蒸馏权重则下载尝试
  （推荐 modelscope 社区），框架本身不支持则跳过（不额外适配）**；与 S4 有损维的 seam 语义：
  **已由首案例界定**——训练感知档可与免训练有损档直接叠加（量化/稀疏在少步档同样生效），
  但**少步档下 Cache 可能整体失效**（步数预算被 warmup 吃满），判据用「开/关两档产物 md5 是否
  相同」的字节级等价，而非加速比（见 `framework-integration/references/train-aware-lossy-method.md` §4.3）。
  - **换入外部训练权重前，先验三条前置契约**（首案例证明这是最贵的一课）：
    ① **步数语义**：声明了 base schedule 的蒸馏适配器，`num_inference_steps=N` = denoiser
    **评估次数**，不是 N-1 个 sigma 区间——按 sigma 点理解会少跑一步；
    ② **装载计数契约**：日志须出现目标模块数 / rank 提升 / 各 worker 激活记录，防「权重没生效
    但请求跑通」的 no-op 假加速；
    ③ **换组件前先读参考实现（如同类框架的官方实现）的判定契约**（用什么张量形状识别该
    checkpoint、用什么参数构造、latent/输入是否被变换）——契约对不上就是**选错权重**，
    扫参不可能补回架构级错配（方法与四个契约项见 `framework-integration/references/train-aware-lossy-method.md` §3.3）。
  - **换入的组件属「预览级」时必须并列两条部署建议**（保画质档 / 预览档），只报最高加速比
    视为交付缺失；质量取证优先用**灰度相关 + 钳位占比 + 输出 std** 三项（PSNR 对丢高频的
    预览级解码器不敏感，见 `quality-gate.md`）。
  - **方法（与产物隔离）**：`framework-integration/references/train-aware-lossy-method.md`
    —— 通用分类、三条前置契约、协同定位比值法、质量分层、归因链与声明纪律；
    **某框架的开启方式属框架差异**，见该框架的 enablement 文档
    （如 `framework-integration/references/vllm-omni-train-aware-enablement.md`）。
  - **数字纪律（看趋势与量级，不看绝对值）**：**不写绝对耗时与绝对质量分值**（只在本次环境
    成立，会被误读为该组合预期值）；**要写**大致加速比（量级/约数）与质量变化度（相对基线的
    差值/降幅）；比例关系与方向结论（标「本组合观测」）保留；原始实测数字留在会话产物目录
    （纪律见 `.agents/README.md` §7「数字纪律（强制）」）。
- **自动化执行要点**：每次只改一个变量；证据链三层递进；正确性先于性能（输出不对性能无意义）；
  同窗口同卡组、多次复现取中位数；单次结果不迁移（换框架/并行配置必须重验）。
- **优化方向动态优选（禁止路径固化）**：**同一方法在不同框架下由不同能力实体承载**（融合 op /
  eager 路径 / 框架自带开关 / 待开发缺口），**收益因此不同、优选方向也随之不同** ⇒ **不得把某个
  模型的优化路径固化成执行序**（案例里的方向结论只作「该组合观测」）。方向必须**在任务内判定**：
  ① 本任务口径下测单点（不引用别处排序）→ ② **比值法**（实测 ÷ 单点连乘）定协同 →
  ③ 结合能力面（`framework-support-matrix.md` 该框架列）与本任务实测收益 → ④ 定方向并登记
  「为何这样排序」；**换框架 / 换模型 / 换负载规模（时长、分辨率、序列长、拓扑）= 重判**。

## 声明纪律与真实性核验（收益数字的归因与边界）

1. **收益声明必带分母**：vs-naive / 同拓扑同卡组 baseline / 上一版本；禁止静默混用分母；
   无匹配基线的运行只报绝对值并注明「不宣称加速」。
2. **证据分级与宣称范围**：CPU/静态检查 → 单次 smoke（运行契约证据）→ formal
   （同窗口同卡组 ≥N 次 hot 取中位数，注明排除项）→ 才允许对外宣称；smoke ≠ formal ≠ 质量门。
3. **计时口径显式化**：写明计时起止边界与排除项（model_load / compile_prime / warmup）；
   中位数配样本数。
4. **真实性核验（计数契约）**：宣称收益前必须证明技术真实参与——记录每技术运行期计数
   （缓存复用次数、稀疏/量化 kernel 调用数、融合 kernel 实际执行次数）；预期计数不符按
   fail-closed 处理（见 framework-integration「特性开关生效验证」），防 no-op 假加速。
5. **off-identity 交叉核验（S4）**：关闭该技术应恢复 baseline 行为，作为伪影/收益归因证据。
6. **质量门禁（S4/闭环）**：有损项端到端质量判定（定量 + 视觉伪影 + off-identity）方法与工具
   见 `accuracy-gate/references/quality-gate.md` + 仓库 `evals/`；本入口只检查
   门禁是否产出 pass/fail 结论并登记进 run-state 备注与 final_report「口径与声明」。
7. **契约变更即新版本**：改变基线/拓扑/口径/特性组合/产物结构（含把某任务结论迁移到另一任务、
   跨框架复用档位）→ 旧 claims 失效并显式关闭，重跑 smoke + formal 且质量门禁重过后方可再宣称
   （见 workflow「阶段推进规则」9）；有损质量对照锚 = 同 seed 冻结 baseline 帧，阈值不跨负载迁移。

## Reference Files

- `workflows/optimization-flow.md` — 加载时机: 入口分流后**必读**（阶段模板/方案确认点/验收
  gate/闭环强制交付/阶段反馈）
- `references/run-state.md` + `scripts/stage_gate.py` — 加载时机: 任务开始（建 run-state）与
  每阶段收尾（回写推进表 + 跑门禁）时
- `workflows/references/dispatch-templates.md` — 加载时机: 派发 subagent 或要求执行者提交自
  验证回执时
- 能力技能 SKILL.md（路由目标）：`env-install` / `remote-access` / `framework-integration` /
  `profiling-collect` / `profiling-analyze` / `performance-optimization` / `dit-parallel-opt` /
  `dummy-run` / `benchmark-dev` / `pattern-dev` / `operator-dev` —— 加载时机: 按「阶段路由表」
  命中对应阶段时加载对应技能
- `references/artifact-layout.md` — 加载时机: 产出/归档闭环产物（manifest / final_report /
  evidence.json）时
- `references/report-contract.md` — 加载时机: 写总览表前核对**报表列契约**（列定义 + lint 指向 +
  审计口径；自 `perf-gate/references/` 迁入，L1 单点）时
- `references/bottleneck-labels.md` — 加载时机: §0 启动确认后判定瓶颈点、把任务交给优化域入口、
  以及判定 S6 进入条件（非 DiT 段 10% 门限，须带步数档）时（标签枚举与门限口径的单一真源）
- `references/overview-report.md` + `references/detail-report.md` — 加载时机: 闭环复验 /
  产物归档时（强制输出总览 + 细分双报表）
- `references/manifest-schema.md` + `scripts/{seam_check.py, manifest_dryrun.py}` — 加载时机:
  生成/校验 manifest、运行前 dry-run 门禁、组合候选静态 seam 判定时（零 NPU/零数据）
- `references/search-orchestration.md` — 加载时机: 需要自动推进组合搜索或并发多 agent 编排时
  （默认人工节奏，护栏必读）
- `accuracy-gate/references/quality-gate.md` + 仓库 `evals/` — 加载时机: S4/闭环
  有损质量判定时（质量工具：`evals/scripts/quality_compare.py` 现算 quality.json、
  `evals/scripts/gen_profile.py` S0 冻结后生成 profile 到 runs/{task}/profiles/、
  `evals/scripts/check_profile.py` close 前强校验——**具体模型 profile 不入库**）
- `scripts/report_lint.py` — 加载时机: 闭环收尾（总览表 8 列/枚举/锚点禁估算机器校验，
  stage_gate close 自动联动；自测 `scripts/report_lint_cases.md`）
- `references/lossless-methodology-notes.md` — 加载时机: S1 融合候选工作流 / S3 内存受限并行
  解锁与量化后复查（§D 量化后融合重审 / §E 量化后通信重审）时
- `references/post-enable-review.md` — 加载时机: S4 每档特性落地后 / 任何改变 kernel 序列/
  数据口径/显存/精度域的特性使能复核时
- `references/effort-estimation.md` — 加载时机: §0 启动确认（时间盒 / 预算 / 候选排序）与每阶段
  计划时
- `dit-perf-opt/references/combination-search.md` — 加载时机: S4 需同时开启 ≥2 个
  有损维度时
- `framework-integration/references/framework-support-matrix.md` — 加载时机: S1/S4
  使能某特性前查框架支持面、使能失败排查、组合候选裁剪时
- `framework-integration/SKILL.md` — 加载时机: §0 判定「框架侧特性没落地」时（分支 A 使能 /
  分支 B 框架自身结构性缺口补齐，comm-stream / 缓存 / 稀疏量化消费者等；原 `framework-extension-dev`
  已并入本技能）
- `.agents/README.md` §2/§4/§7 — 加载时机: 槽位登记与经验/case 回填归属判定时

## 维护与更新

- 阶段/支撑技能/验收点变化 → 同步本表、`workflows/optimization-flow.md` 与 `.agents/README.md`
  对应章节。
- 阶段清单变化 → 同步 `scripts/stage_gate.py` 的 STAGE 常量与 `references/run-state.md`。
- 新经验回填槽位（README §2/§4）→ 视内容量决定是否在对应技能补充 reference。
- 实验复盘产出可复用 case → 按 `.agents/README.md` §7「case 回填规范」沉淀到对应能力技能
  references/，并同步本表引用。
- 本入口的 description 定位句与 README 一致，改动时同步。
