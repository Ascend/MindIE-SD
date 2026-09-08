# 运行状态文件（run-state.md）规范

> 编排层流程推进的单一真相源：回答「谁读 / 谁写 / 信息存哪」，并支撑跨上下文压缩后的状态重建。
> 配套机械门禁：`scripts/stage_gate.py`（按推进表校验阶段验收证据）。

## 定位与分工

| 文件 | 管什么 | 写者 | 读者 |
|------|--------|------|------|
| run-state.md（本规范） | **进行中**流程状态：任务与口径 / 阶段推进表 / 决策与轮次 | 推进表只有编排者（主 agent）写；工作区由执行者先读后追加 | 编排者裁决；执行者进场先读；stage_gate.py 解析推进表 |
| manifest（见 manifest-schema.md） | 任务 scope / 口径声明（启动即定、一次成型） | 编排者 | 运行计划 / dry-run 门禁 |
| final_report + evidence.json（见 artifact-layout.md） | **终态**归档（人读报告 + 机器可读证据） | 编排者（闭环收尾） | 归档 / 交接 / 后续引用 |

不互相复制：看当前进度查 run-state 推进表；看任务口径查 manifest；看终态结论查 final_report。

## 文件位置与布局

默认 `{工作目录}/agentic/run-state.md`；不存在时先按下方模板创建再使用：

```text
{工作目录}/agentic/
├── run-state.md          # 本文件：阶段推进单一真相源
└── evidence/{stage}/     # 验收证据（kernel diff / 墙钟日志 / 复核记录等，一阶段一目录）
    └── {feature}/        # 并行执行时按特性/实现 id 再隔离（一子 agent 一目录，见
                          #   workflows/references/dispatch-templates.md「单点特性并行执行」）
```

## 特性覆盖清单（任务级触发判定 · 前置交付件）

保证"每个候选特性都被显式过一遍触发判定"，防漏想一个特性方向（借鉴 Sana 的
method-baseline catalog / search_space 状态与 cannbot 探索 dashboard 的候选登记）：

- **何时建**：SKILL §0 启动确认时生成（与用户确认目标/验收/缺口补齐策略同一步），任务分析
  过程中可更新（"分析后做 → 做"或"分析后做 → 不做"须记理由与 round）。
- **判定状态**：`做（目标档/经验档）` / `分析后做` / `不做（理由：无对应瓶颈 / 预期收益小
  （整 block <0.5%）/ 框架不支持（引 support-matrix））`；每行附 round/阶段与证据指针。
- **框架档位列**：每特性/能力标 framework×模型组合下的档位——`已支持`（直接用）/
  `待配置`（能力实体已具备、差接线，如"量化有支持但缺 mxfp8 型 w8a8"）/ `待开发`（无支持需
  完整实现）/ `上界`（预留，定义待人工填写）——图例与判例见
  `framework-feature-enablement/references/framework-support-matrix.md` §〇。
- **执行排序**：`已支持 / 待配置` 先做（零开发或接线即可）；`待开发` 先估成本并经 §0 补齐策略
  用户确认再投入；`上界` 只作天花板诊断不宣称（预留空间，语义后续填写）。
- **范围**：固定特性全集（`kernel融合` / `并行` / `Cache` / `量化({修饰符})` / `稀疏` /
  `时间步优化` / 训练感知组可选）+ 任务特定候选（framework-support-matrix 预扫缺口、组合候选）；
  量化按能力组合（w8a8/w4a4/f8，类型 mxFP8/mxFP4）、稀疏按 mindiesd 稀疏算子（rf_v2/ada_bsa…）
  逐能力判档。
- **收口**：闭环前复核清单**无"未裁决"项**——每特性都有 做/不做 结论且理由可查；未裁决视为
  交付缺失（优化-flow 闭环节引用本清单）。

```markdown
## 特性覆盖清单
| 特性/能力 | 框架档位 | 触发判定 | 理由/证据 | round/阶段 |
|-----------|----------|----------|-----------|------------|
| kernel融合(API 接入) | 已支持 | 做 | rope/rms 注册表替换（S1） | S1 |
| 量化(w8a8·mxfp8) | 待配置 | 做 | 能力在 mindiesd 已具备，框架差接线（case §11 P1） | S4 |
| Cache | 待开发 | 分析后做 | 框架无消费且无现成接线，需开发（成本待估） | S4 |
| 稀疏(rf_v2) | 已支持(vLLM)/待配置(L1) | 分析后做 | 图像无 2D 路径待核 | S4 |
| {候选特性} | 上界（预留） | — | 天花板定义待人工填写 | — |
```

## 模板

```markdown
# Run State — {model} × {framework}（{任务类型}）

## 任务与口径
- 目标/验收: <无损/有损档位、目标加速、质量门与时间盒>
- 基线口径: <同拓扑同卡组 baseline / manifest 指针>
- 缺口补齐策略: <开关使能 / framework-extension-dev / fork-monkey / 绕过 / 仅记录>（§0 确认结论）

## 阶段推进表
<!-- stage_gate.py 解析：阶段 | 状态(done/in_progress/blocked) | 验收证据路径(逗号分隔,相对本文件目录) | 备注 -->
| 阶段 | 状态 | 验收证据 | 备注 |
|------|------|----------|------|
| S0 | in_progress |  | 环境准备中 |

## 当前阶段工作区（执行者追加，写入前先读）
### {YYYY-MM-DD} {阶段} — {角色}
- [完成/失败/发现] <描述 — 文件/路径>

## 决策与轮次
- {阶段} 第 {n} 轮 FAIL 原因 / 回退 / 用户决策: {记录}
```

示例（推进表填好后长这样；仅示意，实际按任务删减）：

```text
| 阶段 | 状态 | 验收证据 | 备注 |
|------|------|----------|------|
| S0 | done | evidence/S0/import_check.log, evidence/S0/weights_check.log, evidence/S0/dummy_run.log | 基线跑通 |
| S1 | done | evidence/S1/kernel_diff.csv, evidence/S1/fusion_hit.txt | kernel 融合（含接入）收敛 |
| S3 | in_progress |  | 并行采集中 |
```

## 写者规则与单一真相源

- **推进表只有编排者（主 agent）写**：subagent / 执行者不直接改推进表；自己的产出先落
  evidence/ 或工作区，由编排者核验后镜像进表。
- **并行执行时**：多个单点特性子 agent 各自写入 `evidence/{stage}/{feature}/`（互不覆盖），
  迭代表/推进表仍编排者单写；共享文件与卡组互斥调度（护栏见
  `workflows/references/dispatch-templates.md`「单点特性独立子 agent 与并行执行」）。
- 工作区与 evidence/ 由执行者**先读后追加**：只追加不清空，不覆盖他人记录。
- **即时回写**：阶段结束、裁决、回退、用户确认等关键节点立即落盘，不延后到收尾——任意时刻
  可从 run-state + evidence 重建完整流程状态（上下文压缩 / 断点接力依据）。

## 推进门禁

- 每阶段收尾：编排者更新推进表（status=done + 验收证据路径）→ 跑
  `python scripts/stage_gate.py --stage {Sn} --run-dir {工作目录}/agentic`
  → error=0 才进入下一阶段或宣称闭环；未过不得推进、不得宣称完成。
- 验收证据路径相对 run-state.md 所在目录解析；`<…>` 占位路径跳过存在性校验（仅提示）。
- 闭环（close）行声明路径必须包含 `overview_report.md` 与 `detail_report.md`（强制交付双报表）。

### 跨目录证据与 manifest 拆分（实测用法，env A 2026-09-07）

- **报表/产物目录与 agentic/ 分离是常态**：run-state + evidence 放 `{工作目录}/agentic/`，
  人读产物（overview/detail/final 报表、mp4、manifest 副本）放
  `runs/{date}_{model}_optimization/`（artifact-layout.md）。close 行证据路径写相对 agentic/
  的 `../runs/{date}_{model}_optimization/overview_report.md, ../runs/{date}_{model}_optimization/detail_report.md`
  ——stage_gate 按 basename 校验双报表存在性，跨目录相对路径合法（`target.name` 匹配
  CLOSE_REQUIRED_BASENAMES），无需把报表复制进 agentic/。
- **manifest 按"运行档"拆分，一档一个文件**：基线档 manifest（`features.enable=[]` +
  `baseline_run=true`）与每个特性档 manifest 分开落在产物目录（`runs/.../manifest.baseline.toml`、
  `runs/.../manifest.{feature}.toml`），而不是把所有档塞进一个 manifest 的 enable 列表——
  干跑门禁（manifest_dryrun.py）按档校验 seam 才准确；报表/evidence.json 的 source_binding
  引用具体档 manifest 指针，支持逐档复验。推进表验收证据可只指产物目录内的报表与关键档 manifest。

## 候选与迭代表（探索治理 · JOURNAL 式）

阶段推进表管「阶段是否完成」；本节管**探索过程本身**（S1 融合候选、S3 并行矩阵、S4 组合试验、
S5 训练感知候选等
"候选集 + 尝试 + 评估"的内部迭代），借鉴 bounded search loop：假说先行、一次一候选、拒绝带
签名、预算到点收尾——防无迹可循的探索与无限调优。

**迭代表格式（放在 run-state.md「候选与迭代表」标题下；写者 = 执行者追加，裁决 = 编排者）**：

```markdown
## 候选与迭代表
<!-- 每候选/每档/每次尝试一行；round 全局递增不复用；特性/实现 id = declarations id 或 docs/矩阵特性名 -->
| round | 特性/实现 id | 假说(一行,含预期) | 步数 | 状态 | gate 证据(含计数契约路径) | 拒绝签名/备注 |
|-------|--------------|-------------------|------|------|---------------------------|----------------|
| r1 | kernel融合·mm_swiglu_mxquant(API 接入) | 后端 FFN 三合一，命中后 FFN 执行序收益预期 >0.5% | 1（识别步数） | retain | evidence/S1/r1_kernel_diff.csv | 最终叠加全量核验 |
| r2 | 并行·TP1×USP2 | 图像 20 步短任务比 TP2 快（通信掩盖充分） | 20 | reject | evidence/S3/r2.log | dominated：r1 形态更优 |
```

- **状态枚举**：`retain`（进 frontier）/ `discard`（不采信但无签名价值）/ `reject`（必须有签名）/
  `terminal_pending_review`（预算到点或目标达成，frontier 交用户选档）。
- **步数列（实测步数，必填）**：性能识别行可少步（如 `1（识别步数）`，同口径少步对比，见
  optimization-flow 标准回路「步数口径」）；质量/采纳/最终叠加行 = 全量步数；**最终叠加必须
  全量实测**（少步只筛方向不下结论），并登记到总览报表「步数」列（overview-report §1.1/§2 列 4）。
- **拒绝签名（reject 必填其一）**：crash / implementation-wrong（实现错或未真实参与）/
  degenerate（输出退化）/ dominated（被同组更优候选替代，写被谁替代）/ out-of-scope（越界或
  改变固定口径）/ no-gain（命中但无收益——融合类按「整 block 耗时影响 <0.5%」收益小阈值判定，
  见 optimization-flow S1；其余按噪声/质量口径说明）。
- **特性/实现 id 列**：候选所属特性用固定特性名（`kernel融合`/`并行`/`Cache`/`量化`/`稀疏`/
  `时间步优化` 等，见 overview-report §2.1）+ 实现 id（declarations id 如 `quant_w8a8_mxfp8`、
  `cache_dit`，或融合内容名如 `mm_swiglu_mxquant` + 方法 compile/API 接入）；S4 与融合类候选必填，
  供与计数契约证据对账（防 no-op）。
- **structured negative = 候选素材**：单候选失败不结束迭代——记签名后提出下一假说继续；
  只有当同一阻断条件持续且无新假说时才判 blocked 收尾。
- **预算记录**：迭代表下方一行 `预算: max_rounds={N}, 已用={M}`；预算耗尽 → frontier 收尾，
  未尝试候选保留为清单（进 detail-report §C），不静默丢弃。
- **[MUST] 必测子区（S4-2 组合覆盖清单）**：S4 组合阶段若 `量化`/`稀疏`/`Cache` ≥2 个维度有
  单点 frontier，迭代表须含 [MUST] 行（两两全测 + 三元 `Cache+量化+稀疏`），行首标 `[MUST]`，
  先于自由候选执行；[MUST] 行状态只允许 已测（retain/reject + gate 证据）或 豁免（原因 +
  证据指针），**禁止「未裁决」进入 close**；确未测者必须转豁免并写明阻断（模板与豁免语义见
  performance-optimization/references/combination-search.md「必测组合覆盖集」）——预算记录同时
  注明 `必测集: 已裁决=M/N`，未测必测行不得当作"预算耗尽未尝试"收尾。
- 每轮裁决即时回写（与推进表同纪律）；进入下一阶段前，本阶段迭代表已含每候选的
  gate 证据与裁决。

## 维护与更新

- 阶段清单变化 → 同步 `scripts/stage_gate.py` 的 STAGE 常量与
  `workflows/optimization-flow.md` 阶段模板。
- 迭代表字段/状态枚举变化 → 同步 `workflows/optimization-flow.md`「探索与候选纪律」与
  `references/detail-report.md` §C。
- 本文件是规范模板；实际任务运行状态文件与 evidence 不入 git（产物目录规则见
  artifact-layout.md，远端产物见该文件「仓库零数据」约束）。
