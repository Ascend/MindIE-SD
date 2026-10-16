# .agents/scripts —— 知识层门禁

> 定位：两个**零依赖**（纯 stdlib、零网络、零模型、零密钥、只读、幂等）的静态门禁。
> **规则细节、豁免机制、跨工具纪律与已知盲区都记在本文件**，与脚本**同源同改**；
> `.agents/README.md` §7 只讲「是什么、怎么跑、违规怎么办」；
> 过程记录（历史实测理由、积压登记、会话报告）一律出库到会话产物归档，不入本文件。
> 布置理由（说明随脚本就近）同 `model-auto-optimization/scripts/README.md`。

## 1. `kb_lint.py` —— 知识层结构门禁

把「文档化纪律」变成「可执行门禁」。判据来源是**本仓既有规范**（`dev-workflow`「新增 Skill 规范」+
`.agents/README.md` §7），不是新造规则。

| 规则 | 级别 | 判什么 |
|------|------|--------|
| KB001 | error | 相对引用必须可解析（悬空引用/死链） |
| KB002 | error | `references/*.md` 必须在其所属 SKILL.md 登记（防孤儿） |
| KB003 | error | 技能目录名与 reference 文件名须 kebab-case |
| KB004 | error | SKILL.md frontmatter 必含 `name` / `description` / `compatibility` |
| KB005 | error | `evals/evals.json` 存在、可解析、≥2 条含 `prompt` 与 `expectations` |
| KB006 | error | >300 行的 reference 必须带目录 |
| KB007 | error | README 声明的技能计数须与磁盘目录一致 |
| KB009 | error | 仓内坐标（`docs/` `mindiesd/` `tests/` `benchmarks/` …）必须可解析 |
| KB010 | warn | reference 应含「维护与更新」章节 |
| KB011 | warn | SKILL.md 应含「维护与更新」章节 |
| KB012 | warn | 接线区登记须带「加载时机」语义 |
| KB013 | error | 仓内消费方指向 `.agents/**` 的引用必须可解析（反向边） |
| KB014 | warn | 携带经验结论的文件须含「复核方法/失效信号」 |

**判什么 / 不判什么**：把 `.agents/` 当**有向图**（节点 = SKILL.md / references / scripts / evals，
边 = 相对路径指针），校验图的不变量与接线契约。**不判结论对错**——「结论对不对 / 经验是否仍成立 /
质量是否达标」属语义层（需模型或人），脚本只让「没接线、指向不存在、规范缺失」fail。
与 `model-auto-optimization/scripts/stage_gate.py` 同构：**门禁校验证据与结构齐备性，不校验判断本身**。

**范围界定（各管一段，勿混读）**：

- KB001 = **KB 自身链接图**（`references/` `scripts/` `workflows/` `evals/` `assets/`、技能目录名首段
  与 `.agents/` 根级文件）；
- KB009 = **仓内坐标**（只认本仓真实顶层目录，含 `evals/`、`benchmarks/`）；
- KB013 = **反向边**，只覆盖产品侧根级入口（`README.md` / `README.en.md`）；
- **不判**：外部框架仓路径（`lightx2v_platform/`、`diffsynth_engine/`、`torch/` …）与运行期产物
  （`runs/`、`evidence/`）——不在本仓，纳入只会制造「合法但不在本仓」的假违规。

**豁免机制（既有约定的机械化，非新增规则）**：

- 迁移/更名/删除/移除记录里出现**旧名或缺失名是有意的** ⇒ 同一**连续非空块**出现
  「原名 / 已退役 / 已删除 / 删除 / git rm / 废弃 / 已移除 / 从未提交 / 未合入 / **不存在**」等标记即放行；
- prose 里已显式声明「外部库 / 非本仓 / 本仓不含 / 本仓无」的引用 ⇒ 放行；
- 豁免范围按**连续非空块**判（列表项常跨行：标记在首行、路径在次行）；**表格行例外**（只看本行）；
- 末段主干是占位符的 token（`references/x.md`）不判；
- 缺标记的历史引用仍会被判悬空 —— 这是**有意的**：判定方法比结论更重要（同 §7「结论有寿命」）。

**KB012 的时机语义**按三种来源认：① 表格**表头列名**表达时机（`加载时机`/`何时读`）时，
**该列单元格有值即算**（空单元格仍报）；② 列表项在**连续非空块**内认 `加载时机`/`何时读`/`时读`/`Phase N`；
③ 写出「单点登记于 §X / 此处不重复登记」的行是指针，不按登记判。**已知代价**：`时读` 这类宽写法
带来少量漏判（宁漏不误报——规则要先能被信任）。

```bash
python .agents/scripts/kb_lint.py                    # 全量（error=0 方可提交）
python .agents/scripts/kb_lint.py --list-rules       # 规则与级别
python .agents/scripts/kb_lint.py --rule KB001       # 只跑一条
python .agents/scripts/kb_lint.py --format grouped   # 按规则分组，便于看积压
python .agents/scripts/kb_lint.py --selftest         # 负样本自测：每条规则必须能被触发
```

退出码：0 = 无 error（warn 不阻断）；1 = 存在 error；2 = 前置条件缺失（`.agents/` 不存在）。
`--strict` 时 warn 也计为失败。

## 2. `run_evals.py` —— 技能 evals 裁定覆盖度门禁

补上「纪律可验证」的最后一环。23 个技能的 `evals/evals.json`（prompt + expectations）此前**没有执行器**：
CI 的 `check-json` 只保语法，KB005 只保契约。**确定性编排 + 覆盖度门禁，不调模型**（本仓 CI 无 LLM API）：

- `--pack`：把全部用例导出成**裁定任务包**（每条 expectation 一行，`verdict`/`evidence` 留空），
  由**会话中的 agent** 逐条填写；
- `--check`：校验**裁定覆盖度**——每条必有裁定、取值合法（`pass`/`fail`/`skip`）、`pass`/`fail` 必带
  evidence、`skip` 必写理由、且结果文件的 expectation 原文与当前 `evals.json` **逐字一致**
  （防结果过期或错位）。**这一步纯确定性，可进 CI**；
- `--pack` 产物与裁定过程属会话产物，**不入库**。

```bash
python .agents/scripts/run_evals.py --list                        # 覆盖概览
python .agents/scripts/run_evals.py --pack --out {run_results_dir}/skill_evals.json
python .agents/scripts/run_evals.py --check --results {run_results_dir}/skill_evals.json
python .agents/scripts/run_evals.py --selftest                    # 6 类缺口的负样本自测
```

与 `kb_lint` 同构：**门禁校验裁定齐备性，不校验判断本身的正确性**——「这条 expectation 判得对不对」
由模型/人负责，「有没有判、判得合不合格式」由本门禁负责。

## 3. 跨工具经验（可迁移的门禁纪律）

> 各条规则的**历史实测起点**（当时怎么发现它的）属过程记录，按 §7 判据不成立，已出库到会话产物归档
> `{run_results_dir}/archive/agents-scripts-readme-history.md`；下面只留换模型/换框架仍成立的纪律。

**三条跨工具经验**：

- **markdownlint 版本陷阱**：仓库 pin **v0.44.0**；用更高版本（如 0.49.x）会对同一批文件报**上千条**
  `MD060` 版本性假违规（实测同一批文件：v0.44.0 的 `MD060` = **0** 条 vs 0.49.1 = **1943** 条）。判读前先对齐版本。
- **门禁必须在最后一次编辑之后重跑**：实测教训——`.agents/**` 的 4 处 `MD032`（`**标题**：` 段末**直接接列表**、
  块引用内的列表项前缺一行空引用）是在“跑过门禁之后”才写进文件的，于是当时得出的「0 违规」覆盖的不是最终工作树。
  `kb_lint` 只判链接图与接线契约，**不覆盖 Markdown 格式**，两类门禁不可互相替代；发布前一律以最终工作树复跑。
- **evals 隔离条件**：`prompt` 与 `expectations` 同在 `evals.json`，读 prompt 即已看到期望 ⇒ 本轮 120
  用例中 **66 条自报 `contaminated`**。故 **454 pass 是「技能与测试自洽」的检查结果，不是独立作答质量
  的度量**；有信息量的是 fail 与冲突清单。要干净度量须把 `expected_output` 与 `prompt` 拆到不同文件。

## 4. 已知盲区（本门禁**不**覆盖，须人工留意）

- **KB002 的假阴性**：它以「文件名是否出现在 SKILL.md」判登记，**历史叙述里提到的文件名也算已登记**。
  实测：`operator-dev/SKILL.md` 记录 `mmgelu-flux-wan-qwen-case.md`「已并入 §7 并删除」，但该文件**仍在磁盘上**
  （且未按 `-case.md` 政策登记、数字未出库）——门禁不报。判据是「登记」还是「提及」，机械上无法区分。
- **小节锚点（`§N`）不校验**：实测 `framework-integration/references/vllm-omni-case.md` 曾写「按 §6 协议验证」
  而该文件原无 §6。
- **命令里的 CLI flag 不校验**：实测 `vae-opt/references/verification-and-budget.md` 曾给出 `--axis time`，
  而 `scripts/shard_equivalence_check.py` 无该参数。
- **裸文件名坐标不判**：KB009 只判带 `/` 的坐标（`CHANGELOG.md` 这类裸名不受覆盖）。
- **产品侧不再纳入 KB013**：`evals/`、`benchmarks/` 已改为**不反向引用** `.agents/` 路径，故无需门禁；
  依赖方向为 **skills → 产品侧**（`.agents` 引用 `evals/`、`benchmarks/` 是正向，受 KB009 覆盖）。

## 5. 质量纪律：为什么区分 error / warn

> 建立时的积压计数与逐条收敛过程属过程记录，已出库到 `{run_results_dir}/archive/agents-scripts-readme-history.md`
> （出库时四项积压均为 0）。

**为什么区分 error / warn**：门禁规则必须**先量积压再定级别**。存量积压一次性阻断会让门禁失去信任
并被绕过；error 只留给「当前为 0、可长期保持」的规则（建立时 error=0）。

## 6. 维护与更新

- **改规则** ⇒ 同步三处：本文件表格 + `kb_lint.py` docstring 的规则名/级别 + `.agents/README.md` §7 的当前口径；
- **改脚本** ⇒ 必跑 `--selftest`（`.pre-commit-config.yaml` 的 `kb-lint-selftest` / `run-evals-selftest`
  会在改脚本时自动跑，防「门禁静默失效」）；
- **调级别（warn→error）** ⇒ 前置是该规则积压为 **0**；
- **新增规则前先量积压**：若一次报出几十条，先判「内容真错」还是「规则过宽」——多数情况是后者
  （KB012 的先例），此时改规则而非改内容；
- 门禁契约/结构变化 ⇒ 同步 `.agents/README.md` §7（当前口径）与 `.agents/README.md` §6（目录树）。
