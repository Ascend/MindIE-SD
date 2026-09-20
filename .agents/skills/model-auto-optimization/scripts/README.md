# model-auto-optimization scripts

NPU 前结构性门禁脚本（零数据、零 NPU；产物数据不入 git）。

## seam_check.py —— 特性组合静态判定

特性声明（`feature_declarations.json`）**跨模型**：声明每个 mindiesd/框架特性作用的 seam /
显式互斥 / 依赖模型能力 / 激活窗口；模型只提供能力子集（decl 的 `models` 适配示例，完整适配记 case）。

```bash
# 无冲突组合（跨 seam）：通过
python seam_check.py --features quant_w8a8_dynamic,sparse_rf_v2,cache_dit \
    --model minimax-h3-vllm-omni
# 同 seam 双 writer（量化×量化）→ error；双缓存（DiTCache×AttentionCache）→ error
python seam_check.py --features cache_dit,cache_attention --model minimax-h3-vllm-omni
# 稀疏×缓存同 step_decision → warning（需错开窗口/只留最强档）
python seam_check.py --features sparse_rf_v2,cache_dit --model minimax-h3-vllm-omni
```

## manifest_dryrun.py —— manifest.toml 可复现门禁

```bash
python manifest_dryrun.py --manifest {plan}.toml
```

做：schema/枚举 → 特性 seam（调 seam_check）→ 路径只读检查 → 按框架渲染启动计划
（vllm-omni / lightx2v；其余框架报 unsupported 不猜测）。依赖 python ≥3.11 tomllib，
不依赖 NPU/权重/框架运行。

## stage_gate.py —— 编排层阶段推进门禁

```bash
# 证据按任务隔离（task_id = runs 目录名）；校验 evidence/{task_id}/ 前缀（防旧残留冒充）
python stage_gate.py --stage S3 --task-id 20260908_minimax-h3_optimization --run-dir {工作目录}/agentic
# close 自动联动：report_lint.py（主表写法）+ audit_report.py（表结构+数值自洽）
#              + evals/scripts/check_profile.py（profile 强校验）
python stage_gate.py --stage close --task-id 20260908_minimax-h3_optimization --run-dir {工作目录}/agentic
```

做：解析 run-state.md（`references/run-state.md`）「阶段推进表」→ 校验目标阶段 status=done 且
声明的验收证据路径存在；提供 `--task-id` 时校验证据落 `evidence/{task_id}/`（任务隔离，2026-09-08
起强制——防旧任务残留文件充当本轮证据）；close 额外强制声明 overview_report.md / detail_report.md
（缺任一视为未闭环）并自动跑主表 lint + 表结构/数值审计 + profile 校验（`_run_close_tools`）；
三者任一非 0 ⇒ close 不通过。
退出码 0=通过（可推进/可宣称闭环），1=存在 error（不得推进）。零 NPU、零数据。

**多 agent 契约校验（条件 · 与 `--stage` 无关，每阶段都跑）**：`--run-dir` 下存在 `agentic/dispatch/`
即进入**多 agent 模式**，除阶段校验外再校四件：

1. `handoff.md` 存在，且**最新一条**交接单八字段齐全（状态 / 最后完成动作 / 未决问题 / 下一步 /
   产物指针 / 口径指纹 / 资源占用 / 预算余量）；
2. 每个 `dispatch-*.md` 有对应 `receipt-*.md`，且回执四要素齐（结论 / 命令 / 退出码 / 证据）；
3. 回执里声明的**证据指针可达**——断链即报错（"断链的行不得进总表"的机械兜底）；
4. ≥2 派发单（多单元并行）须有 `lease-*.md` 资源租约（部署与资源分配角色签发）。

**无 `dispatch/` 则整组跳过并打印 `[skip]`**——默认单 agent 自执行必须保持绿（不为"没开多 agent"
报错）。判据单点：`../references/agent-roles-and-handoff.md` §3（交付件与指针链）/§4（交接单）/§5（并行控制）。
**校验边界（强制）**：只校**齐备性**（字段标签齐 + **回执**指针可达）；**不校**交接单自身的指针可达
（其「产物指针」允许是"下一步要交的"）与任何内容正确性——"已完成必须指针可达"只归回执。
`--selftest` 跑 8 个用例的自测（单 agent 不报错 + 缺交接单 / 缺字段 / 缺回执 / 回执缺要素 / 指针断链 /
多单元无租约逐条被触发）；pre-commit 钩子 `stage-gate-selftest` 在本文件变更时自动跑。

## report_lint.py —— 总览表结构 + 可读性校验（close 前置 · 机器校验）

```bash
python report_lint.py {overview_report.md}
python report_lint.py ../runs/20260908_minimax-h3_optimization/overview_report.md
python report_lint.py --selftest        # 负样本自测（内置夹具，零文件依赖）
```

校验 overview_report.md 主表：8 列表头（优化类型|特性名|e2e 耗时|首步耗时|步数|加速比|质量数据|
说明）精确匹配；每行优化类型枚举/特性名非空/e2e·首步·步数·加速比单值；**锚点行 e2e 禁
`[估算]`**（判定两路：① 结构——基线行，或特性名为 §2.2 三元固定名词「同时含 `Cache`+`量化`+`稀疏`」；
② 标记——整行文本含 `三元`/`最强组合`/`最终推荐`。**标记只可能落在说明列**：§2.1/§7 禁这些词进
特性名列，故只查特性名会漏判三元/最终推荐两类锚点）；质量列无损=输出一致、有损=数值非空。

**§2.8 可读性契约**——判据单点 = `references/overview-report.md`
§2.8，常量在脚本内（改判据须两处同改）：

| 级别 | 判据 |
|---|---|
| error | 主表内**开关/环境变量名**（`OMNI_H3_*` / `MINDIESD_*` 族前缀 + 全大写下划线 ≥3 段；前缀表 `SWITCH_FAMILY_PREFIXES`） |
| error | 主表内**裸内部代号**（`O2`/`C2`/`C3b`/`E4` 式单字母+数字；白名单只放 `H3`/`L1–L3`/`S1–S6`） |
| error / warn | 说明列可见文本 >350 字 / >180 字（`NOTE_CAP` / `NOTE_TARGET`） |
| error | `kernel融合` 行说明缺**位置标记**或缺**性能标记**（逐步 `ms/step`·`s/step`，非逐步 `s/请求`） |
| error | 缺「最佳路径下的特性详解」详情节或「附录」，或附录缺 开关对照表 / 内部代号对照表 / 证据指针，或详情节在附录之后 |
| warn | 附录之前的**正文**出现开关名（可追溯性引用放行，提示收敛到附录） |

**降假阳性（刻意的取舍）**：判据取"族前缀 / 显式 `NAME=value` 赋值形态 / 显式白名单"，且族前缀式
token 后紧跟文件扩展名时按**证据指针**放行 —— **报表/证据文件名**（`BASELINE_REBASE_20260920`、
`AMENDMENT_8CARD_ROOTCAUSE`、`CT_CONVTRANSPOSE3D_REPORT`）与**契约内标识符**
（`W' = W + (α/r)·B·A`、`fc1`、`RMSNorm`）一律不判。新增开关族时在 `SWITCH_FAMILY_PREFIXES` 登记。

自测样例 `report_lint_cases.md`（旧 6 项结构检项 + §2.8 的 14 个 `--selftest` 用例，含
"证据文件名不得误报""契约内标识符不得误报"两个回归）。本 lint 强制报表结构与
`references/overview-report.md` §2/§2.8/§7 契约一致，禁止凭人肉对照交付；
失效形态的根因与判据见 `references/overview-table-failure-postmortem.md`。

## audit_report.py —— 表格结构 + 数值自洽审计（改表后 / 复核他人报表）

```bash
python audit_report.py {overview_report.md}                 # 结构与数值一起跑
python audit_report.py {overview_report.md} --only structure # 只查结构（渲染类事故）
python audit_report.py {report.md} --baseline {baseline}      # 细分报表只给相对值时补分母
python audit_report.py {overview_report.md} --selftest      # 负样本自测：审计器本身是否有效
```

做：① **结构**——逐表核对"表头列数 = 分隔行列数 = 每个数据行列数"（按**未转义**竖线
`(?<!\\)\|` 切分）、"表头 → 数据行连续"（空行后的孤立表行报错）、加粗成对性（**跨行成对合法**，
按段落/表行分块配对）、**主表报告必带章节（S5）**——只对"含 `优化类型`+`特性名` 主表"的文件判：
须有「最佳路径下的特性详解」详情节与「附录」（内含 开关对照表 / 内部代号对照表 / 证据指针），
且详情节在附录之前（判据与 `report_lint.py` 同源同改；非报告类 md 不受此判）；
② **数值**——`分母 ÷ e2e ≈ 加速比`、`diffuse ÷ 每步` 为整数、
`实测 ÷ 连乘 ≈ 比值`、"相对上一行"的百分比/倍数与参照行（上一行 or 行 0）、同一测量跨两表的
逐字段一致（契约 §4）。

关键判据（都踩过）：**加速比的分母是"报告级基线"，不是"表内那条叫基线的参考行"**（参考行自身也是
相对报告基线、不是 1.00×）；加粗单元格 `**{数值}**（实测）` 必须剥标记后
判读，否则整片检查被静默跳过；"见 §6"这类文字单元格必须排除，否则会被抽出 `6` 造成假阳性。
每张表所用分母、未参与核算的行都**逐条打印**，跳过原因可见。

退出码：0 = 干净（error=0）；1 = 有 error（改表后不得交付）；2 = 前置条件缺失（无合法表格 /
无法定分母）。**close 阶段由 `stage_gate.py` 强制调用**（与 `report_lint.py` 同级）：非 0 ⇒ 不得宣称闭环。
零 NPU、零数据、只读、幂等（同样输入永远同样结论）。

## 维护

- 特性命名/声明以 `docs/zh/features` 与 `framework-support-matrix.md` 为准；docs 或框架升级时
  同步刷新 `feature_declarations.json`（同 support-matrix 刷新协议）。
- 新框架接入 dry-run → 在 `manifest_dryrun.py` 的 `RENDER_HINTS` 补渲染提示，并在
  `references/manifest-schema.md` 登记。
- **数据随脚本就近**：`feature_declarations.json`（数据）与 `README.md`（文档）放在 `scripts/`
  是因为 `seam_check.py` / `manifest_dryrun.py` 以同目录相对路径读取、脚本与数据同源同改；
  它不属于 `references/` 的命名词表（该词表只管案例/方法论/流程类 md）。
