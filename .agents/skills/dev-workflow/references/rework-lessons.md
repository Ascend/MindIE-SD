# 返工教训（dev-workflow 侧：流程与纪律）

> **定位（本批拆分后）**：本文件只留**与具体领域无关的 dev 流程 / 纪律 / 工具使用教训**（§1–§5、§13 一类），
> 以及**指向 L3 真源的索引**。**领域知识已按 `.agents/README.md` §7 的落位四分类外迁**到各能力技能，
> 本文件对已外迁条目只保留**指针存根**（**编号不变**，`§N` 锚点仍可用，外部引用不会悬空）；
> **实测数字一律出库**（归档于会话产物目录 `{run_results_dir}/archive/`），外迁只搬**判据与流程**。
>
> **改前先看 §39「索引」**：它逐条给出归属（**留** / **已迁到 `路径`** / **已出库**）。
> 编号 **14–22** 的内容已删除、编号保留不复用。

以下问题均在 MindIE-SD 开发中实际发生并导致返工。

## 1. 拒绝未实现功能的前置配置

**问题**：功能未实现，但配置中提前添加了死配置字段。

**规则**：

- 配置字段必须与实现同步添加，禁止提前预留
- 每个配置字段必须有对应的已实现功能
- Review 时检查是否有未使用的新增配置

## 2. 最小必要改动原则

**问题**：为简单功能添加了大量不必要的框架改动。

**规则**：

- 实现功能前先确认：现有基础设施是否已经满足需求
- 每次改动前自问："不改这个，功能能运行吗？"
- 对于使用静态类接口的 pattern，不需要修改注册框架

## 3. 独立任务必须实际并行执行

**问题**：独立任务提前规划了并行，但实际执行时串行化了。多 NPU 卡的并行能力未被利用。

**规则**：

- 无代码依赖的独立任务直接启动并行闭环，不在同一线程串行排队
- 每个闭环独立：写测试 → 实现 → 部署 → 各自 pytest（不同卡）
- 共享文件的修改最后统一合并

## 4. 非代码仓内容不入库

**问题**：部署脚本、临时检查脚本误入代码仓目录。

**规则**：

- 部署脚本、临时检查脚本、一次性验证脚本不纳入代码仓
- 使用后立即删除或放在代码仓外独立目录
- 所有临时脚本统一放入 `tmp/` 目录，`.gitignore` 中追加 `tmp/` 屏蔽
- 合并临时脚本：避免散落多个独立脚本文件，合并为统一入口脚本

## 5. PLAN.md 未随任务变更同步更新

**问题**：任务被跳过或还原后，PLAN.md 中仍保留已废弃的条目。

**规则**：

- 每完成或跳过一个任务，立即更新 PLAN.md
- 删除 PLAN.md 中已废弃的任务条目和文件清单
- 任务粒度变化（如合并/拆分）同步刷新
- PLAN.md 内容必须与代码仓实际状态一致

## 6. triton vs triton-ascend 包名混淆（**留**：已是指针）

详见 [ascend-ops.md](ascend-ops.md)——含**复核方法**（`pip show triton-ascend` + `triton.runtime.driver.active`），
换 triton / CANN 版本后**先复核再套用**绕行。

## 7. `pip install -e .` 新增文件未被索引（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/SKILL.md`「编译原理与何时重装」+ 故障排查表
（`ModuleNotFoundError: mindiesd` → 重新 `python setup.py build_py && pip install -e .`）；
安装域决策树见 `../../env-install/references/troubleshooting-env.md` §A。

## 8. SSH 连接重复创建（**已迁 `remote-access`**）

**判据与处置单点**在 `../../remote-access/SKILL.md`「连接复用原则」+「文件传输」
（单连接复用、`;`/`&&` 串联减少 login shell、增量传输不逐个 `sftp.stat`；
远端 `MaxStartups` 限制与症状见同技能「故障排查」表）。

## 9. 嵌套 Shell 引号转义失败（**已迁 `remote-access`**）

**判据与处置单点**在 `../../remote-access/SKILL.md`「嵌套 shell 引号」+「上传同步纪律」
（复杂逻辑一律写 `.py` / `.sh` 上传后远端执行；必须内联时用 `shlex.quote` 或 base64 传递）。
本文件不复述反例代码。

## 10. Markdown 代码块未指定语言触发 MD040 门禁失败（**留**）

**规则**（格式细则与门禁配置单点在 `markdown-lint` skill，本文件只留流程）：

- 所有围栏代码块必须指定语言或内容类型（`text`/`bash`/`python`/`shell`/`yaml`/`json`/`markdown` 等）；
  目录树、终端输出、日志等非可执行内容用 `text`
- 提交前自检：`pre-commit run markdownlint --files {changed_file}.md`

````markdown
<!-- 正例：围栏后带语言标记 -->
```shell
npu-smi info -l
```

```text
examples/
├── a.py
└── b.py
```

<!-- 反例：裸围栏（无语言标记，触发 MD040） -->
```
examples/
├── a.py
└── b.py
```
````

## 11. `examples/dummy_run` 门禁违规（**留**：门禁纪律；规则表已迁 `code-standards`）

**问题**：文件先通过了阶段性检查（门禁仅报 MD040 / 仅跑 markdownlint），完整门禁扫描时才暴露
代码风格、异常处理、参数设计等多类违规。

**规则**（每条规则的**定义、对应钩子与复核方法**均在
`../../code-standards/references/gate-check-rules.md`，本文件不再罗列规则表）：

- `examples/` 目录与 `mindiesd/` 源目录受**同一套门禁规则**约束，不可放松标准
- 完整门禁扫描可能**分阶段执行**（先 markdownlint，后代码检查）——**首次通过不代表完全通过**
- 提交前应**全面**运行门禁检查，不依赖阶段性通过结果

## 12. 远端日志回传与本地终端编码（**已迁 `remote-access`**）

**判据与处置单点**在 `../../remote-access/SKILL.md`「长任务三段式」（`sys.stdout.reconfigure(encoding="utf-8",
errors="replace")` 后再打印远端日志；远端任务不受影响但轮询/后处理会中断）。
跨平台编码的通用注意事项另见本目录 `cross-platform.md`。

## 13. 通用分析脚本纳入 Skills（**留**）

**问题**：通用性强的脚本（参数化 IP/容器/密码，支持任意 output 格式）曾被当作一次性临时脚本。

**规则**：

- 通用分析 / 部署脚本归入技能的 `{skill}/scripts/` 子目录，不作为一次性临时脚本
- 代码仓内容仅限 `examples/dummy_run/` 示例本身，**不含 profiling 产出的数据与报告**
- 脚本参数化程度应支持不同环境复用

## 23. Pattern 单元测试通过但全模型不命中（**已迁 `pattern-dev`**）

**判据与处置单点**在 `../../pattern-dev/SKILL.md` Phase 4/5 出口纪律 +
`../../pattern-dev/references/mismatch-catalog.md`（类型 7：`placeholder` vs `get_attr`）+
`../../pattern-dev/references/test-templates.md`（单测 model 必须与全模型图结构一致）。

## 24. `nn.Module` 权重（get_attr 形态）的 pattern 表达（**已迁 `pattern-dev`**）

**判据与处置单点**在 `../../pattern-dev/SKILL.md` Phase 2 路径表 +
`../../pattern-dev/references/mismatch-catalog.md` 类型 7 +
`../../pattern-dev/references/graph-pattern-rewrite-guide.md`：
weight 来自模块参数时把它收进 `inputs()` / `pattern()` / `replacement()` 参数（freeze 前窗口命中），
**禁止**手写 FX graph traversal pass。

## 25. （已废弃）自定义 Graph Pass 的 freeze 时序坑（**已迁 `pattern-dev`**，仅留档）

> ⛔ 自定义 FX graph traversal pass 已在 `../../pattern-dev/SKILL.md` Phase 2 **明令禁止并删除**
> （`custom-graph-pass-guide.md` 已退役，`git rm`）——本条只是"**为什么禁止**"的记录，**不得再按此实现**。
> 历史根因（`freeze()` 的 `node_copy` 遇未注册 NPU 自定义 op 报 `KeyError`）见上述 SKILL 的 ⛔ 段。

## 26. 编译开销定位方法（**已迁 `pattern-dev` / `profiling-analyze`**）

**判据与处置单点**在 `../../pattern-dev/SKILL.md` Phase 6「Kernel 级 diff 验证」+
`../../pattern-dev/scripts/compare_profiles.py`（kernel 对比**唯一入口**）+
`../../pattern-dev/references/copy-elimination-guide.md`（Copy 类专有读法：同名 Copy 本体耗时差与
新增 Copy 的前后相邻 kernel）。归因顺序（先排除重编译，再归因 kernel）见 §34。

## 27. 打包排除 `build` 目录误删源码脚本（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/SKILL.md`「部署脚本」节（排除规则只排除编译产物，
**不排除源码 `build/` 本身**）；症状（`build/*.sh` 与第三方 patch 目录一并消失）见同技能故障排查表。

## 28. 权重分片缺失未对照 `index.json` 预检（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/references/weights-prep.md` §5 +
`../../env-install/references/troubleshooting-env.md` §H（**不能只数分片个数**，必须对照
`*.safetensors.index.json` 的 `weight_map` 逐分片核对；缺失分片补下载前先确认远端是否已存在）。

## 29. vllm-omni 源码包缺 `.git` 导致版本非法（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/references/vllm-omni-build.md` Step 2.5
（`setuptools_scm` 无 `.git` 时返回 `dev`，拼 `+npu` 得非法版本 → 设 `VLLM_OMNI_VERSION_OVERRIDE`）。

## 30. pip 依赖解析降级 torch 后未复原（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/references/vllm-omni-build.md`「版本配套矩阵」+
Step 2.5（后装组件用 `--no-deps`，或装完**立即复核全栈版本**；不以后装组件的旧 pin 为准）。

## 31. 容器缺 HCCL ranktable 导致多卡失败（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/SKILL.md` 容器启动节 +
`../../env-install/references/vllm-omni-build.md` Step 2.1 +
`../../env-install/references/troubleshooting-env.md` §A/§F
（必须挂载 `/usr/local/Ascend/driver/topo`；`docker cp` 会在容器重启后丢失）。

## 32. 第三方 wheel 文件名重命名破坏 pip 解析（**已迁 `env-install`**）

**判据与处置单点**在 `../../env-install/references/troubleshooting-env.md` §H
（保留原始 wheel 文件名，勿为下载方便重命名——pip 要求
`{name}-{version}-{build}-{py}-{abi}-{platform}.whl`）。

## 33. 量化层 forward 内就地修改模块状态 → compile 每次重编译（**已迁 `pattern-dev`**）

**判据与处置单点**在 `../../pattern-dev/references/pattern-dev-notes.md` §4
（含 `TORCH_LOGS=recompiles` 定位姿势与 guard 失败文本）+
`../../profiling-analyze/SKILL.md`「快捷判别：先排除 torch.compile 重编译，再归因 kernel」。
规则本体：算子层/模块 `forward` **禁止就地修改模块属性**；dtype 转换用局部变量。

## 34. 先诊断再下结论：性能劣化勿直接归因 kernel（**已迁 `profiling-analyze`**）

**判据与处置单点**在 `../../profiling-analyze/SKILL.md`「快捷判别：先排除 torch.compile 重编译，
再归因 kernel」（`wall_ms / kernel_sum_ms >> 10` + Wait Time 高 + 单个超大设备空闲间隙 → 先查 host 侧）。
结论必须基于 profiling 数据（kernel 时间占比、间隙位置、recompile 日志），不凭直觉。

## 35. 远程多卡实验工具纪律（**已拆迁四处**）

原条目的四项纪律各有单点真源，本文件不再复述：

- 卡组 / 端口**硬编码在本地脚本**（远程执行器会重传覆盖远端 `sed` 修改）→
  `../../framework-integration/references/lightx2v-enablement.md` §2.2；
- `pkill` 模式匹配自杀、SSH 批量命令不堆 shell 嵌套、后台长任务先落盘再轮询 →
  `../../remote-access/SKILL.md`「长任务三段式」+「上传同步纪律」+「故障排查」表；
- 多卡环境劣化**优先换卡组/换端口段再怀疑代码**、避免 SIGKILL 运行中的多卡任务 →
  `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4；
- 结论以**同窗口同卡组 + 多次复现**为准、先 4 步 smoke 再 30 步墙钟 →
  `../../perf-gate/references/window-ab-protocol.md` +
  `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §5。

## 36. npu-smi Health=OK ≠ 卡组功能可用（**已迁 `dit-parallel-opt`**）

**判据与处置单点**在 `../../dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4
（组可用性必须用**真实多卡 run** 核对 per-step cost；每步**均匀**量级级抬升且**无热降频斜率** =
SIGKILL 驱动损伤残留态，跨天不自动恢复，需驱动级复位 / 换健康组）。
测量侧的取数口径（clean-window / 热降频判定）见 `../../perf-gate/references/measurement-discipline.md` §7。

## 37. 质量门禁口径混淆（**已迁 `accuracy-gate`**，纠错纪律留档）

**判据与处置单点**在 `../../accuracy-gate/references/quality-gate.md`
（特性质量 = 与 lossless 基线**同 seed / 同 prompt / 同步数 / 同分辨率、仅该特性单变量**；
跨配置、跨 seed 的 SSIM **不可比**；换分辨率 / 步数 = 新对照；绝对口径 = vs 同构 lossless）。

**留档的纠错纪律**（流程，不属于口径本身）：

- 遇到「质量随强度**异常平坦**」先复核口径（基线配置 / seed / 帧对齐），**再**怀疑算子或几何；
  对算子侧的怀疑必须附「**口径一致的对照实验**」证据
- 纠错后旧结论要**显式作废并注明原因**，避免新会话沿用错误结论

## 38. 性能数字有效性：失败运行会给出更漂亮的耗时（**留**：已是指针）

**判据与上报模板单点**在 `../../perf-gate/references/evidence-toolbox.md` §1（三重证据：
返回码 + 产物字节数 + 成功日志条数，**缺一项该数字即作废**）与
`../../perf-gate/references/measurement-discipline.md` §6（上报模板）。
本文件只留一条流程纪律：**失败运行的数字应单独列出并标注作废**，不得用「大概是网络抖动 / 机器忙」
之类的理由放过缺证据的数字。

## 39. 索引（每条教训现在去哪找）

| 编号 | 主题 | 处理 | 归属（真源） |
|---|---|---|---|
| 1 | 拒绝未实现功能的前置配置 | **留** | 本文件 §1 |
| 2 | 最小必要改动原则 | **留** | 本文件 §2 |
| 3 | 独立任务必须实际并行执行 | **留** | 本文件 §3 |
| 4 | 非代码仓内容不入库 | **留** | 本文件 §4 |
| 5 | PLAN.md 未随任务变更同步更新 | **留** | 本文件 §5 |
| 6 | triton vs triton-ascend 包名混淆 | **留**（已是指针） | `ascend-ops.md`（本目录） |
| 7 | `pip install -e .` 新增文件未被索引 | 已迁 | `env-install/SKILL.md`「编译原理与何时重装」+ `troubleshooting-env.md` §A |
| 8 | SSH 连接重复创建 | 已迁 | `remote-access/SKILL.md`「连接复用原则」/「文件传输」 |
| 9 | 嵌套 Shell 引号转义失败 | 已迁 | `remote-access/SKILL.md`「嵌套 shell 引号」/「上传同步纪律」 |
| 10 | Markdown 代码块未指定语言（MD040） | **留** | 本文件 §10；格式细则在 `markdown-lint` |
| 11 | `examples/dummy_run` 门禁违规 | **留**（规则表已迁） | 本文件 §11；规则定义与复核在 `code-standards/references/gate-check-rules.md` |
| 12 | 远端日志回传与本地终端编码 | 已迁 | `remote-access/SKILL.md`「长任务三段式」；跨平台注意事项见本目录 `cross-platform.md` |
| 13 | 通用分析脚本纳入 Skills | **留** | 本文件 §13 |
| 14–22 | （内容已删除，编号不复用） | — | — |
| 23 | Pattern 单测通过但全模型不命中 | 已迁 | `pattern-dev/SKILL.md` Phase 4/5 + `../../pattern-dev/references/mismatch-catalog.md` 类型 7 + `../../pattern-dev/references/test-templates.md` |
| 24 | `nn.Module` 权重（get_attr）的 pattern 表达 | 已迁 | `pattern-dev/SKILL.md` Phase 2 路径表 + `../../pattern-dev/references/mismatch-catalog.md` 类型 7 + `../../pattern-dev/references/graph-pattern-rewrite-guide.md` |
| 25 | （已废弃）自定义 Graph Pass 的 freeze 时序坑 | 已迁 | `pattern-dev/SKILL.md` Phase 2 ⛔ 段（禁用手写 FX graph traversal） |
| 26 | 编译开销定位方法 | 已迁 | `pattern-dev/SKILL.md` Phase 6 + `../../pattern-dev/scripts/compare_profiles.py` + `../../pattern-dev/references/copy-elimination-guide.md` |
| 27 | 打包排除 `build` 目录误删源码脚本 | 已迁 | `env-install/SKILL.md`「部署脚本」节 |
| 28 | 权重分片缺失未对照 `index.json` 预检 | 已迁 | `env-install/references/weights-prep.md` §5 + `troubleshooting-env.md` §H |
| 29 | vllm-omni 源码包缺 `.git` 导致版本非法 | 已迁 | `env-install/references/vllm-omni-build.md` Step 2.5 |
| 30 | pip 依赖解析降级 torch 后未复原 | 已迁 | `env-install/references/vllm-omni-build.md`「版本配套矩阵」+ Step 2.5 |
| 31 | 容器缺 HCCL ranktable 导致多卡失败 | 已迁 | `env-install/SKILL.md` 容器启动节 + `vllm-omni-build.md` Step 2.1 + `troubleshooting-env.md` §A/§F |
| 32 | 第三方 wheel 文件名重命名破坏 pip 解析 | 已迁（本批新建条目） | `env-install/references/troubleshooting-env.md` §H |
| 33 | 量化层 forward 就地修改模块状态 → 重编译 | 已迁 | `pattern-dev/references/pattern-dev-notes.md` §4 + `profiling-analyze/SKILL.md`「快捷判别」 |
| 34 | 先诊断再下结论：劣化勿直接归因 kernel | 已迁 | `profiling-analyze/SKILL.md`「快捷判别」 |
| 35 | 远程多卡实验工具纪律 | 已拆迁四处 | `framework-integration/references/lightx2v-enablement.md` §2.2 + `remote-access/SKILL.md` + `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4/§5 + `perf-gate/references/window-ab-protocol.md` |
| 36 | `npu-smi` Health=OK ≠ 卡组功能可用 | 已迁 | `dit-parallel-opt/references/ascend-topology-bandwidth-diag.md` §4（口径见 `perf-gate/references/measurement-discipline.md` §7） |
| 37 | 质量门禁口径混淆 | 已迁（纠错纪律留档） | `accuracy-gate/references/quality-gate.md` |
| 38 | 性能数字有效性：失败运行更漂亮 | **留**（已是指针） | `perf-gate/references/evidence-toolbox.md` §1 + `measurement-discipline.md` §6 |

## 维护与更新

新的返工教训按 dev-workflow 的复盘流程补充；**加条目时先判归属**——
与具体领域无关的流程/纪律留本文件，领域知识按 `.agents/README.md` §7 落位四分类直接写进对应技能的
真源并从本文件 §39 索引登记（**本文件不再复制领域正文**）。
