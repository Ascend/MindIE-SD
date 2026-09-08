---
name: operator-dev
compatibility: 外部技能库 cannbot-skills（线上 https://gitcode.com/cann/cannbot-skills，本地可选 clone）；triton / torch_npu / CANN 环境
description: 算子级开发与性能优化：Triton / Ascend C / Catlass / PyPTO / TileLang 算子
             编写、精度对齐与性能调优。优先路由到外部 cannbot-skills 技能库（不重复其内容），
             本 skill 只保留场景 → skill 映射与 MindIE-SD 特有补充。
             当用户需要新增/优化算子、定位算子性能或精度问题时使用此 skill。
             即使用户只提到"写个 triton kernel"或"这个算子怎么加速"而未说算子，也应触发；
             pattern 融合/编译后端见 compilation-dev，算子基准选型/接入测试见 benchmark-dev。
             由 dev-workflow 或 compilation-dev 的 replacement kernel 场景触发。
---

# 算子开发（复用 cannbot-skills）

## 定位

算子级开发/优化任务按场景路由到**外部技能库 `cannbot-skills`**（完整链路：
triton-task-extractor → triton-op-designer → triton-op-coding → triton-latency-optimizer →
triton-simulator-optimizer / triton-precision-debug / triton-op-verifier，以及 Ascend C /
Catlass / PyPTO / TileLang 各 DSL 链）。

**硬约束**：算子开发**必须加载 cannbot 对应技能**，且**与本仓库特有经验并行使用**——两套
经验**不是互斥/替代关系**（cannbot = 通用算子方法论；本仓库 = MindIE-SD/CANN 侧约束与事实，
话题可重叠但不冲突，叠加生效）；融合 DSL 按计算单元分界——**CV（含 matmul）用 catlass、
VV（纯 vector elementwise）推荐 triton**（细则见「使用约束」）。

**本 skill 不复制外部库内容**，只提供：

1. 场景 → skill 映射（见 `references/operator-optimization-skill-map.md`）
2. MindIE-SD 特有补充经验（下表；与 cannbot 经验并行加载使用，不互斥）

## 场景路由

→ `references/operator-optimization-skill-map.md`（完整路由总表）

## 本仓库补充经验（MindIE-SD/CANN 特有；与 cannbot 并行使用，不互斥）

| 场景 | 入口 |
| --- | --- |
| **register_replacement pattern 命中≠收益**：kernel diff → 逐 pass AB → R1-R5 根因目录（含"负收益先查 kernel 形态"教训） | `compilation-dev/references/benefit-rootcause-guide.md` |
| **kernel diff 方法论**：kernel_details.csv 聚合对比、L2-flush bench 必须放计时区外、warm/cold 双档测量 | `compilation-dev/references/benefit-rootcause-guide.md` §2 + `dev-workflow/references/rework-lessons.md` |
| **模型级验证闭环**（不只看单测）：compute-precision 图验证、叠加 AB、远端 NPU 部署流程 | `dev-workflow/references/rework-lessons.md`、`dummy-run/` |
| MiniMax-H3 算子上下文（npu_swiglu 语义、表 [3,D] L2 驻留、真实图形态） | `dummy-run/references/minimax-h3-notes.md` §10 |
| **MindIE-SD/CANN 集成侧经验**：kernel 改动"没生效"排障（tiling-key .o 缓存/全清重建/sentinel 法）、CANN 同名内建算子冲突与改名陷阱、AscendC bf16 Muls/Gather 语义坑、triton 短行地板判定、w8a8 与融合 pattern 冲突、**融合收益前置评估** | `references/mindiesd-fusion-notes.md` |
| **外部/三方 AscendC kernel 接入 mindiesd 内部**（catlass 类）：形态选型（单 .so ASC 混编为终态）、CMake/ASC 链接与静态运行时链接坑、torch custom op C++ 形态（tuple 返回/PrivateUse1/stream）、设备/运行时事实、集成侧数值验证（位级仅 h3 特例，一般融合为 fp8 量化级；接入 compile 图的约束与 compile 前后收益核验归 compilation-dev） | `references/catlass-kernel-integration.md` |
| **只读 catlass 融合算子全链开发**（量化 matmul+激活+输出量化，vendored 头、standalone 对拍计时、mindiesd 集成、compile GraphPatternEntry 真图使能、开关治理）：六段流水线与决策，案例 mm_swiglu_mxquant/mm_gelu_mxquant | `references/catlass-ffn-fusion-guide.md` |
| mm_gelu_mxquant（FLUX/Wan/Qwen）案例细节：真实图链/bias=0/装载 API 坑/计数与 AB/工程坑 | `references/mmgelu-flux-wan-qwen-case.md` |

## 使用约束

- **必须加载 cannbot，且与本仓库经验并行使用（非互斥）**：算子开发/优化开始前，先按
  `references/operator-optimization-skill-map.md` 路由到对应 cannbot skill **并加载其内容**，
  **同时**应用下表与 references 的本仓库特有经验——cannbot 给通用算子方法论（链路/调优/
  精度），本仓库给 MindIE-SD/CANN 约束与事实（融合 DSL 分界、集成机制、真图使能口径、
  开关治理、坑位）；两套经验话题可重叠但**不冲突，不是二选一**。未安装 cannbot 先执行
  `scripts/install_cannbot.sh`（Linux/远端）或 `scripts/install_cannbot.ps1`（Windows 本机）。
- **可用性处理（≠替代/降级关系）**：cannbot 不可用（未 clone / 无网络）时跳过其加载，并把
  场景记入 dev-workflow §6 复盘（触发可用性确认）——这只是暂时跳过外部方法论，本仓库经验
  仍照常并行生效，不代表它顶替 cannbot 的角色；恢复可用后按首条恢复并行加载。
- **融合 DSL 分界（按计算单元，不按模型域）**：
  - **CV 融合（含 matmul：mm + 激活/量化 epilogue，cube+vector）→ 用 catlass**。理由：
    matmul 是融合主体时才有 cube 收益，catlass 提供 GEMM 骨架 + epilogue/量化机制，vendored
    复用流水线见 `references/catlass-ffn-fusion-guide.md`（案例 `mm_swiglu_mxquant` /
    `mm_gelu_mxquant`）。
  - **VV 融合（无 matmul 的纯 vector elementwise：swiglu/gate/激活+量化等）→ 推荐 triton**。
    理由：无 cube 参与，catlass 无收益且重；triton 更轻，是 fusion pattern replacement
    kernel 的默认 DSL（AdaIN/SwiGLU/gate 案例，链路见 skill-map §1.1）。
  - 例外（目标 DSL 无对应能力/形态约束）需说明理由，不许默认抄近路。
- **cannbot 安装支持**：默认装到 `~/.cannbot-skills`，用环境变量 `CANNBOT_SKILLS_DIR` 覆盖
  路径，`CANNBOT_UPDATE=1` 更新，脚本自带关键文件校验。cannbot 仓库根自带
  `install.sh/install.ps1`（插件安装）可按需使用。
- 不把 cannbot-skills 的内容抄入本 skill；引用时给出 skill 名与场景即可。
- 本仓 fusion pattern 的 replacement kernel（triton 自研）走 `compilation-dev` 的
  pattern 生命周期，本 skill 只负责 kernel 本体开发与调优。

## Reference Files

- 🗺️ `references/operator-optimization-skill-map.md` — 加载时机: 任何算子开发/优化任务开始前（场景路由）
- 🔗 `../compilation-dev/references/benefit-rootcause-guide.md` — 加载时机: replacement kernel 命中但收益存疑时
- 📝 `../dummy-run/references/minimax-h3-notes.md` — 加载时机: 涉及 MiniMax-H3 算子语义/图形态时
- 🧩 `references/mindiesd-fusion-notes.md` — 加载时机: kernel 改动未生效/同名算子冲突/AscendC 集成调试/融合收益前置评估时（MindIE-SD/CANN 特有经验，与 cannbot 并行使用）
- 🔌 `references/catlass-kernel-integration.md` — 加载时机: 把 catlass/类 catlass 外部 AscendC kernel 以标准算子形态接入 mindiesd（单 .so ASC 混编、ASC 链接/静态运行时、torch custom op C++ 形态、设备事实、集成侧验证）时；接入 compile 图的 fake/无状态约束与 compile 前后收益核验 → `../compilation-dev/references/pattern-dev-notes.md` §5
- 🔗 `references/catlass-ffn-fusion-guide.md` — 加载时机: 开发/复刻「量化 matmul+激活+输出量化」catlass 类融合算子（vendored 头、bias、GraphPatternEntry 真图命中、开关治理）时（六段流水线；案例 mm_swiglu_mxquant/mm_gelu_mxquant）
- 📋 `references/mmgelu-flux-wan-qwen-case.md` — 加载时机: 对照 mm_gelu_mxquant 案例（真实图链/bias 实况/装载 API 坑/计数与 AB/工程坑）时

## Bundled Scripts

- `scripts/install_cannbot.sh` / `scripts/install_cannbot.ps1` — cannbot-skills 安装/校验/更新（使用 cannbot 特性前执行）

## 维护与更新

当外部 cannbot-skills 结构变化、新增已验证的算子 DSL/优化方法、或本仓沉淀新的算子
教训时，按 dev-workflow 的复盘流程更新本 skill 与 skill-map。

> 依赖提示：本 skill = 「场景路由 + MindIE-SD 特有经验层」——方法论本体在 cannbot，本仓经验
> 在其上叠加（非互斥）；依赖 = cannbot-skills 技能库 + 跨技能引用
> （compilation-dev/references/benefit-rootcause-guide.md、dummy-run/references/minimax-h3-notes.md、
> dev-workflow/references/rework-lessons.md）——外部库结构或任一引用文件变动时同步本文件，避免悬空引用。
