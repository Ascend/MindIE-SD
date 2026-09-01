# Multimodal Skills

面向 MindIE-SD 多模态扩散模型（Wan2.2 / FLUX / Qwen-Image / MiniMax-H3）在昇腾 NPU 上的开发、验证、部署与性能优化技能集合。

## 技能总览

```text
                      ┌─────────────────────┐
                      │    dev-workflow      │  ← 开发总入口
                      └──────┬──────────────┘
       ┌──────────────┬──────┼──────┬───────────┬──────────────┐
       ▼              ▼      ▼      ▼           ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌─────────────┐ ┌──────────┐  ┌────────────────┐
  │ code-    │  │ markdown │  │compilation- │ │dummy-run │  │   ascend-      │
  │ standards│  │ -lint    │  │  dev        │ │  -dev    │  │   deploy       │
  │ Python   │  │ Markdown │  │ Pattern +   │ │ Dummy Run│  │ 本地/远端部署   │
  │ 格式规范  │  │ 格式检查  │  │ Inductor    │ │ 架构验证  │  │ + 编译安装     │
  └──────────┘  └──────────┘  └──┬──────┬───┘ └──────┬────┘  └───────┬────────┘
                                 │      │             │              │
                        ┌────────┴──┐ ┌─┴──────────┐  │     ┌────────┴────────┐
                        │ operator- │ │ aclgraph-  │  │     │ framework-      │
                        │  dev      │ │  dev       │  │     │ integration     │
                        │ 算子开发   │ │ 批量下发   │  │     │ vLLM-Omni/diffusers/
                        │ (cannbot) │ │ (NPUGraph) │  │     │ 魔乐 对接与验证   │
                        └───────────┘ └────────────┘  │     └─────────────────┘
                                                      │
                                            ┌─────────┴─────────┐
                                            ▼                   ▼
                                    ┌──────────────┐   ┌──────────────┐
                                    │ profiling-   │   │ performance  │
                                    │ collection   │   │  -analysis   │
                                    │ NPU 采集+回传 │   │ 5层递进分析   │
                                    └──────┬───────┘   └──────┬───────┘
                                           │                  │
                                           └────────┬─────────┘
                                                    ▼
                                        ┌─────────────────────┐
                                        │  performance-        │
                                        │  optimization        │
                                        │  5步优化闭环         │
                                        └──────────┬──────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │    auto-optimization  │  ← 端到端优化闭环
                                        │ 采集→分析→方案→复验   │
                                        └──────────────────────┘

                                             parallelism-strategy
                                            [WIP] 并行策略选型参考

                                             benchmark-dev
                                            [横切] 核心算子性能基准
                                            （FA/BSA/GMM/MM 微基准）

                             mindie-sd-community-governance
                            [横切] 文档/治理/提交/PR/版本规范
```

---

## 技能列表

### 开发工作流

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[dev-workflow](skills/dev-workflow/SKILL.md)** | 开发总入口：Test-First 流程、并行开发策略、模型验证/部署/性能分析/复盘全流程路由 | ✅ |
| **[compilation-dev](skills/compilation-dev/SKILL.md)** | Pattern matcher 与 Inductor/default 后端：Pattern 创建/注册/调试、Copy 消减（default 路径） | ✅ |
| **[aclgraph-dev](skills/aclgraph-dev/SKILL.md)** | NPU 图批量下发（aclgraph / aclgraph_ex）：NPUGraph 静态 capture、graph pool、lazy capture、replay | ✅ |
| **[operator-dev](skills/operator-dev/SKILL.md)** | 算子级开发与优化：场景路由到外部 cannbot-skills（Triton / Ascend C / Catlass / PyPTO / TileLang）+ 本仓补充 | ✅ |

### 模型验证与部署

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[dummy-run-dev](skills/dummy-run-dev/SKILL.md)** | Dummy Run 构造验证：随机权重快速验证新模型架构兼容性，评估参数量/显存/耗时 | ✅ |
| **[ascend-deploy](skills/ascend-deploy/SKILL.md)** | 部署 MindIE-SD：本地昇腾直接编译安装，或 SSH 推送到远端容器编译。含环境兼容性前置检查、连接复用、NPU 管理 | ✅ |
| **[framework-integration](skills/framework-integration/SKILL.md)** | 三方框架对接与验证：vLLM-Omni 全栈部署（950PR 源码构建）、Cache DiT + diffusers、魔乐社区、DiffSynth-Engine 等外部框架 compile 接入 | ✅ |

### 性能工程

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[benchmark-dev](skills/benchmark-dev/SKILL.md)** | 核心算子（FA/BSA/GMM/MM）性能基准：开发（架构/CLI/口径/schema 约定、扩展、数据异常排查）+ 使用（模型调优中单算子分析、按 dtype/量化档/参数形态对比选型） | ✅ |
| **[profiling-collection](skills/profiling-collection/SKILL.md)** | NPU profiling 数据采集：SSH → 开启 Profiler → 运行推理 → 压缩 → 回传本地。Warmup 自动剔除，数据对接 performance-analysis | ✅ |
| **[performance-analysis](skills/performance-analysis/SKILL.md)** | 5 层递进分析：Warmup 验证 → DiT/VAE 分离 → FA/MatMul/Vector/Comm 分类占比 → Host Bound/通信/融合检测 → 方向级优化建议 | ✅ |
| **[performance-optimization](skills/performance-optimization/SKILL.md)** | 5 步优化闭环：基线 → 分析 → 选方案（从 mindiesd-features.md 选取具体 API） → 实施 → 复验。唯一真相源驱动 | ✅ |

### 端到端优化

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[auto-optimization](skills/auto-optimization/SKILL.md)** | 端到端优化闭环：组合 profiling-collection → performance-analysis → performance-optimization，从 profiling 采集到方案实施到复验的一键流程 | ✅ |

### 并行与治理

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[parallelism-strategy](skills/parallelism-strategy/SKILL.md)** | 并行策略选型与实测：Ulysses USP、CP 通信掩盖（comm-stream masking）、CFG/TP/RSP，含 910B 实测数据与 AlltoAllV 绕过 | ✅ |
| **[code-standards](skills/code-standards/SKILL.md)** | Python 代码格式与 lint 规则（Ruff 配置 / pre-commit 钩子 / 门禁专项） | ✅ |
| **[markdown-lint](skills/markdown-lint/SKILL.md)** | Markdown 文件格式检查规范（MD040 / 验证命令 / 修复模板） | ✅ |
| **[mindie-sd-community-governance](skills/mindie-sd-community-governance/SKILL.md)** | 文档/治理/贡献者工作流/提交及 PR 规范/版本策略 | ✅ |

---

## 快速开始

### 开发一个 MindIE-SD 功能

加载 `dev-workflow`，按路线图执行：

```text
1. 动手前检查清单（模型/精度/分辨率/NPU/CFG 配置确认）
2. 编码实现 → code-standards（Ruff lint）
2.5 编译适配 → compilation-dev（Pattern 注册 / Copy 消减）；算子本体 → operator-dev；批量下发 → aclgraph-dev
3. 模型验证 → dummy-run-dev（Dummy Run）
4. 远端部署 → ascend-deploy（本地编译或 SSH 增量传输）
5. 部署验证 → framework-integration §1（vLLM-Omni / diffusers / 魔乐）
6. NPU profiling → profiling-collection（数据采集）
7. 瓶颈分析 → performance-analysis（5 层递进）
8. 性能优化 → performance-optimization（选取方案 + 复验）
9. 端到端闭环 → auto-optimization（一键采集→分析→方案→复验）
10. 复盘归档 → dev-workflow §6
```

### 验证新模型架构

```text
# 无权重时：Dummy Run 构造验证
python examples/dummy_run/wan_infer.py --config_cache ./configs --device_id 0

# 有真实权重时：framework-integration §1 部署验证（按框架选验证方法）
framework-integration §1 → 1.2 diffusers: from_pretrained → 1 step inference
```

### 部署到昇腾

```bash
# 本地昇腾设备：直接编译安装
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py build_py && pip install -e .

# 远端昇腾设备：SSH 增量部署
python skills/ascend-deploy/scripts/deploy_to_remote.py

# vLLM-Omni 全栈（950PR 源码构建）→ framework-integration §2
```

### Profiling + 分析完整流程

```bash
# 1. 远端采集 profiling
python skills/profiling-collection/scripts/collect_profile.py \
    --script wan_infer.py --device-id 0

# 2. 分析 profiling 数据
python skills/performance-analysis/scripts/analyze_trace.py \
    --profile-dir ./profile_l1 --output-dir ./
```

---

## 目录结构

```text
multimodal-skills/
├── README.md
└── skills/
    ├── dev-workflow/                     # 开发总入口
    │   ├── SKILL.md
    │   └── references/                   # pattern-dev / ascend-ops / cross-platform / rework-lessons
    ├── compilation-dev/                  # Pattern matcher + Inductor/default 后端
    │   ├── SKILL.md
    │   ├── scripts/                      # compare_profiles.py / cmp_kernels.py / analyze_copy_kernels.py
    │   └── references/                   # pattern-templates / registration-checklist / test-templates / mismatch-catalog / graph-comparison-guide / custom-graph-pass / copy-elimination-guide / benefit-rootcause / benchmark-guide / pattern-dev
    ├── aclgraph-dev/                     # NPU 图批量下发（aclgraph/aclgraph_ex）
    │   └── SKILL.md
    ├── operator-dev/                     # 算子开发（复用外部 cannbot-skills）
    │   ├── SKILL.md
    │   └── references/                   # operator-optimization-skill-map
    ├── dummy-run-dev/                    # Dummy Run 构造验证
    │   ├── SKILL.md
    │   └── references/                   # construction-methods / phase-timer / minimax-h3-notes
    ├── ascend-deploy/                    # 部署（本地编译 + 远端 SSH）
    │   ├── SKILL.md
    │   ├── scripts/                      # deploy_to_remote.py / pick_free_device.py / ssh_helper.py
    │   └── references/                   # troubleshooting-tree.md
    ├── framework-integration/            # 三方框架对接与验证（vLLM-Omni 等）
    │   ├── SKILL.md
    │   └── references/                   # troubleshooting-vllm-omni.md
    ├── profiling-collection/             # NPU profiling 采集
    │   ├── SKILL.md
    │   └── scripts/                      # collect_profile.py
    ├── benchmark-dev/                    # 核心算子性能基准工具链开发
    │   ├── SKILL.md
    │   ├── references/                   # debugging.md（数据异常排查）
    │   └── evals/                        # evals.json
    ├── performance-analysis/             # 瓶颈分析（5 层递进）
    │   ├── SKILL.md
    │   ├── scripts/                      # analyze_trace.py / compare_traces.py
    │   └── references/                   # capability-matrix / operator-catalog / heuristics / analysis_workflow
    ├── performance-optimization/         # 优化闭环
    │   ├── SKILL.md
    │   ├── scripts/                      # refresh_features.py
    │   └── references/                   # optimization-dimensions / mindiesd-features
    ├── parallelism-strategy/             # 并行策略（WIP）
    │   └── SKILL.md
    ├── auto-optimization/                # 端到端优化闭环
    │   ├── SKILL.md
    │   └── references/                   # artifact-layout.md
    ├── code-standards/                   # Python 格式规范
    │   ├── SKILL.md
    │   └── references/                   # gate-check-rules.md
    ├── markdown-lint/                    # Markdown 格式检查
    │   └── SKILL.md
    └── mindie-sd-community-governance/   # 文档/治理/提交规范
        ├── SKILL.md
        └── assets/                       # mr_ruleset.xlsx
```

## 技能间数据流

```text
dummy-run-dev (粗粒度时序) ─────────────────────┐
                                                 │
ascend-deploy (部署结果) ──→ framework-integration §1    ├─→ performance-analysis
                          (框架侧验证)           │    (消费 profiling + 粗粒度数据)
                                                 │
profiling-collection (标准 CANN Profiler 数据) ──┘
         │
         ▼
  performance-optimization
  (消费方向级建议 + 查 features.md 选方案)
```

## 贡献

新增 skill 请遵循 `dev-workflow` 中的「新增 Skill 规范」（引用 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) 指南）。

> 每个 skill 均含 `evals/evals.json`（skill-creator 测试用例：≥2 条真实 prompt + expectations）。
> 新增或大改 skill 时必须同步增补 evals。

### 当前开发状态

- ✅ 16 个 skill 已填充实际内容（含 evals/evals.json）

## 参考链接

- [Anthropic skill-creator 规范](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [cannbot-skills（算子开发外部技能库）](https://gitcode.com/cann/cannbot-skills)
- [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
