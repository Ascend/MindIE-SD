# Multimodal Skills

面向 MindIE-SD 多模态扩散模型（Wan2.2 / FLUX / Qwen-Image）在昇腾 NPU 上的开发、验证、部署与性能优化技能集合。

## 技能总览

```text
                      ┌─────────────────────┐
                      │    dev-workflow      │  ← 开发总入口
                      └──────┬──────────────┘
       ┌──────────────┬──────┼──────┬───────────┬──────────────┐
       ▼              ▼      ▼      ▼           ▼              ▼
 ┌──────────┐  ┌──────────┐  ┌─────────────┐ ┌──────────┐  ┌────────────────┐
 │ code-    │  │ markdown │  │compilation- │ │model-    │  │   ascend-      │
 │ standards│  │ -lint    │  │  support    │ │verification│ │   deploy       │
 │ Python   │  │ Markdown │  │ Pattern/    │ │ §A/B     │  │ 本地/远端部署   │
 │ 格式规范  │  │ 格式检查  │  │ Backend/Copy│ │ 验证     │  │ + 编译安装     │
 └──────────┘  └──────────┘  └─────────────┘ └──────┬────┘  └────────────────┘
                                                     │
                                            ┌────────┴────────┐
                                            ▼                 ▼
                                    ┌──────────────┐  ┌──────────────┐
                                    │performance   │  │ profiling-    │
                                    │-evaluation   │  │ collection    │
                                    │ msmodeling   │  │ NPU profiling │
                                    │ CPU模拟/实测  │  │ 数据采集+回传  │
                                    └──────┬───────┘  └──────┬───────┘
                                           │                 │
                                           └────────┬────────┘
                                                    ▼
                                       ┌─────────────────────┐
                                       │  performance-        │
                                       │  analysis            │
                                       │  5层递进分析         │
                                       │  方向级建议          │
                                       └──────────┬──────────┘
                                                  ▼
                                       ┌─────────────────────┐
                                       │  performance-        │
                                       │  optimization        │
                                       │  5步优化闭环         │
                                        │  features.md 方案     │
                                        └──────────┬───────────┘
                                                   │
                                        ┌──────────▼───────────┐
                                        │    auto-optimization  │  ← 端到端优化闭环
                                        │ 采集→分析→方案→复验   │
                                        └──────────────────────┘

                                             parallelism-strategy
                                            [WIP] 并行策略选型参考

                             mindie-sd-community-governance
                            [横切] 文档/治理/提交/PR/版本规范
```

---

## 技能列表

### 开发工作流

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[dev-workflow](skills/dev-workflow/SKILL.md)** | 开发总入口：Test-First 流程、并行开发策略、模型验证/部署/性能分析/复盘全流程路由 | ✅ |
| **[code-standards](skills/code-standards/SKILL.md)** | Python 代码格式与 lint 规则（Ruff 配置 / pre-commit 钩子 / 门禁专项） | ✅ |
| **[markdown-lint](skills/markdown-lint/SKILL.md)** | Markdown 文件格式检查规范（MD040 / 验证命令 / 修复模板） | ✅ |
| **[compilation-dev](skills/compilation-dev/SKILL.md)** | 编译后端适配与分析：Pattern 创建/注册/调试、Copy 算子消减、四后端选择 | ✅ |

### 模型验证与部署

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[model-verification](skills/model-verification/SKILL.md)** | 模型验证：§A Dummy Run 无权重快速验证架构兼容性；§B 部署验证已部署模型的推理正确性（vLLM/diffusers/魔乐 三种框架） | ✅ |
| **[ascend-deploy](skills/ascend-deploy/SKILL.md)** | 部署 MindIE-SD：本地昇腾直接编译安装，或 SSH 推送到远端容器编译。含环境兼容性前置检查、连接复用、NPU 管理 | ✅ |
| **[profiling-collection](skills/profiling-collection/SKILL.md)** | NPU profiling 数据采集：SSH → 开启 Profiler → 运行推理 → 压缩 → 回传本地。Warmup 自动剔除，数据对接 performance-analysis | ✅ |

### 性能工程

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[performance-evaluation](skills/performance-evaluation/SKILL.md)** | msmodeling 性能评估：无 NPU 时 CPU 模拟各硬件性能；有 NPU 时实测并路由到 profiling-collection 采集数据 | ✅ |
| **[performance-analysis](skills/performance-analysis/SKILL.md)** | 5 层递进分析：Warmup 验证 → DiT/VAE 分离 → FA/MatMul/Vector/Comm 分类占比 → Host Bound/通信/融合检测 → 方向级优化建议 | ✅ |
| **[performance-optimization](skills/performance-optimization/SKILL.md)** | 5 步优化闭环：基线 → 分析 → 选方案（从 mindiesd-features.md 选取具体 API） → 实施 → 复验。唯一真相源驱动 | ✅ |
| **[parallelism-strategy](skills/parallelism-strategy/SKILL.md)** | 并行策略选型参考：TP/USP/RSP/CFG 概览。决策树和实测数据待后续补充 | 📝 |

### 端到端优化

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[auto-optimization](skills/auto-optimization/SKILL.md)** | 端到端优化闭环：组合 profiling-collection → performance-analysis → performance-optimization，从 profiling 采集到方案实施到复验的一键流程 | ✅ |

### 治理与规范

| 技能 | 描述 | 状态 |
|------|------|:--:|
| **[mindie-sd-community-governance](skills/mindie-sd-community-governance/SKILL.md)** | 文档/治理/贡献者工作流/提交及 PR 规范/版本策略 | ✅ |

---

## 快速开始

### 开发一个 MindIE-SD 功能

加载 `dev-workflow`，按路线图执行：

```text
1. 动手前检查清单（模型/精度/分辨率/NPU/CFG 配置确认）
2. 编码实现 → code-standards（Ruff lint）
2.5 编译适配 → compilation-dev（Pattern 注册 / Copy 消减 / 后端选择）
3. 模型验证 → model-verification §A（Dummy Run）
4. 远端部署 → ascend-deploy（本地编译或 SSH 增量传输）
5. 部署验证 → model-verification §B（三种框架验证）
6. 性能评估 → performance-evaluation（msmodeling 或实测）
7. NPU profiling → profiling-collection（数据采集）
8. 瓶颈分析 → performance-analysis（5 层递进）
9. 性能优化 → performance-optimization（选取方案 + 复验）
10. 端到端闭环 → auto-optimization（一键采集→分析→方案→复验）
11. 复盘归档 → dev-workflow §6
```

### 验证新模型架构

```text
# 无权重时：§A Dummy Run 构造验证
python examples/dummy_run/wan_infer.py --config_cache ./configs --device_id 0

# 有真实权重时：§B 部署验证（按框架选验证方法）
model-verification §B → B2 diffusers: from_pretrained → 1 step inference
```

### 部署到昇腾

```bash
# 本地昇腾设备：直接编译安装
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python setup.py build_py && pip install -e .

# 远端昇腾设备：SSH 增量部署
python skills/ascend-deploy/scripts/deploy_to_remote.py
```

### 评估 + 分析完整流程

```bash
# 1. msmodeling 评估（无 NPU 模拟或实测）
python -m cli.inference.video_generate \
    <model_path> --device ATLAS_800_A2_376T_64G \
    --height 480 --width 832 --frame-num 81 --dtype bfloat16

# 2. 有 NPU 时：需采集 profiling → 采集 + 分析
python skills/profiling-collection/scripts/collect_profile.py \
    --script wan_infer.py --device-id 0

# 3. 分析 profiling 数据
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
    ├── code-standards/                   # Python 格式规范
    │   ├── SKILL.md
    │   └── references/                   # gate-check-rules.md
    ├── compilation-dev/              # 编译后端适配与分析
    │   ├── SKILL.md
    │   └── references/                   # backend-comparison / graph-comparison-guide / mismatch-catalog / pattern-templates / registration-checklist
    ├── markdown-lint/                    # Markdown 格式检查
    │   └── SKILL.md
    ├── model-verification/               # 模型验证（Dummy Run + 部署验证）
    │   ├── SKILL.md
    │   └── references/                   # construction-methods / phase-timer
    ├── ascend-deploy/                    # 部署（本地编译 + 远端 SSH）
    │   ├── SKILL.md
    │   ├── scripts/                      # deploy_to_remote.py / pick_free_device.py
    │   └── references/                   # troubleshooting-tree.md
    ├── profiling-collection/             # NPU profiling 采集
    │   ├── SKILL.md
    │   └── scripts/                      # collect_profile.py
    ├── performance-evaluation/           # 性能评估（msmodeling）
    │   ├── SKILL.md
    │   ├── scripts/                      # validate_results.py
    │   └── references/                   # setup-guide / evaluation-guide / hardware-specs / ...
    ├── performance-analysis/             # 瓶颈分析（5 层递进）
    │   ├── SKILL.md
    │   ├── scripts/                      # analyze_trace.py / compare_traces.py
    │   └── references/                   # capability-matrix / operator-catalog / heuristics
    ├── performance-optimization/         # 优化闭环
    │   ├── SKILL.md
    │   ├── scripts/                      # refresh_features.py
    │   └── references/                   # optimization-dimensions / mindiesd-features
    ├── parallelism-strategy/             # 并行策略（WIP）
    │   └── SKILL.md
    ├── auto-optimization/                # 端到端优化闭环
    │   ├── SKILL.md
    │   └── references/                   # artifact-layout.md
    └── mindie-sd-community-governance/   # 文档/治理/提交规范
        ├── SKILL.md
        └── assets/                       # mr_ruleset.xlsx
```

## 技能间数据流

```text
model-verification (粗粒度时序) ──────────────────────┐
                                                        │
ascend-deploy (部署结果) ──→ model-verification §B    ├─→ performance-analysis
                          (验证已部署模型)              │    (消费 profiling + 粗粒度数据)
                                                        │
performance-evaluation (msmodeling/实测)               │
                            │                          │
                            └──→ profiling-collection ──┘
                                 (标准 CANN Profiler 数据)
                                         │
                                         ▼
                                  performance-optimization
                                  (消费方向级建议 + 查 features.md 选方案)
```

## 贡献

新增 skill 请遵循 `dev-workflow` 中的「新增 Skill 规范」（引用 [Anthropic skill-creator](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md) 指南）。

### 当前开发状态

- ✅ 12 个 skill 已填充实际内容
- 📝 1 个 skill 待完善（parallelism-strategy）

## 参考链接

- [Anthropic skill-creator 规范](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [msmodeling 文档](https://gitcode.com/Ascend/msmodeling)
- [MindIE-SD](https://gitcode.com/Ascend/MindIE-SD)
