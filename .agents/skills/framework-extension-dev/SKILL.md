---
name: framework-extension-dev
compatibility: 三方框架源码工作区（vLLM-Omni / DiffSynth-Engine / LightX2V 等，editable/fork 安装）、NPU 复验环境；前置：缺口已由 model-auto-optimization §0 与用户确认补齐策略
description: >
  三方框架自身关键优化路径缺失时的特性补齐开发：按框架差异（代码组织/注入点/注册机制/合入姿势）
  从无到有实现 comm-stream 掩盖、缓存、稀疏/量化消费者等结构性能力，推荐「平台注册 + 合入上游」
  收口（参照 LightX2V #1471 `hccl_eager`/`COMPILE_BACKEND_REGISTER`），fork/monkey 备选。
  当 model-auto-optimization §0 判定缺口属框架自身、或 framework-support-matrix 中该框架特性 ❌/❓
  且确认要补齐时使用；即使用户只说"框架没 comm-stream/没缓存"也应触发。
  判据：框架该能力**已存在**（开关/接口/机制）→ framework-feature-enablement 开启使用；**未支持**
  才属本技能（mindiesd 仓内开发走 dev-workflow + compilation-dev/operator-dev）。
  由 model-auto-optimization §0 路由，dev-workflow 承接时加载。
---
# 三方框架特性补齐开发（框架侧缺口）

## 定位与边界

模型优化的「预期关键路径」有时在目标三方框架自身缺失（无 comm-stream 掩盖、无缓存机制、无量化
kernel 消费者、Ring 不支持 attn_mask…）。**判据 = 该能力框架是否已存在**：已存在（有开关/接口/
机制）→ framework-feature-enablement 开启使用；不存在 → 本技能从无到有开发。两类工作差异：

| 场景 | 归属 |
|---|---|
| 框架**已存在**特性的开启/使能（开关/接口已有；含为使能的最小 glue 修复，如 RAINFUSION `supports_packed_mask_free` classmethod） | framework-feature-enablement |
| mindiesd 仓内新增 pattern / kernel / 图下发 / 后端 | dev-workflow + compilation-dev / operator-dev / aclgraph-dev |
| **框架自身结构性能力开发**（comm-stream masking、缓存框架、稀疏/量化消费者、平台注册后端…） | 本技能（承载流程与差异知识；实现经开发子任务） |

### 跨侧重叠与分工判据（框架 × mindiesd 合作界面）

特性落地常横跨两侧：mindiesd 提供能力接口 + 框架提供接入点，而「合作界面」随框架而异（注册表 /
平台分发 / 注入点 / 合入形态），因此**框架开发与 mindiesd 特性开发之间存在天然重叠区**——但
**与特定框架的分工是明确的：以代码落点（仓库）定侧**：

- mindiesd 仓文件（含 `mindiesd/parallel` 等本仓实现）→ dev-workflow + compilation-dev / operator-dev /
  aclgraph-dev 开发子任务；
- 三方框架仓文件 → 结构性开发归本技能、接线/小修归 framework-feature-enablement；
- 重叠区（同一特性两侧都有改动）→ **按文件拆分两侧子任务**，互不越界，对接点统一收口。

收口方式：两侧分别开发后在对接点联调验证（输出一致 + 计数契约 + 三层证据，姿势见
framework-feature-enablement 验证回路）；该框架的合作界面差异记录进
`framework-support-matrix.md` 的姿势列，避免同一框架反复摸索。

## 触发与前置

- 由 `model-auto-optimization` §0「任务启动确认」判定缺口属框架自身并**已与用户确认补齐策略**后路由；
- 或对 `framework-support-matrix.md` 中该框架 `❌/❓` 的关键特性确认要补齐时直接触发。
- 实现本身作为开发子任务执行（Test-First 精神：先最小复现缺口 → 实现 → 冒烟/回归）。

## 框架差异速查（初始版，随 case 回填扩展）

| 框架 | 开发范式 / 注入点 | 注册机制 | 参考 |
|---|---|---|---|
| vLLM-Omni 0.28 | `diffusion_model_runner` 平台分发 + `interface.py` 默认方法 + NPU platform 实现；env 门控 hook（`OMNI_MINDIE_COMPILE`/`OMNI_KPROF*`） | 平台级实现按需扩展 | vllm-omni-minimax-h3-case.md |
| DiffSynth-Engine | `configs/base.py`+`args.py`（`compile_backend="mindie"`）；`pipelines/base.py` `compile_transformer_blocks` 逐个 `_compiled_call_impl` 注入 | —（框架内直接接线） | diffsynth-engine-notes.md |
| LightX2V #1471 | 平台注册表 `COMPILE_BACKEND_REGISTER` / `PLATFORM_COMPILE_BACKEND_REGISTER`；`seq_p_a2a_backend="hccl_eager"`（`ops/a2a/ascend_npu/`，`@torch.compiler.disable`）；rms/rope 注册表 | 平台注册懒工厂 + core 只查注册表 | lightx2v-mindiesd-case.md |

## 补齐实现流程

1. **最小复现缺口**：用最小用例/框架入口复现「关键路径缺失」证据（报错/回退/无机制）。
2. **形态选择**（按 §0 用户确认执行，不静默切换）：① 平台注册 + 合入上游（推荐）；
   ② fork 本地补丁（钉 `base_commit`）；③ monkey-patch（运行期注入，漂移风险最高）。
3. **实现**：按目标框架差异表到对应注入点开发（editable 安装改后即生效；无 `.git` 环境按
   env-install 的版本处理姿势）。
4. **验证**：输出一致性 + 计数契约（预期计数 fail-closed）+ 三层证据（图命中/kernel diff/墙钟，
   见 framework-feature-enablement 验证姿势）；有损类补齐过 evals 质量门。
5. **收口**：合入上游时遵循评审收敛姿势——common 零改动、平台注册 + 配置可选、未知名 **raise**
   不静默回退、backend 实例单例、collective 留 eager（先例：LightX2V #1471 `hccl_eager`）。
6. **回填**：更新 `framework-support-matrix.md`（状态/新特性行 + 版本 + 日期）、补 case 摘要、
   需要时增补本技能 evals。

## 关键纪律

- 框架能力版本漂移：实现前先钉框架 `base_commit`；合入上游前不得把 fork 结果当最终能力宣称。
- 结构性实现不并入 enablement 的 case（enablement 只做接线/小回修），各自单点维护。
- 本技能只提供「怎么补」的知识与流程；是否补、补到哪一档由 model-auto-optimization §0 的用户
  确认决定。

## Reference Files

- `framework-feature-enablement/references/framework-support-matrix.md` — 加载时机: 缺口判定 / 状态与新特性行回填时（特性命名以 docs/zh/features 为准）
- `framework-feature-enablement/references/vllm-omni-minimax-h3-case.md` — 加载时机: vLLM-Omni 侧开发（平台分发/回修/hook 姿势）
- `framework-feature-enablement/references/diffsynth-engine-notes.md` — 加载时机: DiffSynth-Engine 侧开发
- `framework-feature-enablement/references/lightx2v-mindiesd-case.md` — 加载时机: LightX2V 侧开发（#1471 合入姿势）
- `../framework-feature-enablement/SKILL.md` — 加载时机: 验证/使能姿势与计数契约
- `../dev-workflow/SKILL.md` — 加载时机: 开发子任务流程（Test-First/部署/复盘）

## 维护与更新

每个框架的结构性补齐落地后：回填「框架差异速查」行 + 更新 framework-support-matrix + case 摘要；
框架版本升级或上游合入形态变化时按 dev-workflow 复盘流程更新本技能。
