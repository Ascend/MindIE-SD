# 并行作用域「到底有没有生效」的判定（DiT 侧）

> **加载时机**：改了 SP / CP / Ulysses / AllGather-KV 等并行配置后**不报错但结果没变、性能没变**，
> 或**小规模能跑而大规模崩**（如 Ascend `EE1003 coreDim` 超限），或要在 NPU 上定位并行相关崩溃时。
> **这类故障的共同特征是不抛异常**，靠读代码和看日志几乎不可能发现，必须打印状态量并做对照实验。
>
> **边界（各技能对自身使能成功负责）**：本文件只管 **DiT 侧并行改动“是否真的生效”**；
> VAE / 解码器分片侧的同类判据在 `../../vae-opt/references/parallel-scope-effectiveness.md`；
> 具体报错码与版本相关陷阱见同目录 `ascend-parallel-traps.md`；
> 测量与上报口径（三证、ABBA、噪声地板、上报模板）单点在 `perf-gate/references/`，
> 本文件**不复述**，只在 §5 给出指针。

---

## 0. 适用范围：机制可迁移，常数必须现场复验

**本文件的判据是机制（与环境无关），例子里的常量是快照（与环境强绑定）。**

- **可迁移（沉淀）**：静默降级、“日志 ≠ 生效”、小规模掩盖越界错误、作用域深度记账失配、
  必须两侧对照才能下结论 —— 换硬件、换框架、换代际都成立；
- **必须本地重新确认**：错误码字面量、各配置项默认值、后端支持面、对齐倍数、阈值与容差
  —— 引用时必须现场复核，不得当作框架事实直接使用。

陷阱的时间维度纪律（每条陷阱要写“如何判定它仍存在”、升级后逐条复核、不再复现的必须删除）
见 `ascend-parallel-traps.md` §0。**默认动作是枚举 ≥2 个候选 → 同口径实测 → 先正确性后性能 → 择优**
（“先跑对，再跑快；先把候选测全，再谈最优”），不要照抄文档里的参数值；
这条择优纪律的完整形式（含落选读数留档、本环境自有基线）单点在 `../../quantization-dev/SKILL.md` §0.1。

---

## 1. 这类故障为什么难发现

并行代码最常见的失败方式**不是报错，而是静默降级**：

- 分片 hook 没生效 → 每个 rank 冗余地算整条序列 → **结果“看起来正常”，只是慢**；
- 作用域深度记账失配 → 某段逻辑跑在“未分片”分支上 → 小规模无异常，大规模才越界；
- 日志打了 “Applying sequence parallelism…”，但**打印发生在真正注册之前** →
  看日志会得出完全错误的结论。

所以本文件的核心不是“怎么配并行”，而是**“怎么证明它真的生效了”**。

---

## 2. 第一步：先证明是否生效，再谈对错

**不要从“结果对不对”开始，要从“分片有没有发生”开始。** 这两件事的排查成本差一个数量级。

在任何可疑位置打印**状态量**（不是日志字符串）：

```python
import sys
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_world_size as _gsw,
    get_sequence_parallel_rank as _gsr,
)
from vllm_omni.diffusion.forward_context import get_forward_context as _fc

sys.stderr.write(
    "[probe] tensor=%s sp_world=%s sp_rank=%s sp_active=%s depth=%s\n"
    % (tuple(x.shape), _gsw(), _gsr(),
       getattr(_fc(), "sp_active", "n/a"),
       getattr(_fc(), "_sp_shard_depth", "n/a"))
)
sys.stderr.flush()
```

**判读规则**（vLLM-Omni 的 SP 约定）：

| 观测量 | 含义 |
|---|---|
| `x.shape[0] == seq_len` | **未分片**（危险） |
| `x.shape[0] == seq_len // sp_world` | 已分片 ✓ |
| `_sp_shard_depth == 0` | **处于分片作用域之外** → 该处看到的是全序列 |
| `_sp_shard_depth >= 1` | 处于分片作用域内 ✓ |
| `sp_active` | 由 `_sp_shard_depth > 0` 推出，**不要单独相信它** |

`forward_context.py` 的原文就是：*“If `_sp_plan` hooks are applied: use `_sp_shard_depth`
(0 = outside sharded region)”*。**depth 是唯一可靠的判别量。**

### 常见误判：`sp_plan_hooks_applied`

这个字段在**正常与异常两种配置下都可能是 `False`**（它是 contextvar，加载期设置、forward 期读取，
常常读不到）。**用它判断“hooks 没生效”会得到错误结论。** 实测教训：曾据此下结论，
补了对照后发现两侧都是 `False`。

---

## 3. 第二步：每侧都测，不要单侧下结论

**这是最贵的一条。**

并行故障里，任何“差异量”都必须在**生效侧**与**失效侧**各测一次才有意义。
实践中，**连续五次**因只看一侧而得出错误结论：

| 单侧观测 | 由此下的结论 | 补对照后 |
|---|---|---|
| `hooks=False` | hooks 未生效 | 两侧都是 False → 红鲱鱼 |
| 日志无 “Failed to apply…” | 没抛异常 | ✅ 这次对了 |
| `same=False` | 分片正常 | ✅ 这次对了 |
| `ulysses_mode` | 它是元凶 | 真正的元凶是两条结构守卫 |
| `hooks_applied=True` | hook 已注册所以没问题 | 真正原因在同一函数的下一行 |

**操作上**：一次性把同一个探针在 `--usp N`（基线）与复合配置下各跑一遍，**对比表格化**，
再下结论。跑两次的成本远低于在错误假设上迭代五轮。

---

## 4. 第三步：沿“判别量”逐层收窄，而不是猜

当结果不对但分片正常时，**找一个会在正确与错误路径上取不同值的量**，然后一路往上游追。

一条可复用的收敛路径（模板）：

```text
结果全序列 → 找判别量：_sp_shard_depth (基线 1 / 异常 0)
  → 谁改这个深度？  sequence_parallel.py 的 split hook (+1) / gather hook (-1)
  → 打印 split hook：same=False，分片正常 ✓（排除）
  → 打印 gather hook：depth_before=1，在目标行之前就已执行（发现顺序异常）
  → 发现 sp_gather 在模型里被调用两次（两处 decrement，max(0,·) 会归零）
  → 追到调用方的"本地跨度"计算 _sequence_parallel_local_span
  → 打印它：span=(0, seq_len) 全序列（判别量！）
  → 读该函数：两条硬编码守卫"仅支持纯 Ulysses"在复合分解下触发
  → 修复，验证 depth 与 shape 同时恢复
```

**要点**：每一步都产出一个**可打印的判别量**，而不是“我觉得应该是 X 的问题”。
猜错的假设会互相叠加，让问题看起来比实际更复杂。

### 一个通用技巧：身份比较会决定行为

框架里常有 `if new is not original:` 这类**对象身份比较**来判定“是否真的做了变换”。
若某个分支原样返回输入张量，身份比较为假，后续的状态记账（如深度递增）就**不会发生**
—— 而值完全正确，所以不会报错。**排查这类问题时，务必打印 `is` 的结果而不只是 shape。**

---

## 5. 假数字与数值敏感度：单点指针（本节不重复定义）

- **假数字（性能数字来自失败运行）**：三项证据的定义与判据单点在
  `../../perf-gate/references/evidence-toolbox.md` §1。
  并行场景的额外提醒：失败运行**返回得更快**，而并行改动本来就以“更快”为预期，
  于是失败读数会被读成“分片生效了”——报收益前必须先过三证。
- **“差异很大”是不是 bug**：先做**数值敏感度校准**（找一对已知无害的改动量差异当噪声地板），
  SOP 单点在 `../../perf-gate/references/measurement-discipline.md` §3。
- **把收敛后的结论外推到别的形态/步数**：口径与闸门见
  `few-step-multirank-protocol.md`（先在本文件确认分片生效，再用那份协议做跨形态外推）。

---

## 6. 维护与更新

- **触发条件**：`vllm_omni.diffusion.parallel_state` 与 `forward_context` 的接口换代 ——
  §2 的探针依赖 `get_sequence_parallel_world_size`、`get_sequence_parallel_rank`、
  `get_forward_context` 以及 `sp_active` / `_sp_shard_depth` 这组名字，判读规则表也把
  `_sp_shard_depth` 当作**唯一可靠的判别量**；§4 的收敛路径依赖 `sequence_parallel.py` 的
  split hook / gather hook 与 `_sequence_parallel_local_span`；§2 末尾那条
  `sp_plan_hooks_applied` 红鲱鱼提示随 contextvar 的实现方式变化。§0 已声明「错误码字面量、
  各配置项默认值、后端支持面、对齐倍数、阈值与容差」必须现场复核。
- **复核方法**：按 §2 把那段 `[probe] tensor=… sp_world=… sp_rank=… sp_active=… depth=…`
  探针在**同一配置的两侧**各跑一次（基线 `--usp N` 与复合配置），按 §3 表格化对照后再下结论；
  若 `x.shape[0] == seq_len` / `x.shape[0] == seq_len // sp_world` 与判读表里 `_sp_shard_depth`
  的 0（作用域外）/ 非 0（作用域内）两档不再成立（新版本改用别的作用域量），
  本文件的判据链整体作废重写，不做局部打补丁。
- **口径联动**：本文件不复述测量口径，§5 的三条指针（`../../perf-gate/references/evidence-toolbox.md`
  §1、`../../perf-gate/references/measurement-discipline.md` §3、
  `few-step-multirank-protocol.md`）所在文件或章节号变化时一并更新；VAE / 解码器侧的同类判据在
  `../../vae-opt/references/parallel-scope-effectiveness.md`，两边判据须一起复核。
