# Pattern 收益验证与根因分析

> 触发时机：Phase 6 确认 pattern **已命中**（融合 kernel 出现、原 kernel 消失），
> 但逐 pass AB（Phase 6 末尾）显示该 pattern **无收益或负收益**。
> 本文档提供 4 步诊断流程 + 5 类根因目录，均来自真实案例（MiniMax-H3 等，2026-08）。

## 目录

- [1. 核心认知：命中 ≠ 收益](#1-核心认知命中--收益)
- [2. 诊断流程（4 步）](#2-诊断流程4-步)
- [3. 根因目录（R1-R5）](#3-根因目录r1-r5)
- [4. 修复验证](#4-修复验证)
- [5. 案例库](#5-案例库)

---

## 1. 核心认知：命中 ≠ 收益

pattern 匹配成功并生成融合 kernel，**不代表 wall-clock 变快**。融合可能引入新的
开销（dtype 转换、输入物化、format 转换），或匹配到错误子图（fallback）。判据：

| 层面 | 命中判据 | 收益判据 |
| --- | --- | --- |
| 图 | `PatternMatchPass replace N` 增加 | — |
| kernel | 融合 kernel 出现 + 原 kernel 消失 | 新增 kernel 总耗时 < 消失 kernel 总耗时 |
| wall | — | `timed` 段耗时下降（逐 pass AB） |

**只做前两层的 pattern 是"命中未受益"**，必须逐 pass AB 暴露（Phase 6 的 AB 法
是入口，本文档是 AB 发现异常后的根因分析）。

---

## 2. 诊断流程（4 步）

### Step 1: 确认命中的真实情况

- 关闭该 pattern 跑一次 profile，开启该 pattern 跑一次 profile（其余开关一致）
- 用 `kernel_details.csv` 按 kernel 名聚合，diff 两个集合：
  - `gone` = 被融合的原始算子链（如 Neg/Cat/Mul/Split）
  - `new` = 新增的融合 kernel + 新增的辅助 kernel（Copy/Cast/TensorMove）
- 记录：`sum(new_dur)` vs `sum(gone_dur)`

### Step 2: 检查新增辅助 kernel（重点嫌疑）

融合 kernel 之外，任何 `InplaceCopy_*` / `*_Cast` / `*_Slice` / `*_TensorMove` 的
**新增计数与耗时**都是嫌疑。逐个看它们**前后相邻 kernel**（定位在融合算子的输入侧
还是输出侧）：

```text
输入侧 copy:  <RmsNorm> → <InplaceCopy_Slice> → <RotaryV2>     ← 输入物化
输入侧 cast:  <x bf16> → <InplaceCopy_Cast> → <RotaryV2>        ← dtype 提升
输出侧 cast:  <RotaryV2> → <InplaceCopy_Cast> → <Cat>           ← dtype 降回
边界转换:     <RmsNorm> → <InplaceCopy_TensorMove> → <RotaryV2> ← format 不匹配
```

### Step 3: 对照根因目录（§3）归类

每个新增 kernel 按「输入侧/输出侧 + 类型」映射到 R1-R4，读对应修复方案。

### Step 4: 修复 → 重新 AB + kernel diff

修复后重跑：① 该 pattern 的 wall 收益是否转正；② 新增辅助 kernel 是否消失；
③ 单元测试 + 全模型回归（Phase 6 三层）不回归。

---

## 3. 根因目录（R1-R5）

### R1: dtype 提升（replacement 收到 fp32 输入）— 最高频

**症状**：融合 kernel 前后各一个大张量 `InplaceCopy_Cast`（几十 us/处），
pattern 单独收益 ≈ 0 甚至为负。

**机制**：pattern 内 `_to_copy(x, bf16)` 节点被 `register_replacement` 当作
"待匹配子图的一部分"消费掉 → replacement 收到的参数是 cast **前**的 fp32 节点。
若 replacement 调用的融合 API 内部有 `x.to(other.dtype)`（如 `rope.py` 的
`x_in = x.to(cos.dtype)`），bf16 的 x 被提升到 fp32 计算，输出再 `type_as(x)`
降回 bf16 → 每处融合点 2 个大张量 dtype 转换，纯浪费。

**识别**：kernel diff 中 `InplaceCopy_Cast` 新增 ≈ 2 × 融合点数；大小 ≈ x 张量
（非 cos/sin 小张量）。

**修复**：在 replacement 中显式把辅助输入 cast 到 x 的 dtype：

```python
# 修复前: 融合 API 内部隐式提升 x 到 fp32
return mindiesd.layers.rotary_position_embedding(x_rot, cos, sin, ...)

# 修复后: 显式对齐辅助输入 dtype → API 内 x.to(...) 变 no-op
if dtype == torch.bfloat16:
    cos = torch.ops.aten._to_copy.default(cos, dtype=x.dtype)
    sin = torch.ops.aten._to_copy.default(sin, dtype=x.dtype)
return mindiesd.layers.rotary_position_embedding(x_rot, cos, sin, ...)
```

**实测收益**（MiniMax-H3 RoPE，见 §5 案例 1）：修复后每处 Cast 50+67us → 4+4us，
融合 kernel 本体 113us → 35.5us，pattern 收益从 ≈0 → **-0.68ms**。

### R2: 输入物化（slice/view 输入被编译侧物化）

**症状**：融合 kernel 前出现 `InplaceCopy_Slice` / `*_Slice`，耗时与输入张量大小
成正比（几十 us）。

**机制**：pattern 输入是 `x[..., :dim]` 切片（view），融合算子（如 `npu_rotary_mul`）
的 **compiled-graph 输入规范要求独立连续张量** → 编译侧物化 copy。注意：eager 下
`contiguous=False` 的 view 直接调用算子可能成功（probe 可证），但编译图仍会物化。

**识别**：`*_Slice` 新增计数 ≈ 融合点数；前后相邻 = 上游 matmul/norm → 融合算子。

**修复**（按成本递增）：

1. 方向 A：确认融合算子是否真要求独立输入（读 kernel 名/文档；如
   `RotaryPositionEmbeddingV2_Slice` 暗示 slice 可能是 op 内部实现）；
2. 方向 B（更高价值）：**扩大 pattern 匹配范围**，把切片 + passthrough 分支 +
   最终拼接（如 `_apply_rotary_emb` 的完整函数）整体替换为单算子，消除中间物化
   与拼接 cat；
3. 若两者都不可行：接受该 copy 为算子约束，记录在案，评估收益是否仍为正。

### R3: format 转换（相邻算子 format 不匹配）

**症状**：融合 kernel 前/后出现 `InplaceCopy_TensorMove`（几 us-几十 us/个），
且**多个 pattern 叠加后新增**（如 RMSNorm 融合后在其输出侧出现）。

**机制**：上游融合 kernel（如 `npu_rms_norm`）输出 format 与下游算子输入 format
约定不一致（如 BSHD 连续 vs 5HD/internal format）→ 边界 TensorMove。

**识别**：`InplaceCopy_TensorMove` 新增与另一个 pattern 的开启相关（A 开 B 不开时
无此 copy）；位置在 norm → 下一算子边界。

**修复**：对齐两个算子的 format 约定（同 layout）；若上游/下游 kernel 的 format
不可控，评估合并两个 pattern 为一次替换（消除中间边界）。

### R4: 误匹配 fallback（pattern 命中但回退）

**症状**：pattern 单独 AB 为**负收益**；日志出现 `[xxx] fallback (...)`。

**机制**：通用 pattern（如 Wan 的 `x + y*gate` 残差+gate）匹配到**非目标模型**的
相似子图，替换后 `ndim`/shape 校验不满足 → 走 fallback（白匹配，无收益反有
调度开销）。

**识别**：fallback 日志；AB 变慢；kernel diff 中该融合 kernel 出现但计数少于
理论命中数。

**修复**：

1. **注册顺序锚定**（首选）：将更具体的 pattern 注册在通用 pattern **之前**，
   先吃掉同类子图（如 MiniMax RoPE 注册在 wan_residual_gate 之前，消除 4D
   rope 子图的误匹配）；
2. 收紧 pattern 锚定（加 shape/ndim 约束）；
3. 模型场景关闭该通用 pattern 开关（`enable_wan_residual_gate=False`）。

**实测**（MiniMax-H3 residual_gate，见 §5 案例 2）：误匹配 fallback 造成
+0.09~0.26ms 负收益；注册顺序修正 rope 部分后仍有 3D 残差 fallback 残留，
最终建议 MiniMax 场景关闭该开关。

### R5: 替换 kernel 实现低效（自研 kernel 打不过原生算子链）

**症状**：pattern 命中、无任何新增辅助 kernel（无 copy/cast/format 副作用），
但 wall 负收益；kernel diff 中**融合 kernel 本体耗时 > 被替换的原始 kernel 链
耗时之和**。常见于 replacement 是 triton 自研 kernel 而原链是 aclnn 原生
逐元素 kernel 的场景。

**机制**：`register_replacement` 只保证"kernel 数减少"，不保证"总耗时减少"。
Ascend 上 triton 后端的 codegen（grid-stride + mask 谓词、无手调双缓冲/流水）
访存效率显著低于手调 aclnn kernel（实测 ~0.74-1.15 TB/s vs ~2.9 TB/s 有效带宽，
中间张量 L2 驻留使 aclnn 链更占优）。注意两个测量陷阱：

1. 隔离 bench 的热缓存值会**高估** triton（连续复用同一 buffer，L2 温度高），
   必须以**模型内 profile 的 device 时间**为准（本例 bench 92.9us vs 模型内 148us）；
2. 对比基准是"off 时 Inductor 生成的 kernel 链"——NPU default 后端对跨广播
   （`[1,S,D]×[S,D]`）逐元素链**不融合**，off 侧就是多个独立 aclnn kernel，
   每个都是手调最优实现，很难被自研 kernel 整体超越。

**识别**：kernel diff 中融合 kernel 新增耗时 > 消失 kernel 链耗时；无 R1-R4
副作用；逐 pass AB 负收益。

**修复**（按优先级）：

1. **优先用 npu 原生算子做 replacement**（如 `npu_rms_norm`/`npu_swiglu`/
   `npu_rotary_mul`）：手调 C++ kernel，能真正融合且快于被替换链
   （对照：SwiGLU 用 `npu_swiglu` 正收益 -0.35ms）；
2. **排查自研 kernel 的"设计缺陷"而非直接放弃**（本节最重要，见案例 4）：
   先确认被替换链里是否还有 pattern 外的大算子（如 index_select gather）与
   冗余流量（[S,D] 物化+重读）——**把 gather 吸收进 kernel + 行结构/标量
   gather** 后 triton 可反超 aclnn 链（AdaLN：150us/site → 94.7us，转正
   -0.24ms）；"triton 打不过 aclnn" 往往是被 kernel 形态拖累的伪结论；
3. 仍打不过 → 隔离基准调 `BLOCK_SIZE`/grid（本例 1024→8192 提升 36%，
   但最终以模型内 profile 为准），或默认关闭开关（代码保留，待后端改进）。

**实测**（MiniMax-H3 AdaLN，见 §5 案例 4）：初版 BS1024 +0.64ms → BS8192
+0.26ms → gather 融合转正 -0.24ms → cannbot 技能(i32+3行) -0.45ms
（`enable_minimax_h3_adaln` 默认 True）。

---

## 4. 修复验证

每个修复必须闭环验证（缺一不可）：

1. **wall AB**：该 pattern 单独关闭 vs 开启，timed 段收益转正（>0.1ms 视为有效）；
2. **kernel diff**：R1/R2/R3 对应的辅助 copy/cast 消失或显著减少；
3. **正确性回归**：pattern 单元测试 + 全模型 `--compile` 推理 + BF16 图验证
   （`_verify_compute_precision_graph`，无 fp32/tf32/int32 计算节点）；
4. **叠加验证**：在"已有其他 pattern"基础上开启该 pattern，确认叠加收益
   （可能因图变小调度改善而 > 单独收益，也可能因边界冲突而 < 单独收益）。

---

## 5. 案例库

### 案例 1: MiniMax-H3 RoPE — R1 dtype 提升吞掉全部收益（2026-08）

| 项 | 值 |
| --- | --- |
| 现象 | pattern 命中（RotaryPositionEmbeddingV2 ×4 出现，Neg 8→4、Cat 8→4），但 AB 收益 ≈0 |
| 根因 | pattern 内 `_to_copy(cos, bf16)` 被消费，replacement 收到 fp32 cos/sin → `rope.py` `x.to(cos.dtype)` 把 bf16 x 提升 fp32 + `type_as` 降回 |
| 证据 | 每处 rope: `InplaceCopy_Cast 50us + 67us`（大张量 [1,S,H,96]） |
| 修复 | replacement 显式 `_to_copy(cos, dtype=x.dtype)` |
| 结果 | Cast 50+67→4+4us，RotaryV2 113→35.5us，收益 ≈0 → **-0.68ms**（both 26.54→25.76ms） |
| 文件 | `mindiesd/compilation/patterns/minimax_h3_rope_pattern.py`；报告 `refs/minimax_profiles/bf16_compile_ab_report.md` §7 |

### 案例 2: MiniMax-H3 residual_gate — R4 误匹配 fallback（2026-08）

| 项 | 值 |
| --- | --- |
| 现象 | `enable_wan_residual_gate` 对 MiniMax 残差子图误匹配，AB 负收益 |
| 根因 | Wan 通用 `x+y*gate` pattern 命中 MiniMax 的 `x_rot*cos+rotated*sin` 及残差子图，ndim 校验不满足 → fallback |
| 证据 | 日志 `[residual_gate_add] fallback (ndim) x=(1,3967,5376) y=(3967,5376)` ×4 |
| 修复 | RoPE pattern 先注册（消除 rope 子图误匹配）+ MiniMax 场景建议关闭该开关 |
| 结果 | 关闭后 -0.09ms 且消除 fallback 日志噪音 |

### 案例 3（反例）: MiniMax-H3 RMSNorm — 高收益为何高

| 项 | 值 |
| --- | --- |
| 现象 | pattern 收益最大：-4.38ms（-14.2%） |
| 原因 | 被融合的是**多 kernel 链**（freeze 后 pow/mean/add/rsqrt/mul 每处 4-5 kernel，含大 tensor 平方运算），替换为单 `npu_rms_norm` 后 net 收益大；且无 dtype 提升/物化/format 三类副作用 |
| 启示 | 收益与"被替换链的原始成本 - 替换引入的副作用"成正比；链长/大 tensor 处融合收益高，小 kernel 链易被副作用抵消（对照案例 1 修复前） |

### 案例 4: MiniMax-H3 AdaLN — R5 自研 kernel 低效 + 修复路径（2026-08）

| 项 | 值 |
| --- | --- |
| 现象 | 初版 pattern 命中（5×(Add+Mul+Add) 消失、5×scale_shift 新增，339→329 kernels），无新增 copy（`.contiguous()` 对连续输入是 no-op），但逐 pass AB **+0.64ms** |
| 根因(初版) | replacement 是 triton 1D-flatten kernel：BS1024 时 device 230us/site（0.74TB/s），比被替换的 3 个 aclnn 逐元素 kernel（117us/site，中间张量 L2 驻留 ~2.9TB/s）慢 ~2x；**且 2 个 index_select gather（33us/site）在 pattern 外未被吸收，[S,D] scale/shift 物化+重读 170MB 冗余流量** |
| 证据 | kernel diff：`scale_shift +1.15ms` vs `Mul/Add/Adds -0.587ms`（净 +0.56ms）；隔离 bench：BS1024=216us、BS8192=92.9us（warm）、cold 修正后 127us；模型内 device 148us |
| 修复① | BLOCK_SIZE 1024→8192（模型内 230→148us/site，AB +0.64→+0.26ms）——治标不治本 |
| 修复②（转正） | **gather 融合**：MiniMax 调制表只有 [3,D]（3 模态，32KB L2 驻留），pattern 扩展匹配 2 个 index_select 节点 → triton 单 kernel 内按行 gather + scale-shift（工作集 170MB→85MB）→ 每 site 150us→94.7us，**AB -0.24ms 转正**；GatherV2 16→6 |
| 修复③（cannbot 技能） | **i64→i32 索引**（Ascend i64 向量算术降级为标量循环）+ **3 行/program**（triton-latency-optimizer 技能库）→ 每 site 94.7→66us，**AB -0.24→-0.45ms**；grid 1322、尾部 ROWS=1 微 kernel 1.5us |
| 关键教训 | "triton 打不过 aclnn" 是**伪结论**——真实瓶颈是 kernel 设计（流量冗余 + 融合边界 + i64 标量降级 + 并行度），不是 triton 本身；改对 kernel 形态（行结构、标量 gather、int32、多行并行）后 triton 赢了 aclnn 链 |
| 文件 | `mindiesd/layers/scale_shift.py`（`gather_scale_shift` op）、`mindiesd/compilation/patterns/minimax_h3_adaln_pattern.py`；profiles `refs/profiles/adaln_kdiff/{off,on}`；bench `refs/bench_scale_shift*.py`、`refs/adaln_triton_optimization.md`；外部技能 `cannbot-skills/ops/triton-latency-optimizer`（discrete_memory_access / avoid_scalar_lowering / vector_core_partition / multibuffer） |

## 维护与更新

当发现新的负收益根因类型时，按 dev-workflow 的复盘流程更新本文件。
