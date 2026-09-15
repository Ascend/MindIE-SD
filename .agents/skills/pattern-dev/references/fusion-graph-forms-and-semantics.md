# 模型融合的真实图形态与算子语义契约

> **加载时机**：为某个模型写/调融合 pattern 时（先确认真实图形态与目标算子的语义方向），
> 或判断「这条链能不能融合、该用哪个算子承载」时。
> **边界**：pattern 创建/注册/改图的机制见 SKILL 生命周期与
> `graph-pattern-rewrite-guide.md`；收益负转正的根因目录见 `benefit-rootcause-guide.md`；
> 算子本体实现与 DSL 选型见 `operator-dev`；量化契约见 `quantization-dev`。

## 1. 真实图形态（before-freezing dump，逐条实测）

**判据**：pattern 必须按**真实 compile 图**的形态写（`symbolic_trace` / 独立 `make_fx` 构图
形态与真实图不同，其命中结论不得驱动方案选择——见 SKILL 行为约束 #4）。

| 链 | 真实图形态（要点） |
|---|---|
| RMSNorm | `torch.rms_norm` 在 **torch 2.11 已被前置分解**：before-freezing 图直接是 `_to_copy(f32)→pow→mean→add.Scalar→rsqrt→mul→mul`，before-freezing 的 pattern matcher **一次运行即命中**（"必须 after_freezing 二次运行"的旧结论基于 torch 2.9：当时 aot 保留单节点、freeze 才分解） |
| SwiGLU | `matmul → split.Tensor(x, F, -1)` → `getitem`(hidden 前半) / `getitem`(gate 后半) → `silu(getitem_gate)` → `mul(getitem_hidden, silu)` |
| AdaLN 调制 | `index_select(scale_table) → add(·, 1.0) → mul(x, ·) → index_select(shift_table) → add(mul, ·)`（调制表按模态索引取行） |
| FFN hidden（量化） | `Qmm([S,2F]) → view([1,S,2F]) → split → silu → mul → view([S,-1]) → npu_dynamic_mx_quant → Qmm(out)`；两个 `view` 的 S 是**动态**的（同图并存多个 S）⇒ trace 式 pattern 固化常量后永不命中（正解见 `pattern-dev-notes.md` §5.2 与 `graph-pattern-rewrite-guide.md`） |

**手写链 vs 高级 API（RMSNorm 类）**：不要用 `torch.rms_norm` 直接作 pattern——
`make_fx` 对它的自动分解产 `add_.Scalar`（inplace），而真实图产 `add.Scalar`（非 inplace），
**target 不同 ⇒ 0 命中**。手写链要精确固定每个 target：`add.Scalar`、`mean(dim=[x.dim()-1])`、
输入側 `_to_copy(f32)` 的 cast，且**不含输出 cast**（否则会引入 `_to_copy(bf16, layout, device)`
的 kwargs 差异）。备选：`pre_dispatch=True` 的 `make_fx` trace fn 可保留单节点 op（若未来版本
恢复 freeze 后分解，可改用单节点 pattern）。
（同族机制：算子分解差异与 decomp table 见 `mismatch-catalog.md` 类型 1。）

## 2. 目标算子的语义契约（方向错了会静默算错或白融合）

| 目标 | 语义 | 契约 |
|---|---|---|
| `npu_swiglu` | `first_half * silu(second_half)` | 与 diffusers SwiGLU 的 `silu(gate) * hidden` **gate/hidden 顺序相反** ⇒ 必须先把 chunk 顺序对调为 `[gate, hidden]` 再调用；顺序对调后的对拍误差为 **bf16 数值级**（本组合观测） |
| `npu_ffn(act="swiglu")` | 权重方向为 `w1` 的 k 维 = `x` 的 k 维 | 与实体 FFN 图的权重方向**不匹配** ⇒ 不能直接替换 |
| `AdaLayerNorm` 系列（如 `adaln_v2`） | 要求 weight/bias 非 None 或特定 shape | 纯调制链（weight=None）实测 **CheckShape failed** ⇒ 无现成算子可复用；现有 `muls_add` 只支持标量 scale，tensor-scale 调制需自研算子 |
| 自研 triton 算子族 | `gather_scale_shift`（AdaLN，吸收调制表的 index_select）、`gather_residual_gate`（残差+gate）、`swiglu`（免 cat）、`scale_shift`（plain 兜底） | 落地位置 `mindiesd/layers/scale_shift.py`；pattern 侧各模型专用文件（如 `minimax_h3_{swiglu,adaln,gate}_pattern.py`）；**免 cat** 的价值在于省掉 `cat([gate,hidden])` 的大张量物化 |

## 3. 站点数与计数契约（命中验证的锚）

- 融合**是否命中**以 **kernel 计数 == 期望站点数**为准（`scripts/check_fusion_hit.py`），
  不以图命中日志为准；off（flag=False）时应恢复原链。
- 站点数推算要**逐段数清**（例：RMSNorm 站点 = 每层每处 norm 之和，含 refiner / final / norm_out；
  对不上就是覆盖不全或误命中）。同一条链的占位型统计只作量级参照，不作交付证据。

## 4. 融合形态的适用边界（候选判定用）

- 结构判定规则（量化 FFN 类融合）：**hidden 侧输出维 = 2F + out 侧输入维 = F + 该容器内唯一** ⇒
  命中该形态才融合；不满足即"融合 0 站点"，属**预期而非缺陷**，应按未实现行登记
  （`是否完成融合=N（预期可融合：{依据}）`、收益 0）。
- **GELU 单分支 MLP**（如 `FeedForward(activation_fn="gelu-approximate")`）天然不命中 SwiGLU 类
  结构判定：没有 `[hidden|gate]` 两半、没有可配对的 `2F` out。要覆盖需**另起 GELU 语义 kernel**，
  不要试图复用 SwiGLU 融合（此判定适用于同族多模型：Wan / FLUX 等）。
- 同类形态跨模型可泛型覆盖时，**单点收益排序必须逐模型实测**，不得按模型大小外推。

## 5. 收益稳定性判据（跨规模外推）

- 融合收益**比例**在负载时长/序列规模变化时**稳定在同一量级**（本组合观测：同一组 pattern 在
  短/中/长三档时长下收益同向同量级，不随时长漂移）；
- 绝对节省**随规模放大**且呈**超线性**（注意力 O(seq²) 主导）⇒ 结论必须写清是"比例"还是"绝对量"，
  两者不可互换；
- **外推纪律**：站点级收益 ≠ 阶段级收益（见 `../../perf-gate/references/measurement-discipline.md` §5）。

## 维护与更新

新增融合链/新模型形态时补 §1 的图形态与 §2 的算子语义契约；图形态因框架或 torch 版本变化时，
先重跑真实图 dump 再改本文件（形态结论一律以当次 dump 为准）。
