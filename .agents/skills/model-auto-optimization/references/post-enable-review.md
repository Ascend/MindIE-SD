# 特性使能后复核清单（Post-Enable Review）——模型优化流程强制复核点

> 定位：model-auto-optimization 在 S4（及 S1–S3 使能面）**每落地一个会改变 kernel 序列 / 数据口径 /
> 显存余量 / 精度域 的特性档后**，按本清单做一次复核——把「量化后要重新审视融合机会」「量化后要审视
> 通信与量化/压缩通信的可能」等经验固化为**强制动作**，而不是只记录性能数字。
> 方法细节与案例见 `lossless-methodology-notes.md` §B/§D/§E 与
> `framework-feature-enablement/references/vllm-omni-minimax-h3-case.md`；复核记录进 evidence 与子表。

## 0. 触发条件（任一命中即整表复核）

- 使能量化档（INT8/FP8/MXFP8/w8a8/w4a8…）——最典型；
- 使能/改档稀疏（后端 rf_v2/ada_bsa、precision bf16/mix、sparsity 值）；
- 使能/改档缓存（DiT-Cache 类型/阈值）；并行拓扑变化（TP/USP/ring/offload 解锁）；
- compile/融合 pattern 使能或回退；显存占用结构变化（offload、量化降显存）。

## 1. 六面复核（每面：问什么 → 采什么 → 回哪条回路）

### ① kernel 序列与融合（回 S1 融合回路）

- 旧融合结论是否仍有效？——S1 融合判定绑定「使能前序列」，**不得跨序列沿用**；
- 新 kernel 的 producer/consumer 邻接是否产生新候选？（如 w8a8 的 `DynamicQuantV2→QuantBatchMatmulV3`
  对是否紧邻/可并入 GEMM、与上游 norm/SwiGlu epilogue 可融合）；
- **compile 是否按新序列重测**？旧序列负收益不作数（量化序列多出可融合小 kernel 对，结论可能翻转）；
- 输出：融合候选清单，采纳/回退各带三层证据（图命中→kernel diff→墙钟）；**实施后的「使能后融合」
  收益在总览表归入使能特性行**（如 `量化(w8a8)` 行含其融合收益），拆分在特性 §5 子表展示
  （规则见 overview-report.md §2.4），未实施候选标 ❓ 不计入当前行收益。

### ② 通信（回 S3 回路）

- 单步 compute/comm(未重叠)/free 占比是否变化（GEMM 加速 → comm 占比升 → 掩盖空间重估）？
- 通信数据可否**量化/压缩**：TP allreduce 部分和、USP/SP 注意力跨 rank K/V 交换、量化 allreduce——
  每个候选是**有损维度**：误差走质量门禁 + off-identity + 计数契约，再进 S4-2 组合协议叠加；
- 实现归属：集合通信/comm-stream/量化 allreduce 多属框架结构性缺口 → 按 SKILL §0 补齐策略经
  `framework-extension-dev` 确认，**不静默改三方框架**。

### ③ 显存与并行解锁（回 S3 回路）

- 量化/offload 后显存余量是否解锁新拓扑？（量化降显存 → 跑一遍「并行 × 新余量」候选矩阵，
  至少 10 步快测；H3 实证：INT8 online 后 TP1×USP4 可行 = 85.1s@60）。

### ④ 质量与口径（回 S4/质量门禁）

- 有损档质量 = **vs 同构 lossless 绝对口径**（有损×并行交叉值只作次级参考）；
- 视觉判卷 + off-identity；阈值引用 `evals/profiles/{model}.toml`（未校准不宣称通过）；
- 报表命名与说明列：量化修饰符 token（w8a8/f8/w8a8f8）语义与**覆盖范围**、稀疏度数值、数据类型必写。

### ⑤ 组合 seam（回 S4-2 组合协议）

- 新档与既有组合的 seam 判定：同 seam 只留最强档、跨 seam 可叠加；frontier 保留 / 回退名单更新
  （组合前查 `combination-search.md`，量化 × 融合 pattern 先确认 kernel 接受量化输入）；
- **覆盖完整性核验**：`量化`/`稀疏`/`Cache` 维度就绪后，逐项核对迭代表 [MUST] 覆盖清单——
  跨 seam 两两（量化×稀疏 / 稀疏×Cache / 量化×Cache）与**三元 `Cache+量化+稀疏`** 均有裁决
  （已测或带证据豁免），无未裁决 [MUST] 行；缺行即组合覆盖缺口，打回补测或登记豁免。

### ⑥ 计数契约与证据（真实性核验）

- 技术真实参与：量化 kernel 调用数 / 缓存命中步 / 稀疏 staying-dense / 融合 kernel 实际执行数；
  off-identity 独立档留痕；预期计数不符 fail-closed；
- 复核记录进 evidence.json + 相应子表/说明列（如 overview_report §5 子表、step_trace 占比）。

## 2. 首案例对照（H3 × vLLM-Omni 0.28 / 950PR；2026-09-05/06 + env A 2026-09-07）

| 使能 | 复核发现 | 去向 |
|---|---|---|
| INT8 online（w8a8） | 266 MatMul → 6 遗留 + 260 对 DQ/QuantBatchMatmulV3（2885 行，零新增布局）；GEMM 级已融合 → 新机会 O1–O7（O1/O6 优先） | §D + 归档 `H3_w8a8_fusion_analysis.md` |
| INT8 | Comm(未重叠) 0.81s 不变、占比 16.6%→19.6%（+mix 30.8%、Overlapped=0）→ 量化通信/GEMM-comm 重叠候选 | §E |
| INT8 降显存 | TP1×USP4 + DLO 解锁（114.4s 无损 / 85.1·52.05s 有损）；USP4 质量补测为绝对值（19.43/0.692、15.97/0.523） | §B + overview §5.2 |
| 单独 Cache | 组合表缺「单点行」被审阅发现 → 补测 38.4s（576p，与 768P 单点 -66.7% 交叉一致）→ 单点行纳入报表强制项 | overview-report §2.3 |
| 显式 TORCH_SDPA 基线（editable mindiesd 宿主，env A） | 不显式指定时默认路由 FLASH_ATTN → 基线口径污染；显式 backend + resolve 日志 + 输出 md5 三方一致才视为冻结 | troubleshooting-vllm-omni §P0（基线显式化最佳实践） |
| mxfp8+FFN-MX+Cache 组合（60 步 TP2，env A） | 融合计数 fused 0→52/52、Qmm/DxQ 各 -52（kernel csv 交叉）；质量 18.53/0.673 ≥ 阈值 16/0.51 但视觉 inconclusive → 不宣称质量通过 | 计数契约核验；视觉不确定不宣称 pass |
| 稀疏 rf_v2（end_step 语义，env A） | `end_step=60`（=全程保留 dense）输出与上档 md5 一致 = staying-dense no-op；`end_step=0` 才真实参与（-38%）→ 参数语义先核 + 输出 off-identity 确认参与 | staying-dense 检查链（fail-closed） |
| 共享宿主多租户热节流（env A） | 同档 e2e 高 20-100% 的异常窗 → 剔除留痕；结论取同窗相邻对（r1/r3 稳定对）+ 反转 AB | troubleshooting-vllm-omni §P0（时间窗纪律） |

## 3. 接线与维护

- 编排：model-auto-optimization SKILL S4 纪律 6/7 + 阶段路由表 S4 验收点/产物；本清单为 S4 每档落地时
  的强制执行工具；
- 工具：profiling-collect 单步 hook（同口径 kernel/step_trace）、quality-gate、combination-search、
  framework-extension-dev（结构性缺口实现）；
- 维护：六面内容变化 → 同步 SKILL S4 纪律与 `lossless-methodology-notes.md` §D/§E（方法细节不在此重复）。
