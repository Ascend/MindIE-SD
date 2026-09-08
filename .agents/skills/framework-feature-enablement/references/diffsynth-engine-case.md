# 实例：DiffSynth-Engine 接入 MindIE-SD（Qwen-Image，性能/使能视角）

> ⚠️ **本文件是 DiffSynth-Engine 特定实例**。部署/compile 接入细节见
> `framework-feature-enablement/references/diffsynth-engine-notes.md`（互补，不重复）；
> 本文聚焦**性能使能方法 + kernel 级验证**，按 SKILL.md §6.2 迁移清单组织。
> 结论仅在该框架 + Qwen-Image + 该容器环境成立；移植他处需重新验证。

## 1. 框架画像（§6.2 迁移清单速答）

| 检查项 | DiffSynth-Engine 实况 |
|---|---|
| 算子注册机制 | 无 `rms_type`/`rope_type` 类抽象接口（模型层硬编码 torch 算子）→ 走 compile 接入 |
| 并行实现 | 单卡/无 Ulysses（`_repeated_blocks` 逐个编译），无 a2a → 无"a2a 留 eager"问题 |
| compile 图形态 | `Pipeline.compile_transformer_blocks` 对 submodule 编译；chunk/split 展开形态与 dummy 类似 |
| 动态 shape | 序列/分辨率固定（1024²），无 `Sym` 符号 → dynamic 参数无 guard 负担 |
| 长序列通信 | 单卡无通信；融合收益不被通信稀释 |

**结论**：DiffSynth-Engine 与 LightX2V 差异大（无注册表接口、无 a2a）——
LightX2V 的运行时接入路径（§5.1）**不适用**，必须走 compile（§5.2 的 a2a 修复也不需要）。

## 2. 使能方法（compile 接入 3 处改动）

> 完整代码见 diffsynth-engine-notes.md §2；此处只列**性能相关要点**。

1. **必须写入 `_compiled_call_impl`**（等价 `nn.Module.compile()`）：
   `torch.compile(submodule, backend=...)` **不赋值不生效**——
   直接调用返回包装对象不改模块，warmup 与 eager 一致、pattern 0 命中
2. **RoPE 实数域改写**：`apply_rotary_emb_qwen(use_real=False)` 原实现是复数域
   （`view_as_complex`）→ 图内 fp32 复数岛 + `qwen_rope_pattern` 不命中。
   改实数域等价（`unbind → mul/add → stack → flatten`）才命中
3. **text encoder key 归一化**：transformers>=5.x 布局与 4.x 权重不匹配 →
   注入 key_mapping 才能加载（正确性前提，非性能）

## 3. kernel 级验证（三层证据）

### 3.1 使能集合（与 dummy run 一致，实测）

| 融合 pattern | 融合 kernel | dummy run | DiffSynth-Engine |
|---|---|---|---|
| qwen_rope | `RotaryPositionEmbeddingV2` / `npu_rotary_mul` | ✅ | ✅ |
| qk_norm RMSNorm | `RmsNorm`（npu_rms_norm） | ✅ | ✅ |
| 残差 gate | `residual_gate_add_kernel` | ✅ | ✅（仅 3D 站点） |
| 调制 | `AdaLayerNormV2` | ✅ | ✅ |
| GELU | `FastGelu` | ✅ | ✅ |

> ⚠️ 使能判断**不依赖层数/权重**（pattern 匹配基于图结构）→ 用 dummy run（2 层）
> 即可验证"外部框架下 pattern 是否使能"，真实权重（60 层）直接复用结论。

### 3.2 三个陷阱

1. **日志 2048 截断**：`MINDIE_LOG_LEVEL=DEBUG` 的 graph 行被 `MAX_LOG_STRING_LEN=2048`
   截断 → grep 搜不到融合 kernel 会**误判 0 命中**。以 `graph_log_url` 落盘 DOT
   （需 `pip install pydot`）或 kernel_details.csv 为准
2. **图命中 ≠ 运行期全部生效**：`residual_gate_add` 对 4D attention 张量运行期
   fallback（日志 `fallback (ndim)`）→ 看 kernel_details.csv 中融合 kernel 的
   **实际执行次数**（3D 站点融合、4D 站点原生）
3. **不做耗时比较**：dummy（2 层 0.72B）vs 真实（60 层 20B）耗时由层数主导，
   dummy 编译的融合收益被新增 kernel 淹没 → "dummy 反而更慢"是假象。
   耗时评估必须真实权重 + kernel diff

## 4. 与 LightX2V 的关键差异（照搬清单）

| 做法 | LightX2V（有效） | DiffSynth-Engine（是否需要） |
|---|---|---|
| a2a 留 eager（`torch._dynamo.disable`） | ✅ 必须（Ulysses a2a 在 block 内） | ❌ 不需要（无 a2a，单卡） |
| backend 实例复用 | ✅ 必须 | ✅ 同样必须（防 BACKEND_MATCH 重编译） |
| swiglu 双 split 变体 | ✅ 必须（chunk(2) 展开成双 split） | ❌ 按实际图形态验证（Qwen 无 swiglu） |
| 运行时注册表接入（rms_type/rope_type） | ✅ 有接口 | ❌ 无接口，走 compile |

> 结论：**backend 实例复用是跨框架通用**；a2a/pattern 形态是**框架特定**，必须各自验证。

## 5. 实测结论（真实权重 60 层，已完成）

- 使能集合与 dummy 一致 → 融合 kernel 实际生效（rope/rms/gate/adaln/gelu），
  kernel diff 显示融合 kernel 出现（`residual_gate_add_kernel` / `AdaLayerNormV2` /
  `FastGelu` / `RotaryPositionEmbeddingV2` / `RmsNorm`）且逐元素 `Mul`/`Stack`/`Cat`
  大幅减少——融合收益在 kernel 级可见。
- 单卡无通信 → 收益完全来自算子融合（无 LightX2V 的通信重叠红利）。
- 端到端（真实权重 1024²，latent 输出）：compile 相对 eager 有稳定正收益
  （per-step 约 5% 量级，warmup 后稳态测量），输出正确性保持
  （latent 与 eager 高相似、VAE 解码图像肉眼一致）。
- ⚠️ 结论仅在"该框架 + Qwen-Image + 该容器环境 + 该分辨率"成立；换配置须按
  「使能与验证回路」重验（dummy 只判使能，不判收益）。

> 过程细节与完整数据不外泄：本文只给方向性结论与相对量级；详细逐步耗时 /
> 显存 / kernel 明细属内部测量记录，不入库（见 .agents 治理与隐私约定）。

## 6. 复核与扩展结论（同窗口冷卡口径，收益声明纪律）

> 收益声明纪律：全部为同窗口同卡组、中位数口径；<3% 视为噪声不宣称。
> MatMul/addmm 占 kernel 总耗时 ~67%、FA ~15%——两者不参与 pattern 融合，
> 无损融合的绝对上限即"norm/rope/激活/元素级"区段（本模型 ≈5–6% 量级）。

### 6.1 无损基线复核（8 步×5 次中位数，冷卡）

- eager ~571.8 ms/step vs compile all-on ~538.6–552.2 ms/step（≈ −5.8%，与 §5 −3~5% 一致）。
- ⚠️ 热卡/同机其他 vllm 服务争用会把 per-step 拉到 2 倍+（如 1200ms 级失真值）——
  **基线必须冷卡 + 同窗口 + 中位数**，否则收益方向都可能判反。

### 6.2 逐 pass AB（单变量 disable，8 步中位数）

| disable | 相对 all-on | 结论 |
|---|---|---|
| qwen_rope | 变慢 ~5–7%（最大单项） | rope 融合保留 |
| wan_residual_gate | 变慢 ~1–2% | 3D gate 净正（4D 误匹配有噪音，见 §6.4） |
| fast_gelu / rms_norm | 变慢 ~1–2% | 保留 |
| adaln×3（wan+norm_out） | 变慢 ~2–3% | 保留 |

- 逐项均净正、无负贡献 → 无损融合已近 floor。

### 6.3 attention 进图编译：实测中性→回退（kernel diff + interleaved A/B）

- 姿势：`USPAttention.forward` 的 `@torch.compiler.disable` 仅在 SP 多卡路径保留
  （collective 留 eager），单卡路径放行进图。kernel 级 −0.4%（仅边界 copy 小减），
  FA 本体 kernel 不变（eager 与图内均为同一 fused FA）；interleaved A/B 墙钟中性~略慢。
- 附加风险：Ascend950 上 `npu_fusion_attention` 被 compile 后 seed/offset 可能被优化，
  固定 seed 时结果可能偏离 eager（torch_npu 官方警告）。
- 结论：**单卡 attention 进图无收益且引入 seed 语义风险 → 回退**；FA 区段收益
  应找 FA 算子侧/布局侧，不是"把它编进 compile 图"。

### 6.4 residual_gate_add 4D fallback（噪音，非致命）

- 现象：text-stream rope（动态 s31）的 `out_imag=xr*sin+xi*cos` add 链被通用
  `x + y*gate` pattern 误匹配（qwen_rope 只吃掉 image 静态 rope），运行期 4D
  `[B,S,H,D//2]` 触发 triton 3D-only fallback + 每调用一次 `print(flush=True)`
  → 每次推理 ~240 次 python dispatch + 日志刷屏（量级约 1–3% 墙钟）。
- 处置建议（按需）：qwen_rope pattern 补 text 动态 rope 变体（吃回子图）优先于
  给 residual_gate_add 补 4D kernel；或 fallback print 限流。当前影响小，优先级中低。

### 6.5 有损使能（S4）—— 用库现有能力，未新增框架功能

- 背景边界：DSE 框架本身没有有损特性（量化只匹配 nn.Linear，DSE 主算力为自定义
  TP linear；稀疏无抽象接口）→ 按用户指示**不向框架添加任何新特性**，有损验证全部
  使用 mindiesd **库现有 CacheAgent 能力**（DiTBlockCache / AttentionCache）在 bench
  脚本侧接入（DSE 源码零改动，git status + 远端 grep 双核实）。

DiTBlockCache 参数矩阵（CFG-off 单调用路径，compile 262.7 ms/step 基线）：

| 档位 | per-step (ms) | vs compile | 质量门禁（latent cos / 像素 mean-abs/255） |
|---|---|---|---|
| [30,60) ss2 iv2 | 213.8 | −18.6% | 0.991 / 4.52 pass |
| [15,60) ss2 iv2 | 190.8 | **−27.4%** | 0.991 / 4.60 pass（收益最大） |
| [30,60) ss4 iv2 | 239.0 | −9.0% | 0.999 / 2.42 pass（质量最优档） |
| [30,60) ss2 iv3 | 198.1 | −24.6% | 0.983 / 5.99 pass |
| [45,60) ss2 iv2 | 239.9 | −8.7% | 0.991 / ~4.6 pass |

计数契约：reuse=90/compute=390 = 3 复用步 × 30/60 block（与期望一致，非 no-op）。

- **AttentionCache [30,60) ss2 iv2**：216.5 ms/step（−17.6%），latent cos 0.997、
  像素 2.90/255（质量最优方法，粒度更细）。
- **无损/有损叠加复核（off-identity）**：eager+DiT cache = −24.1%、eager+AttentionCache
  = −18.1%，latent 偏差与 compile+同档一致 → 无损/有损可独立叠加、互不抵消；
  cache=0 档复现基线（disable 恢复）。
- **双缓存叠加 = fail-closed**：DiT+Attention 同时使能墙钟看似更快（176.4）但 latent
  cos 0.822（接受线 ~0.98）→ 假加速；DiT 复用步跳过 block 使内层 AttentionCache 计数
  失步。两缓存方法**互斥**，只能二选一（docs 建议 DiTCache 优先、AttentionCache 备选）。
- **CFG-on 生产路径 + DiTBlockCache = shape 冲突**（pos/neg 文本长度 12 vs 14 → 缓存
  delta 形状不匹配）：库语义约束（cache 假定每 denoise step 单次调用 shape 恒定），
  非 bug——需按分支独立 agent / text padding / batch-concat（框架集成项，未做）。
- 最优组合（综合收益×质量）：compile + DiTBlockCache [15,60) ss2 iv2 = 190.8 ms/step
  （vs eager 281.2 的 −32.1%）；质量敏感选 AttentionCache 或 DiT ss4 档。

> 过程细节与完整数据不外泄：本文只给方向性结论与相对量级；详细逐步耗时 /
> 显存 / kernel 明细属内部测量记录，不入库（见 .agents 治理与隐私约定）。
