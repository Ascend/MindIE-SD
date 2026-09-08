# 三方框架使能 mindiesd 有损缓存（CacheAgent/DiTBlockCache/AttentionCache）通用姿势

> 来源：DiffSynth-Engine（Qwen-Image）真实权重实测提炼。适用于"三方框架本身没有有损特性、
> 但 mindiesd 库自带 CacheAgent 缓存能力"的场景——**优先用库现有能力、bench/脚本侧接入，
> 不向框架添加新特性**（用户指示的边界，本文件是方法论，不是代码合入指南）。
> 硬件：Ascend 950PR 单卡；结论仅在该框架+模型+分辨率成立，换配置须按
> framework-feature-enablement「使能与验证回路」重验。

## 1. 何时用 / 何时不用

- **用**：框架在 denoise 循环里按 block 顺序调用重复块（如 60 层 DiT），希望用
  缓存跳过相邻 step 的冗余计算；mindiesd 工作区有 `cache_agent/`（CacheAgent/CacheConfig/
  DiTBlockCache/AttentionCache）。
- **不用/替代**：量化、稀疏 FA 等在框架侧无抽象接口或无量化资产时——不硬接，
  记录阻塞证据即可（不宣称）；框架已有原生缓存则优先框架原生。

## 2. 接入姿势（三要素）

1. **CacheConfig 语义对齐**：`blocks_count = 每 denoise step 的 block 数`，
   `steps_count = denoise step 数`。⚠️ **CFG 双分支（pos/neg 各一次 transformer 调用）
   会让 CacheAgent 计步翻倍** → 用 CFG-off 单调用路径验证（1 transformer 调用 ==
   1 cache step）；CFG-on 生产路径会因 pos/neg 文本长度不同触发缓存 delta shape 冲突
   （库假定每 step shape 恒定，非 bug）。
2. **返回值约定换序**：DSE 类框架 block 返回 `(encoder_hidden_states, hidden_states)`，
   DiTBlockCache 约定 `res=(hidden, encoder)` → 包一层 adapter 换序（胶水，不动框架源码）。
3. **bench 侧接入**：在验证脚本里替换 `model.transformer_blocks` 为 adapter ModuleList
   （或包 block 的 attention 模块），DSE 源码零改动；compile（block 级 `_compiled_call_impl`）
   保持不变。

## 3. 验证三件套（缺一不可，防假加速）

1. **墙钟**：同窗口同卡组、8 步 × 3–5 次 hot 取中位数；分母 = compile 基线（不是 eager）。
   热卡/同机其他服务争用会拉出 2 倍+ 失真值 → 冷卡 + 同窗口。
2. **计数契约（真实性核验）**：直接统计库 apply 路径 reuse/compute 次数，与理论期望比对
   （如 3 复用步 × 30/60 block ⇒ reuse=90 / compute=390）。**计数对不上或为 0 = no-op，禁止宣称**。
3. **质量门禁**：latent cosine（接受线 ~0.98）+ VAE 解码像素 mean-abs（参考 eager-vs-compile
   无损基线 ~5.6/255）；方法见 performance-optimization `references/quality-gate.md`。

## 4. 实测结论（Qwen-Image 60 层，CFG-off 单调用，compile 基线 262.7 ms/step）

- DiTBlockCache 参数矩阵：窗口 [15,60)/[30,60)/[45,60)、step_start {2,4}、interval {2,3}
  → 墙钟 −8.7% ~ −27.4%，全档质量门禁过（cos 0.983–0.999 / 像素 2.4–6.0）。
- AttentionCache [30,60) ss2 iv2 → −17.6%，质量最优（cos 0.997 / 像素 2.90）。
- 收益/质量权衡：窗口越大收益越大（[15,60) −27.4%）、step_start 越晚质量越好（ss4 像素 2.42）。
- 无损/有损可叠加（eager+cache 与 compile+cache 偏差一致）→ 无损项叠加复核通过。
- **双缓存（DiT+Attention）互斥**：同时使能墙钟看似更快但质量 FAIL（cos 0.822）——
  DiT 复用步跳过 block 使 AttentionCache 计步失步。二选一，DiTCache 优先、AttentionCache 备选。

## 5. 常见坑速查

| 症状 | 原因 | 处理 |
|---|---|---|
| CFG-on 报 tensor size mismatch | pos/neg 文本长度不同 → 缓存 delta shape 冲突 | CFG-off 路径验证；生产需分支独立 agent / padding / batch-concat（框架集成） |
| 墙钟快但 latent cosine <0.98 | 双缓存叠加 / 计数失步复用错步 | 只开一种缓存；看计数契约 |
| 日志刷屏 residual_gate fallback | text-rope 动态子图被通用 pattern 误匹配（4D） | qwen_rope 补 text 变体 / fallback print 限流（低优先） |

## 维护与更新

当 mindiesd cache_agent 接口或跨框架姿势变化、出现新的缓存组合约束时，按 dev-workflow 复盘更新；
框架特定数字回填对应框架 case（如 diffsynth-engine-case.md §6），本文只保留通用姿势。
