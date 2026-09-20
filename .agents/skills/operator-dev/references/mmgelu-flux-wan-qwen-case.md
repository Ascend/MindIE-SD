# 案例：mm_gelu_mxquant（FLUX/Wan/Qwen FFN 融合）实战细节

> **本件是案例细节留档，不作为推荐加载入口**：常规路径读 `mindiesd-fusion-notes.md` §7 的集成侧要点；
> 只有在需要本文 §1–§5 的原始细节（真实图链、bias 实况、装载 API 坑、图级计数）时才读本件。
> 2026-09，目标设备档为 MXFP8 代际（`{soc_version}`，现场取）；框架/依赖版本按现场实际安装
> （版本坐标与复核方式见会话产物归档 `{run_results_dir}/archive/`）。与 h3 `mm_swiglu_mxquant`
> 同族；工作流见 `catlass-ffn-fusion-guide.md`。数据/报告：会话产物归档 `{run_results_dir}/archive/`。

## 1. 语义与真实图链

- 目标：`out = MXQuant(gelu_tanh(Qmm(x)))`，单 N=F（flux/wan/qwen 均为 FeedForward 形态：
  `net[0]=GELU(proj up), net[1]=Dropout, net[2]=Linear down`；qwen 的 img/txt_mlp 同构）。
- 真实 compile 图 FFN 站点（fusion off 时 probe 到）：
  `adaln_v2 → getitem → reshape([M,-1]) → DxQ → getitem(x1/x_scale) →
  Qmm(hidden)(x1, transpose(w1), transpose(ws1), kwargs{bias: _to_copy(fp32)}) →
  reshape([1,M,F]) → npu_fast_gelu → reshape([M,-1]) → DxQ(act) → … → Qmm(out)`。
- seam：`aten.reshape` ×2（尺寸随 M 变）+ `_to_copy`（functionalization）；register_replacement
  canonical 链 miss → GraphPatternEntry 命中（P5）。

## 2. kernel 要点

- gelu 用恒等式 `x·σ(1.59577·(x+0.044715·x³))`；mx 量化尾直接复用
  `TileSwigluAndMxQuant::ComputeMaxExp/ComputeScale/QuantToFp8`。
- **bias 实测全 0**（flux up/down Linear absmean/max=0）：kernel 可选 bias 支持按通用性保留，
  真机融合等价 no-bias（zeros-bias 对拍 98.3%）。
- bias 装载坑：GM→UB 用 `AscendC::DataCopy(Local, Global, uint32 元素数)`（count=元素数，
  内部 32B 对齐检查），同步 **MTE2_V**；曾用 MTE3_V → 竞态；曾用 DataCopyPad 3 参（gm→ub 需
  4 参带 pad）与 DataCopyParams blockLen 单位——先查 CANN header 再写。
- 判别 bias 是否加对列：常数 bias(±1)/ramp 对拍 94-100%、逐列误差均匀 → 列位正确；
  小随机 bias 低匹配来自 torch Qmm bias 数值路径差异（fp8 边界翻转），真实 bias=0 场景不受影响。

## 3. 使能（compile GraphPatternEntry）

- pattern 树：`Qmm(Arg×3) → reshape(Ignored) → npu_fast_gelu(另注 aten.gelu 变体)
  → reshape(Ignored) → DxQ(_users=MULTIPLE) → getitem0(_users=MULTIPLE) → Qmm(Arg×3)`。
- handler：从 out-proj Qmm 反向取 x1/w1/ws1/x_scale/bias，重建 fused+out Qmm（kwargs 复制）。
- 注册：`register_ffn_gelu_fusion_graph_entries` 在 `passes/__init__.py` 于
  `enable_flux_wan_ffn_gelu_fusion`(默认 True) 开启时调用。

## 4. 结果（同窗同卡，compile w8a8）

三个模型均为**正收益**：两个在**个位数百分比量级**、一个在**亚百分比量级**（同窗同卡 off→on 读数见会话产物归档 `{run_results_dir}/archive/`）。

| 模型 | 收益量级 | 计数契约 | 验证 |
|---|---|---|---|
| FLUX.1-dev | 个位数百分比 | off-identity + fused 数 == 命中站点数 | PASSED |
| Wan2.2 | 亚百分比 | 同上 | PASSED |
| Qwen-Image | 个位数百分比 | 同上 | PASSED |

- 计数契约/off-identity ✓：**off 臂 fused 计数 = 0 且原链计数不变；on 臂 fused 数 == 命中站点数**
  （各模型的具体计数见会话产物归档 `{run_results_dir}/archive/`）。
- 未覆盖：flux single-block（2，act_mlp→concat→proj_out 非 FeedForward）、wan refiner（1）——
  §C 未实现行标注。
- 数值 fp8 级近似（smoke：no-bias 98.3%）；真机质量门未跑（报告标注）。

## 5. 工程坑速查

- 同步同 basename 的 ops/plugin cpp 互相覆盖（上传用不同暂存名）。
- 共享远端 .so 可能被他方构建（不同 torch ABI）覆盖 → 复现前 `check_mindie_operator_exists`。
- qwen-image 在该 diffusers 版本线上有 `QwenEmbedRope` str-device bug ⇒ 降档到可用版本（版本坐标与复核方式见会话产物归档 `{run_results_dir}/archive/`）。
- profile 并行必须 `--profile-dir` 隔离。

## 6. 维护与更新

- **触发条件**：本件声明的数据/报告先出库到会话产物归档 `{run_results_dir}/archive/` 后有更新；
  §4 结果表的收益量级与命中站点数在 FLUX.1-dev /
  Wan2.2 / Qwen-Image 任一新跑中变化；§1 站点链、§2 kernel 要点、§3 pattern 树随
  `catlass-ffn-fusion-guide.md` 与 `mindiesd-fusion-notes.md` §7 更新。
- **复核方法**：逐条比对 `mindiesd-fusion-notes.md` §7 与本文 §1–§5，
  并核对本文 §4 计数契约（off 臂 fused=0 且原链计数不变；on 臂 fused 数 == 命中站点数）与 §3 的
  `enable_flux_wan_ffn_gelu_fusion` 注册路径；同时按 `.agents/README.md` §7 的 `-case.md` 政策核对
  本件在 SKILL.md 的登记状态与实测数字出库情况。
