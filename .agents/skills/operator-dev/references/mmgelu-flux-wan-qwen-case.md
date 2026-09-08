# 案例：mm_gelu_mxquant（FLUX/Wan/Qwen FFN 融合）实战细节

> 2026-09，A310-50（A5/MXFP8），diffusers 0.40（qwen 0.38）。与 h3 `mm_swiglu_mxquant`
> 同族；工作流见 `catlass-ffn-fusion-guide.md`。数据/报告：`tmp/wanflux_ffn/`（ge_hit、
> compilation_ffn_fusion_report.md、phase1_3_delivery_report.md）。

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

| 模型 | off→on (ms) | 收益 | fused | gelu(残) | Qmm | 验证 |
|---|---:|---:|---:|---:|---:|---:|
| FLUX.1-dev | 15.24 → 14.37 | −5.7% | ×4 | 2 | 50→46 | PASSED |
| Wan2.2 | (≈841.8) → 835.05 | ≈−0.8% | ×2 | 1 | 24→22 | PASSED |
| Qwen-Image | (≈7.01) → 6.72 | ≈−4% | ×3 | 0 | 31→28 | PASSED |

- 计数契约/off-identity ✓（off: fused 0/gelu 6；on: fused 4/gelu 2）。
- 未覆盖：flux single-block（2，act_mlp→concat→proj_out 非 FeedForward）、wan refiner（1）——
  §C 未实现行标注。
- 数值 fp8 级近似（smoke：no-bias 98.3%）；真机质量门未跑（报告标注）。

## 5. 工程坑速查

- 同步同 basename 的 ops/plugin cpp 互相覆盖（上传用不同暂存名）。
- 共享远端 .so 可能被他方构建（不同 torch ABI）覆盖 → 复现前 `check_mindie_operator_exists`。
- qwen-image diffusers 0.40 `QwenEmbedRope` str-device bug → 用 0.38。
- profile 并行必须 `--profile-dir` 隔离。
