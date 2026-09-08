# compile vs 非 compile（eager）双报表模板

> dummy-run §C 强制项配套模板。每次 compile vs eager 收益对比必须按 §C 口径产出
> overview + detail 两张表。本文件给出命令、口径与 MiniMax-H3 示例（示例数值为
> kernel_details 家族聚合的**示意近似 ✻**，正式报表按站点映射精确填写；数据源见文末）。

## 0. 采集与聚合命令（口径统一）

```shell
# eager：NR_PROFILE=1 跑 minimax_h3_infer.py --quant w8a8（无 --compile）
# compile：同配置 + --compile
# 各自 kernel_details.csv → 按算子家族/站点聚合耗时（us→ms）
# block 基准 = eager transformer timed（ms）
```

## 1. 总览表（overview：按融合算子一行）

| 序号 | 融合算子 | 融合前组成（eager 被替代链） | 是否完成融合 | 融合前耗时(ms) | 融合后耗时(ms) | 相对融合前 block 耗时的收益 |
|---:|---|---|:--:|---:|---:|---:|
| 1 | npu_rms_norm | Pow+Mean+Rsqrt+Add/Mul 分解链（eager 多 kernel 聚合 ✻1.9） | Y | 1.90 | 0.46 | -6.5% ✻ |
| 2 | npu_rotary_mul（RoPE 融合） | Slice/neg/cat/Mul/Add + Copy 物化链 ✻1.2 | Y | 1.20 | 0.46 | -3.3% ✻ |
| 3 | gather_scale_shift（AdaLN） | index_select + Mul/Add 调制链 ✻1.5 | Y | 1.50 | 0.68 | -3.7% ✻ |
| 4 | gather_residual_gate | index_select + Mul/Add ✻1.2 | Y | 1.20 | 0.36 | -3.8% ✻ |
| 5 | FFN act（swiglu）并入 FUSED | split/chunk→silu→mul ✻0.3 | Y（并入 mm_swiglu_mxquant） | 0.30 | 0.00 | -1.4% ✻ |
| 6 | mm_swiglu_mxquant（FFN hidden：mm+swiglu+mxquant） | Qmm([S,2F])→swiglu→DxQ ✻2.03 | Y | 2.03 | 1.85 | -0.8% ✻ |
| 7 | out-proj 输入 DxQ 消减 | DynamicMxQuant（FFN out A 量化）✻0.25 | Y | 0.25 | 0.00 | -1.1% ✻ |
| 8 | —（占位示例：预期可融合未实现） | 示例：AdaLN 调制与 FFN 同 kernel 化 | N（预期可融合：epilogue 侧） | 0.00 | 0.00（=融合前） | 0（未实现不虚填） |

> ✻ 示例数值为 kernel_details 家族聚合近似；正式报表必须按「图 dump 站点→kernel」映射后
> 以站点口径填写，并附聚合脚本与数据文件路径。收益分母 = eager transformer timed（22.19ms）。

## 2. 明细表（detail：按 block 算子执行序列行）

| 算子归属（Attn/FFN(MoE)） | 融合后所属算子 | 未融合时的组成 | 融合后性能(ms) | 未融合性能(ms) | 相对未 compile block 耗时的收益 |
|---|---|---|---:|---:|---:|
| FFN | mm_swiglu_mxquant（FFN hidden） | DxQ→Qmm([S,2F])→swiglu→DxQ→Qmm(out) | 1.85 ✻ | 2.03 ✻ | -0.8% ✻ |
| FFN | npu_quant_matmul（out-proj，输入已 fp8） | 同上行末 Qmm（原 out DxQ 已消减） | 0.83 ✻ | 0.95 ✻ | -0.5% ✻ |
| Attn | npu_dynamic_mx_quant + QuantMatmulV5（q/k/v） | 同左（GEMM 未融合，compile 不变） | 1.25 ✻ | 1.25 ✻ | 0% |
| Attn | FlashAttentionScoreV4 | 同左（FA 未融合） | 2.56 ✻ | 2.56 ✻ | 0% |
| Attn | npu_rms_norm（qk_norm/norm1） | Pow+Mean+Rsqrt+Add/Mul 分解链 | 0.13 ✻ | 0.55 ✻ | -1.9% ✻ |

> detail 行按 block 内算子执行序排列（可合并同单元连续实例并标注 ×N）；GEMM/FA 等未融合
> 行"未融合时的组成=原算子名"、收益 0。收益分母 = eager block 耗时（22.19ms）。

## 3. 数据来源（示例）

- MiniMax-H3 w8a8（FFN 融合**默认开启**，diffusers 0.40）：eager 21.77ms / compile 15.65ms
  （-28.1% wall；kernel_sum 22.92→16.60ms，kernel 数 484→338）；eager/compile 两份
  kernel_details.csv 见 `tmp/mmx_w8a8/fusion_default_{eager,compile}_kernel_details.csv`，
  聚合脚本 `tmp/mmx_w8a8/agg_fusion_default.py`；完整 §C 双报表见
  `tmp/mmx_w8a8/dummy_ab_reports_mmx_fusion_default_on.md`。
- diffusers 0.40 全模型 wall（三模型并行，一模型一卡，2026-09-06）：Wan2.2 951.49→841.76
  （-11.5%）、MiniMax-H3 21.77→15.65（-28.1%）、FLUX.1-dev 17.2→15.14（-12.0%）。
- 收益归因方法论（GEMM/FA 不变、小 kernel 融合）：
  `../compilation-dev/references/pattern-dev-notes.md` §5。

## 维护与更新

口径/分母/模板字段变更时更新 dummy-run/SKILL.md §C 与本文；新增真实填表示例时替换 ✻ 占位。
