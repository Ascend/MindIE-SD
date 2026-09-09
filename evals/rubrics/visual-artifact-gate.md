# 视觉伪影门禁判定标准（Visual Artifact Gate）

> 来源改编：跨栈视觉伪影判卷方法论模板
> （判定方法与时序规则与硬件无关，仅借结构）。本文件为 MindIE-SD
> （昇腾 NPU / 多模态扩散，图像与视频模型）适用版。

## 用途与输入

比较 config（优化后）与官方/冻结 baseline 的端到端产物，判定是否引入**新的、可感知的**
视觉伪影。输入：

- baseline 采样帧 + config 采样帧（**帧索引必须对齐**；同 seed/提示词/分辨率/步数，见
  `evals/profiles/README.md`）
- 可选并排视频或并排采样帧
- config manifest 与性能摘要

> 按帧序列观察（时序输入），不把独立截图当结论：相邻帧的闪烁/跳变/块边界 popping
> 即使单帧看起来正常也属于时序伪影；块边界只在运动时可见，仍须报告
> `patch_boundary_discontinuity` 与 `temporal_flicker_popping`。

## 伪影类别

**视频模型判全部 12 类；图像模型只判非时序 8 类（剔除带 * 的时序类）。**

| key | 类别 | 适用 |
|---|---|---|
| snow_static_speckle | 雪花/静态噪点 | 图像 + 视频 |
| blur_detail_loss | 模糊/细节丢失 | 图像 + 视频 |
| mosaic_blocking_patch_artifacts | 马赛克/块状伪影 | 图像 + 视频 |
| patch_boundary_discontinuity | 分块边界不连续/拼贴纹理错位 | 图像 + 视频 |
| banding_posterization | 色带/色阶断层 | 图像 + 视频 |
| oversaturation_color_shift | 过饱和/色偏 | 图像 + 视频 |
| ghosting_smearing | 鬼影/拖影 | 图像 + 视频 |
| melting_morphing_structure | 结构融化/形变 | 图像 + 视频 |
| degraded_text_faces_hands | 文字/人脸/手部退化 | 图像 + 视频 |
| composition_or_motion_regression | 构图回归（视频另含运动回归） | 图像 + 视频 |
| temporal_flicker_popping * | 帧间闪烁/跳动/不稳定明暗细节 | 仅视频 |
| loss_of_temporal_coherence * | 时序连贯性丢失（运动卡顿/碎裂/鬼影化） | 仅视频 |

## 判定输出（JSON）

```json
{
  "overall": "pass | fail | inconclusive",
  "new_artifacts": [
    {
      "category": "blur_detail_loss",
      "severity": "low | medium | high",
      "frame_indices": [12, 24],
      "evidence": "短句证据（位置/现象）"
    }
  ],
  "temporal_checks": {
    "flicker_or_popping": "pass | fail | uncertain",
    "patch_boundary_stability": "pass | fail | uncertain",
    "motion_coherence": "pass | fail | uncertain",
    "detail_degradation": "pass | fail | uncertain"
  },
  "baseline_notes": "baseline 现象简述",
  "config_notes": "config 现象简述",
  "recommendation": "promote | tune | reject | rerun"
}
```

## Pass 规则

- `overall=pass` 仅当**没有 medium/high 级新伪影**。
- low 级差异可通过，但必须满足：config 有明显收益，且该伪影**不属于**以下禁放行类——
  时序闪烁/popping、时序连贯性丢失、块边界不连续、马赛克/块状、雪花/静态、鬼影/拖影、
  运动断裂、重大模糊。
- 单帧正常但同一区域跨帧不一致 → 判时序伪影；仅运动时可见的块边界 → 判
  `patch_boundary_discontinuity` + `temporal_flicker_popping`。

## 判卷方式

- **VLM/多模态判卷（有则用）**：把「伪影类别」与「Pass 规则」两节作为 judge 提示，
  输入对齐帧/并排视频，要求输出上述 JSON。
- **无 VLM**：人工并排对照 + `evals/scripts/quality_compare.py` 定量结果，输出
  `inconclusive` 或人工结论，并记录依据；不得把「无 VLM」当作质量通过的等价物。
