# Baseline 对照 Profile（契约 + 判定 + 指针）

有损质量对照必须使用「可比 target 基线」：同一模型、同一推理入口、同一提示词集/seed/
分辨率/帧数/步数，只允许优化项不同。具体模型 profile（`profiles/{model}.toml`）
**不提交仓库**：由流程在 S0 冻结基线后经 `scripts/gen_profile.py` 生成到
`runs/{task_id}/profiles/{model}.toml`，close 前由 `scripts/check_profile.py` 强校验。

## Profile 只存三类内容（2026-09 重构）

| 内容 | 是否入库 | 说明 |
|------|----------|------|
| **契约**：model/geometry/steps/seed/prompts/入口占位 | ✅ | 复现所需，跨运行不变 |
| **判定结论**（不可推导的一次性语义判定） | ✅ | `[domain]` 归类依据、visual pass/fail/inconclusive、off-identity 结论、"弱视觉域无绝对门槛"等语义 |
| **指针**：冻结 baseline 引用 + `last_verified` | ✅ | runs 目录 + md5/hash + 日期 |
| **可推导数值**（psnr/ssim/delta/md5 等） | ❌ **不入库** | 由 `scripts/quality_compare.py` 对 `runs/` 冻结产物**现算**，存 `{run}/quality.json`；每次运行的标杆质量信息**强制随报表登记**（detail-report §E，见 .agents 侧） |

> 原则：仓库里 profile 是"契约 + 判定记录 + 指针"，数值永远可复现、不漂移；
> 与 `artifact-layout.md`「仓库零数据」（帧/媒体/原始产物只留 runs/，evidence.json 只存指针）一致。

## 模板

仓库唯一 profile 形态 = `profiles/_template.toml`（契约骨架）。生成具体模型 profile：

```bash
python evals/scripts/gen_profile.py --model {model} \
    --task-dir runs/{task_id}_{model}_optimization \
    --domain video_chaos|image_seed_deterministic \
    --resolution {WxH} --frames {N} --steps {N} \
    --topology <同拓扑同卡组> --frozen-hash <baseline 帧集 md5> \
    --seed {N} --off-identity <pass/说明>
```

字段语义与骨架见 `_template.toml` 注释（必填 section/字段、domain 协议、
数值不预填原则）。模板到生成物的字段映射由 gen_profile.py 实现，改模板须同步
gen_profile.py 与 check_profile.py 的 REQUIRED_* 校验。

## 约定

- 定量指标失败可 `explicitly_deferred`，但报告必须说明原因并附并排对照产物，不允许静默跳过。
- 换分辨率/步数/seed = 新对照；禁止跨 profile 混比。
- 基线帧冻结后不再改动；config 帧与 baseline 帧索引一一对应
  （`evals/scripts/quality_compare.py` 按文件名或排序索引对齐）。
- 阈值未校准前不得宣称「质量通过」；弱视觉域（video_chaos）只做基线登记与回归对比，
  不设绝对门槛（首个案例 minimax-h3 校准经验，2026-09-05）。
- **图像模型取样约定（2026-09-05/06 第二案例校准，Qwen-Image-2512 × vllm-omni，V3）**：
  图像无帧轴 → 质量对照用**固定 21-seed 同 seed 像素对**（同 seed 生成确定性 → 每 seed 1 张；
  21 seed 均化统计，目录内文件名 sNNNN.png 对齐，quality_compare.py 按名配对），等价于视频的
  21 采样帧；**图像同 seed 像素质量域（如 w8a8 PSNR 37.5/SSIM 0.986）远高于视频档（H3 19.6/0.679，
  无轨迹混沌）→ 阈值/数值域不可跨任务类型迁移**；`domain` 字段即为此归类，驱动协议自动选择。
- **每次运行的标杆质量信息 = 报表强制项**：有损/组合每档的质量数值与判定结论随
  `overview_report.md` 质量列 + `detail_report.md` §E 逐行登记（数值现算引用
  runs/{id}/quality.json；判定结论引用本 profile [decisions]）——缺质量证据的运行不得宣称
  quality pass，质量证据缺失视为交付缺失。
- 首个校准案例：MiniMax-H3 × vllm-omni（2026-09-05，950PR，video_chaos 域）与
  Qwen-Image-2512（2026-09-06，image_seed_deterministic 域）的**校准结论已沉淀到
  本节约定与 case 文件**；其运行时 profile 由流程按 `_template.toml` 生成（不入库），
  历史入库版 `profiles/minimax-h3.toml` / `profiles/qwen-image-2512.toml` 已废弃删除。
  新模型 profile 参照它建（契约 + domain + decisions + last_verified；数值不预填）。
