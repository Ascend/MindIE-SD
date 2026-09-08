# manifest 运行契约与 dry-run（可复现门禁）

> 用途：把每次模型优化的「运行契约」结构化（`manifest.toml`），并在 **占用 NPU 之前**用 dry-run 校验
> 自洽性与可复现性（schema / 枚举 / 路径 / 特性 seam / 渲染计划）。配合
> `overview-report.md`（基线=TP 多卡未优化）保证数字可复现。
> 仓库零数据：manifest 只描述与指路（版本/指针/env/特性组合），产物数据（profile/帧/日志）保留在
> 远端 `runs/{date}_{model}_optimization/`，不入 git。

## 1. 与 dummy-run 的区别

| | dry-run（本文件） | dummy-run（dummy-run skill） |
|---|---|---|
| 执行 | 结构性预检：解析/渲染/静态判定 | 真实执行：随机权重/精简模型上 NPU 跑通 |
| 消耗 | 零 NPU 占用、零权重 | 需 NPU（少层数） |
| 回答 | 「配置是否自洽、可复现、无 seam 冲突」 | 「模型架构/算子是否真能跑」 |
| 顺序 | 先（每次运行/闭环前门禁） | 后（S0 出基线/验证接入姿势） |

## 2. manifest.toml schema（最小字段）

```toml
[meta]
kind = "optimization"          # optimization / smoke / control
model = "Wan2.2-T2V-14B"
framework = "vllm-omni"
framework_version = "0.28"
mindiesd_commit = "{git}"
purpose = "closed_loop"        # closed_loop / frontier / evidence / blocker_probe

[run]
topology = "TP2xUSP2"          # 基线口径：TP 多卡未优化（说明列标注 fallback 时写单卡）
gpus = "0,1,2,3"
baseline_run = true            # 本 manifest 是否为基线运行

[features]
enable = ["quant_w8a8_dynamic", "sparse_rf_v2", "cache_dit"]   # 必须能被 seam_check 通过
disable = []

[env]
PYTHON_BIN = "{ptr}"
MODEL_PATH = "{model_weight_dir}/{model}"
SEED = "1101"
FEATURE_FLAGS = "..."          # 框架侧开关/environment（渲染器转成启动命令）

[thresholds]
psnr_db = 16.0                 # 引用 evals/profiles/{model}.toml
ssim = 0.51

[off_identity]
required = true

[artifacts]
profile_dir = "<远端 runs/.../stepN_profile>"   # 仅指针
output_media = "<远端 runs/.../out.mp4>"
```

## 3. dry-run 做什么（`scripts/manifest_dryrun.py`）

1. schema 与枚举校验（kind/purpose/topology/features 合法）；
2. 特性 seam/能力检查（复用 `scripts/seam_check.py` + `feature_declarations.json`）；
3. 引用的仓库/远端路径存在性（只读检查，不产生数据）；
4. 按框架渲染「启动计划」（env + 入口提示），打印计划文本；
   - 渲染器当前覆盖：`vllm-omni` / `lightx2v`（其余框架报 unsupported，不做静默猜测）；
5. 全绿才允许进入真实运行/闭环复验。

依赖：python ≥3.11（tomllib 标准库）；声明数据 `feature_declarations.json`；**不依赖 NPU/权重/框架运行**。
局限：dry-run 不做真实启动与耗时；真实命令以各框架 case 的入口为准，渲染器只做结构化提示。

## 4. 仓库零数据与版本绑定

- evidence.json 存 `source_binding`：mindiesd/framework commit、framework_version、manifest 指针、
  profile 名、远端产物目录指针（见 artifact-layout.md 模板）——**全部是指标/指针，不含数据**。
- 帧/媒体/原始 profile/日志只存在于远端产物目录，git 只提交文档/代码/元数据。

## 维护与更新

schema/枚举变化、框架升级或新框架接入 → 更新本文件 + `manifest_dryrun.py` 渲染器 +
`feature_declarations.json`；特性命名以 docs/zh/features 为准。
