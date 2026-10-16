# 三方框架模型权重准备（modelscope 默认优先，下载到远端容器）

> **目录** · [1. 适用场景与结论速览](#1-适用场景与结论速览) · [2. 下载源优先级与分区落位](#2-下载源优先级与分区落位) ·
> [3. 下载命令（modelscope CLI）](#3-下载命令modelscope-cli) · [4. 后台化 + 监控](#4-后台化--监控ssh-断开不中断) ·
> [5. 完整性校验](#5-完整性校验) · [6. 已知坑](#6-已知坑) · [7. 权重就绪后交接](#7-权重就绪后交接服务路径语义) · [维护与更新](#维护与更新)
>
> 使用任何三方框架（vLLM-Omni / LightX2V / DiffSynth-Engine / diffusers）跑**真实权重**前的
> 前置步骤：把模型权重从 **modelscope（默认）** 下载到远端昇腾容器的模型根目录并校验完整。
> 依据：2026-08 实测 MiniMax-H3 T2VA（FL2VA 分区 ~134 GiB，16 并发，**数小时量级**完成，
> 81 文件全部校验通过）。本文件是**通用方法**，适用于所有三方框架；具体模型的分区/格式
> 差异按 §2 判定。

## 1. 适用场景与结论速览

- **何时需要**：三方框架（vLLM-Omni / LightX2V / DiffSynth-Engine / diffusers）需要真实权重
  时；dummy run（随机权重）不需要，见 dummy-run。
- **下载源优先级**：**默认 modelscope**（国内可达、免代理；HF gated 仓库在 modelscope 镜像通常
  免鉴权），HuggingFace / 其他 gated 仓库作**次选**（需 token）——判据见 §2.1。
- **目录约定**：`{model_weight_dir}/{模型名}/{任务变体}/`（如 `{model_weight_dir}/MiniMax-H3/FL2VA`），
  模型根目录直接 serve；各模型实测落位见 §2.2 落位表（**未实测的格子留空，不推测**）。
- **核心命令**（详见 §3）：
  `modelscope download {模型} --local_dir {root} --include '{分区}/**' --max-workers 16`

## 2. 下载源优先级与分区落位

### 2.1 下载源优先级（默认 modelscope）

1. **modelscope（默认首选）**：国内可达、无需代理；**HF gated 仓库在 modelscope 镜像通常免鉴权**
   （如 `MiniMaxAI/MiniMax-H3`（HF，需审批）↔ `MiniMax/MiniMax-H3`（modelscope，直接下））。
   两种用法（落位约定一致，都要求**直落模型根目录**）：
   - CLI：`modelscope download {repo_id} --local_dir {root} --include '{分区}/**' --max-workers 16`
     （参数要点见 §3）；
   - Python：`snapshot_download('{repo_id}', local_dir='{root}', allow_patterns=['{分区}/*'])`
     ——⚠️ 若改用 `cache_dir=` 会得到 snapshot 哈希嵌套目录（服务路径会变、难排查），
     **权重落位一律用 `local_dir`**。
   - 实测聚合速率为**个位数 MB/s 量级**（16 并发），单分区 134 GiB 需**数小时量级**完成
     （规划输入，非性能宣称；具体速率/耗时见会话产物归档，换机器/换网络必须重测）。
2. **HuggingFace / 其他 gated 仓库（次选）**：modelscope 无镜像、或必须用 HF 侧特定 revision 时。
   gated 仓库须先拿访问授权并带 token：
   - 登录：`hf auth login`（旧版 `huggingface-cli login`）或环境变量 `HF_TOKEN`；
   - 下载：`hf download {repo_id} --local-dir {root} --include '{分区}/*' --token $HF_TOKEN`
     （旧版 `huggingface-cli download --resume-download`）；
   - 网络受限时把镜像 endpoint 指向 hf-mirror；**分片缺失补下载**同样走 modelscope / hf-mirror
     两条（见 §6 与 `framework-integration/references/troubleshooting-vllm-omni.md`）。
3. **只取配置文件的场合**（KB 级，dummy run 用）：modelscope 离线下载后 `--config_cache` 指定路径，
   不需要 `HF_TOKEN`（见 dummy-run §A2）。

### 2.2 权重分区落位约定与落位表

- **落位约定（统一）**：`{model_weight_dir}/{模型名}/{任务变体}/` —— **模型根目录直接 serve**，
  任务变体（分区 / 精度档 / 对话与编辑变体）作为子目录；不在 `{模型名}` 之下再加厂商 / 组织层
  （同一模型在不同框架文档里出现两套路径会导致服务路径与排查口径分裂）。
- **落位表**（新增模型按同列补齐；**未实测的格子一律留空并在说明列标「未实测」，不得按命名习惯推测**）：

| 模型 | 目录落位（`{model_weight_dir}/…`） | 任务变体 / 分区 | modelscope 仓库 id | 来源 / 说明 |
|---|---|---|---|---|
| MiniMax-H3 | `MiniMax-H3/`（根目录 + `FL2VA/`、`Ref2VA/`） | t2va / fl2va → `FL2VA/**`；ref2va → `Ref2VA/**`；**根目录 = diffusers 格式给 dummy run** | `MiniMax/MiniMax-H3` | 实测日志（§2.3 / §7，81 文件；双分区全量约 270 GiB 量级） |
| Qwen-Image（基础版） | `Qwen-Image/` | 未实测（diffusers 布局，真实权重 60 层 / 1024²） | 未实测 | 目录落位见 `framework-integration/references/diffsynth-engine-enablement.md` §6；**仓库 id / 分区未实测** |
| Qwen-Image-2512 | `Qwen-Image-2512/` | 未实测（diffusers 布局，`QwenImagePipeline`） | 未实测 | 目录落位见 `framework-integration/references/vllm-omni-enablement.md` §6；**仓库 id / 分区未实测** |
| Qwen-Image-Edit-2511 | 未实测 | Edit / I2I（走 `/v1/images/edits`，multipart） | 未实测 | `framework-integration/SKILL.md`「Edit 类模型验证」示例路径按本节「落位约定」取 `{model_weight_dir}/Qwen-Image-Edit-2511`；**框架侧历史写法** `{model_weight_dir}/qwen/Qwen/Qwen-Image-Edit-2511`（多一层厂商/组织）与落位约定不符，勿照抄；**实际落位未实测** |
| Wan2.2 | 未实测 | 未实测 | 未实测 | 无真实权重落位记录（dummy run 用随机权重，见 dummy-run） |
| FLUX.1-dev | 未实测 | 未实测 | 未实测 | 同上（dummy run 用随机权重） |

- **回填纪律**：新增模型时三项同记——**仓库 id + 实际落位 + 任务变体**，依据列写来源
  （实测日志 / 框架文档 / 会话产物）；**只写实测过的**，其余留空并标「未实测」。

### 2.3 仓库布局与按任务选分区（MiniMax-H3 为例）

- **同一仓库可能混两种布局**（与 dummy-run minimax-h3-notes §1 一致）：
  - 仓库**根目录** = diffusers 格式（`model_index.json` 的 `_class_name:
    MiniMaxH3ModularPipeline`）→ dummy run 用
  - `FL2VA/`、`Ref2VA/` **子目录** = vLLM-Omni 格式（`MiniMaxH3DiTModel` 等）→ 部署用
  - 判定方法：看子目录 `model_index.json` 的类名 / `transformer/config.json` 键
    （`ffn_hidden_size`、`latents_dim` 等仅 vLLM-Omni 格式有）
- **按任务选分区**（MiniMax-H3）：
  - `t2va` / `fl2va` 任务 → **FL2VA 分区**（~134 GiB BF16：13 片 transformer + 14 片
    text_encoder + VAE 代码/配置，共 81 文件）
  - `ref2va` 任务 → `Ref2VA/**`
  - 双分区全量约 **270 GiB**，按需下载，避免一次性拉全

## 3. 下载命令（modelscope CLI）

```bash
# 容器内（modelscope 未预装时先装）
python -m pip install -U modelscope

# 按分区下载到模型根目录（MiniMax-H3 T2VA 示例）
modelscope download MiniMax/MiniMax-H3 \
  --local_dir {model_weight_dir}/MiniMax-H3 \
  --include 'FL2VA/**' --max-workers 16
```

参数要点：

- **`--local_dir`**：直接落盘布局（无 snapshot 哈希嵌套），`{root}/FL2VA/...`，模型根目录
  直接 serve；避免 `snapshot_download(cache_dir=...)` 的哈希目录（服务路径会变、难排查）。
- **`--include '{分区}/**'`**：只拉所需分区；不带则拉全仓库（含根目录 diffusers 大权重，
  徒增存储）。
- **`--max-workers 16`**：并发流数。实测单流为**数百 kB/s 量级**，16 并发聚合为**个位数 MB/s 量级**；
  初期 ramp-up 慢，以稳定段速率估算 ETA。

## 4. 后台化 + 监控（SSH 断开不中断）

```bash
# 容器内 nohup 后台（docker exec -d 使下载进程脱离 SSH 会话）
nohup modelscope download MiniMax/MiniMax-H3 \
  --local_dir {model_weight_dir}/MiniMax-H3 \
  --include 'FL2VA/**' --max-workers 16 \
  > {model_weight_dir}/h3_download.log 2>&1 &
echo $! > {model_weight_dir}/h3_download.pid
```

- **宿主侧轮询脚本**：每分钟记录 `du -sh {root}` + `pgrep -fc modelscope` 到日志；
  用户要求低频查询（如 30 分钟一次）时配合定时任务即可。
- ⚠️ **轮询脚本自身也要 nohup**：SSH 连接重置（`client_loop: send disconnect:
  Connection reset`）会杀掉未脱离会话的宿主进程；但容器内 nohup 的下载进程不受影响，
  无需重启下载。
- **完成判定**（三者齐备）：① modelscope 进程退出（`pgrep` 为 0）；②
  `find {root} -name '*.incomplete'` 为 0；③ 日志尾部出现 `Snapshot ready at {root}`。

## 5. 完整性校验

**权重确认纪律（先确认、再校验）**：下载前先确认远端是否已有该模型（存在则复用，不重复下载）；
下载后按下表逐项确认，**任一不通过不得当"权重就绪"**。

| 检查项 | 命令 / 依据 | 通过标准 |
|---|---|---|
| 目录大小 | `du -sh {root}` | 与仓库说明一致（FL2VA ≈ 134–135G） |
| 分区入口 | `ls {root}/FL2VA/model_index.json` | 存在（vLLM-Omni 用分区识别） |
| 残留未完成 | `find {root} -name '*.incomplete' \| wc -l` | 0 |
| 文件总数 | `find {root} -type f \| wc -l` | 与下载进度 "81/81" 一致 |
| 分片齐全 | `ls {root}/FL2VA/transformer/` 等 | `model-0000X-of-00013` 无缺号 |
| 校验和 | 仓库提供 `*.md5` / `*.sha256` / `checksums.json` 时逐文件核对 | 全部一致；**无校验和文件时以「无 `.incomplete` + 分片号齐全 + 文件总数一致」为准**（不自造校验和） |
| 下载日志 | `tail` 日志 | `100% ... 81/81` + `Snapshot ready` |

## 6. 已知坑

- **DNS/超时重试警告（易误报）**：容器 DNS 抖动时日志出现
  `urllib3.connectionpool: Retrying ... NameResolutionError / ReadTimeoutError`，
  modelscope 自动重试成功，**非致命**。`grep -i error` 会把 WARNING 行误报为错误——
  判断失败必须区分 WARNING（可忽略）与 ERROR/Traceback（才需处理）。
- **速率波动**：个别分片可低至**数百 kB/s 量级**、各流进度不均；聚合速率以 `du` 差分 /
  监控日志为准，不要用单条进度条估算 ETA。
- **`.incomplete` 后缀**：下载中的 safetensors 带 `.incomplete` 后缀，完成后自动去除；
  该后缀文件不计入最终文件数。
- **分片缺失**：框架启动报 `ValueError: ... weights were not initialized from checkpoint`
  时，对照 `*.safetensors.index.json` 的 `weight_map` 逐个核对分片，缺失的从
  hf-mirror / modelscope 补下载（framework-integration references/troubleshooting-vllm-omni.md
  有同款结论）。
- **模型仓库双格式混用**：vLLM-Omni 部署目录（如 `{root}/FL2VA`）**不能**当 dummy run 的
  `--config_cache`（类名/配置键不兼容，见 §2）。

## 7. 权重就绪后交接（服务路径语义）

- **本技能只管"权重就位"**：按 §2.2 落位约定落盘的模型根目录**可直接被三方框架 serve / 加载**
  （LightX2V、DiffSynth-Engine 用各自的 `--model_path` / `from_pretrained` 指向它即可；
  vLLM-Omni 的 pipeline 按任务档自动识别 `FL2VA/` / `Ref2VA/` 分区）。
- **服务启动与请求级档位不属本技能**（`vllm serve` 目标二选一、`--task-type`、
  `extra_params.task` 请求级切换、换档后的生效复核）→
  `framework-integration/references/run-entry-and-request-tiers.md` §1。
- 交接判据：§5 完整性校验全过 ⇒ 判"权重就位"；框架侧起服务后的验证归 `framework-integration`。

## 维护与更新

当 modelscope / HF CLI 参数变化、**§2.2 落位表中未实测的格子被实测补齐**、
或下载/校验流程有改进时，按 dev-workflow 的复盘流程更新本文件。
