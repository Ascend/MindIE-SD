# 三方框架模型权重准备（modelscope 下载到远端容器）

> **目录** · [1. 适用场景与结论速览](#1-适用场景与结论速览) · [2. 仓库与分区选择](#2-仓库与分区选择) ·
> [3. 下载命令（modelscope CLI）](#3-下载命令modelscope-cli) · [4. 后台化 + 监控](#4-后台化--监控ssh-断开不中断) ·
> [5. 完整性校验](#5-完整性校验) · [6. 已知坑](#6-已知坑) · [7. 服务路径](#7-服务路径) · [维护与更新](#维护与更新)
>
> 使用任何三方框架（vLLM-Omni / LightX2V / DiffSynth-Engine / diffusers）跑**真实权重**前的
> 前置步骤：把模型权重从 modelscope 下载到远端昇腾容器的模型根目录并校验完整。
> 依据：2026-08 实测 MiniMax-H3 T2VA（FL2VA 分区 ~134 GiB，16 并发，4h23m 完成，
> 81 文件全部校验通过）。本文件是**通用方法**，适用于所有三方框架；具体模型的分区/格式
> 差异按 §2 判定。

## 1. 适用场景与结论速览

- **何时需要**：三方框架（vLLM-Omni / LightX2V / DiffSynth-Engine / diffusers）需要真实权重
  时；dummy run（随机权重）不需要，见 dummy-run。
- **目录约定**：`{model_weight_dir}/{模型名}/`（如 `{model_weight_dir}/MiniMax-H3`），
  模型根目录直接 serve。
- **首选 modelscope**：HF gated 模型在 modelscope 镜像通常**免鉴权**；实测下载聚合速率
  ~9 MB/s（16 并发），单分区 134 GiB 约 4.5 小时。
- **核心命令**（详见 §3）：
  `modelscope download {模型} --local_dir {root} --include '{分区}/**' --max-workers 16`

## 2. 仓库与分区选择

- **HF gated → modelscope 免鉴权**：如 `MiniMaxAI/MiniMax-H3`（HF，需审批）↔
  `MiniMax/MiniMax-H3`（modelscope，直接下）。优先走 modelscope。
- **同一仓库可能混两种布局**（以 MiniMax-H3 为例，与 dummy-run minimax-h3-notes §1 一致）：
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
- **`--max-workers 16`**：并发流数。实测单流 400–900 kB/s，16 并发聚合 ~9 MB/s；
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

| 检查项 | 命令 / 依据 | 通过标准 |
|---|---|---|
| 目录大小 | `du -sh {root}` | 与仓库说明一致（FL2VA ≈ 134–135G） |
| 分区入口 | `ls {root}/FL2VA/model_index.json` | 存在（vLLM-Omni 用分区识别） |
| 残留未完成 | `find {root} -name '*.incomplete' \| wc -l` | 0 |
| 文件总数 | `find {root} -type f \| wc -l` | 与下载进度 "81/81" 一致 |
| 分片齐全 | `ls {root}/FL2VA/transformer/` 等 | `model-0000X-of-00013` 无缺号 |
| 下载日志 | `tail` 日志 | `100% ... 81/81` + `Snapshot ready` |

## 6. 已知坑

- **DNS/超时重试警告（易误报）**：容器 DNS 抖动时日志出现
  `urllib3.connectionpool: Retrying ... NameResolutionError / ReadTimeoutError`，
  modelscope 自动重试成功，**非致命**。`grep -i error` 会把 WARNING 行误报为错误——
  判断失败必须区分 WARNING（可忽略）与 ERROR/Traceback（才需处理）。
- **速率波动**：个别分片可低至 ~400 kB/s、各流进度不均；聚合速率以 `du` 差分 /
  监控日志为准，不要用单条进度条估算 ETA。
- **`.incomplete` 后缀**：下载中的 safetensors 带 `.incomplete` 后缀，完成后自动去除；
  该后缀文件不计入最终文件数。
- **分片缺失**：框架启动报 `ValueError: ... weights were not initialized from checkpoint`
  时，对照 `*.safetensors.index.json` 的 `weight_map` 逐个核对分片，缺失的从
  hf-mirror / modelscope 补下载（framework-feature-enablement references/troubleshooting-vllm-omni.md
  有同款结论）。
- **模型仓库双格式混用**：vLLM-Omni 部署目录（如 `{root}/FL2VA`）**不能**当 dummy run 的
  `--config_cache`（类名/配置键不兼容，见 §2）。

## 7. 服务路径

```bash
# vLLM-Omni：serve 模型根目录，pipeline 自动识别 FL2VA/Ref2VA 分区
vllm serve {model_weight_dir}/MiniMax-H3 --task-type t2va ...
# 或直接 serve 分区目录 {model_weight_dir}/MiniMax-H3/FL2VA
# 请求级任务切换：extra_params.task = "t2va" / "fl2va" / "ref2va"
```

- LightX2V / DiffSynth-Engine 等框架用各自的 `--model_path` / `from_pretrained` 指向
  同一模型根目录即可（目录布局对三方框架通用）。

## 维护与更新

当 modelscope CLI 参数变化、新模型出现新的分区/格式约定、或下载/校验流程有改进时，
按 dev-workflow 的复盘流程更新本文件。
