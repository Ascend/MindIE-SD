---
name: env-install
compatibility: paramiko（deploy_to_remote.py）；远端 SSH + Docker + CANN；本地 cmake/build/wheel；triton-ascend；950PR 源码构建全栈（torch/torch_npu/vllm/vllm-ascend/vllm-omni/LightX2V）；modelscope
description: >
  环境安装与准备：把部署环境从零安装就绪——mindiesd 编译安装（本地昇腾直装 / SSH 推远端容器 /
  Docker 镜像直装）与三方推理框架全栈安装（vLLM-Omni 950PR 源码构建、DiffSynth-Engine 部署、
  LightX2V editable 部署），并负责模型权重确认与下载（下载前先确认远端是否已存在）。不含特性使能与验证
  （framework-feature-enablement）与 profiling（profiling-collect）；SSH 工具由 remote-access 提供。
  当用户需要安装 MindIE-SD、源码构建/直装 vLLM-Omni 或 LightX2V（editable + PLATFORM=ascend_npu）、
  或确认/下载模型权重时使用此技能；
  即使用户只提到"把代码推到服务器""在容器里装 vllm 全栈""准备 lightx2v 调优环境"而未说昇腾，
  只要上下文涉及环境安装与准备都应触发。由 model-auto-optimization S0 与 dev-workflow
  部署阶段指引加载。
---

# 环境安装与准备

## 定位

本技能是能力层的「环境安装与准备」技能，覆盖两端职责：

- **安装部署环境**：mindiesd 编译安装（本地昇腾直装 / SSH 推远端容器）+ 三方推理框架全栈安装
  （vLLM-Omni 950PR 源码构建、DiffSynth-Engine 部署）。
- **模型权重确认/下载**：三方框架跑真实权重前，把权重从 modelscope 下载到远端容器并校验完整。

分工边界：

- **remote-access**：提供 SSH/SFTP 连接、传输与容器执行等远程工具；本技能决定「装什么、怎么装」，
  remote-access 决定「怎么连、怎么传」。本技能内的 deploy_to_remote.py 也可独立使用。
- **framework-feature-enablement**：特性使能与框架侧验证（服务启动、特性开关、推理验证）；
  本技能止于安装完成（import mindiesd 成功、版本配套就位、权重就位）。
- **dummy-run**：随机权重模型验证，不需要真实权重；本技能不为其下载权重。
- **profiling-collect**：性能 profiling 采集；本技能不含。

触发：当用户需要部署安装 MindIE-SD、在远端容器内装三方框架全栈或确认/下载模型权重时使用本技能；
即使用户只提到"把代码推到服务器""在容器里装 vllm 全栈"，只要上下文涉及环境安装与准备都应触发。
由 model-auto-optimization 的 S0 阶段调用，亦由 dev-workflow 的部署阶段指引加载。

## 安装路径总览

```text
你当前在哪里？
├─ 已在昇腾设备上 → 走「MindIE-SD 编译安装」本地直装路径
├─ 本地开发机，远端昇腾容器已就绪 → 「部署脚本」增量推送 + 容器内编译安装
├─ 已有可用容器/镜像，或官方预构建镜像覆盖目标芯片 → 「Docker 镜像直装」直接使用（容器内按需补装 mindiesd）
├─ 目标用 vLLM-Omni 托管扩散模型（Qwen-Image / Wan / MiniMax-H3，950PR）→ 「三方框架全栈安装」
├─ 目标跑 LightX2V（editable 源码 + mindiesd 同容器）→ 「三方框架全栈安装」→ LightX2V 部署要点
├─ 三方框架需要真实权重 → 「权重确认与下载」（先确认远端已存在，存在则跳过）
└─ 只需随机权重快速验证架构 → dummy-run（本技能不下载权重）
```

| 路径 | 场景 | 编译位置 |
| --- | --- | --- |
| 本地昇腾直装 | 已在昇腾设备上 | 本机执行 `python setup.py build_py && pip install -e .` |
| SSH 推远端容器 | 本地开发机 → 远端昇腾 Docker 容器 | `scripts/deploy_to_remote.py` 增量传输后在容器内编译 |
| Docker 镜像直装 | 官方预构建镜像覆盖目标芯片（如 Atlas A2/A3 aarch64），或已有可用容器/镜像 | 镜像内环境已就绪 → 免编译，容器内 `pip install mindiesd`；需自定义时才按需补装 |

三条路径共用同一份「MindIE-SD 编译安装」流程与「兼容性前置检查」；镜像直装路径在镜像内
CANN/torch/torch_npu 已就绪时免源码构建（仅补装 mindiesd），只有镜像内缺失基础环境或需改动
框架源码时才回到编译安装/源码构建；远端路径额外需要部署脚本与部署前参数确认（见「部署脚本」）。

### 远端前置（本地 → 远端路径）

部署前必须向用户确认以下信息（无默认值，禁止猜测）：

| 参数 | 说明 |
| --- | --- |
| 远端 IP / 用户名 / 密码 | SSH 登录信息（底层连接工具由 remote-access 提供） |
| 远端工作目录 | 代码存放路径 |
| Docker 镜像 | 已配置好 CANN+PyTorch 的镜像（从 GitCode 或官方文档验证最新版本） |
| 容器名 / 容器状态 | 已运行 / 需新建；容器内外路径映射是否一致 |

确认阻断点：镜像版本、容器配置等未经验证的参数必须由用户确认后方可执行。

> **远端**: {IP} / {容器名} / {工作目录}
> **镜像**: {镜像名}（从 GitCode 或官方文档验证最新版本）
> **NPU 可用卡数**: {空闲数} / {总数}（通过 `npu-smi info -l` 确认）
> **是否继续？** [Y/N]

## Docker 镜像直装路径

**适用场景**（镜像内环境已就绪时直接使用，或在容器内按需补装 mindiesd）：

- **官方预构建镜像覆盖目标芯片**（如 `quay.io/ascend/vllm-omni:*`，仅 Atlas A2/A3 aarch64）：
  镜像内 CANN/torch/torch_npu 与 vLLM-Omni 全栈已就绪，可直接使用镜像、免源码构建；缺 mindiesd 时补装。
- **Ascend 950PR / 950DT（x86_64）无官方全栈镜像**：可用 CANN 基础镜像（如 `cann:9.1.0-950-*`）起容器，
  再按本技能「MindIE-SD 编译安装」源码路径补装，或参考 `references/vllm-omni-build.md` 源码构建
  vLLM-Omni 全栈。

**`docker run` 直装要点**（容器创建时一次配好，避免事后反复 `docker cp`）：

```bash
# 方式一：Ascend Docker 运行时（自动注入设备），仍须显式挂载 HCCL ranktable 目录
docker run -it --rm --runtime=ascend \
  -v /usr/local/Ascend/driver/topo:/usr/local/Ascend/driver/topo \
  -v {model_weight_dir}:/workspace/model_weight \
  quay.io/ascend/vllm-omni:{tag}
```

- **无 Ascend Docker 运行时**：手动加 `--device=/dev/davinci0`、`--device=/dev/davinci_manager`、
  `--device=/dev/hisi_hdc`、`--device=/dev/devmm_svm` 与 `-v /usr/local/Ascend/driver:/usr/local/Ascend/driver`
  （topo / 权重挂载同上）。
- **必须挂载 HCCL ranktable 目录** `/usr/local/Ascend/driver/topo`（含 `950/atlas_350_*.json`）：
  缺失时多卡启动报 `hcclCommInitRootInfoConfig error code is 4`（见本文件「故障排查」表与
  `references/vllm-omni-build.md` Step 2.1）；已运行的容器可 `docker cp` 补挂
  `/usr/local/Ascend/driver/topo` 到 `{容器}:/usr/local/Ascend/driver/topo`。
- **同时挂载模型权重根目录** `{model_weight_dir}:/workspace/model_weight`：容器内外路径语义一致、服务直接
  serve；起容器前先按「权重确认与下载」确认权重是否已存在（存在则复用，同款原则）。
- **容器内先验证环境再补装**（避免驱动/设备未就绪时白跑安装），命令如下：

```bash
npu-smi info -l   # NPU 卡可见（驱动与设备挂载正确）
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"
```

镜像内已带 CANN/torch/torch_npu 时，无需源码构建基础环境，只需容器内补装 mindiesd：直接
`pip install mindiesd`，或按「MindIE-SD 编译安装」源码安装 `python setup.py build_py && pip install -e .`
（源码需挂载/拷入容器，注意「上传打包规则」勿排除源码树 `build/` 目录）。

**拉取注意事项**（镜像较大、多架构、内网/代理都会影响拉取，先确认再动手）：

- **多架构 manifest**：`quay.io/ascend/vllm-omni:*` 等多架构仓库默认按宿主架构拉取；x86_64 宿主上
  aarch64-only 镜像不可用时不要硬 `--platform`，改走 950PR 源码路径。
- **内网/代理**：先确认拉取源可达（registry mirror / 代理）；大镜像用 `docker pull --retry` 或后台拉取，
  避免 SSH/终端断连中断传输。
- **与本地已有镜像/容器复用**：起新容器前先确认是否已有可用镜像或容器（`docker images` / `docker ps -a`），
  能复用就不重拉/重装——与「权重确认与下载」先确认再下载是同一原则。

**直装 vs 源码构建（一句话取舍）**：A2/A3 有官方镜像时直装省时（免源码构建）；950PR/950DT（x86_64）
无官方全栈镜像、或需要改动 vllm / vllm-omni / mindiesd 框架源码时，走「三方框架全栈安装」源码构建路径。

## MindIE-SD 编译安装

本地直装与远端容器内编译的流程一致，以下步骤通用。编译前先 source CANN 环境。

### 兼容性前置检查

远端容器内验证环境版本兼容性：

```bash
# PyTorch + TorchNPU 版本匹配
docker exec {容器名} bash -lc 'python -c "
import torch, torch_npu
print(f\"PyTorch={torch.__version__}, torch_npu={torch_npu.__version__}\")
assert torch_npu.__version__ >= \"2.6\", \"torch_npu too old\"
"'

# CANN 环境
docker exec {容器名} bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    cat /usr/local/Ascend/ascend-toolkit/version.cfg 2>/dev/null | head -3'
```

本地路径直接用 `python -c` + `source set_env.sh` 验证。

| 检查项 | 最低要求 | 不满足时 |
| --- | --- | --- |
| PyTorch | >= 2.6 | 升级 PyTorch 版本 |
| TorchNPU | >= 2.6，与 PyTorch 主版本匹配 | 升级 TorchNPU |
| CANN | >= 9.0.0，含 bisheng 编译器 | 升级 CANN SDK |
| Python | >= 3.10 | 升级 Python |
| cmake / build / wheel | 可用 | `pip install cmake build wheel` |

> 版本不匹配时中止，参考「故障排查」表或 `references/troubleshooting-env.md`。

### 编译依赖（远端容器内最低编译条件）

| 依赖 | 要求 |
| --- | --- |
| CANN | >= 9.0.0，含 bisheng 编译器 |
| Python | >= 3.10 |
| PyTorch | 2.6 / 2.7 / 2.8 / 2.9 / 2.10（950PR 全栈用 2.11.0，见 `references/vllm-omni-build.md`「版本配套矩阵」） |
| TorchNPU | 与 PyTorch 版本匹配 |
| triton | 3.5.0（部署使用时需要） |
| triton-ascend | 3.2.1（部署使用时需要，Ascend 版 triton） |
| 环境变量 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| 编译工具 | cmake, build, wheel（`pip install build wheel`） |

### build_tik_ops.sh 规避

⚠️ `build_tik_ops.sh` 在部分环境会失败（参考 [issue#64](https://gitcode.com/Ascend/MindIE-SD/issues/64)），
部署前注释掉 `build/build_ops.sh` 中的对应行：

```bash
# 修改前
source ${current_script_dir}/build_tik_ops.sh
```

```bash
# 修改后
# source ${current_script_dir}/build_tik_ops.sh
```

容器内可用 sed 一键完成等效修改：

```bash
sed -i 's|^source ${current_script_dir}/build_tik_ops.sh|# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh
```

### 编译 + 安装

**远端（Docker 容器内）**：

```bash
docker exec {容器名} bash -lc '
source /usr/local/Ascend/ascend-toolkit/set_env.sh &&
cd {工作目录}/MindIE-SD &&
pip install build wheel -q &&
python setup.py build_py &&
pip install -e . &&
echo DEPLOY_SUCCESS
'
```

**本地（已在昇腾设备上）**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd {项目根目录}
pip install build wheel -q
python setup.py build_py
pip install -e .
```

### 上传打包规则

⚠️ 上传打包时**不要排除源码树中的 `build/` 目录**（含 build_ops.sh / build_plugin.sh 等脚本），
否则 `python setup.py build_py` 报 `No such file or directory: .../MindIE-SD/build`。
`deploy_to_remote.py` 的排除列表只过滤编译产物子目录（`build/build/`、`build/output/`、
`build/vendors/`、`build/custom_project_tik/`、`mindiesd/ops/`、`mindiesd/plugin/`、
`docs/_build/`）与 `.git`/`__pycache__`/`dist`/egg-info 等，不会排除源码 `build/` 本身。

### 验证安装

```bash
# 远端
docker exec {容器名} bash -lc 'python3 -c "import mindiesd; print(mindiesd.__version__)"'

# 本地
python -c "import mindiesd; print(mindiesd.__version__)"
```

### 编译原理与何时重装

`python setup.py build_py` 执行以下步骤：

1. 通过 `build/build_ops.sh` 编译 AscendC 自定义算子（laser_attention, la_preprocess 等）
2. 通过 `build/build_plugin.sh` 用 cmake 编译 C++ PyTorch 插件，生成 `.so` 文件
3. 将编译产物拷贝到 `mindiesd/plugin/` 和 `mindiesd/ops/`

`pip install -e .` 以可编辑模式安装，使代码修改即时生效。

> `pip install -e .` 仅在以下目录有新增或变更时才需要重新执行：
>
> - `mindiesd/` — Python 包源码索引需刷新
> - `csrc/` — C++ 源码需重新编译为 `.so`
> - `build/` — 编译脚本变更
>
> 若变更仅涉及 `examples/`、`tests/`、`docs/` 等非包目录，可跳过此步骤，
> 直接使用远端已有的安装版本。

## 三方框架全栈安装（vLLM-Omni 950PR 源码构建 + DiffSynth-Engine + LightX2V）

当目标是在远端容器内用 **vLLM-Omni 托管扩散模型**（Qwen-Image-2512 / Wan2.2 / MiniMax-H3 等），
需安装完整栈 `torch + torch_npu + vllm + vllm-ascend + vllm-omni + mindiesd`。⚠️ 官方预构建镜像
（`quay.io/ascend/vllm-omni:*`）仅覆盖 Atlas A2/A3（aarch64），**Ascend 950PR / 950DT（x86_64）
必须从源码构建**。示例（950PR + vllm 0.26.0，2026-08 实测）：torch 2.11.0+cpu → torch_npu
2.11.0 → vllm 0.26.0 源码构建 → vllm-ascend（releases/v0.26.0rc）→ vllm-omni（main）→ mindiesd（dev）。

完整构建流程（版本配套矩阵、Step 2.1–2.6 分步命令、内联已知坑与 ⚠️ 警告，含 setuptools 升级 /
rust 跳过 / numpy 阿里云镜像等）见 `references/vllm-omni-build.md`。**加载时机**：需要源码构建
vLLM-Omni 全栈（950PR/950DT）或排障 vllm / vllm-ascend / vllm-omni 构建问题时，加载该 reference 执行。

### DiffSynth-Engine 部署要点

`diffsynth_engine`（Qwen-Image 等扩散模型的独立推理框架）接入 MindIE-SD compile 前，
先把框架包部署到远端容器：

- 纯 Python 包、无需编译：无 `setup.py build_py` / CANN 构建步骤，增量传输源码后
  `pip install -e . --no-deps` 即可（安装要点见 deploy_to_remote.py 的传输姿势）
- ⚠️ `--no-deps` 的原因：pyproject 锁旧版 `transformers==4.57.6` / `diffusers==0.36.0`，
  而容器内是较新版本（vllm-omni / mindiesd 依赖）——带依赖安装会把容器环境降级破坏；
  装完用框架自有的 import 兼容检查确认可导入（如 Qwen2_5_VL / QwenImagePipelineOutput 等）
- 无 `.git` 时 setuptools-scm 报错 →
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DIFFSYNTH_ENGINE=1.0.0`
- mindiesd 用 compile 工作区：脚本内 `sys.path.insert(0, "{工作区}/mindie-sd-compile")`
  （避免用 pip 替换容器内已装 mindiesd，替换会波及共享该容器镜像的其他框架依赖）
- 真实权重就绪：DiffSynth-Engine 的 Qwen-Image 用 diffusers 布局权重目录
  （transformer/vae/text_encoder/tokenizer/scheduler + model_index.json），确认见 weights-prep

> compile 适配（`compile_backend="mindie"`、`_compiled_call_impl` 写入、text encoder key
> 归一化等）与融合算子使能判断属 framework-feature-enablement，不在本技能范围。

### LightX2V 部署要点

LightX2V 与 vLLM-Omni 形态不同：**editable 源码 + `PLATFORM=ascend_npu` 环境变量**，无独立服务进程：

- lightx2v 与 mindiesd 都在同一容器内 `pip install -e`（源码目录）；更新代码（rsync 新文件）后
  无需重装、重启进程即生效
- ⚠️ **`import lightx2v` 前必须 `export PLATFORM=ascend_npu`**：否则设备初始化按默认平台走，
  报 `ERR99999 UNKNOWN application exception` 类异常
- 版本配套实测矩阵（python 3.12 / torch 2.11 / torch_npu 2.11 / CANN 9.1）、就绪验证、
  运行入口、MiniMax-H3 权重分区（t2av 不需要 FL2VA 135G）：见 `references/lightx2v-env.md`

> LightX2V 侧 compile 使能（`compile_backend` / `hccl_eager`）与 kernel diff 方法属
> framework-feature-enablement，不在本技能范围。

## 权重确认与下载

⚠️ **部署/准备时，先与用户确认远端环境是否已存在该模型权重或可复用服务**（例如
`{model_weight_dir}/{模型名}/` 已完整、或该模型已在别的容器/服务中 serve）。
存在则跳过下载，直接复用路径；只有确认缺失才执行下载。

完整流程见 `references/weights-prep.md`，速览：

- **何时需要**：三方框架（vLLM-Omni / LightX2V / DiffSynth-Engine / diffusers）需要真实权重时；
  dummy run（随机权重）不需要，见 dummy-run。
- **目录约定**：`{model_weight_dir}/{模型名}/`（如 `{model_weight_dir}/MiniMax-H3`），
  模型根目录直接 serve；仓库内 `FL2VA/`、`Ref2VA/` 等子目录 = vLLM-Omni 格式，按任务分区下载。
  ⚠️ LightX2V t2av 运行不需要 `FL2VA`（135G，其他任务组件），其实际分区口径见
  `references/lightx2v-env.md` §5。
- **首选 modelscope**：HF gated 模型在 modelscope 镜像通常**免鉴权**；实测下载聚合速率
  ~9 MB/s（16 并发），单分区 134 GiB 约 4.5 小时。

核心命令（MiniMax-H3 T2VA 示例）：

```bash
# 容器内（modelscope 未预装时先装）
python -m pip install -U modelscope

# 按分区下载到模型根目录
modelscope download MiniMax/MiniMax-H3 \
  --local_dir {model_weight_dir}/MiniMax-H3 \
  --include 'FL2VA/**' --max-workers 16
```

- `--local_dir`：直接落盘布局（无 snapshot 哈希嵌套），模型根目录直接 serve，服务路径稳定。
- `--include '{分区}/**'`：只拉所需分区；不带则拉全仓库，徒增存储。
- `--max-workers 16`：并发流数；初期 ramp-up 慢，以稳定段速率估算 ETA。

后台化 + 监控（SSH 断开不中断）：

```bash
# 容器内 nohup 后台（docker exec -d 使下载进程脱离 SSH 会话）
nohup modelscope download MiniMax/MiniMax-H3 \
  --local_dir {model_weight_dir}/MiniMax-H3 \
  --include 'FL2VA/**' --max-workers 16 \
  > {model_weight_dir}/h3_download.log 2>&1 &
echo $! > {model_weight_dir}/h3_download.pid
```

⚠️ **轮询脚本自身也要 nohup**：SSH 连接重置（`client_loop: send disconnect: Connection reset`）
会杀掉未脱离会话的宿主进程；但容器内 nohup 的下载进程不受影响，无需重启下载。

**完成判定**（三者齐备）：① modelscope 进程退出（`pgrep` 为 0）；②
`find {root} -name '*.incomplete'` 为 0；③ 日志尾部出现 `Snapshot ready at {root}`。

完整性校验（详见 `references/weights-prep.md` §5）：

| 检查项 | 命令 / 依据 | 通过标准 |
| --- | --- | --- |
| 目录大小 | `du -sh {root}` | 与仓库说明一致（FL2VA ≈ 134–135G） |
| 分区入口 | `ls {root}/FL2VA/model_index.json` | 存在（vLLM-Omni 用分区识别） |
| 残留未完成 | `find {root} -name '*.incomplete' \| wc -l` | 0 |
| 文件总数 | `find {root} -type f \| wc -l` | 与下载进度 "81/81" 一致 |
| 下载日志 | `tail` 日志 | `100% ... 81/81` + `Snapshot ready` |

已知坑：

- **DNS/超时重试警告（易误报）**：容器 DNS 抖动时日志出现
  `urllib3.connectionpool: Retrying ... NameResolutionError / ReadTimeoutError`，
  modelscope 自动重试成功，**非致命**。`grep -i error` 会把 WARNING 行误报为错误——
  判断失败必须区分 WARNING（可忽略）与 ERROR/Traceback（才需处理）。
- **`.incomplete` 后缀**：下载中的 safetensors 带 `.incomplete` 后缀，完成后自动去除；
  该后缀文件不计入最终文件数。
- **分片缺失**：框架启动报 `ValueError: ... weights were not initialized from checkpoint` 时，
  对照 `*.safetensors.index.json` 的 `weight_map` 逐个核对分片，缺失的从 hf-mirror / modelscope
  补下载。
- **模型仓库双格式混用**：vLLM-Omni 部署目录（如 `{root}/FL2VA`）**不能**当 dummy run 的
  `--config_cache`（类名/配置键不兼容）。

服务侧直接用模型根目录 serve（`vllm serve {root} --task-type t2va ...` 或 serve 分区目录；
启动细节见 framework-feature-enablement）。

## 部署脚本

`scripts/deploy_to_remote.py` 是本技能的部署主脚本（本地开发机 → 远端昇腾容器），
自带 paramiko 连接；底层 SSH/SFTP 连接复用等通用工具由 remote-access 提供。

命令行参数（均为必填）：

| 参数 | 说明 |
| --- | --- |
| `--host` | 远端服务器 IP |
| `--user` | SSH 登录用户名 |
| `--password` | SSH 登录密码 |
| `--workspace` | 远端工作目录 |
| `--container` | 远端容器名 |
| `--local-root` | 本地源码根目录 |

执行传输 + 编译：

```bash
python deploy_to_remote.py --host <远端IP> --user {用户名} --password {密码} \
  --workspace {远端工作目录} --container {容器名} \
  --local-root {本地源码根目录}
```

脚本自动完成：

1. **收集本地文件**：按排除规则过滤（`.git`、`__pycache__`、`dist`、egg-info、
   `build/` 下编译产物子目录、`mindiesd/ops/`、`mindiesd/plugin/`、`docs/_build/` 等）
2. **CRLF 换行符处理**：文本文件（`.py`/`.sh`/`.json`/`.txt`/`.md`/`.yaml`/`.yml`/`.cfg`）
   传输时 CRLF→LF，转换仅作用于远端副本，**不在本地修改源码换行符**（避免无意义 git diff）
3. **增量传输**：远端已存在且大小相同的文件跳过；缺失目录自动 `mkdir -p`
4. **容器内编译**：`docker exec {容器名} bash -lc 'cd {workspace} && source set_env.sh &&
   cd MindIE-SD && pip install build wheel -q && python setup.py build_py && pip install -e . &&
   echo DEPLOY_SUCCESS'`

完成后检查输出中的 `DEPLOY_SUCCESS`。

> ⚠️ **路径语义核对**：脚本把文件传到 `{workspace}/{local_root.name}`
> （如 `/home/{user}/code/MindIE-SD_compile`），但构建命令固定 `cd {workspace}/MindIE-SD`。
> 若远端实际布局与本仓约定（如 `/home/{user}/code/mindie-sd-compile`）不一致，不要直接用该脚本，
> 改精准 SFTP 同步；另外 `refs/` 等大目录不在脚本排除列表，全量传输会带上 profiling 数据。
> ⚠️ 传输完成前不要独立多次开 SSH 连接（远端 `MaxStartups` 限制会拒绝访问），
> 遵循连接复用原则（单个连接从传输到验证全程复用）。

## 故障排查

高危问题速查（完整决策树见 `references/troubleshooting-env.md`）：

| 症状 | 原因 | 解决 |
| --- | --- | --- |
| 编译失败 | CANN 环境未 source | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| `build_ops.sh` exit code 101 | build_tik_ops.sh 失败 | 注释掉 build_ops.sh 中 `source build_tik_ops.sh` 行 |
| `import triton` 成功但 `0 active drivers` | 安装了标准 triton（非 Ascend 版本） | `pip uninstall triton -y && pip install triton-ascend && pip install pybind11` |
| `ModuleNotFoundError: mindiesd` | `pip install -e .` 未重新索引 | 重新执行 `python setup.py build_py && pip install -e .` |
| git/curl HTTPS 报 `SEC_E_NO_CREDENTIALS`（Windows 开发机） | git 默认 schannel TLS 后端在受限进程里握手失败 | `git -c http.sslBackend=openssl fetch`（可全局 `git config --global http.sslBackend openssl`） |
| vllm 多卡启动报 `hcclCommInitRootInfoConfig error code is 4` | 容器缺 HCCL ranktable | `docker cp /usr/local/Ascend/driver/topo {容器}:/usr/local/Ascend/driver/topo` 或挂载（见 `references/vllm-omni-build.md` Step 2.1） |

> 其他问题（SSH 认证、docker exec 引号转义、CRLF、环境依赖、运行时 OOM、NPU 崩溃、输出异常等）
> 见 `references/troubleshooting-env.md`；vLLM-Omni 全栈构建期问题另见
> framework-feature-enablement 的 `references/troubleshooting-vllm-omni.md`。

## Reference Files

- `references/weights-prep.md` — 加载时机: 三方框架需要真实权重，把模型权重从 modelscope
  下载到远端容器并校验完整时（分区选择、nohup 后台化、完整性校验、已知坑）
- `references/lightx2v-env.md` — 加载时机: 部署/复现 LightX2V 调优环境（editable 安装、
  PLATFORM=ascend_npu、版本配套矩阵、运行入口、MiniMax-H3 t2av 权重分区）时
- `references/troubleshooting-env.md` — 加载时机: 部署/编译/安装遇到异常，需系统排查定位根因时
- `references/vllm-omni-build.md` — 加载时机: 需要源码构建 vLLM-Omni 全栈（950PR/950DT）或排障 vllm / vllm-ascend / vllm-omni 构建问题时（版本配套矩阵、Step 2.1–2.6 全量细节）

## 维护与更新

当远端昇腾环境变化（torch/TorchNPU 版本升级、CANN SDK 更新）、vllm/vllm-ascend/vllm-omni
版本矩阵变化、950PR 构建路径调整、modelscope 下载流程或容器配置调整、或发现新的安装/部署问题时，
按 dev-workflow 的复盘流程更新本 skill。
